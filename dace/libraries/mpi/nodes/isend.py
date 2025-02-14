# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import dtypes


@dace.library.expansion
class ExpandIsendMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), dest, tag, req = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        code = ""
        if ddt is not None:
            code = f"""static MPI_Datatype newtype;
                        static int init=1;
                        if (init) {{
                           MPI_Type_vector({ddt['count']}, {ddt['blocklen']}, {ddt['stride']}, {ddt['oldtype']}, &newtype);
                           MPI_Type_commit(&newtype);
                           init=0;
                        }}
                            """
            mpi_dtype_str = "newtype"
            count_str = "1"
        buffer_offset = 0
        code += f"MPI_Isend(&(_buffer[{buffer_offset}]), {count_str}, {mpi_dtype_str}, _dest, _tag, MPI_COMM_WORLD, _request);"
        if ddt is not None:
            code += f"""// MPI_Type_free(&newtype);
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class Isend(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandIsendMPI,
    }
    default_implementation = "MPI"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_buffer", "_dest", "_tag"}, outputs={"_request"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: buffer, count, mpi_dtype, req of the input data
        """

        buffer, dest, tag, req = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_request":
                req = sdfg.arrays[e.data.data]

        if dest.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Source must be an integer!")
        if tag.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Tag must be an integer!")

        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)
                # compute buffer offset
                minelem = data.subset.min_element()
                dims_data = sdfg.arrays[data.data].strides
                buffer_offsets = []
                for idx, m in enumerate(minelem):
                    buffer_offsets += [(str(m) + "*" + str(dims_data[idx]))]
                buffer_offset = "+".join(buffer_offsets)

                # create a ddt which describes the buffer layout IFF the sent data is not contiguous
                ddt = None
                if dace.libraries.mpi.utils.is_access_contiguous(data, sdfg.arrays[data.data]):
                    pass
                else:
                    ddt = dace.libraries.mpi.utils.create_vector_ddt(data, sdfg.arrays[data.data])
        return (buffer, count_str, buffer_offset, ddt), dest, tag, req

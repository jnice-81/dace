from dace.sdfg.state import SDFGState
from dace import dtypes, subsets, memlet
from dace.transformation import dataflow
from dace.transformation.dataflow import hbm_copy_transform
from typing import Iterable
from dace.sdfg import utils
from dace.sdfg.sdfg import SDFG
import dace
import dace.libraries.blas.nodes as blas
import numpy as np
import dace.sdfg.nodes as nd
from dace.transformation import optimizer

################
# Helpers

def expandlibnode(sdfg: dace.SDFG, state: dace.SDFGState, 
    node: nd.LibraryNode, impl: str, dryExpand: bool = False, *args, **kwargs):
    node.implementation = impl
    if not dryExpand:
        node.expand(sdfg, state, *args, **kwargs)

def expand_first_libnode(sdfg: dace.SDFG, impl: str, dryExpand: bool=False,
    *args, **kwargs):
    for node, state in sdfg.all_nodes_recursive():
        if(isinstance(node, nd.LibraryNode)):
            expandlibnode(state.parent, state, node, impl, dryExpand, *args, **kwargs)
            return

def create_hbm_access(state: SDFGState, name, locstr, shape, lib_node,
    lib_conn, is_write, mem_str, dtype=dace.float32):
    state.parent.add_array(name, shape, dtype)
    state.parent.arrays[name].location["bank"] = locstr
    access = state.add_access(name)
    if is_write:
        state.add_edge(lib_node, lib_conn, access, None,
            memlet.Memlet(mem_str))
    else:
        state.add_edge(access, None, lib_node, lib_conn, 
            memlet.Memlet(mem_str))

def random_array(size, type = np.float32, fix_constant = None):
    if not isinstance(size, Iterable):
        size = (size,)
    if fix_constant is None:
        a = np.random.rand(*size)
        a = a.astype(type)
    else:
        a = np.ones(size, type) * fix_constant
    return a

def create_or_load(load_from, create_method, compile=True):
    if load_from is None:
        sdfg = create_method()
        if compile:
            sdfg.compile()
    else:
        sdfg = utils.load_precompiled_sdfg(load_from)
    return sdfg

################
# Execution methods

def exec_dot_hbm(data_size_per_bank: int, banks_per_input: int, load_from=None):
    def create_dot_sdfg():
        N = dace.symbol("N")

        sdfg = SDFG("hbm_dot")
        state = sdfg.add_state("sdot")
        dot_node = blas.Dot("sdot_node")
        dot_node.implementation = "FPGA_HBM_PartialSums"
        create_hbm_access(state, "in1", f"hbm.0:{banks_per_input}", 
            [banks_per_input, N], dot_node, "_x", False, "in1")
        create_hbm_access(state, "in2", f"hbm.{banks_per_input}:{2*banks_per_input}",
            [banks_per_input, N], dot_node, "_y", False, "in2")
        create_hbm_access(state, "out", "ddr.0", [1],
            dot_node, "_result", True, "out", dace.float32)
        dot_node.expand(sdfg, state, partial_width=16)

        sdfg.apply_fpga_transformations(False, validate=False)

        utils.update_array_shape(sdfg, "in1", [banks_per_input*N])
        utils.update_array_shape(sdfg, "in2", [banks_per_input*N])
        sdfg.arrays["in1"].storage = dtypes.StorageType.CPU_Heap
        sdfg.arrays["in2"].storage = dtypes.StorageType.CPU_Heap
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            xform.apply(sdfg)

        return sdfg

    sdfg = create_or_load(load_from, create_dot_sdfg)
    
    x = random_array(data_size_per_bank*banks_per_input, fix_constant=1)
    y = random_array(data_size_per_bank*banks_per_input, fix_constant=1)
    result = np.zeros(1, dtype=np.float32)
    check = np.dot(x, y)
    sdfg(in1=x, in2=y, out=result, N=data_size_per_bank)
    print(check)
    print(result)
    assert np.allclose(result, check)

def exec_axpy(data_size_per_bank: int, banks_per_array: int, load_from=None):
    N = dace.symbol("N")

    def create_axpy_sdfg():
        sdfg = SDFG("hbm_axpy")
        state = sdfg.add_state("axpy")
        axpy_node = blas.Axpy("saxpy_node")
        axpy_node.implementation = "fpga_hbm"
        create_hbm_access(state, "in1", f"hbm.0:{banks_per_array}", 
            [banks_per_array, N], axpy_node, "_x", False, "in1")
        create_hbm_access(state, "in2", f"hbm.{banks_per_array}:{2*banks_per_array}",
            [banks_per_array, N], axpy_node, "_y", False, "in2")
        create_hbm_access(state, "out", f"hbm.{2*banks_per_array}:{3*banks_per_array}",
            [banks_per_array, N], axpy_node, "_res", True, "out")
        axpy_node.expand(sdfg, state)

        sdfg.apply_fpga_transformations(False)
        utils.update_array_shape(sdfg, "in1", [banks_per_array*N])
        utils.update_array_shape(sdfg, "in2", [banks_per_array*N])
        utils.update_array_shape(sdfg, "out", [banks_per_array*N])
        for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
            patterns=[hbm_copy_transform.HbmCopyTransform]):
            xform.apply(sdfg)

        return sdfg

    sdfg = create_or_load(load_from, create_axpy_sdfg, False)
    sdfg.view()


def createGemv(target : str = None):
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def sgemvtest(A : dace.float32[M, N], x : dace.float32[N], y : dace.float32[M]):
        y[:] = A @ x

    sdfg = sgemvtest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA_TilesByColumn")
    sdfg.compile()

def createGemm(target : str = None):
    N = dace.symbol("N")
    M = dace.symbol("M")
    
    @dace.program
    def sgemmtest(A : dace.float32[M, N], B : dace.float32[N, 100], C : dace.float32[M, 100]):
        C[:] = A @ B

    sdfg = sgemmtest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA1DSystolic")
    sdfg.compile()
    
#exec_dot_hbm(1000, 16)
#sdfg = utils.load_precompiled_sdfg("mycompiledstuff")
exec_axpy(10, 2)
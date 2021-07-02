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

def random_array(size, type = np.float32):
    if not isinstance(size, Iterable):
        size = (size,)
    #a = np.random.rand(*size)
    #a = a.astype(type)
    a = np.ones(size, type)
    return a

def createDot(target : str = None):
    N = dace.symbol("N")

    sdfg = SDFG("hbm_dot")
    state = sdfg.add_state("sdot")
    dot_node = blas.Dot("sdot_node")
    dot_node.implementation = "FPGA_HBM_PartialSums"
    create_hbm_access(state, "in1", "hbm.0:2", [2, N], 
        dot_node, "_x", False, "in1")
    create_hbm_access(state, "in2", "hbm.2:4", [2, N],
        dot_node, "_y", False, "in2")
    create_hbm_access(state, "out", "ddr.0", [1],
        dot_node, "_result", True, "out")
    dot_node.expand(sdfg, state, partial_width=16)

    sdfg.apply_fpga_transformations(False, validate=False)

    utils.update_array_shape(sdfg, "in1", [2*N])
    utils.update_array_shape(sdfg, "in2", [2*N])
    sdfg.arrays["in1"].storage = dtypes.StorageType.CPU_Heap
    sdfg.arrays["in2"].storage = dtypes.StorageType.CPU_Heap
    for xform in optimizer.Optimizer(sdfg).get_pattern_matches(
        patterns=[hbm_copy_transform.HbmCopyTransform]):
        xform.apply(sdfg)
    
    sdfg.view()
    sdfg.compile(target)
    return sdfg

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

def runDot(csdfg : dace.SDFG, data_size: int):
    x = random_array(data_size)
    y = random_array(data_size)
    result = random_array(1)
    check = np.dot(x, y)
    csdfg(in1=x, in2=y, out=result, N=data_size)
    print(result)
    print(check)
    assert np.allclose(result, check)

csdfg = createDot("mycompiledstuff/")
runDot(csdfg, 300)
#sdfg = utils.load_precompiled_sdfg("mycompiledstuff")
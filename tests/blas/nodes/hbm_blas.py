from dace.sdfg import utils
from dace.sdfg.sdfg import SDFG
import dace
import dace.libraries.blas.nodes as blas
import numpy as np
import dace.sdfg.nodes as nd

def expandlibnode(sdfg : dace.SDFG, state : dace.SDFGState, 
    node : nd.LibraryNode, impl : str, *args, **kwargs):
    node.implementation = impl
    node.expand(sdfg, state, *args, **kwargs)

def expand_first_libnode(sdfg : dace.SDFG, impl : str, *args, **kwargs):
    for node, state in sdfg.all_nodes_recursive():
        if(isinstance(node, nd.LibraryNode)):
            expandlibnode(state.parent, state, node, impl, *args, **kwargs)
            return

def dotTest(target : str = None):
    N = dace.symbol("N")

    @dace.program
    def sdottest(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[1]):
        np.dot(in1, in2, out)

    sdfg = sdottest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "FPGA_PartialSums")
    sdfg.view()
    sdfg.compile()

def gemvTest(target : str = None):
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def sgemvtest(A : dace.float32[M, N], x : dace.float32[N], y : dace.float32[M]):
        y[:] = A @ x

    sdfg = sgemvtest.to_sdfg()
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA_Accumulate")
    sdfg.apply_fpga_transformations(False)
    sdfg.apply_strict_transformations()
    #sdfg.view()
    sdfg.compile()

def gemmTest(target : str = None):
    N = dace.symbol("N")
    M = dace.symbol("M")
    

    @dace.program
    def sgemmtest(A : dace.float32[M, N], B : dace.float32[N, 100], C : dace.float32[M, 100]):
        C[:] = A @ B

    sdfg = sgemmtest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA1DSystolic")
  # sdfg.view()
    sdfg.compile()



dotTest()
#gemvTest()
#gemmTest()
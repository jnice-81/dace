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
    for state in sdfg.nodes():
        for node in state.nodes():
            if(isinstance(node, nd.LibraryNode)):
                expandlibnode(sdfg, state, node, impl, *args, **kwargs)

def dotTest():
    N = dace.symbol("N")

    @dace.program
    def sdottest(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[1]):
        np.dot(in1, in2, out)

    sdfg = sdottest.to_sdfg()
    expand_first_libnode(sdfg, "FPGA_PartialSums")
    sdfg.apply_fpga_transformations()
    #sdfg.view()
    sdfg.compile()

def gemvTest():
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def sgemvtest(A : dace.float32[M, N], x : dace.float32[N], y : dace.float32[M]):
        y[:] = A @ x

    sdfg = sgemvtest.to_sdfg()
    expand_first_libnode(sdfg, "specialize")
    expand_first_libnode(sdfg, "FPGA_Accumulate")
    sdfg.view()


dotTest()
#gemvTest()
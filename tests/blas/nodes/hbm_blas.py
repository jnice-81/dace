from typing import Iterable
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

def random_array(size, type = np.float32):
    if not isinstance(size, Iterable):
        size = (size,)
    a = np.random.rand(*size)
    a = a.astype(type)
    print(a.dtype)
    return a

def createDot(target : str = None):
    N = dace.symbol("N")

    @dace.program
    def sdottest(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[1]):
        np.dot(in1, in2, out)

    sdfg = sdottest.to_sdfg()
    sdfg.apply_fpga_transformations(False)
    expand_first_libnode(sdfg, "FPGA_HBM_PartialSums")
    #sdfg.view()
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
    expand_first_libnode(sdfg, "FPGA_Accumulate")
    sdfg.view()
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
  # sdfg.view()
    sdfg.compile()

def runDot(csdfg : dace.SDFG, datasize):
    x = random_array(datasize)
    y = random_array(datasize)
    result = random_array(1)
    check = np.dot(x, y)
    csdfg(in1=x, in2=y, out=result, N=datasize)
    assert np.allclose(result, check)


sdfg = createDot("mycompiledstuff/")
#sdfg = utils.load_precompiled_sdfg("mycompiledstuff")
runDot(sdfg, 100)
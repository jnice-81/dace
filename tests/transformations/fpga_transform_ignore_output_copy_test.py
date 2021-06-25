from dace.sdfg.sdfg import SDFG
import dace
import dace.libraries.blas.nodes as blas
import numpy as np
import dace.sdfg.nodes as nd

def assert_copy_to_device(sdfg, arrayname, assertexists):
    count_copy = 0
    for state in sdfg.states():
        if state.label.startswith("pre_"):
            for node in state.nodes():
                if isinstance(node, nd.AccessNode):
                    if node.data == arrayname:
                        count_copy += 1
    assert(count_copy > 0 or not assertexists)
    assert(count_copy == 0 or assertexists)

def test_ignore_one_output():
    N = dace.symbol("N")

    @dace.program
    def saddtest(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[N]):
        out[:] = in1[:] + in2[:]

    sdfg = saddtest.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_fpga_transformations(False)
    assert_copy_to_device(sdfg, "out", False)
                        
def test_copy_wcr_output():
    N = dace.symbol("N")

    @dace.program
    def sdottest(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[1]):
        np.dot(in1, in2, out)

    sdfg = sdottest.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_fpga_transformations(False)
    assert_copy_to_device(sdfg, "out", True)

def test_copy_input_output():
    N = dace.symbol("N")

    @dace.program
    def sdottest(inout : dace.float32[N]):
        inout[:] = inout[:] + 15

    sdfg = sdottest.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_fpga_transformations(False)
    assert_copy_to_device(sdfg, "inout", True)

if __name__ == "__main__":
    test_ignore_one_output()
    test_copy_wcr_output()
    test_copy_input_output()
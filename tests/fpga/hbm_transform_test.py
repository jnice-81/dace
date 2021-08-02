# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.codegen.targets import fpga
from typing import List, Tuple, Union
from dace.sdfg import SDFG
import dace
from dace.transformation.dataflow import HbmTransform

def set_assignment(sdfg: SDFG, assignments: List[Tuple[str, str, str]]):
    for array, memorytype, bank in assignments:
        desc = sdfg.arrays[array]
        desc.location["memorytype"] = memorytype
        desc.location["bank"] = bank

def check_assignment(sdfg: SDFG, assignments: List[Union[Tuple[str, int], Tuple[str, str, str]]]):
    for val in assignments:
        if len(val) == 3:
            array, memorytype, bank = val
            assert sdfg.arrays[array].location["memorytype"] == memorytype
            assert sdfg.arrays[array].location["bank"] == bank
        else:
            array, banks = val
            if sdfg.arrays[array].location["memorytype"] == "HBM":
                low, high = fpga.get_multibank_ranges_from_subset(sdfg.arrays[array].location["bank"], sdfg)
            else:
                low, high = (0, 1)
            assert banks == high - low
            
def _exec_test(sdfgsource, assign, checkassign):
    sdfg = sdfgsource()
    set_assignment(sdfg, assign)
    sdfg.apply_transformations(HbmTransform, validate=False)
    #sdfg.view()
    #xform = HbmTransform(sdfg.sdfg_id, -1, {}, -1)
    #xform.apply(sdfg)
    #sdfg.view()
    check_assignment(sdfg, checkassign)
    sdfg.validate()
    assert not HbmTransform.can_be_applied(sdfg, {}, -1, sdfg, False)


def create_axpy_sdfg(array_shape=dace.symbol("n"), map_range=dace.symbol("n")):
    @dace.program
    def axpy(x: dace.float32[array_shape], y: dace.float32[array_shape]):
        for i in dace.map[0:map_range]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                yout >> y[i]
                yout = xin + yin
    return axpy.to_sdfg()

def create_nd_sdfg():
    n = dace.symbol("n")
    m = dace.symbol("m")
    @dace.program
    def nd_sdfg(x: dace.float32[n, m], y: dace.float32[m, n], z: dace.float32[n, m]):
        for i in dace.map[0:n]:
            for j in dace.map[0:m]:
                with dace.tasklet:
                    yin << y[j, i]
                    xin << x[i, j]
                    zout >> z[i, j]
                    xout = yin + xin
    return nd_sdfg.to_sdfg()

def create_not_splitable_dependence_sdfg():
    # All arrays would be splitable in principle, if they were alone, but because of
    # the dependencies on the same map variables, it's not possible to split each in just one dimension
    n = dace.symbol("n")
    @dace.program
    def no_split_sdfg(x: dace.float32[n, n], y: dace.float32[n, n], z: dace.float32[n+1, n]):
        for i in dace.map[0:n]:
            for j in dace.map[0:n]:
                with dace.tasklet:
                    yin << y[j, i]
                    xin << x[i, j]
                    zout >> z[i, j]
                    zout = yin + xin
        for i in dace.map[0:n]:
            with dace.tasklet:
                xin << x[i, 0]
                yin << y[i, 0]
                zout >> z[n, i]
                zout = xin + yin
    return no_split_sdfg.to_sdfg()

def create_multiple_range_map_sdfg():
    @dace.program
    def multi_range_sdfg(x: dace.float16[16, 32, 16], y: dace.float16[16, 32, 16]):
        for i, j, w in dace.map[0:16, 0:32, 0:16]:
            y[i, j, w] = x[i, j, w] + y[i, j, w]
    return multi_range_sdfg.to_sdfg()

def test_direct_axpy():
    _exec_test(create_axpy_sdfg, [], [("x", 16), ("y", 16)])

def test_assigned_axpy_unroll_3():
    _exec_test(create_axpy_sdfg, [("x", "HBM", "3:6")], [("x", "HBM", "3:6"), ("y", "HBM", "0:3")])
    
def test_assigned_axpy_unroll_1():
    _exec_test(create_axpy_sdfg, [("x", "DDR", "0")], [("x", "DDR", "0"), ("y", "HBM", "0")])

def test_fixed_array_size_axpy_17():
    _exec_test(lambda: create_axpy_sdfg(17), [], [("x", 1), ("y", 1)])

def test_fixed_map_range_axpy_17():
    _exec_test(lambda: create_axpy_sdfg(map_range=17), [], [("x", 1), ("y", 1)])

def test_fixed_axpy_17():
    _exec_test(lambda: create_axpy_sdfg(17, 17), [], [("x", 1), ("y", 1)])

def test_fixed_axpy_21():
    _exec_test(lambda: create_axpy_sdfg(21, 21), [], [("x", 7), ("y", 7)])

def test_nd_split():
    _exec_test(create_nd_sdfg, [], [("x", 10), ("y", 10), ("z", 10)])

def test_no_split():
    _exec_test(create_not_splitable_dependence_sdfg, [], [("x", 1), ("y", 1), ("z", 1)])

def test_multiple_range_map():
    # SDFG defines a third non splitable temporary array which is placed on 17, thats why 16 cannot be taken
    _exec_test(create_multiple_range_map_sdfg, [], [("x", 8), ("y", 8)])

test_direct_axpy()
test_assigned_axpy_unroll_3()
test_assigned_axpy_unroll_1()
test_fixed_array_size_axpy_17()
test_fixed_map_range_axpy_17()
test_fixed_axpy_17()
test_fixed_axpy_21()
test_nd_split()
test_no_split()
test_multiple_range_map()
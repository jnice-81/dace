# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation.dataflow.strip_mining import StripMining
from typing import List, Tuple, Union
from dace.sdfg import SDFG, nodes
import dace
from dace.libraries.blas.nodes import dot
from dace.transformation.dataflow import HbmTransform, hbm_transform
from dace.transformation.interstate import NestSDFG


def set_assignment(sdfg: SDFG, assignments: List[Tuple[str, str, str]]):
    for array, memorytype, bank in assignments:
        desc = sdfg.arrays[array]
        desc.location["memorytype"] = memorytype
        desc.location["bank"] = bank


def check_assignment(sdfg: SDFG, assignments: List[Union[Tuple[str, int],
                                                         Tuple[str, str,
                                                               str]]]):
    for val in assignments:
        array, memorytype, bank = val
        assert sdfg.arrays[array].location["memorytype"] == memorytype
        assert sdfg.arrays[array].location["bank"] == bank


def _exec_hbmtransform(sdfg_source, assign):
    sdfg = sdfg_source()
    set_assignment(sdfg, assign)
    sdfg.apply_transformations(HbmTransform, validate=False)    
    sdfg.view()
    exit()
    sdfg.compile()

def create_vadd_sdfg(array_shape=dace.symbol("n"), map_range=dace.symbol("n")):
    @dace.program
    def vadd(x: dace.float32[array_shape], y: dace.float32[array_shape], z: dace.float32[array_shape]):
        for i in dace.map[0:map_range]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                zout >> z[i]
                zout = xin + yin
    sdfg = vadd.to_sdfg()
    sdfg.apply_strict_transformations()
    return sdfg


def create_nd_sdfg():
    n = dace.symbol("n")
    m = dace.symbol("m")

    @dace.program
    def nd_sdfg(x: dace.float32[n, m], y: dace.float32[m, n],
                z: dace.float32[n, m]):
        for i in dace.map[0:n]:
            for j in dace.map[0:m]:
                with dace.tasklet:
                    yin << y[j, i]
                    xin << x[i, j]
                    zout >> z[i, j]
                    xout = yin + xin

    sdfg = nd_sdfg.to_sdfg()
    sdfg.apply_strict_transformations()
    return sdfg


def create_not_splitable_dependence_sdfg():
    # All arrays would be splitable in principle, if they were alone, but because of
    # the dependencies on the same map variables, it's not possible to split each in just one dimension
    n = dace.symbol("n")

    @dace.program
    def no_split_sdfg(x: dace.float32[n, n], y: dace.float32[n, n],
                      z: dace.float32[n + 1, n]):
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

    sdfg = no_split_sdfg.to_sdfg()
    sdfg.apply_strict_transformations()
    return sdfg


def create_multiple_range_map_sdfg():
    @dace.program
    def multi_range_sdfg(x: dace.float16[16, 32, 16], y: dace.float16[16, 32,
                                                                      16]):
        for i, j, w in dace.map[0:16, 0:32, 0:16]:
            y[i, j, w] = x[i, j, w] + y[i, j, w]

    sdfg = multi_range_sdfg.to_sdfg()
    sdfg.apply_strict_transformations()
    return sdfg


def create_gemv_blas_sdfg(tile_size_y=None, tile_size_x=None, m=None):
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def gemv(A: dace.float32[M, N], x: dace.float32[N], y: dace.float32[M]):
        y[:] = A @ x

    sdfg = gemv.to_sdfg()
    sdfg.apply_strict_transformations()
    if m is not None:
        sdfg.specialize({M: m})
    libnode = list(
        filter(lambda x: isinstance(x, nodes.LibraryNode),
               sdfg.nodes()[0].nodes()))[0]
    libnode.expand(sdfg, sdfg.nodes()[0])
    libnode = list(
        filter(lambda x: isinstance(x, nodes.LibraryNode),
               sdfg.nodes()[0].nodes()))[0]
    libnode.implementation = "FPGA_TilesByColumn"
    libnode.expand(sdfg,
                   sdfg.nodes()[0],
                   tile_size_y=tile_size_y,
                   tile_size_x=tile_size_x)
    sdfg.apply_strict_transformations()
    return sdfg

def create_deeply_nested_sdfg():
    N = dace.symbol("N")

    sdfg = create_axpy_sdfg(N, N)
    sdfg.apply_transformations(NestSDFG)
    sdfg.apply_transformations(NestSDFG)
    map_node = next(filter(lambda x: isinstance(x[0], nodes.MapEntry), sdfg.all_nodes_recursive()))[0]
    array_banks = {"x": ("HBM", f"0:{16}", [16]), 
                "y": ("HBM", f"{16}:{32}", [16])}
    hbm_transform.transform_sdfg_for_hbm(sdfg, ("k", 16), 
        array_banks, {(map_node.map, 0): 16}, True)

    sdfg.apply_fpga_transformations()
    return sdfg

def test_assigned_axpy_unroll_3():
    _exec_hbmtransform(create_vadd_sdfg, [("x", "HBM", "3:6"), ("y", "HBM", "0:3"), ("z", "HBM", "6:9")])


def test_assigned_axpy_unroll_1():
    _exec_hbmtransform(create_axpy_sdfg, [("x", "DDR", "0")],
                       [("x", "DDR", "0"), ("y", "HBM", "0:1")])


def test_fixed_array_size_axpy_17():
    _exec_hbmtransform(lambda: create_axpy_sdfg(17), [], [("x", "HBM", "0:1"),
                                                          ("y", "HBM", "1:2")])


def test_fixed_map_range_axpy_17():
    _exec_hbmtransform(lambda: create_axpy_sdfg(map_range=17), [],
                       [("x", "HBM", "0:1"), ("y", "HBM", "1:2")])


def test_fixed_axpy_17():
    _exec_hbmtransform(lambda: create_axpy_sdfg(17, 17), [],
                       [("x", "HBM", "0:1"), ("y", "HBM", "1:2")])


def test_fixed_axpy_21():
    _exec_hbmtransform(lambda: create_axpy_sdfg(21, 21), [],
                       [("x", "HBM", "0:7"), ("y", "HBM", "7:14")])


def test_nd_split():
    _exec_hbmtransform(create_nd_sdfg, [], [("x", "HBM", "0:10"),
                                            ("y", "HBM", "10:20"),
                                            ("z", "HBM", "20:30")])


def test_no_split():
    _exec_hbmtransform(create_not_splitable_dependence_sdfg, [],
                       [("x", "HBM", "0:1"), ("y", "HBM", "1:2"),
                        ("z", "HBM", "2:3")])


def test_multiple_range_map():
    # SDFG defines a third non splitable temporary array which is placed on 17, thats why 16 cannot be taken
    _exec_hbmtransform(create_multiple_range_map_sdfg, [],
                       [("x", "HBM", "0:8"), ("y", "HBM", "8:16")])


def test_gemv_blas_nudging():
    # Tests the ability to influence the found splits via asserting arrays
    # that must be splitable. Note that this graph is wrong because it splits
    # along the inner map x which has to be a pipeline/sequential. (The map reads
    # from and writes to y_local).
    _exec_hbmtransform(lambda: create_gemv_blas_sdfg(m=150),
                       [("x", "HBM", "0:2")], [("x", "HBM", "0:2"),
                                               ("A", "HBM", "2:4"),
                                               ("y", "HBM", "4:5")])


def test_gemv_blas():
    # Because split happens using the outermost map, there needs to be a positive tile size to actually split
    _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32), [],
                       [("x", "HBM", "30:31"), ("y", "HBM", "15:30"),
                        ("A", "HBM", "0:15")])

def test_deeply_nested():
    sdfg = create_deeply_nested_sdfg()
    sdfg.validate()

if __name__ == "__main__":
    test_assigned_axpy_unroll_3()
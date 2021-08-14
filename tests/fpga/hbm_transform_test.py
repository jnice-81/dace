# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
from dace import dtypes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.dataflow.strip_mining import StripMining
from typing import List, Tuple, Union
from dace.sdfg import SDFG, nodes
import dace
from dace.transformation.dataflow import HbmTransform, hbm_transform
from dace.transformation.interstate import NestSDFG


def set_assignment(sdfg: SDFG, assignments: List[Tuple[str, str, str]]):
    for array, memorytype, bank in assignments:
        desc = sdfg.arrays[array]
        desc.location["memorytype"] = memorytype
        desc.location["bank"] = bank


def _exec_hbmtransform(sdfg_source, assign, nest=False):
    sdfg = sdfg_source()
    set_assignment(sdfg, assign)
    assert sdfg.apply_transformations(HbmTransform, {
        "new_dim": "kw",
        "move_to_FPGA_global": False
    },
                                      validate=False) == 1
    if nest:
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.Default:
                desc.storage = dtypes.StorageType.FPGA_Global
        sdfg.apply_transformations(NestSDFG, validate=False)
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.FPGA_Global:
                desc.storage = dtypes.StorageType.Default
    sdfg.apply_fpga_transformations(validate=False)
    sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
    sdfg.view()
    csdfg = sdfg.compile()
    return csdfg

def create_vadd_sdfg(array_shape=dace.symbol("n"), map_range=dace.symbol("n")):
    @dace.program
    def vadd(x: dace.float32[array_shape], y: dace.float32[array_shape],
             z: dace.float32[array_shape]):
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

def rand_float(input_shape):
    a = np.random.rand(*input_shape)
    a = a.astype(np.float32)
    return a

# TODO: Write tester for matrix; Let vadd stuff use the tester; Write application with multiple locations
def validate_vadd_sdfg(csdfg, input_shape):
    a = rand_float(input_shape)
    b = rand_float(input_shape)
    c = rand_float(input_shape)
    expect = a + b

    csdfg(x=a, y=b, z=c)
    assert np.allclose(expect, c)

def test_axpy_unroll_3():
    _exec_hbmtransform(create_vadd_sdfg, [("x", "HBM", "3:6"),
                                          ("y", "HBM", "0:3"),
                                          ("z", "HBM", "6:9")])


def test_axpy_unroll_1():
    _exec_hbmtransform(create_vadd_sdfg,
                       [("x", "DDR", "0"), ("y", "HBM", "0:1"),
                        ("z", "DDR", "1")])


def test_axpy_unroll_mixed():
    _exec_hbmtransform(create_vadd_sdfg, [("x", "DDR", "0"),
                                          ("y", "HBM", "0:3"),
                                          ("z", "HBM", "0:3")])


def test_nd_split():
    _exec_hbmtransform(create_nd_sdfg, [("x", "HBM", "0:10"),
                                        ("y", "HBM", "10:20"),
                                        ("z", "HBM", "20:30")])


def test_gemv_blas_1():
    _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32),
                       [("x", "HBM", "31:32"), ("y", "HBM", "30:31"),
                        ("A", "HBM", "0:30")], True)

def test_gemv_blas_2():
    _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32),
                       [("x", "HBM", "31:32"), ("y", "HBM", "15:30"),
                        ("A", "HBM", "0:15")], True)

def test_axpy_inconsistent_no_apply():
    N = dace.symbol("N")
    sdfg = create_vadd_sdfg(N, N)
    set_assignment(sdfg, [("x", "HBM", "0:2"), ("y", "DDR", "0"),
                          ("z", "HBM", "0:3")])
    assert sdfg.apply_transformations(HbmTransform, validate=False) == 0


if __name__ == "__main__":
    """
    test_axpy_unroll_3()
    test_axpy_unroll_1()
    test_axpy_unroll_mixed()
    test_nd_split()
    test_gemv_blas_1()
    test_gemv_blas_2()
    test_axpy_inconsistent_no_apply()
    """
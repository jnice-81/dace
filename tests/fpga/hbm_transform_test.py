# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation.dataflow.streaming_memory import StreamingMemory
import numpy as np
from dace import dtypes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from typing import List, Tuple
from dace.sdfg import SDFG, nodes
import dace
from dace.transformation.dataflow import HbmTransform
from dace.transformation.interstate import NestSDFG
from functools import reduce


def set_assignment(sdfg: SDFG, assignments: List[Tuple[str, str, str]]):
    for array, memorytype, bank in assignments:
        desc = sdfg.arrays[array]
        desc.location["memorytype"] = memorytype
        desc.location["bank"] = bank


def rand_float(input_shape):
    a = np.random.rand(*input_shape)
    a = a.astype(np.float32)
    return a


def _exec_hbmtransform(sdfg_source, assign, nest=False, num_apply=1,
    compile=True):
    sdfg = sdfg_source()
    set_assignment(sdfg, assign)
    assert sdfg.apply_transformations_repeated(HbmTransform, {
        "new_dim": "kw",
        "move_to_FPGA_global": False
    },
                                               validate=False) == num_apply
    if num_apply == 0:
        return
    #sdfg.view()
    if nest:
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.Default:
                desc.storage = dtypes.StorageType.FPGA_Global
        sdfg.apply_transformations(NestSDFG, validate=False)
        for _, desc in sdfg.arrays.items():
            if desc.storage == dtypes.StorageType.FPGA_Global:
                desc.storage = dtypes.StorageType.Default
    sdfg.apply_fpga_transformations(validate=False) == 1
    sdfg.apply_transformations_repeated(InlineSDFG, validate=False)
    if compile:
        sdfg = sdfg.compile()
    return sdfg


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


def create_multi_access_sdfg():
    N = dace.symbol("N")

    @dace.program
    def sth(z: dace.float32[N], x: dace.float32[N], y: dace.float32[N],
            w: dace.float32[N], o1: dace.float32[N], o2: dace.float32[N]):
        for i in dace.map[0:N]:
            o1[i] = z[i] + x[i]
        for i in dace.map[0:N]:
            o2[i] = w[i] + y[i]

    sdfg = sth.to_sdfg()
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


def validate_vadd_sdfg(csdfg, input_shape):
    a = rand_float(input_shape)
    b = rand_float(input_shape)
    c = rand_float(input_shape)
    expect = a + b

    csdfg(x=a, y=b, z=c, n=reduce(lambda x, y: x * y, input_shape))
    assert np.allclose(expect, c)


def validate_gemv_sdfg(csdfg, matrix_shape, x_shape, y_shape):
    # A and potentially y is assumed to be split along dim 0
    A = rand_float(matrix_shape)
    x = rand_float(x_shape)
    y = rand_float(y_shape)
    expect = np.matmul(A, x)

    csdfg(A=A, x=x, y=y, M=matrix_shape[0] * matrix_shape[1], N=matrix_shape[2])
    if len(y_shape) == 1:
        y = np.reshape(y, y_shape[0] * y_shape[1])
    assert np.allclose(y, expect)


def test_axpy_unroll_10():
    csdfg = _exec_hbmtransform(create_vadd_sdfg, [("x", "HBM", "0:10"),
                                                  ("y", "HBM", "10:20"),
                                                  ("z", "HBM", "20:30")])
    validate_vadd_sdfg(csdfg, [3, 20])


def test_axpy_unroll_1():
    # This SDFG is fine, but we would do nothing at all
    csdfg = _exec_hbmtransform(create_vadd_sdfg, [("x", "DDR", "0"),
                                                  ("y", "HBM", "0:1"),
                                                  ("z", "DDR", "1")],
                               num_apply=0)


def test_axpy_unroll_mixed():
    csdfg = _exec_hbmtransform(create_vadd_sdfg, [("x", "DDR", "0"),
                                                  ("y", "HBM", "0:2"),
                                                  ("z", "HBM", "0:2")])
    validate_vadd_sdfg(csdfg, [2, 20])


def test_nd_split():
    _exec_hbmtransform(create_nd_sdfg, [("x", "HBM", "0:10"),
                                        ("y", "HBM", "10:20"),
                                        ("z", "HBM", "20:30")])


def test_gemv_blas_1():
    csdfg = _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32),
                               [("x", "HBM", "31:32"), ("y", "HBM", "30:31"),
                                ("A", "HBM", "0:30")], True)
    validate_gemv_sdfg(csdfg, [30, 10, 5], [5], [10 * 30])


def test_gemv_blas_2():
    csdfg = _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32),
                               [("x", "HBM", "31:32"), ("y", "HBM", "15:30"),
                                ("A", "HBM", "0:15")], True)
    validate_gemv_sdfg(csdfg, [15, 10, 5], [5], [15, 10])


def test_axpy_inconsistent_no_apply():
    _exec_hbmtransform(create_vadd_sdfg,
                       [("x", "HBM", "0:2"), ("y", "DDR", "0"),
                        ("z", "HBM", "0:3")],
                       num_apply=0)


def test_multiple_applications():
    _exec_hbmtransform(create_multi_access_sdfg, [("x", "HBM", "0:2"),
                                                  ("y", "HBM", "2:4"),
                                                  ("z", "HBM", "4:6"),
                                                  ("w", "HBM", "10:12"),
                                                  ("o1", "HBM", "6:8"),
                                                  ("o2", "HBM", "8:10")],
                       num_apply=2)

def test_streaming():
    sdfg = _exec_hbmtransform(lambda: create_gemv_blas_sdfg(32),
                               [("x", "HBM", "31:32"), ("y", "HBM", "30:31"),
                                ("A", "HBM", "0:30")], True, 1, False)
    sdfg.apply_transformations_repeated(StreamingMemory)
    sdfg.view()

if __name__ == "__main__":
    test_streaming()
    exit() # View is active
    test_axpy_unroll_10()
    test_axpy_unroll_1()
    test_axpy_unroll_mixed()
    test_nd_split()
    test_gemv_blas_1()
    test_gemv_blas_2()
    test_axpy_inconsistent_no_apply()
    test_multiple_applications()

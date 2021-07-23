#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import functools
from dace.sdfg import utils
import numpy as np

import argparse
import scipy

import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, hbm_transform, hbm_copy_transform


def pure_graph(implementation, dtype, veclen):

    sdfg_name = f"dot_{implementation}_{dtype.ctype}_w{veclen}"
    sdfg = dace.SDFG(sdfg_name)

    state = sdfg.add_state("dot")

    n = dace.symbol("n")
    a = dace.symbol("a")

    vtype = dace.vector(dtype, veclen)

    if implementation == "FPGA_HBM_PartialSums":
        bank_count = 8
        input_lenght = n // (veclen * bank_count)
    else:
        input_lenght = n / veclen

    sdfg.add_array("x", [input_lenght], vtype)
    sdfg.add_array("y", [input_lenght], vtype)
    sdfg.add_array("r", [1], dtype)

    x = state.add_read("x")
    y = state.add_read("y")
    result = state.add_write("r")

    dot_node = blas.Dot("dot")
    dot_node.implementation = implementation
    dot_node.n = n

    state.add_memlet_path(x,
                          dot_node,
                          dst_conn="_x",
                          memlet=Memlet(f"x[0:{n}/{veclen}]"))
    state.add_memlet_path(y,
                          dot_node,
                          dst_conn="_y",
                          memlet=Memlet(f"y[0:{n}/{veclen}]"))
    state.add_memlet_path(dot_node,
                          result,
                          src_conn="_result",
                          memlet=Memlet(f"r[0]"))
    
    if implementation == "FPGA_HBM_PartialSums":
        xform = hbm_transform.HbmTransform(sdfg.sdfg_id, -1, {}, -1)
        xform.update_array_banks = [("x", "HBM", f"0:{bank_count}"),
            ("y", "HBM", f"{bank_count}:{2*bank_count}"), 
            ("r", "DDR", "0")]
        xform.apply(sdfg)
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                if node.label != "r":
                    utils.update_path_subsets(state, node, f"0:{bank_count}, 0:{input_lenght}")
        
    return sdfg


def fpga_graph(implementation, dtype, veclen):
    sdfg = pure_graph(implementation, dtype, veclen)
    sdfg.apply_transformations_repeated([FPGATransformSDFG, InlineSDFG])
    if implementation == "FPGA_HBM_PartialSums":
        utils.update_array_shape(sdfg, "x", [functools.reduce(lambda x, y: x*y, sdfg.arrays["x"].shape)])
        utils.update_array_shape(sdfg, "y", [functools.reduce(lambda x, y: x*y, sdfg.arrays["y"].shape)])
        for edge in sdfg.states()[1].edges():
            if edge.src.data != "r":
                hbm_copy_transform.HbmCopyTransform.apply_to(
                    sdfg, 
                    verify=False,
                    _src_node=edge.src,
                    _dst_node=edge.dst,
                )
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated(
        [InlineSDFG, StreamingMemory], [{}, {
            "storage": dace.StorageType.FPGA_Local
        }])
    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=128)
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vector-length", type=int, default=16)
    args = parser.parse_args()
    size = args.N

    if args.target == "pure":
        sdfg = pure_graph("pure", dace.float32, args.vector_length)
    elif args.target == "intel_fpga":
        dace.Config.set("compiler", "fpga_vendor", value="intel_fpga")
        sdfg = fpga_graph("FPGA_Accumulate", dace.float32, args.vector_length)
    elif args.target == "xilinx":
        dace.Config.set("compiler", "fpga_vendor", value="xilinx")
        sdfg = fpga_graph("FPGA_PartialSums", dace.float32, args.vector_length)
    elif args.target == "xilinx_hbm":
        dace.Config.set("compiler", "fpga_vendor", value="xilinx")
        sdfg = fpga_graph("FPGA_HBM_PartialSums", dace.float32, args.vector_length)
    else:
        print(f"Unsupported target: {args.target}")
        exit(-1)

    dot = sdfg.compile()

    x = np.ndarray(size, dtype=np.float32)
    y = np.ndarray(size, dtype=np.float32)
    result = np.ndarray(1, dtype=np.float32)

    x[:] = np.random.rand(size).astype(np.float32)
    y[:] = np.random.rand(size).astype(np.float32)

    result[0] = 0

    dot(x=x, y=y, r=result, n=size)

    ref = scipy.linalg.blas.sdot(x, y)

    diff = abs(result[0] - ref)
    if diff >= 1e-6 * ref:
        raise ValueError("Unexpected result returned from dot product: "
              "got {}, expected {}".format(result[0], ref))

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
import dace
import numpy as np

N = dace.symbol('N', positive=True)


@dace.program(dace.float32[N], dace.float32[N], dace.uint32[1], dace.float32)
def pbf(A, out, outsz, ratio):
    ostream = dace.define_stream(dace.float32, N)

    @dace.map(_[0:N])
    def filter(i):
        a << A[i]
        r << ratio
        b >> ostream(-1)
        osz >> outsz(-1, lambda x, y: x + y, 0)

        filter = (a > r)

        if filter:
            b = a

        osz = filter

    ostream >> out


def regression(A, ratio):
    return A[np.where(A > ratio)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=64)
    parser.add_argument("ratio", type=float, nargs="?", default=0.5)
    args = vars(parser.parse_args())

    N.set(args["N"])
    ratio = np.float32(args["ratio"])

    print('Predicate-Based Filter. size=%d, ratio=%f' % (N.get(), ratio))

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.zeros_like(A)
    outsize = dace.scalar(dace.uint32)
    outsize[0] = 0

    pbf(A, B, outsize, ratio)

    if dace.Config.get_bool('profiling'):
        dace.timethis('filter', 'numpy', 0, regression, A, ratio)

    filtered = regression(A, ratio)

    if len(filtered) != outsize[0]:
        print("Difference in number of filtered items: %d (DaCe) vs. %d (numpy)" % (outsize[0], len(filtered)))
        totalitems = min(outsize[0], N.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))
        exit(1)

    # Sort the outputs
    filtered = np.sort(filtered)
    B[:outsize[0]] = np.sort(B[:outsize[0]])

    if len(filtered) == 0:
        print("==== Program end ====")
        exit(0)

    diff = np.linalg.norm(filtered - B[:outsize[0]]) / float(outsize[0])
    print("Difference:", diff)
    if diff > 1e-5:
        totalitems = min(outsize[0], N.get())
        print('DaCe:', B[:totalitems].view(type=np.ndarray))
        print('Regression:', filtered.view(type=np.ndarray))

    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)

import dace
import numpy as np

N = dace.symbol('N')
N.set(12345)


@dace.program
def nested(A: dace.float64[N], B: dace.float64[N], factor: dace.float64):
    B[:] = A * factor


@dace.program
def nested_symbol(A: dace.float64[N], B: dace.float64[N]):
    nested(A[0:5], B[0:5], 0.5)
    nested(A=A[5:N], B=B[5:N], factor=2.0)


def test_nested_symbol():
    A = np.random.rand(20)
    B = np.random.rand(20)
    nested_symbol(A, B)
    assert np.allclose(B[0:5], A[0:5] / 2) and np.allclose(B[5:20], A[5:20] * 2)


if __name__ == '__main__':
    test_nested_symbol()

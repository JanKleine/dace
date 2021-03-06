import dace
import numpy as np

# Declaration of symbolic variables
N, BS = (dace.symbol(name) for name in ['N', 'BS'])


@dace.program
def seq_cond(HD: dace.complex128[N, BS, BS], HE: dace.complex128[N, BS, BS],
             HF: dace.complex128[N, BS, BS],
             sigmaRSD: dace.complex128[N, BS, BS],
             sigmaRSE: dace.complex128[N, BS, BS],
             sigmaRSF: dace.complex128[N, BS, BS]):

    for n in range(N):
        if n < N - 1:
            HE[n] -= sigmaRSE[n]
        else:
            HE[n] = -sigmaRSE[n]
        if n > 0:
            HF[n] -= sigmaRSF[n]
        else:
            HF[n] = -sigmaRSF[n]
        HD[n] = HD[n] - sigmaRSD[n]


if __name__ == '__main__':
    seq_cond.compile()

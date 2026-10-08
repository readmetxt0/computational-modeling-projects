"""ПЗ-2. Бинарная обучающая матрица X[1..m, 1..N, 1..n] (бинарное пространство Хемминга)."""
import numpy as np

DELTA = 50  # поле контрольных допусков (delta = ±50)


def tolerances(Y, delta=DELTA):
    """Система контрольных допусков: NDK[k,i] = mean - delta, VDK[k,i] = mean + delta,
    где mean - среднее i-го признака по n реализациям класса k."""
    mean = Y.mean(axis=2)                 # [m, N]
    return mean - delta, mean + delta


def binarize(Y, ndk, vdk):
    """X[k,i,j] = 1, если NDK[k,i] <= Y[k,i,j] <= VDK[k,i], иначе 0."""
    return ((Y >= ndk[:, :, None]) & (Y <= vdk[:, :, None])).astype(np.int8)


def format_binary(Xk, rows=None):
    """Строка = реализация, столбец = признак."""
    mat = Xk.T if rows is None else Xk.T[:rows]
    return "\n".join(" ".join(str(v) for v in row) for row in mat)

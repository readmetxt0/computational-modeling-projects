"""ПЗ-3. Эталонные векторы (геометрические центры) классов распознавания EV[1..m, 1..N]."""
import numpy as np

RO = 0.5  # уровень селекции координат эталонного вектора


def reference_vectors(X, ro=RO):
    """EV[k,i] = 1, если (1/n) * sum_j X[k,i,j] > ro, иначе 0."""
    return (X.mean(axis=2) > ro).astype(np.int8)

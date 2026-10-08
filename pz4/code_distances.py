"""ПЗ-4. Кодовые расстояния между центрами классов и реализациями (метрика Хемминга)."""
import numpy as np


def hamming(a, b):
    """Кодовое расстояние Хемминга между двоичными векторами."""
    return int(np.sum(a != b))


def center_distance_matrix(EV):
    """Матрица M x M кодовых расстояний между эталонными векторами."""
    M = EV.shape[0]
    return np.array([[hamming(EV[a], EV[b]) for b in range(M)] for a in range(M)])


def neighbors(EV):
    """PARA[k] - ближайший сосед класса k (минимум в строке матрицы расстояний, k != l).
    При нескольких равных минимумах берётся первый (они равноправны)."""
    D = center_distance_matrix(EV).astype(float)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1)


def realization_distances(X, EV, k, c):
    """Расстояния от центра класса c до всех n реализаций класса k: массив длины n."""
    return np.array([hamming(EV[c], X[k, :, j]) for j in range(X.shape[2])])


def build_SK(X, EV, k, l):
    """SK[1,j]  - расстояние от центра текущего класса k до его реализаций;
       SK[2,j]  - расстояние от центра класса k до реализаций ближайшего соседа l."""
    return np.vstack([realization_distances(X, EV, k, k),
                      realization_distances(X, EV, l, k)])


def build_SK_PARA(X, EV, k, l):
    """То же, но текущим классом считается ближайший сосед l:
       SK_PARA[1,j] - от центра l до реализаций l, SK_PARA[2,j] - от центра l до реализаций k."""
    return build_SK(X, EV, l, k)


def line_positions(X, EV, k, l):
    """Положение реализаций на горизонтальной прямой: центр k в точке 0, центр l в точке D.
    По расстояниям r1 (до центра k) и r2 (до центра l) координата
    t = (D^2 + r1^2 - r2^2) / (2D); при D = 0 центры совпадают, используется t = r1."""
    D = hamming(EV[k], EV[l])
    r1k = realization_distances(X, EV, k, k); r2k = realization_distances(X, EV, k, l)
    r1l = realization_distances(X, EV, l, k); r2l = realization_distances(X, EV, l, l)
    if D == 0:
        return D, r1k.astype(float), r1l.astype(float)
    t = lambda r1, r2: (D**2 + r1.astype(float)**2 - r2.astype(float)**2) / (2 * D)
    return D, t(r1k, r2k), t(r1l, r2l)

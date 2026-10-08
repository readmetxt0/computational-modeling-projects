"""ПЗ-1. Формирование входной обучающей матрицы Y[1..m, 1..N, 1..n].

Вариант 10: два изображения-текстуры (классы распознавания X1, X2).
"""
import numpy as np
from PIL import Image

M = 2      # m - количество классов
N = 100    # N - количество признаков распознавания
n = 100    # n - количество реализаций


def split_textures(path):
    """Вырезает из общей картинки две текстуры (левая - класс 1, правая - класс 2)."""
    im = Image.open(path).convert("L")
    # границы подобраны по однородным (белым) полосам исходного файла
    left = im.crop((0, 8, 152, 160))
    right = im.crop((156, 8, 302, 160))
    return [left, right]


def brightness(img, size=(N, n)):
    """Яркость пикселя (R+G+B)/3 -> матрица size[1] x size[0] (целые 0..255).
    Изображение центрально обрезается до N x n пикселей."""
    a = np.asarray(img.convert("RGB"), dtype=np.int32)
    h, w = a.shape[:2]
    top, lft = (h - size[1]) // 2, (w - size[0]) // 2
    a = a[top:top + size[1], lft:lft + size[0]]
    return (a[..., 0] + a[..., 1] + a[..., 2]) // 3


def build_Y(images):
    """Y[k, i, j] - значение i-го признака (столбец изображения) в j-й реализации (строка)."""
    Y = np.zeros((len(images), N, n), dtype=np.int32)
    for k, img in enumerate(images):
        b = brightness(img)          # b[строка j, столбец i]
        Y[k] = b.T                   # -> [признак i, реализация j]
    return Y


def format_matrix(Yk, rows=None):
    """Текстовое представление матрицы класса: строка = реализация, столбец = признак."""
    mat = Yk.T if rows is None else Yk.T[:rows]
    return "\n".join(" ".join(f"{v:3d}" for v in row) for row in mat)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "images/textures_var10.png"
    Y = build_Y(split_textures(path))
    print("Y.shape =", Y.shape, "(m, N, n)")
    for k in range(M):
        print(f"Класс {k+1}: min={Y[k].min()}, max={Y[k].max()}, среднее={Y[k].mean():.1f}")
    np.save("Y.npy", Y)

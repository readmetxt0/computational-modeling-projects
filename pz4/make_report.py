import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from training_matrix import split_textures, build_Y, brightness, M, N, n
from binary_matrix import tolerances, binarize, DELTA
from reference_vectors import reference_vectors, RO
from code_distances import (center_distance_matrix, neighbors, build_SK, build_SK_PARA,
                            line_positions, hamming)


imgs = split_textures("images/textures_var10.png")
Y = build_Y(imgs)
ndk, vdk = tolerances(Y, DELTA)
X = binarize(Y, ndk, vdk)
EV = reference_vectors(X, RO)
PARA = neighbors(EV)
DM = center_distance_matrix(EV)
k, l = 0, int(PARA[0])
SK, SKP = build_SK(X, EV, k, l), build_SK_PARA(X, EV, k, l)
SK2, SKP2 = build_SK(X, EV, l, k), build_SK_PARA(X, EV, l, k)   # для текущего класса 2

# ---------- рисунки ----------
D, t1, t2 = line_positions(X, EV, k, l)
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.axhline(0, color="gray", lw=.8)
ax.scatter(t1, np.full(n, .12), s=10, alpha=.6, label="реализации класса 1")
ax.scatter(t2, np.full(n, -.12), s=10, alpha=.6, label="реализации класса 2")
ax.scatter([0, D], [0, 0], c="k", marker="x", s=80, label="центры классов")
ax.set_yticks([]); ax.set_xlabel("положение на прямой (кодовое расстояние)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("images/line.png", dpi=150); plt.close()

# ---------- помощники для docx ----------
def new_doc():
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(14)
    for s in d.sections:
        s.left_margin = Cm(3); s.right_margin = Cm(1.5); s.top_margin = s.bottom_margin = Cm(2)
    return d

def P(d, text, bold=False, center=False, italic=False):
    par = d.add_paragraph(); r = par.add_run(text); r.bold = bold; r.italic = italic
    if center: par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return par

def CODE(d, text):
    par = d.add_paragraph(); r = par.add_run(text); r.font.name = "Courier New"; r.font.size = Pt(9)

def PIC(d, path, cap, w=11):
    d.add_picture(path, width=Cm(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(d, cap, center=True)

def QA(d, items):
    P(d, "Ответы на контрольные вопросы", True)
    for i, (q, a) in enumerate(items, 1):
        P(d, f"{i}. {q}", True); P(d, a)

def head(d, num, title, goal):
    P(d, f"Практическое занятие {num}", True, True); P(d, title, True, True); P(d, "Вариант 10", italic=True, center=True)
    P(d, "Цель работы: ", True).add_run(goal).bold = False

def frag(arr, r=8, c=20):
    return "\n".join(" ".join(f"{int(v):3d}" if arr.max() > 1 else str(int(v)) for v in row[:c]) for row in arr[:r])


# ================= ПЗ-4 =================
d = new_doc()
head(d, 4, "Формирование массива кодовых расстояний между центрами классов распознавания",
     "разработать и программно реализовать алгоритм формирования массива кодовых расстояний между центрами классов распознавания.")
P(d, "1. Краткие теоретические сведения", True)
P(d, "Матрица кодовых расстояний используется для разбиения множества эталонных векторов на пары ближайших соседей "
     "ℜ_m^|2| = <x_m, x_l>, где x_l – эталонный вектор соседнего класса. Алгоритм: а) структурируется множество эталонных "
     "векторов, начиная с вектора x_1 базового класса; б) строится матрица кодовых расстояний между эталонными векторами "
     "размерности M×M; в) для каждой строки находится минимальный элемент (при равных минимумах берётся любой), его столбец "
     "определяет ближайший класс-«соседа»; г) формируется структурированное множество пар {ℜ_m | m = 1..M}, задающее план обучения.")
P(d, "Кодовое расстояние Хемминга d(x, y) = Σ_i (x_i ⊕ y_i) – число несовпадающих разрядов двоичных векторов.")
P(d, "2. Ход работы", True)
P(d, "1) Построена матрица кодовых расстояний между эталонными векторами. 2) Определён ближайший сосед PARA[k] для каждого класса. "
     "3) Сформирован массив SK[1..2, 1..n]: SK[1,j] – расстояние от центра текущего класса до j-й реализации этого класса, "
     "SK[2,j] – расстояние от центра текущего класса до j-й реализации ближайшего соседнего класса (эти массивы нужны далее "
     "для расчёта ошибок 1-го и 2-го рода). 4) Массив SK_PARA строится так же, но за текущий класс берётся ближайший сосед. "
     "5) Положение каждой реализации на прямой, на которой лежат центры классов, находится по SK и SK_PARA: при расстоянии "
     "между центрами D координата t = (D² + r1² − r2²) / (2D), где r1, r2 – расстояния до первого и второго центров.")
P(d, "3. Результаты", True)
P(d, "Матрица кодовых расстояний между эталонными векторами:", italic=True)
CODE(d, "\n".join(" ".join(f"{int(v):3d}" for v in row) for row in DM))
P(d, f"Ближайший сосед: для класса 1 – класс {PARA[0]+1}, для класса 2 – класс {PARA[1]+1}; план обучения: ℜ_1 = <x_1, x_{PARA[0]+1}>, "
     f"ℜ_2 = <x_2, x_{PARA[1]+1}>.")
P(d, f"Средние значения: SK[1] = {SK[0].mean():.2f}, SK[2] = {SK[1].mean():.2f}; SK_PARA[1] = {SKP[0].mean():.2f}, "
     f"SK_PARA[2] = {SKP[1].mean():.2f}.")
P(d, "Фрагмент массива SK (первые 20 реализаций, текущий класс 1):", italic=True)
CODE(d, "SK[1]: " + " ".join(f"{v:2d}" for v in SK[0][:20]) + "\nSK[2]: " + " ".join(f"{v:2d}" for v in SK[1][:20]))
P(d, "Фрагмент массива SK_PARA (первые 20 реализаций):", italic=True)
CODE(d, "SK_PARA[1]: " + " ".join(f"{v:2d}" for v in SKP[0][:20]) + "\nSK_PARA[2]: " + " ".join(f"{v:2d}" for v in SKP[1][:20]))
PIC(d, "images/line.png", "Рисунок 1 – Распределение реализаций между текущим классом и его ближайшим соседом", 14)
P(d, "Примечание. Реализации второго класса лежат ближе к центру второго класса, реализации первого – ближе к первому; "
     "области частично перекрываются, что и определяет ошибки 1-го и 2-го рода в ПЗ-5.", italic=True)
P(d, "4. Программная реализация", True)
CODE(d, open("code_distances.py", encoding="utf-8").read())
QA(d, [
 ("Алгоритм формирования массива кодовых расстояний между центрами классов распознавания.",
  "Для пары классов вычисляются расстояния Хемминга от эталонного вектора текущего класса до всех его реализаций (SK[1]) и "
  "до реализаций ближайшего соседа (SK[2]); аналогичный массив SK_PARA строится для соседа как для текущего класса."),
 ("Что такое кодовое расстояние? Как строится матрица кодовых расстояний?",
  "Кодовое расстояние – число разрядов, в которых двоичные векторы различаются (расстояние Хемминга). Матрица M×M "
  "содержит попарные расстояния между всеми эталонными векторами; диагональ нулевая."),
 ("По какому алгоритму определяется ближайший «сосед» класса распознавания?",
  "Для строки матрицы кодовых расстояний, соответствующей классу, выбирается минимальный ненулевой (недиагональный) элемент; "
  "его столбец задаёт ближайший соседний класс. При нескольких равных минимумах берётся любой."),
 ("Какой компонент использован для отображения распределения реализаций?",
  "В методичке – графический компонент (PictureBox) C# с рисованием точек на горизонтальной прямой; в работе – matplotlib "
  "(scatter), встроенный в интерфейс через FigureCanvasTkAgg."),
])
P(d, "Вывод", True)
P(d, f"Построена матрица кодовых расстояний ({int(DM[0,1])} между центрами), определены ближайшие соседи, сформированы массивы "
     "SK и SK_PARA и визуализировано распределение реализаций на прямой между центрами классов. Эти данные будут "
     "использованы для расчёта информационного критерия Шеннона в ПЗ-5.")
d.save("PZ4_otchet.docx")

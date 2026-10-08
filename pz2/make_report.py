import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from training_matrix import split_textures, build_Y, brightness, M, N, n
from binary_matrix import tolerances, binarize, DELTA

imgs = split_textures("images/textures_var10.png")
Y = build_Y(imgs)
ndk, vdk = tolerances(Y, DELTA)
X = binarize(Y, ndk, vdk)

# ---------- рисунки ----------
# ---------- рисунки ----------
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
for c in range(M):
    ax[0][c].imshow(brightness(imgs[c]), cmap="gray", vmin=0, vmax=255)
    ax[0][c].set_title(f"Класс {c+1}: яркость"); ax[0][c].axis("off")
    ax[1][c].imshow(X[c].T, cmap="gray", vmin=0, vmax=1)
    ax[1][c].set_title(f"Класс {c+1}: бинарная матрица"); ax[1][c].axis("off")
plt.tight_layout(); plt.savefig("images/binary.png", dpi=150); plt.close()


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


# ================= ПЗ-2 =================
d = new_doc()
head(d, 2, "Формирование бинарной обучающей матрицы",
     "разработать и программно реализовать алгоритм формирования бинарной обучающей матрицы.")
P(d, "1. Краткие теоретические сведения", True)
P(d, "Обучающая матрица типа «объект–свойство» ||y_m,i^(j)|| характеризует m-е функциональное состояние (класс "
     "распознавания): строка – реализация образа (N признаков), столбец – случайная обучающая выборка i-го признака "
     "объёмом n. Для изображения матрица формируется сканированием рецепторного поля: в каждом пикселе получают яркость "
     "(0…255 градаций). Для дальнейшего обучения матрица Y переводится в бинарное пространство Хемминга – "
     "это позволяет вычислять кодовые расстояния и работать с эталонными двоичными векторами.")
P(d, "Правило бинаризации: X[k,i,j] = 1, если NDK[i] ≤ Y[k,i,j] ≤ VDK[i], и X[k,i,j] = 0, если Y[k,i,j] < NDK[i] "
     "или Y[k,i,j] > VDK[i], где NDK[i], VDK[i] – нижний и верхний контрольные допуски i-го признака.")
P(d, "2. Ход работы", True)
P(d, f"1) Задано поле контрольных допусков delta = ±{DELTA}. 2) Для каждого признака i и класса k контрольные допуски "
     f"заданы вокруг среднего значения признака по реализациям: NDK[k,i] = mean − {DELTA}, VDK[k,i] = mean + {DELTA}. "
     "3) Матрица Y[1..2, 1..100, 1..100] переведена в бинарную X[1..2, 1..100, 1..100] по правилу выше. "
     "4) Результат выведен в виде изображения и числовой матрицы (интерфейс gui.py).")
P(d, "3. Результаты", True)
PIC(d, "images/binary.png", "Рисунок 1 – Яркость и бинарная обучающая матрица двух классов", 10)
P(d, f"Доля единиц в бинарной матрице: класс 1 – {X[0].mean():.3f}, класс 2 – {X[1].mean():.3f}.")
for c in range(M):
    P(d, f"Фрагмент бинарной матрицы класса {c+1} (8 реализаций × 20 признаков):", italic=True)
    CODE(d, frag(X[c].T))
P(d, "4. Программная реализация", True)
CODE(d, open("binary_matrix.py", encoding="utf-8").read())
QA(d, [
 ("Как описать систему контрольных допусков на признаки распознавания?",
  "Для каждого i-го признака задаётся пара значений: нижний NDK[i] и верхний VDK[i] контрольные допуски, "
  "ограничивающие интервал значений признака, при попадании в который признак считается соответствующим классу."),
 ("С какой целью проводится перевод обучающей матрицы в бинарное пространство Хемминга?",
  "Чтобы перейти от непрерывных значений к двоичным векторам, для которых определена метрика Хемминга (кодовое расстояние), "
  "можно строить эталонные векторы, контейнеры классов и вычислять информационный критерий."),
 ("Какой компонент (средство) использован для графического отображения бинарной матрицы?",
  "В методичке – PictureBox (C#) с покомпонентной раскраской Bitmap (SetPixel); в данной работе на Python – "
  "matplotlib (imshow) в отчёте и PIL/Canvas в интерфейсе."),
 ("Что называется контрольным полем допусков на признаки распознавания?",
  "Заданное экспертом (или рассчитанное) поле допусков δ_K,i, внутри которого значение признака считается принадлежащим классу; "
  "в процессе обучения оно расширяется с шагом step_DK для поиска оптимального значения."),
 ("Что называется нормированным полем допусков на признаки распознавания?",
  "Поле допусков δ_H,i, которое получено по результатам обучения и используется на экзамене: оптимальное (по информационному "
  "критерию) из полей, полученных при расширении контрольных допусков."),
 ("Как формируется обучающая матрица?",
  "Строки – реализации образа, столбцы – признаки; для изображения это матрица яркости пикселей (0…255), полученная сканированием "
  "рецепторного поля."),
])
P(d, "Вывод", True)
P(d, f"Реализован перевод матрицы яркости Y в бинарную матрицу X для двух классов текстур варианта 10 при delta = ±{DELTA}. "
     "Бинарная матрица используется для формирования эталонных векторов (ПЗ-3).")
d.save("PZ2_otchet.docx")

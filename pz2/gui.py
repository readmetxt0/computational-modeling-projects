"""Интерфейс (tkinter + matplotlib) для ПЗ-2: загрузка двух текстур и вывод результатов этапов ПЗ-1..ПЗ-2.
Запуск: python gui.py"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
from PIL import Image, ImageTk
from training_matrix import build_Y, N, n
from binary_matrix import tolerances, binarize, format_binary, DELTA
STAGE = 2  # номер практического занятия
if STAGE >= 3:
    from reference_vectors import reference_vectors
if STAGE >= 4:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from code_distances import neighbors, build_SK, build_SK_PARA, line_positions


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"ПЗ-{STAGE}. Интеллектуальная система распознавания текстур")
        self.images = [None, None]
        self.panels = []
        top = tk.Frame(self); top.pack(padx=6, pady=6)
        for k in range(2):
            col = tk.Frame(top); col.grid(row=0, column=k, padx=6)
            tk.Button(col, text=f"Загрузить класс {k+1}", command=lambda k=k: self.load(k)).pack()
            lbl = tk.Label(col, text="(нет изображения)"); lbl.pack(pady=3)
            txt = scrolledtext.ScrolledText(col, width=50, height=12, font=("Courier", 8)); txt.pack()
            self.panels.append((lbl, txt))
        ctl = tk.Frame(self); ctl.pack()
        tk.Label(ctl, text="delta:").pack(side=tk.LEFT)
        self.delta = tk.StringVar(value=str(DELTA))
        tk.Entry(ctl, textvariable=self.delta, width=5).pack(side=tk.LEFT)
        tk.Button(ctl, text="Рассчитать", command=self.calc).pack(side=tk.LEFT, padx=6)
        self.info = tk.Label(self, justify=tk.LEFT, font=("Courier", 9)); self.info.pack()
        if STAGE >= 4:
            self.fig = Figure(figsize=(7, 2.4)); self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack()

    def load(self, k):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png *.jpg *.bmp *.tif")])
        if not path:
            return
        try:
            img = Image.open(path).convert("L")
        except OSError as e:
            messagebox.showerror("Ошибка", str(e)); return
        self.images[k] = img
        photo = ImageTk.PhotoImage(img.resize((N, n)))
        lbl = self.panels[k][0]; lbl.configure(image=photo, text=""); lbl.image = photo

    def calc(self):
        if any(i is None for i in self.images):
            messagebox.showwarning("Внимание", "Загрузите оба изображения"); return
        try:
            delta = float(self.delta.get())
        except ValueError:
            messagebox.showerror("Ошибка", "delta должна быть числом"); return
        Y = build_Y(self.images)
        ndk, vdk = tolerances(Y, delta); X = binarize(Y, ndk, vdk)           # ПЗ-2
        for k, (_, txt) in enumerate(self.panels):
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, "Бинарная матрица (30 реализаций):\n" + format_binary(X[k], 30))
        lines = [f"Y: {Y.shape}, доля единиц в X: {X[0].mean():.2f} / {X[1].mean():.2f}"]
        if STAGE >= 3:                                                       # ПЗ-3
            EV = reference_vectors(X)
            lines.append(f"Эталонные векторы: единиц {int(EV[0].sum())} / {int(EV[1].sum())}")
        if STAGE >= 4:                                                       # ПЗ-4
            PARA = neighbors(EV); k, l = 0, int(PARA[0])
            SK, SKP = build_SK(X, EV, k, l), build_SK_PARA(X, EV, k, l)
            lines.append(f"Сосед класса 1: {l+1}; SK[1]={SK[0].mean():.1f}, SK[2]={SK[1].mean():.1f}, "
                         f"SK_PARA[1]={SKP[0].mean():.1f}, SK_PARA[2]={SKP[1].mean():.1f}")
            D, t1, t2 = line_positions(X, EV, k, l)
            self.ax.clear(); self.ax.axhline(0, color="gray", lw=.8)
            self.ax.scatter(t1, np.full(n, .1), s=8, label="класс 1", alpha=.6)
            self.ax.scatter(t2, np.full(n, -.1), s=8, label="класс 2", alpha=.6)
            self.ax.scatter([0, D], [0, 0], c="k", marker="x", s=60, label="центры")
            self.ax.set_yticks([]); self.ax.legend(fontsize=7); self.fig.tight_layout(); self.canvas.draw()
        self.info.config(text="\n".join(lines))


if __name__ == "__main__":
    App().mainloop()

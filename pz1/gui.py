"""ПЗ-1, задание 3: предварительный интерфейс (диалоговый режим) на tkinter.
Загрузка двух изображений, формирование и вывод обучающей матрицы Y."""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from training_matrix import build_Y, format_matrix, N, n


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ПЗ-1. Формирование входной обучающей матрицы")
        self.images, self.panels = [], []
        top = tk.Frame(self); top.pack(padx=8, pady=8)
        for k in range(2):
            col = tk.Frame(top); col.grid(row=0, column=k, padx=8)
            tk.Button(col, text=f"Загрузить класс {k+1}",
                      command=lambda k=k: self.load(k)).pack()
            lbl = tk.Label(col, text="(нет изображения)", width=N // 7, height=N // 14)
            lbl.pack(pady=4)
            txt = scrolledtext.ScrolledText(col, width=48, height=18, font=("Courier", 8))
            txt.pack()
            self.panels.append((lbl, txt))
            self.images.append(None)
        tk.Button(self, text="Сформировать матрицу Y", command=self.calc).pack(pady=6)
        self.info = tk.Label(self, text=""); self.info.pack()

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
        lbl = self.panels[k][0]; lbl.configure(image=photo, text="", width=N, height=n); lbl.image = photo

    def calc(self):
        if any(i is None for i in self.images):
            messagebox.showwarning("Внимание", "Загрузите оба изображения"); return
        Y = build_Y(self.images)
        for k, (_, txt) in enumerate(self.panels):
            txt.delete("1.0", tk.END); txt.insert(tk.END, format_matrix(Y[k]))
        self.info.config(text=f"Y[1..{Y.shape[0]}, 1..{Y.shape[1]}, 1..{Y.shape[2]}] сформирована")


if __name__ == "__main__":
    App().mainloop()

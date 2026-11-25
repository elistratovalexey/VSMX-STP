"""Графический интерфейс (tkinter) - Windows 11."""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
import webbrowser
from typing import List

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from . import APP_TITLE, __version__
from . import config as cfg
from . import updater
from .converter import ConvertOptions, convert


class App(tk.Tk):
    def __init__(self, files: List[str]):
        super().__init__()
        self.settings = cfg.load()
        self.title(f"{APP_TITLE} {__version__}")
        self.geometry("920x680")
        self.minsize(760, 520)
        try:
            if sys.platform == "win32":
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # pragma: no cover
            pass
        self._set_icon()
        self.q: "queue.Queue" = queue.Queue()
        self.busy = False
        self._build_menu()
        self._build_ui()
        for f in files:
            self._add_file(f)
        self.after(100, self._poll)
        if self.settings.check_updates_on_start:
            self.after(1500, lambda: self.check_updates(silent=True))

    # ------------------------------------------------------------ UI
    def _set_icon(self):
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        for p in (os.path.join(base, "vcmx2stp", "assets", "app.ico"), os.path.join(base, "assets", "app.ico")):
            if os.path.isfile(p):
                try:
                    self.iconbitmap(p)
                except tk.TclError:  # pragma: no cover
                    pass
                break

    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Добавить файлы .vcmx...", command=self.pick_files, accelerator="Ctrl+O")
        fm.add_separator()
        fm.add_command(label="Выход", command=self.destroy)
        m.add_cascade(label="Файл", menu=fm)
        sm = tk.Menu(m, tearoff=0)
        sm.add_command(label="Репозиторий обновлений GitHub...", command=self.set_repo)
        self.var_autoupd = tk.BooleanVar(value=self.settings.check_updates_on_start)
        sm.add_checkbutton(label="Проверять обновления при запуске", variable=self.var_autoupd,
                           command=self._save)
        m.add_cascade(label="Настройки", menu=sm)
        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Проверить обновления", command=lambda: self.check_updates(False))
        hm.add_command(label="Страница проекта", command=self.open_page)
        hm.add_separator()
        hm.add_command(label="О программе", command=self.about)
        m.add_cascade(label="Справка", menu=hm)
        self.config(menu=m)
        self.bind("<Control-o>", lambda e: self.pick_files())

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}
        top = ttk.Frame(self)
        top.pack(fill="both", expand=False, **pad)

        # --- файлы
        lf = ttk.LabelFrame(top, text="Файлы KUKA.Sim / Visual Components (.vcmx)")
        lf.pack(fill="x", **pad)
        self.lst = tk.Listbox(lf, height=6, selectmode="extended")
        self.lst.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        bf = ttk.Frame(lf)
        bf.pack(side="right", fill="y")
        ttk.Button(bf, text="Добавить...", command=self.pick_files).pack(fill="x", pady=2)
        ttk.Button(bf, text="Удалить", command=self.remove_sel).pack(fill="x", pady=2)
        ttk.Button(bf, text="Очистить", command=lambda: self.lst.delete(0, "end")).pack(fill="x", pady=2)

        # --- выход
        of = ttk.LabelFrame(top, text="Папка для .stp (пусто - рядом с исходным файлом)")
        of.pack(fill="x", **pad)
        self.var_out = tk.StringVar(value=self.settings.output_dir)
        ttk.Entry(of, textvariable=self.var_out).pack(side="left", fill="x", expand=True, padx=4, pady=4)
        ttk.Button(of, text="Обзор...", command=self.pick_outdir).pack(side="right", padx=4)

        # --- папки поиска
        sf = ttk.LabelFrame(top, text="Папки поиска невстроенных компонентов (eCatalog, My Models)")
        sf.pack(fill="x", **pad)
        self.lst_dirs = tk.Listbox(sf, height=3)
        self.lst_dirs.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        for d in self.settings.search_dirs:
            self.lst_dirs.insert("end", d)
        bf2 = ttk.Frame(sf)
        bf2.pack(side="right", fill="y")
        ttk.Button(bf2, text="Добавить...", command=self.add_dir).pack(fill="x", pady=2)
        ttk.Button(bf2, text="Удалить", command=self.del_dir).pack(fill="x", pady=2)

        # --- параметры
        pf = ttk.LabelFrame(top, text="Параметры")
        pf.pack(fill="x", **pad)
        self.var_hidden = tk.BooleanVar(value=self.settings.include_hidden)
        self.var_env = tk.BooleanVar(value=self.settings.include_envelopes)
        self.var_colors = tk.BooleanVar(value=self.settings.colors)
        self.var_tol = tk.DoubleVar(value=self.settings.angle_tol)
        ttk.Checkbutton(pf, text="Скрытые элементы", variable=self.var_hidden).grid(row=0, column=0, sticky="w", padx=6)
        ttk.Checkbutton(pf, text="Рабочие зоны роботов", variable=self.var_env).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(pf, text="Цвета", variable=self.var_colors).grid(row=0, column=2, sticky="w", padx=6)
        ttk.Label(pf, text="Допуск компланарности, °:").grid(row=0, column=3, sticky="e", padx=6)
        ttk.Spinbox(pf, from_=0.01, to=5.0, increment=0.1, width=6, textvariable=self.var_tol).grid(row=0, column=4)
        self.var_fit = tk.BooleanVar(value=getattr(self.settings, "surface_fit", True))
        self.var_fit_tol = tk.DoubleVar(value=getattr(self.settings, "fit_tol", 0.1))
        ttk.Checkbutton(pf, text="Распознавать поверхности: цилиндры, конусы, сферы, торы, B-сплайны (меньше размер)",
                        variable=self.var_fit).grid(row=1, column=0, columnspan=3, sticky="w", padx=6)
        ttk.Label(pf, text="Допуск распознавания, мм:").grid(row=1, column=3, sticky="e", padx=6)
        ttk.Spinbox(pf, from_=0.005, to=1.0, increment=0.01, width=6,
                    textvariable=self.var_fit_tol).grid(row=1, column=4)

        # --- запуск
        rf = ttk.Frame(top)
        rf.pack(fill="x", **pad)
        self.btn = ttk.Button(rf, text="Конвертировать в STEP", command=self.start)
        self.btn.pack(side="left")
        self.pb = ttk.Progressbar(rf, mode="indeterminate")
        self.pb.pack(side="left", fill="x", expand=True, padx=8)
        self.status = ttk.Label(rf, text="Готово")
        self.status.pack(side="right")

        # --- журнал
        lg = ttk.LabelFrame(self, text="Журнал")
        lg.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(lg, wrap="word", height=12, font=("Consolas", 9))
        sb = ttk.Scrollbar(lg, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("warn", foreground="#b35c00")
        self.txt.tag_configure("err", foreground="#c00000")
        self.txt.tag_configure("ok", foreground="#006400")

    # ------------------------------------------------------------ действия
    def _add_file(self, f: str):
        if f and f not in self.lst.get(0, "end"):
            self.lst.insert("end", f)

    def pick_files(self):
        fs = filedialog.askopenfilenames(title="Выберите файлы .vcmx", initialdir=self.settings.last_dir or None,
                                         filetypes=[("KUKA.Sim / Visual Components", "*.vcmx *.vcm"), ("Все файлы", "*.*")])
        for f in fs:
            self._add_file(os.path.normpath(f))
        if fs:
            self.settings.last_dir = os.path.dirname(fs[0])

    def remove_sel(self):
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Папка для файлов STEP")
        if d:
            self.var_out.set(os.path.normpath(d))

    def add_dir(self):
        d = filedialog.askdirectory(title="Папка с компонентами .vcmx")
        if d:
            self.lst_dirs.insert("end", os.path.normpath(d))

    def del_dir(self):
        for i in reversed(self.lst_dirs.curselection()):
            self.lst_dirs.delete(i)

    def set_repo(self):
        r = simpledialog.askstring("Обновления", "Репозиторий GitHub (владелец/имя):",
                                   initialvalue=self.settings.github_repo, parent=self)
        if r:
            self.settings.github_repo = r.strip()
            self._save()

    def open_page(self):
        webbrowser.open(f"https://github.com/{self.settings.github_repo}")

    def about(self):
        messagebox.showinfo("О программе", f"{APP_TITLE}\nВерсия {__version__}\n\n"
                            "Конвертация .vcmx (KUKA.Sim Pro / Visual Components) в STEP AP214 "
                            "с сохранением структуры сборок, без триангуляции.\n"
                            "Геометрическое ядро: Open CASCADE Technology (LGPL).\n\n"
                            f"https://github.com/{self.settings.github_repo}")

    def _save(self):
        s = self.settings
        s.check_updates_on_start = bool(self.var_autoupd.get())
        s.search_dirs = list(self.lst_dirs.get(0, "end"))
        s.include_hidden = bool(self.var_hidden.get())
        s.include_envelopes = bool(self.var_env.get())
        s.colors = bool(self.var_colors.get())
        try:
            s.angle_tol = float(self.var_tol.get())
        except (tk.TclError, ValueError):
            s.angle_tol = 0.5
        s.surface_fit = bool(self.var_fit.get())
        try:
            s.fit_tol = max(0.001, float(self.var_fit_tol.get()))
        except (tk.TclError, ValueError):
            s.fit_tol = 0.1
        s.output_dir = self.var_out.get().strip()
        cfg.save(s)

    def log(self, msg: str, tag: str = ""):
        self.q.put(("log", msg, tag))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "log":
                    _, msg, tag = item
                    if not tag:
                        low = msg.lower()
                        if "не хватает" in low or "предупреждение" in low or "внимание" in low:
                            tag = "warn"
                        elif "ошибка" in low:
                            tag = "err"
                    self.txt.insert("end", msg + "\n", tag)
                    self.txt.see("end")
                elif kind == "status":
                    self.status.config(text=item[1])
                elif kind == "missing":
                    messagebox.showwarning("Предупреждение: не хватает встроенных моделей", item[1])
                elif kind == "done":
                    self._finish(item[1])
                elif kind == "call":
                    item[1]()
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ------------------------------------------------------------ конвертация
    def start(self):
        if self.busy:
            return
        files = list(self.lst.get(0, "end"))
        if not files:
            self.pick_files()
            files = list(self.lst.get(0, "end"))
            if not files:
                return
        self._save()
        s = self.settings
        opt = ConvertOptions(search_dirs=s.search_dirs, include_hidden=s.include_hidden,
                             include_envelopes=s.include_envelopes, colors=s.colors, angle_tol=s.angle_tol,
                             surface_fit=s.surface_fit, fit_tol=s.fit_tol)
        outdir = s.output_dir
        self.busy = True
        self.btn.state(["disabled"])
        self.pb.start(12)
        threading.Thread(target=self._worker, args=(files, outdir, opt), daemon=True).start()

    def _worker(self, files, outdir, opt):
        ok = 0
        for i, f in enumerate(files, 1):
            self.q.put(("status", f"{i}/{len(files)}: {os.path.basename(f)}"))
            out = None
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                out = os.path.join(outdir, os.path.splitext(os.path.basename(f))[0] + ".stp")
            try:
                res = convert(f, out, opt, log=self.log)
                ok += 1
                self.log(f"Готово: {res.output}", "ok")
                if res.needs_attention:
                    self.q.put(("missing", f"{os.path.basename(f)}\n\n{res.missing_message()}"))
            except Exception as ex:
                self.log(f"ОШИБКА при конвертации {f}: {ex}", "err")
                self.log(traceback.format_exc(), "err")
            self.log("-" * 60)
        self.q.put(("done", (ok, len(files))))

    def _finish(self, r):
        self.busy = False
        self.pb.stop()
        self.btn.state(["!disabled"])
        self.status.config(text=f"Готово: {r[0]} из {r[1]}")

    # ------------------------------------------------------------ обновления
    def check_updates(self, silent: bool):
        def work():
            try:
                info = updater.check(self.settings.github_repo)
            except Exception as ex:
                if not silent:
                    self.q.put(("call", lambda e=ex: messagebox.showerror("Обновления", f"Не удалось проверить обновления:\n{e}")))
                return
            if info is None:
                if not silent:
                    self.q.put(("call", lambda: messagebox.showinfo("Обновления", f"Установлена последняя версия {__version__}")))
                return
            self.q.put(("call", lambda: self._offer_update(info)))

        threading.Thread(target=work, daemon=True).start()

    def _offer_update(self, info):
        notes = info.notes.strip()
        if len(notes) > 800:
            notes = notes[:800] + "..."
        if not messagebox.askyesno("Доступно обновление",
                                   f"Доступна версия {info.version} (установлена {__version__}).\n\n{notes}\n\n"
                                   "Скачать и установить сейчас? Программа будет перезапущена."):
            return
        self.log(f"Загрузка обновления {info.version}...")
        self.pb.configure(mode="determinate", maximum=100, value=0)

        def prog(done, total):
            if total:
                self.q.put(("call", lambda v=done * 100 / total: self.pb.configure(value=v)))

        def work():
            try:
                path = updater.download(info, prog)
                self.log("Запуск установщика...")
                self.q.put(("call", lambda p=path: updater.run_installer(p)))
            except Exception as ex:
                self.q.put(("call", lambda e=ex: messagebox.showerror("Обновления", f"Ошибка загрузки обновления:\n{e}")))

        threading.Thread(target=work, daemon=True).start()


def main(files: List[str]) -> int:
    app = App(files)
    app.mainloop()
    return 0

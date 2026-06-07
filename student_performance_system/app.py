import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .data_utils import StudentInput, validate_student_input
from .database import HistoryDatabase
from .model_service import PredictionService


APP_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = APP_DIR / "dataset.csv"
DB_PATH = APP_DIR / "student_predictions.db"

COLORS = {
    "bg": "#eef3f8",
    "surface": "#ffffff",
    "surface_alt": "#f7fafc",
    "border": "#d7e0ea",
    "text": "#172033",
    "muted": "#667085",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "bar_track": "#e8eef7",
}

PREDICTION_COLORS = {
    "Low": COLORS["danger"],
    "Medium": COLORS["warning"],
    "High": COLORS["success"],
}


class StudentPerformanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Prediction System")
        self.geometry("1120x720")
        self.minsize(1040, 660)
        self.configure(bg=COLORS["bg"])

        self.service = PredictionService(DATASET_PATH)
        self.db = HistoryDatabase(DB_PATH)
        self.last_student = None
        self.last_prediction = None
        self.current_probabilities = {"Low": 0.0, "Medium": 0.0, "High": 0.0}

        self._configure_style()
        self._build_layout()
        self._refresh_history()
        self._draw_feature_importance()
        self._show_metrics()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        default_font = ("Times New Roman", 10)
        style.configure(".", font=default_font, foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Surface.TFrame", background=COLORS["surface"])
        style.configure("Panel.TFrame", background=COLORS["surface"], relief="solid", borderwidth=1)
        style.configure("Toolbar.TFrame", background=COLORS["surface"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=default_font)
        style.configure("Surface.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=default_font)
        style.configure("Muted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Times New Roman", 9))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Times New Roman", 22, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Times New Roman", 10))
        style.configure("HomeTitle.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Times New Roman", 30, "bold"))
        style.configure("HomeIntro.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Times New Roman", 12))
        style.configure("HomeAction.TButton", font=("Times New Roman", 12, "bold"), padding=(24, 12), background=COLORS["primary"], foreground="#ffffff")
        style.map(
            "HomeAction.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
            foreground=[("disabled", "#dbeafe"), ("!disabled", "#ffffff")],
        )
        style.configure("Section.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Times New Roman", 13, "bold"))
        style.configure("Result.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Times New Roman", 20, "bold"))
        style.configure("TButton", font=("Times New Roman", 10, "bold"), padding=(12, 8), borderwidth=0)
        style.configure("Accent.TButton", background=COLORS["primary"], foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
            foreground=[("disabled", "#dbeafe"), ("!disabled", "#ffffff")],
        )
        style.configure("TEntry", padding=(8, 5), relief="solid", bordercolor=COLORS["border"], lightcolor=COLORS["border"])
        style.configure("TCombobox", padding=(8, 5), relief="solid")
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Times New Roman", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", COLORS["surface"])], foreground=[("selected", COLORS["primary"])])
        style.configure(
            "Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=30,
            bordercolor=COLORS["border"],
            borderwidth=1,
        )
        style.configure("Treeview.Heading", background=COLORS["surface_alt"], foreground=COLORS["muted"], font=("Times New Roman", 9, "bold"))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", COLORS["text"])])
        style.configure("Horizontal.TProgressbar", troughcolor=COLORS["bar_track"], background=COLORS["primary"], bordercolor=COLORS["bar_track"])

    def _build_layout(self):
        self.root_frame = ttk.Frame(self, padding=(22, 18))
        self.root_frame.pack(fill="both", expand=True)

        self.home_frame = ttk.Frame(self.root_frame)
        self.workspace_frame = ttk.Frame(self.root_frame)

        self._build_home_screen(self.home_frame)
        self._build_workspace(self.workspace_frame)
        self._show_home()

    def _build_home_screen(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        content = ttk.Frame(parent)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)

        ttk.Label(
            content,
            text="Student Performance Prediction System",
            style="HomeTitle.TLabel",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 14))

        intro = (
            "This system uses student learning records and previous grades to predict final "
            "performance categories, helping teachers quickly review academic trends and "
            "prediction history."
        )
        ttk.Label(
            content,
            text=intro,
            style="HomeIntro.TLabel",
            anchor="center",
            justify="center",
            wraplength=720,
        ).grid(row=1, column=0, pady=(0, 34))

        actions = ttk.Frame(content)
        actions.grid(row=2, column=0)
        ttk.Button(
            actions,
            text="Predict",
            style="HomeAction.TButton",
            command=self._show_prediction_screen,
        ).pack(side="left", ipadx=18, padx=(0, 10))
        ttk.Button(
            actions,
            text="Feature Importance",
            style="HomeAction.TButton",
            command=self._show_feature_importance_screen,
        ).pack(side="left", ipadx=18, padx=10)
        ttk.Button(
            actions,
            text="History",
            style="HomeAction.TButton",
            command=self._show_history_screen,
        ).pack(side="left", ipadx=18, padx=(10, 0))

    def _build_workspace(self, root):
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Button(header, text="Home", command=self._show_home).pack(side="right", padx=(14, 0))
        self.workspace_title = ttk.Label(header, text="Student Performance Prediction System", style="Title.TLabel")
        self.workspace_title.pack(anchor="w")
        self.workspace_subtitle = ttk.Label(header, text="Random Forest prediction workspace for academic performance review", style="Subtitle.TLabel")
        self.workspace_subtitle.pack(anchor="w", pady=(3, 0))
        self.metrics_label = ttk.Label(header, text="", style="Subtitle.TLabel")
        self.metrics_label.pack(anchor="w", pady=(6, 0))

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_tabs(body)

    def _build_input_panel(self, parent):
        self.input_panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.input_panel.columnconfigure(1, weight=1)

        ttk.Label(self.input_panel, text="Student Input", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        self.vars = {
            "name": tk.StringVar(value="Student A"),
            "matric_no": tk.StringVar(value="A0001"),
            "sex": tk.StringVar(value="F"),
            "age": tk.StringVar(value="17"),
            "study_time": tk.StringVar(value="2"),
            "failures": tk.StringVar(value="0"),
            "activities": tk.StringVar(value="yes"),
            "absences": tk.StringVar(value="4"),
            "g1": tk.StringVar(value="60"),
            "g2": tk.StringVar(value="65"),
        }

        fields = [
            ("Name", "name", "entry"),
            ("Matric No.", "matric_no", "entry"),
            ("Gender", "sex", ["F", "M"]),
            ("Age", "age", "entry"),
            ("Study Time (1-4)", "study_time", ["1", "2", "3", "4"]),
            ("Failures", "failures", ["0", "1", "2", "3", "4"]),
            ("Activities", "activities", ["yes", "no"]),
            ("Absences", "absences", "entry"),
            ("Previous Grade G1 (0-100)", "g1", "entry"),
            ("Midterm Grade G2 (0-100)", "g2", "entry"),
        ]

        for row, (label, key, control) in enumerate(fields, start=1):
            ttk.Label(self.input_panel, text=label, style="Surface.TLabel").grid(row=row, column=0, sticky="w", pady=5)
            if isinstance(control, list):
                widget = ttk.Combobox(self.input_panel, textvariable=self.vars[key], values=control, state="readonly", width=19)
            else:
                widget = ttk.Entry(self.input_panel, textvariable=self.vars[key], width=22)
            widget.grid(row=row, column=1, sticky="ew", pady=5, padx=(12, 0))

        button_row = len(fields) + 1
        ttk.Button(self.input_panel, text="Predict", style="Accent.TButton", command=self._predict).grid(row=button_row, column=0, columnspan=2, sticky="ew", pady=(16, 5))
        ttk.Button(self.input_panel, text="Save Record", command=self._save_record).grid(row=button_row + 1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Button(self.input_panel, text="Batch CSV Predict", command=self._batch_predict).grid(row=button_row + 2, column=0, columnspan=2, sticky="ew", pady=5)

    def _build_tabs(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=1, sticky="nsew")

        self.result_tab = ttk.Frame(self.notebook, style="Surface.TFrame", padding=18)
        self.history_tab = ttk.Frame(self.notebook, style="Surface.TFrame", padding=18)

        self.notebook.add(self.result_tab, text="Prediction")
        self.notebook.add(self.history_tab, text="History")

        self.result_text = ttk.Label(self.result_tab, text="Ready for prediction", style="Result.TLabel")
        self.result_text.pack(anchor="w", pady=(0, 12))

        ttk.Label(
            self.result_tab,
            text="Enter student information and click Predict to see confidence by class.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 18))
        self.probability_frame = ttk.Frame(self.result_tab, style="Surface.TFrame")
        self.probability_frame.pack(fill="x", pady=(0, 12))
        self.probability_bars = {}
        for klass in ["Low", "Medium", "High"]:
            row = ttk.Frame(self.probability_frame, style="Surface.TFrame")
            row.pack(fill="x", pady=6)
            label = ttk.Label(row, text=klass, style="Surface.TLabel", width=9)
            label.pack(side="left")
            bar = tk.Canvas(row, height=18, bg=COLORS["surface"], highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=(8, 12))
            bar.bind("<Configure>", lambda _event, k=klass: self._redraw_probability_bar(k))
            value = ttk.Label(row, text="0.00%", style="Muted.TLabel", width=8)
            value.pack(side="right")
            self.probability_bars[klass] = (bar, value)

        self.feature_frame = ttk.Frame(parent, style="Surface.TFrame", padding=18)
        self.feature_frame.grid(row=0, column=1, sticky="nsew")
        self.feature_frame.grid_remove()
        self.feature_canvas = tk.Canvas(self.feature_frame, bg=COLORS["surface"], height=440, highlightthickness=1, highlightbackground=COLORS["border"])
        self.feature_canvas.pack(fill="both", expand=True)
        self.feature_canvas.bind("<Configure>", lambda _event: self._draw_feature_importance())

        controls = ttk.Frame(self.history_tab, style="Toolbar.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Refresh", command=self._refresh_history).pack(side="left")
        ttk.Button(controls, text="View Detail", command=self._view_history_detail).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Delete Selected", command=self._delete_selected_history).pack(side="left", padx=8)
        ttk.Button(controls, text="Export CSV", command=self._export_history).pack(side="left")

        columns = ("id", "timestamp", "student_name", "matric_no", "g1", "g2", "prediction_result", "confidence_score")
        self.history_tree = ttk.Treeview(self.history_tab, columns=columns, show="headings", height=16)
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=110, anchor="center")
        self.history_tree.column("timestamp", width=150)
        self.history_tree.column("student_name", width=140)
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.tag_configure("odd", background=COLORS["surface"])
        self.history_tree.tag_configure("even", background=COLORS["surface_alt"])
        self.history_tree.bind("<Double-1>", lambda _event: self._view_history_detail())

    def _show_home(self):
        self.workspace_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def _show_prediction_screen(self):
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Student Performance Prediction System",
            "Random Forest prediction workspace for academic performance review",
            show_metrics=True,
        )
        self.feature_frame.grid_remove()
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.notebook.select(self.result_tab)

    def _show_feature_importance_screen(self):
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Feature Importance",
            "This chart shows which student attributes contribute most strongly to the trained Random Forest model.",
            show_metrics=False,
        )
        self.input_panel.grid_remove()
        self.notebook.grid_remove()
        self.feature_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._draw_feature_importance()

    def _show_history_screen(self):
        self._refresh_history()
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Student Performance Prediction System",
            "Random Forest prediction workspace for academic performance review",
            show_metrics=True,
        )
        self.feature_frame.grid_remove()
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.notebook.select(self.history_tab)

    def _set_workspace_header(self, title, subtitle, show_metrics):
        self.workspace_title.configure(text=title)
        self.workspace_subtitle.configure(text=subtitle)
        if show_metrics:
            self.metrics_label.pack(anchor="w", pady=(6, 0))
        else:
            self.metrics_label.pack_forget()

    def _show_metrics(self):
        metrics = self.service.metrics
        self.metrics_label.configure(
            text=(
                f"Dataset: {metrics['train_size']} training / {metrics['test_size']} testing records | "
                f"Holdout accuracy: {metrics['accuracy']:.2%} | "
                f"Classes: {metrics['class_distribution']}"
            )
        )

    def _predict(self):
        try:
            student = validate_student_input({key: var.get() for key, var in self.vars.items()})
            result = self.service.predict(student)
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.last_student = student
        self.last_prediction = result
        self.result_text.configure(text=f"Prediction: {result['prediction']} | Confidence: {result['confidence']:.2%}")
        self._set_probability_text(result["probabilities"])

    def _save_record(self):
        if not self.last_student or not self.last_prediction:
            messagebox.showwarning("No Prediction", "Please run a prediction before saving.")
            return
        self.db.add_prediction(
            self.last_student,
            self.last_prediction["prediction"],
            self.last_prediction["confidence"],
        )
        self._refresh_history()
        messagebox.showinfo("Saved", "Prediction record saved successfully.")

    def _batch_predict(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="batch_predictions.csv",
        )
        if not output_path:
            return

        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            results = []
            for index, row in enumerate(rows, start=1):
                student = StudentInput(
                    name=row.get("name", f"Student {index}"),
                    matric_no=row.get("matric_no", "-"),
                    sex=row["sex"],
                    age=int(float(row["age"])),
                    study_time=int(float(row.get("study_time", row.get("weeklystudytime", 2)))),
                    failures=int(float(row["failures"])),
                    activities=row["activities"],
                    absences=int(float(row["absences"])),
                    g1=self._read_score_100(row, "G1"),
                    g2=self._read_score_100(row, "G2"),
                )
                result = self.service.predict(student)
                row["prediction_result"] = result["prediction"]
                row["confidence_score"] = f"{result['confidence']:.4f}"
                results.append(row)
            fieldnames = list(results[0].keys()) if results else ["prediction_result", "confidence_score"]
            with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            messagebox.showinfo("Batch Complete", f"Saved predictions to:\n{output_path}")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))

    def _set_probability_text(self, probabilities):
        self.current_probabilities = dict(probabilities)
        for klass in ["Low", "Medium", "High"]:
            probability = probabilities.get(klass, 0.0)
            bar, value_label = self.probability_bars[klass]
            value_label.configure(text=f"{probability:.2%}")
            self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _redraw_probability_bar(self, klass):
        if not hasattr(self, "probability_bars"):
            return
        probability = self.current_probabilities.get(klass, 0.0)
        bar, _value_label = self.probability_bars[klass]
        self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _draw_probability_bar(self, canvas, probability, color):
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 260)
        height = 18
        radius = 9
        canvas.delete("all")
        self._rounded_rect(canvas, 0, 0, width, height, radius, fill=COLORS["bar_track"], outline="")
        fill_width = max(int(width * probability), radius * 2 if probability > 0 else 0)
        if fill_width:
            self._rounded_rect(canvas, 0, 0, fill_width, height, radius, fill=color, outline="")

    def _rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    @staticmethod
    def _read_score_100(row, key):
        value = float(row[key])
        if 0 <= value <= 20:
            return int(round(value * 5))
        return int(round(value))

    def _draw_feature_importance(self):
        if not hasattr(self, "feature_canvas"):
            return
        self.feature_canvas.delete("all")
        items = self.service.feature_importances()
        canvas_width = max(self.feature_canvas.winfo_width(), 760)
        chart_width = 760
        chart_left = max((canvas_width - chart_width) // 2, 22)
        left = chart_left + 210
        top = 30
        bar_height = 24
        gap = 16

        self.feature_canvas.create_text(chart_left, 16, text="Random Forest Feature Importance", anchor="nw", fill=COLORS["text"], font=("Times New Roman", 14, "bold"))
        self.feature_canvas.create_text(chart_left, 42, text="Higher bars indicate stronger influence in the trained model.", anchor="nw", fill=COLORS["muted"], font=("Times New Roman", 9))
        max_value = max((value for _, value in items), default=1) or 1
        for i, (label, value) in enumerate(items):
            y = top + 52 + i * (bar_height + gap)
            bar_width = int((value / max_value) * 430)
            self.feature_canvas.create_text(left - 14, y + bar_height / 2, text=label, anchor="e", fill=COLORS["text"], font=("Times New Roman", 10))
            self._rounded_rect(self.feature_canvas, left, y, left + 430, y + bar_height, 7, fill=COLORS["bar_track"], outline="")
            self._rounded_rect(self.feature_canvas, left, y, left + bar_width, y + bar_height, 7, fill=COLORS["primary"], outline="")
            self.feature_canvas.create_text(left + 446, y + bar_height / 2, text=f"{value:.1%}", anchor="w", fill=COLORS["muted"], font=("Times New Roman", 10, "bold"))
        self.feature_canvas.configure(scrollregion=(0, 0, canvas_width, top + 58 + len(items) * (bar_height + gap)))

    def _refresh_history(self):
        if not hasattr(self, "history_tree"):
            return
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for index, row in enumerate(self.db.list_predictions()):
            values = (
                row["id"],
                row["timestamp"],
                row["student_name"],
                row["matric_no"],
                row["g1"],
                row["g2"],
                row["prediction_result"],
                f"{row['confidence_score']:.2%}",
            )
            tag = "even" if index % 2 else "odd"
            self.history_tree.insert("", "end", values=values, tags=(tag,))

    def _delete_selected_history(self):
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a history record to delete.")
            return

        values = self.history_tree.item(selected[0], "values")
        prediction_id = values[0]
        confirm = messagebox.askyesno(
            "Delete Record",
            f"Delete prediction record ID {prediction_id}?",
        )
        if not confirm:
            return

        deleted = self.db.delete_prediction(prediction_id)
        if deleted:
            self._refresh_history()
            messagebox.showinfo("Deleted", "History record deleted successfully.")
        else:
            messagebox.showerror("Delete Failed", "The selected record no longer exists.")

    def _view_history_detail(self):
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a history record to view.")
            return

        values = self.history_tree.item(selected[0], "values")
        row = self.db.get_prediction(values[0])
        if not row:
            messagebox.showerror("Record Not Found", "The selected history record no longer exists.")
            self._refresh_history()
            return

        student = StudentInput(
            name=row["student_name"],
            matric_no=row["matric_no"],
            sex=row["sex"],
            age=row["age"],
            study_time=row["study_time"],
            failures=row["failures"],
            activities=row["activities"],
            absences=row["absences"],
            g1=row["g1"],
            g2=row["g2"],
        )
        explanation = self.service.local_feature_importances(student)
        self._show_detail_window(row, explanation)

    def _show_detail_window(self, row, explanation):
        window = tk.Toplevel(self)
        window.title(f"Prediction Detail - {row['student_name']}")
        window.geometry("720x620")
        window.minsize(640, 520)

        body = ttk.Frame(window, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Prediction Detail", font=("Times New Roman", 15, "bold")).pack(anchor="w")
        probabilities = explanation["probabilities"]
        summary = (
            f"Student: {row['student_name']} ({row['matric_no']})\n"
            f"Saved: {row['timestamp']}\n"
            f"Prediction: {row['prediction_result']} | Saved confidence: {row['confidence_score']:.2%}\n"
            f"Current model confidence: {explanation['confidence']:.2%}\n"
            f"Class probabilities: Low {probabilities.get('Low', 0.0):.2%}, "
            f"Medium {probabilities.get('Medium', 0.0):.2%}, High {probabilities.get('High', 0.0):.2%}"
        )
        ttk.Label(body, text=summary, justify="left").pack(anchor="w", pady=(8, 12))

        columns = ("feature", "current", "reference", "impact", "direction")
        tree = ttk.Treeview(body, columns=columns, show="headings", height=10)
        headings = {
            "feature": "Feature",
            "current": "Student Value",
            "reference": "Typical Value",
            "impact": "Impact",
            "direction": "Effect",
        }
        widths = {"feature": 170, "current": 110, "reference": 110, "impact": 100, "direction": 130}
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center")
        tree.column("feature", anchor="w")
        tree.pack(fill="both", expand=True)

        for item in explanation["items"]:
            tree.insert(
                "",
                "end",
                values=(
                    item["feature"],
                    item["current_value"],
                    item["reference_value"],
                    f"{item['impact']:+.2%}",
                    "Supports prediction" if item["direction"] == "supports" else "Reduces prediction",
                ),
            )

        note = (
            "Impact means the change in predicted-class probability when that single feature is replaced "
            "with a typical training-set value. Larger absolute values have stronger influence for this student."
        )
        ttk.Label(body, text=note, wraplength=660, justify="left").pack(anchor="w", pady=(12, 0))

    def _export_history(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="prediction_history.csv",
        )
        if not path:
            return
        self.db.export_csv(path)
        messagebox.showinfo("Exported", f"History exported to:\n{path}")


def main():
    app = StudentPerformanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()

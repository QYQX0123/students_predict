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


class StudentPerformanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Prediction System")
        self.geometry("1040x680")
        self.minsize(980, 620)

        self.service = PredictionService(DATASET_PATH)
        self.db = HistoryDatabase(DB_PATH)
        self.last_student = None
        self.last_prediction = None

        self._configure_style()
        self._build_layout()
        self._refresh_history()
        self._draw_feature_importance()
        self._show_metrics()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f7f8fa")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f7f8fa", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#f7f8fa", font=("Segoe UI", 18, "bold"))
        style.configure("Result.TLabel", background="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))

    def _build_layout(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Student Performance Prediction System", style="Title.TLabel").pack(anchor="w")
        self.metrics_label = ttk.Label(root, text="")
        self.metrics_label.pack(anchor="w", pady=(4, 12))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_tabs(body)

    def _build_input_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 14))

        ttk.Label(panel, text="Student Input", style="Panel.TLabel", font=("Segoe UI", 13, "bold")).grid(
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
            ttk.Label(panel, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            if isinstance(control, list):
                widget = ttk.Combobox(panel, textvariable=self.vars[key], values=control, state="readonly", width=18)
            else:
                widget = ttk.Entry(panel, textvariable=self.vars[key], width=21)
            widget.grid(row=row, column=1, sticky="ew", pady=4, padx=(10, 0))

        button_row = len(fields) + 1
        ttk.Button(panel, text="Predict", command=self._predict).grid(row=button_row, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        ttk.Button(panel, text="Save Record", command=self._save_record).grid(row=button_row + 1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(panel, text="Batch CSV Predict", command=self._batch_predict).grid(row=button_row + 2, column=0, columnspan=2, sticky="ew", pady=4)

    def _build_tabs(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=1, sticky="nsew")

        self.result_tab = ttk.Frame(notebook, padding=14)
        self.feature_tab = ttk.Frame(notebook, padding=14)
        self.history_tab = ttk.Frame(notebook, padding=14)

        notebook.add(self.result_tab, text="Prediction")
        notebook.add(self.feature_tab, text="Feature Importance")
        notebook.add(self.history_tab, text="History")

        self.result_text = ttk.Label(self.result_tab, text="Enter student information and click Predict.", style="Result.TLabel")
        self.result_text.pack(anchor="w", pady=(0, 12))

        self.probability_text = tk.Text(self.result_tab, height=8, width=60, borderwidth=0, font=("Consolas", 11))
        self.probability_text.pack(fill="x")
        self.probability_text.configure(state="disabled")

        self.feature_canvas = tk.Canvas(self.feature_tab, bg="#ffffff", height=420, highlightthickness=1, highlightbackground="#d0d5dd")
        self.feature_canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self.history_tab)
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
        self.history_tree.bind("<Double-1>", lambda _event: self._view_history_detail())

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
        self.probability_text.configure(state="normal")
        self.probability_text.delete("1.0", "end")
        self.probability_text.insert("end", "Class probabilities\n\n")
        for klass in ["Low", "Medium", "High"]:
            self.probability_text.insert("end", f"{klass:<8}: {probabilities.get(klass, 0.0):.2%}\n")
        self.probability_text.configure(state="disabled")

    @staticmethod
    def _read_score_100(row, key):
        value = float(row[key])
        if 0 <= value <= 20:
            return int(round(value * 5))
        return int(round(value))

    def _draw_feature_importance(self):
        self.feature_canvas.delete("all")
        items = self.service.feature_importances()
        width = 760
        left = 190
        top = 28
        bar_height = 28
        gap = 18

        self.feature_canvas.create_text(20, 12, text="Random Forest Feature Importance", anchor="nw", font=("Segoe UI", 13, "bold"))
        max_value = max((value for _, value in items), default=1) or 1
        for i, (label, value) in enumerate(items):
            y = top + 34 + i * (bar_height + gap)
            bar_width = int((value / max_value) * 430)
            self.feature_canvas.create_text(left - 12, y + bar_height / 2, text=label, anchor="e", font=("Segoe UI", 10))
            self.feature_canvas.create_rectangle(left, y, left + bar_width, y + bar_height, fill="#2f80ed", outline="")
            self.feature_canvas.create_text(left + bar_width + 8, y + bar_height / 2, text=f"{value:.1%}", anchor="w", font=("Segoe UI", 10))
        self.feature_canvas.configure(scrollregion=(0, 0, width, top + 34 + len(items) * (bar_height + gap)))

    def _refresh_history(self):
        if not hasattr(self, "history_tree"):
            return
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for row in self.db.list_predictions():
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
            self.history_tree.insert("", "end", values=values)

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

        ttk.Label(body, text="Prediction Detail", font=("Segoe UI", 15, "bold")).pack(anchor="w")
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

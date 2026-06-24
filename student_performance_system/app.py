"""Tkinter desktop interface for the student performance prediction system.

中文：负责主窗口、页面导航、学生表单、预测结果、模型评估、历史记录和文件对话框。
模型与数据库细节分别交给 PredictionService 和 HistoryDatabase。
English: Builds the main window, navigation, student form, prediction results,
model evaluation, history views, and file dialogs. Model and persistence details
are delegated to PredictionService and HistoryDatabase.
"""

import csv
import posixpath
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from xml.etree import ElementTree
from zipfile import ZipFile

from .data_utils import STUDY_TIME_COLUMN, StudentInput, validate_student_input
from .database import HistoryDatabase
from .model_service import PredictionService


APP_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = APP_DIR / "dataset.csv"
DB_PATH = APP_DIR / "student_predictions.db"

BATCH_COLUMN_ALIASES = {
    "name": ("name", "student name", "student_name"),
    "matric_no": (
        "matric_no",
        "matric no",
        "matric no.",
        "student number",
        "student_number",
        "student id",
        "student_id",
        "matric",
    ),
    "sex": ("sex", "gender"),
    "age": ("age",),
    "study_time": ("study_time", "study time", "weekly study time", "weeklystudytime", STUDY_TIME_COLUMN),
    "failures": ("failures", "failure", "previous failures"),
    "activities": ("activities", "activity", "extracurricular activities"),
    "absences": ("absences", "absence", "absent"),
    "g1": ("g1", "previous grade", "previous grade g1", "previous grade (g1)", "previous grade g1 (0-100)"),
    "g2": ("g2", "midterm grade", "midterm grade g2", "midterm grade (g2)", "midterm grade g2 (0-100)"),
}

BATCH_FIELD_LABELS = {
    "name": "Name",
    "matric_no": "Matric No.",
    "sex": "Gender/Sex",
    "age": "Age",
    "study_time": "Study Time",
    "failures": "Failures",
    "activities": "Activities",
    "absences": "Absences",
    "g1": "G1 (0-100)",
    "g2": "G2 (0-100)",
}

# 中文：集中管理颜色，使所有页面风格一致，并方便后续统一更换主题。
# English: A centralized palette keeps every screen consistent and easy to restyle.
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

# 中文：预测类别与颜色的映射，用于三条置信度进度条。
# English: Prediction-class colors used by the three confidence bars.
PREDICTION_COLORS = {
    "Low": COLORS["danger"],
    "Medium": COLORS["warning"],
    "High": COLORS["success"],
}


class StudentPerformanceApp(tk.Tk):
    """Main application window containing all screens and event handlers.

    中文：类继承 tk.Tk，因此实例本身就是根窗口。以下划线开头的方法是按钮、
    窗口尺寸事件和初始化流程使用的内部回调。
    English: This class inherits tk.Tk, so its instance is the root window.
    Underscore-prefixed methods are internal callbacks for buttons, resize events,
    and initialization.
    """

    def __init__(self):
        """Initialize the window, services, application state, and widgets.

        中文：PredictionService 创建时同步训练模型，HistoryDatabase 确保数据表存在。
        所有页面只构建一次，之后通过显示或隐藏容器实现导航。
        English: PredictionService trains synchronously during creation, and
        HistoryDatabase ensures its table exists. Screens are built once and later
        navigation only shows or hides their containers.
        """
        super().__init__()
        self.title("Student Performance Prediction System")
        self.geometry("1160x720")
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
        self._populate_evaluation()
        self._show_metrics()

    def _configure_style(self):
        """Define reusable ttk typography, colors, spacing, and widget states.

        中文：clam 主题提供较稳定的跨平台自定义效果；style.map 配置按钮悬停、
        按下、禁用以及表格选中等动态状态。
        English: The clam theme provides predictable cross-platform customization.
        style.map defines active, pressed, disabled, and selected states.
        """
        style = ttk.Style(self)
        style.theme_use("clam")
        default_font = ("Times New Roman", 11)
        style.configure(".", font=default_font, foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Surface.TFrame", background=COLORS["surface"])
        style.configure("Panel.TFrame", background=COLORS["surface"], relief="solid", borderwidth=1)
        style.configure("Toolbar.TFrame", background=COLORS["surface"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=default_font)
        style.configure("Surface.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=default_font)
        style.configure("Muted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Times New Roman", 10))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Times New Roman", 22, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Times New Roman", 11))
        style.configure("HomeTitle.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Times New Roman", 30, "bold"))
        style.configure("HomeIntro.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Times New Roman", 13))
        style.configure("HomeAction.TButton", font=("Times New Roman", 13, "bold"), padding=(24, 12), background=COLORS["primary"], foreground="#ffffff")
        style.map(
            "HomeAction.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
            foreground=[("disabled", "#dbeafe"), ("!disabled", "#ffffff")],
        )
        style.configure("Section.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Times New Roman", 14, "bold"))
        style.configure("Result.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Times New Roman", 20, "bold"))
        style.configure("ResultCard.TFrame", background=COLORS["surface_alt"], relief="solid", borderwidth=1)
        style.configure(
            "ResultCardTitle.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["text"],
            font=("Times New Roman", 20, "bold"),
        )
        style.configure(
            "ResultCardMuted.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["muted"],
            font=("Times New Roman", 10),
        )
        style.configure("Insight.TFrame", background=COLORS["surface_alt"], relief="solid", borderwidth=1)
        style.configure(
            "InsightTitle.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["text"],
            font=("Times New Roman", 13, "bold"),
        )
        style.configure(
            "InsightBody.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["muted"],
            font=("Times New Roman", 10),
        )
        style.configure(
            "Positive.InsightBody.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["success"],
            font=("Times New Roman", 10, "bold"),
        )
        style.configure(
            "Negative.InsightBody.TLabel",
            background=COLORS["surface_alt"],
            foreground=COLORS["danger"],
            font=("Times New Roman", 10, "bold"),
        )
        style.configure("TButton", font=("Times New Roman", 11, "bold"), padding=(12, 6), borderwidth=0)
        style.configure("Accent.TButton", background=COLORS["primary"], foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_hover"])],
            foreground=[("disabled", "#dbeafe"), ("!disabled", "#ffffff")],
        )
        style.configure("Success.TButton", background=COLORS["success"], foreground="#ffffff")
        style.map(
            "Success.TButton",
            background=[("active", "#15803d"), ("pressed", "#166534")],
            foreground=[("disabled", "#dcfce7"), ("!disabled", "#ffffff")],
        )
        style.configure("TEntry", padding=(8, 3), relief="solid", bordercolor=COLORS["border"], lightcolor=COLORS["border"])
        style.configure("TCombobox", padding=(8, 3), relief="solid")
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Times New Roman", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected", COLORS["surface"])], foreground=[("selected", COLORS["primary"])])
        style.configure(
            "Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=32,
            bordercolor=COLORS["border"],
            borderwidth=1,
        )
        style.configure("Treeview.Heading", background=COLORS["surface_alt"], foreground=COLORS["muted"], font=("Times New Roman", 10, "bold"))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", COLORS["text"])])
        style.configure("Horizontal.TProgressbar", troughcolor=COLORS["bar_track"], background=COLORS["primary"], bordercolor=COLORS["bar_track"])

    def _build_layout(self):
        """Create home and workspace containers / 创建主页与工作区顶层容器。"""
        self.root_frame = ttk.Frame(self, padding=(20, 14))
        self.root_frame.pack(fill="both", expand=True)

        self.home_frame = ttk.Frame(self.root_frame)
        self.workspace_frame = ttk.Frame(self.root_frame)

        self._build_home_screen(self.home_frame)
        self._build_workspace(self.workspace_frame)
        self._show_home()

    def _build_home_screen(self, parent):
        """Build the centered home screen and its three navigation actions / 构建居中主页和三个入口。"""
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
            text="Model Evaluation",
            style="HomeAction.TButton",
            command=self._show_evaluation_screen,
        ).pack(side="left", ipadx=18, padx=10)
        ttk.Button(
            actions,
            text="History",
            style="HomeAction.TButton",
            command=self._show_history_screen,
        ).pack(side="left", ipadx=18, padx=(10, 0))

    def _build_workspace(self, root):
        """Build the shared header and responsive body used by work screens.

        中文：预测页使用左侧输入、右侧结果；评估页和历史页隐藏这两部分并横跨两列。
        English: Prediction uses left-side input and right-side results. Evaluation
        and history hide them and span both grid columns.
        """
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(header, text="Home", command=self._show_home).pack(side="right", padx=(14, 0))
        ttk.Button(header, text="History", command=self._show_history_screen).pack(side="right")
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
        """Build the student form and bind controls to Tkinter StringVar values.

        中文：固定选项使用只读 Combobox，自由文本和数字使用 Entry。
        English: Fixed choices use readonly Combobox widgets; free text and numbers
        use Entry widgets.
        """
        self.input_panel = ttk.Frame(parent, style="Panel.TFrame", padding=(16, 12))
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.input_panel.columnconfigure(1, weight=1)

        ttk.Label(self.input_panel, text="Student Input", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
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

        # 中文：每个元组定义一行的标签、变量键，以及控件类型或下拉选项。
        # English: Each tuple defines a label, variable key, and control type/options.
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
            ttk.Label(self.input_panel, text=label, style="Surface.TLabel").grid(row=row, column=0, sticky="w", pady=2)
            if isinstance(control, list):
                widget = ttk.Combobox(self.input_panel, textvariable=self.vars[key], values=control, state="readonly", width=19)
            else:
                widget = ttk.Entry(self.input_panel, textvariable=self.vars[key], width=22)
            widget.grid(row=row, column=1, sticky="ew", pady=2, padx=(12, 0))

        button_row = len(fields) + 1
        ttk.Button(self.input_panel, text="Predict", style="Accent.TButton", command=self._predict).grid(row=button_row, column=0, columnspan=2, sticky="ew", pady=(10, 3))
        ttk.Button(self.input_panel, text="Save Record", style="Success.TButton", command=self._save_record).grid(row=button_row + 1, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(self.input_panel, text="Import CSV/XLSX to History", command=self._batch_predict).grid(row=button_row + 2, column=0, columnspan=2, sticky="ew", pady=3)

    def _build_tabs(self, parent):
        """Build prediction, model-evaluation, and history widgets.

        中文：控件仅创建一次并在页面切换时复用，避免重复绑定事件和重建表格。
        English: Widgets are created once and reused, avoiding duplicate event
        bindings and unnecessary reconstruction.
        """
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=1, sticky="nsew")

        self.result_tab = ttk.Frame(self.notebook, style="Surface.TFrame", padding=16)
        self.result_tab.columnconfigure(0, weight=1, uniform="insight")
        self.result_tab.columnconfigure(1, weight=1, uniform="insight")
        self.result_tab.rowconfigure(2, weight=1)

        self.notebook.add(self.result_tab, text="Prediction")

        result_card = ttk.Frame(self.result_tab, style="ResultCard.TFrame", padding=(16, 13))
        result_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.result_text = ttk.Label(
            result_card,
            text="Ready for prediction",
            style="ResultCardTitle.TLabel",
        )
        self.result_text.pack(anchor="w")
        ttk.Label(
            result_card,
            text="Enter student information and click Predict to see confidence by class.",
            style="ResultCardMuted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.probability_frame = ttk.Frame(self.result_tab, style="Surface.TFrame")
        self.probability_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(
            self.probability_frame,
            text="Class Probability",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 4))
        self.probability_bars = {}
        for klass in ["Low", "Medium", "High"]:
            # 中文：Canvas 可为每个类别提供独立颜色和圆角，因此不使用 ttk.Progressbar。
            # English: Canvas supports per-class colors and rounded ends unlike ttk.Progressbar.
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

        factors_card = ttk.Frame(self.result_tab, style="Insight.TFrame", padding=(15, 13))
        factors_card.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        ttk.Label(factors_card, text="Key Factors", style="InsightTitle.TLabel").pack(anchor="w")
        ttk.Label(
            factors_card,
            text="Largest local effects compared with typical training values.",
            style="InsightBody.TLabel",
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(3, 10))
        self.key_factor_labels = []
        for index in range(3):
            label = ttk.Label(
                factors_card,
                text=f"{index + 1}. Run a prediction to calculate this factor.",
                style="InsightBody.TLabel",
                wraplength=300,
                justify="left",
            )
            label.pack(anchor="w", fill="x", pady=4)
            self.key_factor_labels.append(label)

        attention_card = ttk.Frame(self.result_tab, style="Insight.TFrame", padding=(15, 13))
        attention_card.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        ttk.Label(attention_card, text="Suggested Attention", style="InsightTitle.TLabel").pack(anchor="w")
        self.attention_label = ttk.Label(
            attention_card,
            text=(
                "Run a prediction to receive a focused review note. Suggestions are "
                "for academic support and should not be treated as certain outcomes."
            ),
            style="InsightBody.TLabel",
            wraplength=300,
            justify="left",
        )
        self.attention_label.pack(anchor="w", fill="x", pady=(10, 0))

        self.evaluation_frame = ttk.Frame(parent, style="Surface.TFrame", padding=18)
        self.evaluation_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.evaluation_frame.grid_remove()
        self.evaluation_frame.columnconfigure(0, weight=1)
        self.evaluation_frame.rowconfigure(2, weight=1)

        self.evaluation_summary = ttk.Label(
            self.evaluation_frame,
            text="",
            style="Section.TLabel",
            justify="left",
        )
        self.evaluation_summary.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        tables = ttk.Frame(self.evaluation_frame, style="Surface.TFrame")
        tables.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        tables.columnconfigure(0, weight=3)
        tables.columnconfigure(1, weight=2)

        metrics_panel = ttk.Frame(tables, style="Surface.TFrame")
        metrics_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        ttk.Label(metrics_panel, text="Classification Metrics (5-Fold Out-of-Fold)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        metric_columns = ("class", "precision", "recall", "f1", "support")
        self.metrics_tree = ttk.Treeview(metrics_panel, columns=metric_columns, show="headings", height=4)
        for column, heading, width in [
            ("class", "Class", 90),
            ("precision", "Precision", 90),
            ("recall", "Recall", 90),
            ("f1", "F1-score", 90),
            ("support", "Support", 75),
        ]:
            self.metrics_tree.heading(column, text=heading)
            self.metrics_tree.column(column, width=width, anchor="center")
        self.metrics_tree.pack(fill="x")

        matrix_panel = ttk.Frame(tables, style="Surface.TFrame")
        matrix_panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(matrix_panel, text="Confusion Matrix (Actual x Predicted)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        matrix_columns = ("actual", "low", "medium", "high")
        self.matrix_tree = ttk.Treeview(matrix_panel, columns=matrix_columns, show="headings", height=3)
        for column, heading in zip(matrix_columns, ("Actual", "Low", "Medium", "High")):
            self.matrix_tree.heading(column, text=heading)
            self.matrix_tree.column(column, width=78, anchor="center")
        self.matrix_tree.pack(fill="x")

        self.feature_canvas = tk.Canvas(
            self.evaluation_frame,
            bg=COLORS["surface"],
            height=280,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.feature_canvas.grid(row=2, column=0, sticky="nsew")
        self.feature_canvas.bind("<Configure>", lambda _event: self._draw_feature_importance())

        self.history_frame = ttk.Frame(parent, style="Surface.TFrame", padding=18)
        self.history_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.history_frame.grid_remove()

        controls = ttk.Frame(self.history_frame, style="Toolbar.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Refresh", command=self._refresh_history).pack(side="left")
        ttk.Button(controls, text="View Detail", command=self._view_history_detail).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Delete Selected", command=self._delete_selected_history).pack(side="left", padx=8)
        ttk.Button(controls, text="Export CSV", command=self._export_history).pack(side="left")

        columns = ("id", "timestamp", "student_name", "matric_no", "g1", "g2", "prediction_result", "confidence_score")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings", height=16, selectmode="extended")
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
        """Hide the workspace and return home / 隐藏工作区并返回主页。"""
        self.workspace_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def _show_prediction_screen(self):
        """Show the single-student prediction workflow / 显示单个学生预测页。"""
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Student Performance Prediction System",
            "Random Forest prediction workspace for academic performance review",
            show_metrics=True,
        )
        self.history_frame.grid_remove()
        self.evaluation_frame.grid_remove()
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.notebook.select(self.result_tab)

    def _show_evaluation_screen(self):
        """Show cross-validation metrics, confusion matrix, and global importance.

        中文：进入页面时重新填充表格并重绘图表，以适应可能变化的窗口尺寸。
        English: Tables and charts refresh on entry to accommodate window-size changes.
        """
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Model Evaluation",
            "Review 5-fold cross-validation performance, classification metrics, errors, and feature importance.",
            show_metrics=False,
        )
        self.input_panel.grid_remove()
        self.notebook.grid_remove()
        self.history_frame.grid_remove()
        self.evaluation_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._populate_evaluation()
        self._draw_feature_importance()

    def _show_history_screen(self):
        """Refresh and show saved prediction records / 刷新并显示预测历史。"""
        self._refresh_history()
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "History",
            "Review saved prediction records, open record details, delete old entries, or export the history to CSV.",
            show_metrics=False,
        )
        self.input_panel.grid_remove()
        self.notebook.grid_remove()
        self.evaluation_frame.grid_remove()
        self.history_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

    def _set_workspace_header(self, title, subtitle, show_metrics):
        """Set page headings and toggle the compact metrics line / 更新页头并控制指标行。"""
        self.workspace_title.configure(text=title)
        self.workspace_subtitle.configure(text=subtitle)
        if show_metrics:
            self.metrics_label.pack(anchor="w", pady=(6, 0))
        else:
            self.metrics_label.pack_forget()

    def _show_metrics(self):
        """Display the production model's holdout summary.

        中文：这里是 80/20 留出测试结果；完整五折结果位于 Model Evaluation 页面。
        English: This is the 80/20 holdout result; full five-fold results appear on
        the Model Evaluation screen.
        """
        metrics = self.service.metrics
        self.metrics_label.configure(
            text=(
                f"Dataset: {metrics['train_size']} training / {metrics['test_size']} testing records | "
                f"Holdout accuracy: {metrics['accuracy']:.2%} | "
                f"Classes: {metrics['class_distribution']}"
            )
        )

    def _populate_evaluation(self):
        """Populate all evaluation widgets from PredictionService metrics.

        中文：显示每折准确率、均值和标准差，然后写入各类别 Precision、Recall、F1、
        Support、宏平均，以及“实际类别 × 预测类别”混淆矩阵。
        English: Displays fold accuracies, mean/std, per-class precision, recall, F1,
        support, macro averages, and the actual-by-predicted confusion matrix.
        """
        if not hasattr(self, "metrics_tree"):
            return
        evaluation = self.service.metrics["evaluation"]
        fold_text = ", ".join(f"{value:.2%}" for value in evaluation["fold_accuracies"])
        self.evaluation_summary.configure(
            text=(
                f"5-Fold Cross-Validation Accuracy: {evaluation['mean_accuracy']:.2%} "
                f"(standard deviation {evaluation['std_accuracy']:.2%})\n"
                f"Fold results: {fold_text}"
            )
        )

        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for klass in evaluation["classes"]:
            values = evaluation["per_class"][klass]
            self.metrics_tree.insert(
                "",
                "end",
                values=(
                    klass,
                    f"{values['precision']:.2%}",
                    f"{values['recall']:.2%}",
                    f"{values['f1']:.2%}",
                    values["support"],
                ),
            )
        macro = evaluation["macro_average"]
        self.metrics_tree.insert(
            "",
            "end",
            values=("Macro Avg", f"{macro['precision']:.2%}", f"{macro['recall']:.2%}", f"{macro['f1']:.2%}", macro["support"]),
        )

        for item in self.matrix_tree.get_children():
            self.matrix_tree.delete(item)
        matrix = evaluation["confusion_matrix"]
        for actual in ("Low", "Medium", "High"):
            self.matrix_tree.insert(
                "",
                "end",
                values=(actual, matrix[actual]["Low"], matrix[actual]["Medium"], matrix[actual]["High"]),
            )

    def _predict(self):
        """Validate the form, request a prediction, and update result widgets.

        中文：失败时显示异常并停止；成功后缓存学生和结果，Save Record 只保存最近一次
        有效预测。
        English: Errors are displayed and abort the action. On success, the student
        and result are cached so Save Record stores only the latest valid prediction.
        """
        try:
            student = validate_student_input({key: var.get() for key, var in self.vars.items()})
            result = self.service.predict(student)
            explanation = self.service.local_feature_importances(student)
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.last_student = student
        self.last_prediction = result
        self.result_text.configure(text=f"Prediction: {result['prediction']} | Confidence: {result['confidence']:.2%}")
        self._set_probability_text(result["probabilities"])
        self._update_prediction_insights(result["prediction"], explanation["items"])

    def _update_prediction_insights(self, prediction, items):
        """Refresh per-student factors and a cautious academic support note."""
        top_items = items[:3]
        for index, label in enumerate(self.key_factor_labels):
            if index >= len(top_items):
                label.configure(
                    text=f"{index + 1}. No additional factor available.",
                    style="InsightBody.TLabel",
                )
                continue

            item = top_items[index]
            impact = item["impact"]
            if impact > 0:
                effect = f"Supports predicted class (+{impact:.2%})"
                label_style = "Positive.InsightBody.TLabel"
            elif impact < 0:
                effect = f"Reduces predicted-class confidence ({impact:.2%})"
                label_style = "Negative.InsightBody.TLabel"
            else:
                effect = "No measurable local effect"
                label_style = "InsightBody.TLabel"
            label.configure(
                text=f"{index + 1}. {item['feature']}\n{effect}",
                style=label_style,
            )

        reducing_items = [item for item in items if item["impact"] < 0]
        focus = reducing_items[0]["feature"] if reducing_items else None
        if prediction == "Low":
            message = (
                "Priority review is recommended. Check recent grades, attendance, "
                "and learning progress, then consider targeted academic support."
            )
        elif prediction == "Medium":
            message = (
                "Continue monitoring progress and reinforce weaker areas before the "
                "final assessment. Short, targeted follow-up may help maintain momentum."
            )
        else:
            message = (
                "Maintain the current learning pattern while continuing routine progress "
                "checks. Strong predicted performance still benefits from consistent support."
            )

        if focus:
            message += f"\n\nModel review focus: {focus} currently has the strongest negative local effect."
        else:
            message += "\n\nNo input showed a negative local effect for the predicted class."
        self.attention_label.configure(text=message)

    def _save_record(self):
        """Persist the latest valid prediction / 保存最近一次有效预测。"""
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

    def _batch_predict_legacy(self):
        """Predict every CSV row and write an enriched output CSV.

        中文：输出保留原始列，并追加 prediction_result 和 confidence_score。任何读取、
        字段或数值错误都会通过消息框报告。
        English: The output preserves original columns and appends prediction_result
        and confidence_score. File, field, and numeric errors are reported in a dialog.
        """
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
                # 中文：兼容应用字段 study_time 和原数据集字段 weeklystudytime。
                # English: Accept both app field study_time and dataset field weeklystudytime.
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
                # 中文：带 BOM 的 UTF-8 可减少 Windows Excel 打开中文时的乱码。
                # English: UTF-8 with BOM improves Chinese detection in Windows Excel.
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            messagebox.showinfo("Batch Complete", f"Saved predictions to:\n{output_path}")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))

    def _set_probability_text(self, probabilities):
        """Update text and redraw all probability bars / 更新文本并重绘概率条。"""
        self.current_probabilities = dict(probabilities)
        for klass in ["Low", "Medium", "High"]:
            probability = probabilities.get(klass, 0.0)
            bar, value_label = self.probability_bars[klass]
            value_label.configure(text=f"{probability:.2%}")
            self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _redraw_probability_bar(self, klass):
        """Redraw one bar after Canvas resize / Canvas 尺寸变化后重绘一条概率条。"""
        if not hasattr(self, "probability_bars"):
            return
        probability = self.current_probabilities.get(klass, 0.0)
        bar, _value_label = self.probability_bars[klass]
        self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _draw_probability_bar(self, canvas, probability, color):
        """Draw a rounded track and proportional colored fill.

        中文：非零值至少显示一个圆角直径，避免很小的概率完全不可见。
        English: Nonzero values receive at least one rounded-end diameter so tiny
        probabilities remain visible.
        """
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
        """Approximate a rounded rectangle using a smoothed Canvas polygon.

        中文：Canvas 没有原生圆角矩形，因此利用重复控制点和 smooth=True 模拟。
        English: Canvas has no native rounded rectangle, so repeated control points
        and smooth=True approximate curved corners.
        """
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _batch_predict(self):
        """Import CSV/XLSX rows, validate them, predict, and save to history."""
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV and Excel files", "*.csv *.xlsx"),
                ("CSV files", "*.csv"),
                ("Excel workbooks", "*.xlsx"),
            ]
        )
        if not path:
            return

        try:
            students = self._load_batch_students(Path(path))
            if not students:
                raise ValueError("The selected file does not contain any student rows.")
            for student in students:
                result = self.service.predict(student)
                self.db.add_prediction(student, result["prediction"], result["confidence"])
            self._refresh_history()
            self._show_history_screen()
            messagebox.showinfo("Batch Complete", f"Imported {len(students)} prediction records into History.")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))

    @classmethod
    def _load_batch_students(cls, path):
        """Read supported batch files and return fully validated student inputs."""
        rows = cls._read_batch_rows(path)
        students = []
        errors = []
        for row_number, row in rows:
            values = cls._batch_row_to_form_values(row)
            try:
                students.append(validate_student_input(values, require_identity=True))
            except ValueError as exc:
                errors.append(f"Row {row_number}: {exc}")

        if errors:
            visible_errors = errors[:10]
            remaining = len(errors) - len(visible_errors)
            message = "Batch import failed. Please fix these rows:\n" + "\n".join(visible_errors)
            if remaining > 0:
                message += f"\n... and {remaining} more row(s)."
            raise ValueError(message)
        return students

    @classmethod
    def _read_batch_rows(cls, path):
        """Read .csv or .xlsx files as row dictionaries keyed by header names."""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls._read_csv_batch_rows(path)
        if suffix == ".xlsx":
            return cls._read_xlsx_batch_rows(path)
        raise ValueError("Only .csv and .xlsx files can be imported.")

    @classmethod
    def _read_csv_batch_rows(cls, path):
        """Read CSV rows while preserving spreadsheet-like row numbers."""
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cls._validate_batch_headers(reader.fieldnames or [])
            rows = []
            for row_number, row in enumerate(reader, start=2):
                clean_row = {key: value for key, value in row.items() if key is not None}
                if cls._row_has_value(clean_row):
                    rows.append((row_number, clean_row))
            return rows

    @classmethod
    def _read_xlsx_batch_rows(cls, path):
        """Read the first worksheet from an XLSX file using only the standard library."""
        try:
            with ZipFile(path) as workbook:
                sheet_path = cls._first_worksheet_path(workbook)
                shared_strings = cls._read_shared_strings(workbook)
                root = ElementTree.fromstring(workbook.read(sheet_path))
        except KeyError as exc:
            raise ValueError("The XLSX file is missing required worksheet data.") from exc

        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        raw_rows = []
        for row_element in root.findall(".//main:sheetData/main:row", namespace):
            row_number = int(row_element.attrib.get("r", len(raw_rows) + 1))
            values = {}
            for cell in row_element.findall("main:c", namespace):
                reference = cell.attrib.get("r", "")
                match = re.match(r"([A-Z]+)", reference)
                if not match:
                    continue
                values[cls._column_number(match.group(1))] = cls._xlsx_cell_text(cell, shared_strings, namespace)
            if any(value.strip() for value in values.values()):
                raw_rows.append((row_number, values))

        if not raw_rows:
            raise ValueError("The XLSX file is empty.")

        _header_row_number, header_cells = raw_rows[0]
        max_column = max(header_cells)
        headers = [header_cells.get(index, "").strip() for index in range(1, max_column + 1)]
        cls._validate_batch_headers(headers)

        rows = []
        for row_number, cells in raw_rows[1:]:
            row = {
                header: cells.get(index, "")
                for index, header in enumerate(headers, start=1)
                if header
            }
            if cls._row_has_value(row):
                rows.append((row_number, row))
        return rows

    @classmethod
    def _first_worksheet_path(cls, workbook):
        """Resolve the first sheet path from workbook relationships."""
        workbook_root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        namespace = {
            "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        first_sheet = workbook_root.find("main:sheets/main:sheet", namespace)
        if first_sheet is None:
            raise ValueError("The XLSX file does not contain any worksheet.")
        relation_id = first_sheet.attrib.get(f"{{{namespace['rel']}}}id")

        relationships_root = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_namespace = {"pkg": "http://schemas.openxmlformats.org/package/2006/relationships"}
        for relationship in relationships_root.findall("pkg:Relationship", rel_namespace):
            if relationship.attrib.get("Id") == relation_id:
                target = relationship.attrib["Target"]
                if target.startswith("/"):
                    return target.lstrip("/")
                return posixpath.normpath(posixpath.join("xl", target))
        raise ValueError("The XLSX workbook relationship for the first sheet is invalid.")

    @staticmethod
    def _read_shared_strings(workbook):
        """Return the workbook shared string table, if one exists."""
        if "xl/sharedStrings.xml" not in workbook.namelist():
            return []
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = []
        for item in root.findall("main:si", namespace):
            strings.append("".join(text.text or "" for text in item.findall(".//main:t", namespace)))
        return strings

    @classmethod
    def _xlsx_cell_text(cls, cell, shared_strings, namespace):
        """Convert one XLSX cell to the text used by the batch validator."""
        cell_type = cell.attrib.get("t")
        if cell_type == "s":
            value = cell.find("main:v", namespace)
            if value is None or value.text is None:
                return ""
            return shared_strings[int(value.text)].strip()
        if cell_type == "inlineStr":
            inline = cell.find("main:is", namespace)
            if inline is None:
                return ""
            return "".join(text.text or "" for text in inline.findall(".//main:t", namespace)).strip()

        value = cell.find("main:v", namespace)
        if value is None or value.text is None:
            return ""
        return cls._display_spreadsheet_value(value.text)

    @staticmethod
    def _display_spreadsheet_value(value):
        """Keep spreadsheet numbers readable for validation error messages."""
        try:
            number = float(value)
        except ValueError:
            return str(value).strip()
        if number.is_integer():
            return str(int(number))
        return str(number).strip()

    @staticmethod
    def _column_number(column_letters):
        """Convert Excel column letters into a 1-based index."""
        number = 0
        for letter in column_letters:
            number = number * 26 + ord(letter) - ord("A") + 1
        return number

    @classmethod
    def _validate_batch_headers(cls, headers):
        """Ensure the import file contains every column required by the form."""
        normalized_headers = {cls._normalize_header(header) for header in headers if header}
        missing = []
        for key, aliases in BATCH_COLUMN_ALIASES.items():
            if not any(cls._normalize_header(alias) in normalized_headers for alias in aliases):
                missing.append(BATCH_FIELD_LABELS[key])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

    @classmethod
    def _batch_row_to_form_values(cls, row):
        """Map flexible spreadsheet headers into the exact form field keys."""
        normalized_row = {}
        for key, value in row.items():
            normalized_key = cls._normalize_header(key)
            if normalized_key and normalized_key not in normalized_row:
                normalized_row[normalized_key] = "" if value is None else str(value).strip()

        values = {}
        for key, aliases in BATCH_COLUMN_ALIASES.items():
            values[key] = ""
            for alias in aliases:
                value = normalized_row.get(cls._normalize_header(alias), "")
                if value != "":
                    values[key] = value
                    break
        return values

    @staticmethod
    def _normalize_header(header):
        """Normalize headers so CSV/XLSX templates can use friendly labels."""
        return re.sub(r"[^a-z0-9]+", "", str(header).strip().lower())

    @staticmethod
    def _row_has_value(row):
        """Return True when at least one imported cell contains content."""
        return any(str(value or "").strip() for value in row.values())

    @staticmethod
    def _read_score_100(row, key):
        """Normalize batch scores to 0-100 / 将批量 CSV 的 0-20 或 0-100 成绩统一为百分制。"""
        value = float(row[key])
        if 0 <= value <= 20:
            return int(round(value * 5))
        return int(round(value))

    def _draw_feature_importance(self):
        """Render normalized global feature importance as horizontal bars / 绘制全局特征重要性。"""
        if not hasattr(self, "feature_canvas"):
            return
        self.feature_canvas.delete("all")
        items = self.service.feature_importances()
        canvas_width = max(self.feature_canvas.winfo_width(), 760)
        chart_width = 760
        chart_left = max((canvas_width - chart_width) // 2, 22)
        left = chart_left + 210
        top = 16
        bar_height = 14
        gap = 7

        self.feature_canvas.create_text(chart_left, 16, text="Random Forest Feature Importance", anchor="nw", fill=COLORS["text"], font=("Times New Roman", 14, "bold"))
        self.feature_canvas.create_text(chart_left, 40, text="Higher bars indicate stronger influence in the trained model.", anchor="nw", fill=COLORS["muted"], font=("Times New Roman", 10))
        max_value = max((value for _, value in items), default=1) or 1
        for i, (label, value) in enumerate(items):
            # 中文：相对最大值缩放，既保留比例又充分利用图表宽度。
            # English: Scale against the maximum to preserve ratios and use chart width.
            y = top + 48 + i * (bar_height + gap)
            bar_width = int((value / max_value) * 430)
            self.feature_canvas.create_text(left - 14, y + bar_height / 2, text=label, anchor="e", fill=COLORS["text"], font=("Times New Roman", 11))
            self._rounded_rect(self.feature_canvas, left, y, left + 430, y + bar_height, 7, fill=COLORS["bar_track"], outline="")
            self._rounded_rect(self.feature_canvas, left, y, left + bar_width, y + bar_height, 7, fill=COLORS["primary"], outline="")
            self.feature_canvas.create_text(left + 446, y + bar_height / 2, text=f"{value:.1%}", anchor="w", fill=COLORS["muted"], font=("Times New Roman", 11, "bold"))
        self.feature_canvas.configure(scrollregion=(0, 0, canvas_width, top + 54 + len(items) * (bar_height + gap)))

    def _refresh_history(self):
        """Reload database rows and apply alternating table colors.

        中文：先删除旧 Treeview 行，再插入数据库最新内容，避免显示过期记录。
        English: Existing Treeview rows are removed before current database rows are
        inserted, preventing stale history from remaining visible.
        """
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

    def _delete_selected_history_single_legacy(self):
        """Validate selection, confirm deletion, then refresh / 检查选择、确认删除并刷新。"""
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

    def _delete_selected_history(self):
        """Delete one or more selected history rows after confirmation."""
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select one or more history records to delete.")
            return

        prediction_ids = [self.history_tree.item(item, "values")[0] for item in selected]
        id_text = ", ".join(str(prediction_id) for prediction_id in prediction_ids[:10])
        if len(prediction_ids) > 10:
            id_text += f", ... and {len(prediction_ids) - 10} more"

        confirm = messagebox.askyesno(
            "Delete Records",
            f"Delete {len(prediction_ids)} selected prediction record(s)?\n\nIDs: {id_text}",
        )
        if not confirm:
            return

        deleted = self.db.delete_predictions(prediction_ids)
        if deleted:
            self._refresh_history()
            messagebox.showinfo("Deleted", f"Deleted {deleted} history record(s) successfully.")
        else:
            messagebox.showerror("Delete Failed", "The selected records no longer exist.")

    def _view_history_detail(self):
        """Reconstruct the selected student and calculate a current explanation.

        中文：数据库保留当时结果，详情页使用当前模型重新计算概率和影响；模型或数据集
        改变后，两者可能不同。
        English: The database preserves the original result, while the detail screen
        recalculates probabilities and impacts with the current model. They may differ
        after model or dataset changes.
        """
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
        """Display saved facts and one-feature-at-a-time local influences.

        中文：影响值为原概率减去替换成典型值后的概率。正值支持当前预测，负值降低该
        预测类别概率。
        English: Impact is original probability minus probability after reference
        replacement. Positive values support the prediction; negative values reduce it.
        """
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
        """Choose a destination and export history / 选择目标路径并导出历史 CSV。"""
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
    """Create the root window and start Tkinter / 创建根窗口并启动事件循环。"""
    app = StudentPerformanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()

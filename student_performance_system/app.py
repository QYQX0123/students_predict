"""Tkinter user interface for the student performance prediction system.

中文：这个模块实现整个桌面应用界面：主页、预测表单、结果卡片、模型评估、历史记录、
CSV/XLSX 批量导入、详情窗口和导出操作。界面层只负责组织用户操作与显示状态；训练、
预测和数据库读写分别交给 PredictionService 与 HistoryDatabase。这样写可以把“用户
交互”与“业务逻辑”分开，后续修改模型或存储时不必重写窗口布局。

English: This module implements the desktop UI: home page, prediction form, result
cards, model evaluation, history table, CSV/XLSX batch import, detail window, and
export actions. The UI organizes user interaction and display state, while training,
prediction, and persistence are delegated to PredictionService and HistoryDatabase.
This separation lets model or storage changes happen without rewriting the window
layout.
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

STUDY_TIME_OPTIONS = [
    "1 (<5 hours)",
    "2 (5 - 10 hours)",
    "3 (10 - 20 hours)",
    "4 (>20 hours)",
]

FAILURE_OPTIONS = ["0", "1", "2", "3", "4+"]

# 中文：所有颜色集中在一个字典里，页面、表格、按钮和图表都引用同一套值；
# 这样实现主题一致，也让后期换色只需要改一处。
# English: All colors live in one dictionary and are reused by pages, tables,
# buttons, and charts. This keeps the theme consistent and makes restyling localized.
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

# 中文：预测类别到颜色的映射，用于把 Pass 和 Fail 的含义直接体现在概率条上。
# English: Prediction-to-color mapping makes Pass and Fail meanings visible in bars.
PREDICTION_COLORS = {
    "Fail": COLORS["danger"],
    "Pass": COLORS["success"],
}

PROBABILITY_BAR_LABELS = (("Pass", "Pass Probability"), ("Fail", "Fail Risk"))


class StudentPerformanceApp(tk.Tk):
    """Root window, screen manager, and event-handler collection.

    中文：这个类继承 tk.Tk，实例本身就是应用根窗口。它保存模型服务、数据库连接、
    表单变量、最近一次预测结果和各个页面控件引用。这样写的功能是让所有界面事件都能
    访问同一份应用状态，同时把窗口逻辑限制在一个类中。

    English: This class inherits tk.Tk, so the instance is the root window. It owns
    the prediction service, database layer, form variables, latest prediction, and
    widget references. Keeping these in one class lets UI callbacks share application
    state while containing window logic in one place.
    """

    def __init__(self):
        """Create services, build widgets, migrate history, and show initial metrics.

        中文：启动时立即训练模型、打开历史数据库并迁移旧记录，然后一次性创建所有页面。
        页面后续通过 grid/pack 的显示与隐藏切换，而不是反复销毁重建；这样能减少重复
        事件绑定，也让窗口响应更稳定。

        English: Startup trains the model, opens the history database, migrates old
        records, and builds all screens once. Later navigation shows or hides
        grid/pack containers instead of destroying and recreating widgets, reducing
        duplicate bindings and keeping the UI stable.
        """
        super().__init__()
        self.title("Student Performance Prediction System")
        self.geometry("1160x720")
        self.minsize(1040, 660)
        self.configure(bg=COLORS["bg"])

        self.service = PredictionService(DATASET_PATH)
        self.db = HistoryDatabase(DB_PATH)
        self._migrate_legacy_history_records()
        self.last_student = None
        self.last_prediction = None
        self.current_probabilities = {"Fail": 0.0, "Pass": 0.0}

        self._configure_style()
        self._build_layout()
        self._refresh_history()
        self._populate_evaluation()
        self._show_metrics()

    def _configure_style(self):
        """Register shared ttk styles for the whole application.

        中文：这里统一设置字体、颜色、按钮状态、表格行高、Notebook 标签和结果卡片样式。
        选择 ttk.Style 而不是逐个控件手动设置，是为了让界面保持一致，并减少重复参数。

        English: This method centralizes fonts, colors, button states, table row
        height, Notebook tabs, and result-card styling. ttk.Style is used instead of
        per-widget options so the UI stays consistent with less repetition.
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
        """Build the two top-level screen containers.

        中文：应用只有主页和工作区两个顶层容器；预测、评估和历史都属于工作区内部视图。
        这样写让“回到主页”和“在工作区内切换功能”成为两个清晰层次。

        English: The app has two top-level containers: home and workspace. Prediction,
        evaluation, and history are workspace views. This keeps home navigation and
        workspace navigation as separate levels.
        """
        self.root_frame = ttk.Frame(self, padding=(20, 14))
        self.root_frame.pack(fill="both", expand=True)

        self.home_frame = ttk.Frame(self.root_frame)
        self.workspace_frame = ttk.Frame(self.root_frame)

        self._build_home_screen(self.home_frame)
        self._build_workspace(self.workspace_frame)
        self._show_home()

    def _build_home_screen(self, parent):
        """Build the home page and its navigation buttons.

        中文：主页只提供三个入口：Predict、Model Evaluation 和 History。它不放复杂表格，
        是为了让用户启动后先选择任务，再进入具体工作区。

        English: The home screen provides only three entry points: Predict, Model
        Evaluation, and History. Complex controls are kept out so users choose a task
        before entering a focused workspace.
        """
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
            "pass probability, helping teachers quickly review academic risk and prediction "
            "history."
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
        """Build the shared header and body grid for workspace screens.

        中文：工作区顶部放标题、说明和模型摘要；主体区域采用两列网格。预测页使用左列
        表单和右列结果，评估页/历史页则隐藏这些部件并横跨两列显示完整表格或图表。

        English: The workspace header contains title, subtitle, and model summary.
        The body uses a two-column grid: prediction uses form-left/result-right,
        while evaluation and history hide those widgets and span both columns.
        """
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(header, text="Home", command=self._show_home).pack(side="right", padx=(14, 0))
        ttk.Button(header, text="History", command=self._show_history_screen).pack(side="right")
        self.workspace_title = ttk.Label(header, text="Student Performance Prediction System", style="Title.TLabel")
        self.workspace_title.pack(anchor="w")
        self.workspace_subtitle = ttk.Label(header, text="Random Forest workspace for pass-probability review", style="Subtitle.TLabel")
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
        """Build the prediction input form.

        中文：每个输入控件绑定一个 StringVar，按钮回调可以一次性读取当前值。性别、
        学习时间、失败次数和课外活动使用只读下拉框，是为了避免无效枚举值；姓名、
        学号和分数使用文本框，交给 validate_student_input 做统一校验。

        English: Each control is bound to a StringVar so callbacks can read the
        current form at once. Gender, study time, failures, and activities use
        readonly comboboxes to prevent invalid categories; names, IDs, and grades use
        entries and are validated centrally by validate_student_input.
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
            "study_time": tk.StringVar(value="2 (5 - 10 hours)"),
            "failures": tk.StringVar(value="0"),
            "activities": tk.StringVar(value="yes"),
            "absences": tk.StringVar(value="4"),
            "g1": tk.StringVar(value="60"),
            "g2": tk.StringVar(value="65"),
        }

        # 中文：用数据列表生成表单行，避免为每个字段重复写一段几乎相同的控件创建代码。
        # English: Generate form rows from data to avoid repeating similar widget code.
        fields = [
            ("Name", "name", "entry"),
            ("Matric No.", "matric_no", "entry"),
            ("Gender", "sex", ["F", "M"]),
            ("Age", "age", "entry"),
            ("Study Time", "study_time", STUDY_TIME_OPTIONS),
            ("Failures", "failures", FAILURE_OPTIONS),
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
        """Create prediction output, evaluation, and history widgets.

        中文：这里的“tabs”不只包含 Notebook，也创建评估页和历史页的独立 Frame。所有
        控件只构建一次，之后由页面切换函数决定显示哪个容器；这样可以保持 Treeview
        列设置、Canvas 绑定和结果状态不丢失。

        English: Despite the name, this creates the Notebook plus separate evaluation
        and history frames. Widgets are built once and shown by navigation methods,
        preserving Treeview columns, Canvas bindings, and result state.
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
            text="Enter student information and click Predict to see the estimated pass probability.",
            style="ResultCardMuted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.probability_frame = ttk.Frame(self.result_tab, style="Surface.TFrame")
        self.probability_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(
            self.probability_frame,
            text="Pass Probability",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 4))
        self.probability_bars = {}
        for klass, display_label in PROBABILITY_BAR_LABELS:
            # 中文：Canvas 允许自定义颜色、圆角和重绘逻辑，比 ttk.Progressbar 更适合显示两类概率。
            # English: Canvas supports custom colors, rounded ends, and redraw logic for two probabilities.
            row = ttk.Frame(self.probability_frame, style="Surface.TFrame")
            row.pack(fill="x", pady=6)
            label = ttk.Label(row, text=display_label, style="Surface.TLabel", width=15)
            label.pack(side="left")
            bar = tk.Canvas(row, height=18, bg=COLORS["surface"], highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=(8, 12))
            bar.bind("<Configure>", lambda _event, k=klass: self._redraw_probability_bar(k))
            value = ttk.Label(row, text="0.00%", style="Muted.TLabel", width=8)
            value.pack(side="right")
            self.probability_bars[klass] = (bar, value)

        key_factors_card = ttk.Frame(self.result_tab, style="Insight.TFrame", padding=(15, 13))
        key_factors_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(key_factors_card, text="Key Factors", style="InsightTitle.TLabel").pack(anchor="w")
        self.key_factors_label = ttk.Label(
            key_factors_card,
            text="Run a prediction to see the strongest factors for this student.",
            style="InsightBody.TLabel",
            wraplength=280,
            justify="left",
        )
        self.key_factors_label.pack(anchor="w", fill="x", pady=(10, 0))

        attention_card = ttk.Frame(self.result_tab, style="Insight.TFrame", padding=(15, 13))
        attention_card.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
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
        self.evaluation_frame.rowconfigure(1, weight=1)

        self.evaluation_summary = ttk.Label(
            self.evaluation_frame,
            text="",
            style="Section.TLabel",
            justify="left",
        )
        self.evaluation_summary.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        evaluation_grid = ttk.Frame(self.evaluation_frame, style="Surface.TFrame")
        evaluation_grid.grid(row=1, column=0, sticky="nsew")
        evaluation_grid.columnconfigure(0, weight=1)
        evaluation_grid.columnconfigure(1, weight=1)
        evaluation_grid.rowconfigure(0, weight=1)
        evaluation_grid.rowconfigure(1, weight=1)

        comparison_panel = ttk.Frame(evaluation_grid, style="Surface.TFrame")
        comparison_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        ttk.Label(
            comparison_panel,
            text="Model Comparison (Same 80/20 Holdout Split)",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        comparison_columns = ("model", "accuracy", "macro_f1")
        self.comparison_tree = ttk.Treeview(comparison_panel, columns=comparison_columns, show="headings", height=3)
        for column, heading, width, anchor in [
            ("model", "Model", 180, "w"),
            ("accuracy", "Accuracy", 95, "center"),
            ("macro_f1", "Macro F1", 95, "center"),
        ]:
            self.comparison_tree.heading(column, text=heading)
            self.comparison_tree.column(column, width=width, anchor=anchor)
        self.comparison_tree.pack(fill="x")

        matrix_panel = ttk.Frame(evaluation_grid, style="Surface.TFrame")
        matrix_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        ttk.Label(matrix_panel, text="Confusion Matrix (Actual x Predicted)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        matrix_columns = ("actual", "fail", "pass")
        self.matrix_tree = ttk.Treeview(matrix_panel, columns=matrix_columns, show="headings", height=2)
        for column, heading in zip(matrix_columns, ("Actual", "Fail", "Pass")):
            self.matrix_tree.heading(column, text=heading)
            self.matrix_tree.column(column, width=84, anchor="center")
        self.matrix_tree.pack(fill="x")

        metrics_panel = ttk.Frame(evaluation_grid, style="Surface.TFrame")
        metrics_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(10, 0))
        ttk.Label(metrics_panel, text="Pass/Fail Metrics (5-Fold Out-of-Fold)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
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

        feature_panel = ttk.Frame(evaluation_grid, style="Surface.TFrame")
        feature_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(10, 0))
        self.feature_canvas = tk.Canvas(
            feature_panel,
            bg=COLORS["surface"],
            height=230,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.feature_canvas.pack(fill="both", expand=True)
        self.feature_canvas.bind("<Configure>", lambda _event: self._draw_feature_importance())

        self.history_frame = ttk.Frame(parent, style="Surface.TFrame", padding=18)
        self.history_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.history_frame.grid_remove()

        controls = ttk.Frame(self.history_frame, style="Toolbar.TFrame")
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Refresh", command=self._refresh_history).pack(side="left")
        ttk.Button(controls, text="View Detail", command=self._view_history_detail).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Delete Selected", command=self._delete_selected_history).pack(side="left", padx=8)
        ttk.Button(controls, text="Export Table", command=self._export_history).pack(side="left")

        columns = ("id", "timestamp", "student_name", "matric_no", "g1", "g2", "predicted_g3", "prediction_result", "pass_probability")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings", height=16, selectmode="extended")
        headings = {
            "id": "ID",
            "timestamp": "Timestamp",
            "student_name": "Student",
            "matric_no": "Matric No.",
            "g1": "G1",
            "g2": "G2",
            "predicted_g3": "Predicted G3",
            "prediction_result": "Result",
            "pass_probability": "Pass Probability",
        }
        for col in columns:
            self.history_tree.heading(col, text=headings[col])
            self.history_tree.column(col, width=110, anchor="center")
        self.history_tree.column("timestamp", width=150)
        self.history_tree.column("student_name", width=140)
        self.history_tree.column("predicted_g3", width=120)
        self.history_tree.column("pass_probability", width=130)
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.tag_configure("odd", background=COLORS["surface"])
        self.history_tree.tag_configure("even", background=COLORS["surface_alt"])
        self.history_tree.bind("<Double-1>", lambda _event: self._view_history_detail())

    def _show_home(self):
        """Switch back to the home page.

        中文：隐藏工作区、显示主页。没有销毁控件，所以用户返回预测页时之前的输入仍在。

        English: Hides the workspace and shows the home page. Widgets are not
        destroyed, so previous form values remain when returning to prediction.
        """
        self.workspace_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def _show_prediction_screen(self):
        """Show the single-student prediction workflow.

        中文：此视图显示左侧输入表单和右侧预测结果，并隐藏评估与历史容器。它还显示
        模型摘要，帮助用户知道当前模型基于多少训练/测试记录。

        English: This view shows the left input form and right prediction result
        while hiding evaluation and history containers. It also shows model summary
        text so users know the train/test counts behind the current model.
        """
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Student Performance Prediction System",
            "Random Forest workspace for pass-probability review",
            show_metrics=True,
        )
        self.history_frame.grid_remove()
        self.evaluation_frame.grid_remove()
        self.input_panel.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.notebook.select(self.result_tab)

    def _show_evaluation_screen(self):
        """Show model-quality diagnostics.

        中文：评估页整合五折交叉验证、分类指标、混淆矩阵和特征重要性。进入页面时重新
        填表和画图，是因为窗口尺寸可能变化，Canvas 图表需要按当前宽度重新布局。

        English: The evaluation view combines five-fold validation, classification
        metrics, confusion matrix, and feature importance. Tables and charts refresh
        on entry because the window may have resized and the Canvas chart needs a
        current layout.
        """
        self.home_frame.pack_forget()
        self.workspace_frame.pack(fill="both", expand=True)
        self._set_workspace_header(
            "Model Evaluation",
            "Review 5-fold pass/fail metrics, errors, and feature importance.",
            show_metrics=False,
        )
        self.input_panel.grid_remove()
        self.notebook.grid_remove()
        self.history_frame.grid_remove()
        self.evaluation_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._populate_evaluation()
        self._draw_feature_importance()

    def _show_history_screen(self):
        """Show saved prediction history.

        中文：进入历史页前先刷新数据库内容，避免用户刚保存、删除或批量导入后看到旧表格。

        English: The database is refreshed before showing history so saves, deletes,
        and imports are reflected immediately.
        """
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
        """Update the shared workspace title area.

        中文：不同页面复用同一个 header，只替换标题、副标题和是否显示模型摘要；这样
        页面切换时视觉结构保持一致。

        English: All workspace views reuse one header and only change the title,
        subtitle, and model-summary visibility, keeping navigation visually stable.
        """
        self.workspace_title.configure(text=title)
        self.workspace_subtitle.configure(text=subtitle)
        if show_metrics:
            self.metrics_label.pack(anchor="w", pady=(6, 0))
        else:
            self.metrics_label.pack_forget()

    def _show_metrics(self):
        """Render the compact model summary shown on the prediction page.

        中文：这里展示训练集/测试集大小、留出准确率和类别分布。完整评估信息放到独立
        评估页，避免预测页过于拥挤。

        English: Shows train/test size, holdout accuracy, and class distribution.
        Full diagnostics live on the evaluation page so the prediction page stays
        focused.
        """
        metrics = self.service.metrics
        self.metrics_label.configure(
            text=(
                f"Dataset: {metrics['train_size']} training / {metrics['test_size']} testing records | "
                f"Holdout accuracy: {metrics['accuracy']:.2%} | "
                f"Pass/Fail distribution: {metrics['class_distribution']}"
            )
        )

    def _populate_evaluation(self):
        """Fill evaluation summary, metric table, and confusion matrix.

        中文：数据来自 PredictionService.metrics["evaluation"]。函数实现三块内容：
        五折准确率摘要、每类 Precision/Recall/F1/Support、真实类别乘预测类别的混淆
        矩阵。把填表逻辑集中在这里，可以在初始化和进入评估页时复用。

        English: Data comes from PredictionService.metrics["evaluation"]. The method
        fills three areas: fold-accuracy summary, per-class precision/recall/F1/
        support, and the actual-by-predicted confusion matrix. Centralizing this
        logic lets initialization and page entry reuse it.
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

        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        for row in self.service.metrics["model_comparison"]:
            self.comparison_tree.insert(
                "",
                "end",
                values=(
                    row["model"],
                    f"{row['accuracy']:.2%}",
                    f"{row['macro_f1']:.2%}",
                ),
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
        for actual in ("Fail", "Pass"):
            self.matrix_tree.insert(
                "",
                "end",
                values=(actual, matrix[actual]["Fail"], matrix[actual]["Pass"]),
            )

    def _current_form_values(self):
        """Return current form values in validation-ready form.

        中文：下拉框会显示带说明的选项，例如 "2 (5 - 10 hours)"；校验函数只需要数字
        代码。因此这里先读取 StringVar，再去掉显示说明。

        English: Comboboxes show friendly labels such as "2 (5 - 10 hours)", while
        validation expects only the numeric code. This method reads StringVars and
        strips display-only text first.
        """
        return self._normalize_choice_values({key: var.get() for key, var in self.vars.items()})

    @staticmethod
    def _normalize_choice_values(values):
        """Convert display labels from comboboxes into stored codes.

        中文：实现方式是取字符串第一个空格前的部分，所以 "4+" 会变成 "4"，带说明的
        学习时间也会变成 "1" 到 "4"。这样界面可以友好显示，底层仍使用简单数字。

        English: The implementation keeps the part before the first space, so "4+"
        becomes "4" and descriptive study-time labels become "1" through "4". The UI
        can be friendly while the data layer receives simple numbers.
        """
        normalized = dict(values)
        normalized["study_time"] = str(normalized.get("study_time", "")).split(" ", 1)[0]
        normalized["failures"] = str(normalized.get("failures", "")).rstrip("+")
        return normalized

    def _predict(self):
        """Validate input, run the model, and refresh the result area.

        中文：预测前先调用统一校验，失败则弹窗并停止；成功后缓存 student/result，更新
        结果标题、概率条和文字建议。缓存最近一次结果是为了让 Save Record 明确保存
        用户刚刚看到的预测。

        English: Validation runs before prediction; failures show a dialog and stop
        the action. On success, student/result are cached and the result title, bars,
        and support note are refreshed. Caching ensures Save Record stores the exact
        prediction the user just saw.
        """
        try:
            student = validate_student_input(self._current_form_values())
            result = self.service.predict(student)
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.last_student = student
        self.last_prediction = result
        self.result_text.configure(
            text=(
                f"Pass Probability: {result['pass_probability']:.2%} | "
                f"Predicted G3: {result['predicted_g3']:.1f} | Result: {result['prediction']}"
            )
        )
        self._set_probability_text(result["probabilities"])
        self._update_key_factors(result["key_factors"])
        self._update_prediction_insights(result["prediction"], result["pass_probability"])

    def _update_key_factors(self, key_factors):
        """Display the strongest local factors for the current prediction.

        English: The service returns perturbation-based factors. The UI keeps the
        text compact so it remains readable beside Suggested Attention.
        """
        if not key_factors:
            self.key_factors_label.configure(text="No key factors are available for this prediction.")
            return
        lines = []
        for factor in key_factors:
            impact = factor["impact"] * 100
            if factor["abs_impact"] > 0.005:
                signal = f"{impact:+.1f} pp"
            else:
                signal = f"model weight {factor['importance']:.1%}"
            lines.append(
                f"{factor['feature']}: {factor['value']} ({signal})\n"
                f"{factor['message']}"
            )
        self.key_factors_label.configure(text="\n\n".join(lines))

    def _update_prediction_insights(self, prediction, pass_probability):
        """Update the short support note below the prediction result.

        中文：这个提示不改变模型输出，只把通过概率解释成可行动的教学提醒。阈值分为
        高风险、中等风险和较稳定三档，是为了让教师快速判断是否需要额外关注。

        English: This note does not change the model output; it translates pass
        probability into an actionable teaching reminder. Three thresholds separate
        high risk, moderate risk, and stable performance for quick review.
        """
        if prediction == "Fail" or pass_probability < 0.5:
            message = (
                "Priority review is recommended because the estimated pass probability "
                "is below 50%. Check recent grades, attendance, and learning progress."
            )
        elif pass_probability < 0.75:
            message = (
                "Continue monitoring progress and reinforce weaker areas before the final "
                "assessment. The student is likely to pass, but the margin is not strong."
            )
        else:
            message = (
                "Maintain the current learning pattern while continuing routine progress checks. "
                "The estimated pass probability is strong, but support should remain consistent."
            )

        message += "\n\nThis note is based on the predicted pass probability and should support, not replace, teacher judgment."
        self.attention_label.configure(text=message)

    def _save_record(self):
        """Save the latest successful prediction to history.

        中文：如果用户还没有点击 Predict，则不允许保存，避免写入空结果。保存后立即
        刷新历史缓存，让 History 页面可以看到新记录。

        English: Saving is blocked until Predict has produced a result, preventing
        empty history records. After insertion, history data is refreshed so the
        History screen can show the new row.
        """
        if not self.last_student or not self.last_prediction:
            messagebox.showwarning("No Prediction", "Please run a prediction before saving.")
            return
        self.db.add_prediction(
            self.last_student,
            self.last_prediction["prediction"],
            self.last_prediction["pass_probability"],
            self.last_prediction["predicted_g3"],
        )
        self._refresh_history()
        messagebox.showinfo("Saved", "Prediction record saved successfully.")

    def _batch_predict_legacy(self):
        """Legacy CSV-to-CSV batch prediction flow.

        中文：这个函数保留旧功能：读取 CSV，逐行预测，再写出带 prediction_result 和
        pass_probability 的新 CSV。当前界面入口已改为直接导入历史，但保留旧函数可以
        兼容已有调用或以后恢复“导出批量预测文件”的需求。

        English: This preserves the older flow: read a CSV, predict each row, and
        write a new CSV with prediction_result and pass_probability. The current UI
        imports directly into history, but keeping this method preserves compatibility
        and a possible future export workflow.
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
                # 中文：同时接受界面字段名和原始数据集字段名，降低批量文件模板要求。
                # English: Accept UI and dataset column names to make batch templates less strict.
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
                row["pass_probability"] = f"{result['pass_probability']:.4f}"
                row["predicted_g3"] = f"{result['predicted_g3']:.1f}"
                results.append(row)
            fieldnames = list(results[0].keys()) if results else ["prediction_result", "pass_probability", "predicted_g3"]
            with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                # 中文：utf-8-sig 带 BOM，更适合被 Windows Excel 直接打开。
                # English: utf-8-sig includes a BOM, which Windows Excel opens more reliably.
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            messagebox.showinfo("Batch Complete", f"Saved predictions to:\n{output_path}")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))

    def _set_probability_text(self, probabilities):
        """Store probabilities and redraw both probability bars.

        中文：概率值保存在 current_probabilities 中，这样窗口缩放触发 Canvas Configure
        事件时，可以用同一份数值重新绘制。

        English: Probabilities are cached in current_probabilities so Canvas
        Configure events can redraw bars with the same values after resizing.
        """
        self.current_probabilities = dict(probabilities)
        for klass, _display_label in PROBABILITY_BAR_LABELS:
            probability = probabilities.get(klass, 0.0)
            bar, value_label = self.probability_bars[klass]
            value_label.configure(text=f"{probability:.2%}")
            self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _redraw_probability_bar(self, klass):
        """Redraw one probability bar after its Canvas changes size.

        中文：Tkinter Canvas 不会自动缩放已经画好的图形，因此尺寸变化后必须手动重画。

        English: Tkinter Canvas does not automatically scale drawn shapes, so bars
        must be redrawn manually after a size change.
        """
        if not hasattr(self, "probability_bars"):
            return
        probability = self.current_probabilities.get(klass, 0.0)
        bar, _value_label = self.probability_bars[klass]
        self._draw_probability_bar(bar, probability, PREDICTION_COLORS[klass])

    def _draw_probability_bar(self, canvas, probability, color):
        """Draw the background track and filled probability segment.

        中文：用自定义 Canvas 图形实现圆角概率条。非零概率至少绘制一个圆角直径，避免
        1% 这类很小的值在视觉上消失。

        English: A custom Canvas shape creates rounded probability bars. Nonzero
        probabilities draw at least one rounded-end diameter so tiny values such as
        1% remain visible.
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
        """Create a rounded rectangle on a Tkinter Canvas.

        中文：Tkinter 没有内置圆角矩形，所以这里用平滑多边形模拟。集中成工具函数后，
        概率条和特征重要性条可以复用同一绘制方式。

        English: Tkinter has no native rounded rectangle, so a smoothed polygon is
        used. Keeping this as a helper lets probability bars and importance bars
        share one drawing technique.
        """
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _batch_predict(self):
        """Import CSV/XLSX students, predict each one, and save them to history.

        中文：这是当前批量入口。它不再写出单独预测文件，而是把每一行作为一次历史记录
        保存，便于教师直接在 History 页面查看、删除和导出。

        English: This is the current batch entry point. Instead of writing a separate
        prediction file, each row becomes a history record so teachers can review,
        delete, and export results from the History screen.
        """
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
                self.db.add_prediction(student, result["prediction"], result["pass_probability"], result["predicted_g3"])
            self._refresh_history()
            self._show_history_screen()
            messagebox.showinfo("Batch Complete", f"Imported {len(students)} prediction records into History.")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))

    @classmethod
    def _load_batch_students(cls, path):
        """Load a batch file and validate every student row.

        中文：这个函数把“读取文件”“表头映射”和“学生输入校验”串起来。它收集前若干
        行错误再一次性报出，原因是批量导入时用户通常希望一次修复多个问题。

        English: This chains file reading, header mapping, and student validation.
        It collects several row errors before raising so users can fix multiple
        problems in one pass.
        """
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
        """Dispatch batch reading by file extension.

        中文：CSV 与 XLSX 的解析方式完全不同，所以先按扩展名分派到专门函数；不支持的
        类型立即报错，避免后续解析产生误导性信息。

        English: CSV and XLSX parsing are very different, so extension dispatch sends
        each file to a specialized reader. Unsupported types fail early with a clear
        message.
        """
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls._read_csv_batch_rows(path)
        if suffix == ".xlsx":
            return cls._read_xlsx_batch_rows(path)
        raise ValueError("Only .csv and .xlsx files can be imported.")

    @classmethod
    def _read_csv_batch_rows(cls, path):
        """Read CSV data into numbered row dictionaries.

        中文：DictReader 使用第一行作为表头，数据行从第 2 行开始编号，与电子表格中用户
        看到的行号一致。空行会被忽略，减少无意义错误。

        English: DictReader uses the first row as headers, so data row numbering
        starts at 2, matching what users see in spreadsheets. Empty rows are skipped
        to avoid meaningless errors.
        """
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
        """Read the first worksheet of an XLSX file without external packages.

        中文：XLSX 本质上是 ZIP 包中的 XML 文件。这里用标准库 ZipFile 和 ElementTree
        读取第一张工作表，可以避免新增依赖，同时满足本项目只需要简单表格导入的需求。

        English: XLSX is a ZIP package of XML files. This method uses only ZipFile
        and ElementTree to read the first worksheet, avoiding new dependencies while
        supporting the simple table import this app needs.
        """
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
        """Find the XML path for the first worksheet inside an XLSX workbook.

        中文：workbook.xml 只告诉我们第一张 sheet 的关系 ID，真正文件路径在
        workbook.xml.rels 中。按关系解析路径可以兼容不同 Excel 生成的内部目录结构。

        English: workbook.xml gives only the relationship ID of the first sheet; the
        real XML path lives in workbook.xml.rels. Resolving through relationships
        supports workbooks with different internal layouts.
        """
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
        """Read the shared string table used by many XLSX files.

        中文：Excel 经常把重复文本放在 sharedStrings.xml 中，单元格只保存索引。读取这个
        表后，_xlsx_cell_text 才能把字符串索引还原成真实文本。

        English: Excel often stores repeated text in sharedStrings.xml and cells
        contain only indexes. Reading this table lets _xlsx_cell_text restore the
        actual strings.
        """
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
        """Convert one XLSX cell element into plain text.

        中文：XLSX 单元格可能是共享字符串、内联字符串或数字。本函数把三种表示统一
        转成字符串，后续校验逻辑就不需要理解 XML 细节。

        English: An XLSX cell can be a shared string, inline string, or number. This
        method turns all three forms into plain text so later validation does not
        need to know XML details.
        """
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
        """Normalize numeric XML text for user-facing validation.

        中文：Excel XML 常把整数写成浮点形式。把 2.0 显示为 2 可以让导入错误信息更接近
        用户在表格中看到的内容。

        English: Excel XML often stores integers as float-looking text. Showing 2.0
        as 2 keeps validation messages close to what users see in the spreadsheet.
        """
        try:
            number = float(value)
        except ValueError:
            return str(value).strip()
        if number.is_integer():
            return str(int(number))
        return str(number).strip()

    @staticmethod
    def _column_number(column_letters):
        """Convert Excel column letters to a 1-based number.

        中文：XLSX 单元格引用使用 A、B、AA 这类列字母；转换成数字后才能和表头位置
        对齐，生成每一行的字典。

        English: XLSX cell references use letters such as A, B, and AA. Converting
        them to numbers lets row cells align with header positions.
        """
        number = 0
        for letter in column_letters:
            number = number * 26 + ord(letter) - ord("A") + 1
        return number

    @classmethod
    def _validate_batch_headers(cls, headers):
        """Check whether imported headers can provide every required form field.

        中文：表头会先归一化，再和别名表匹配。这样用户可以使用 Name、Student Name、
        Matric No. 等友好名称，而不必严格记住程序内部字段名。

        English: Headers are normalized and matched against aliases. This lets users
        write friendly names such as Name, Student Name, or Matric No. instead of
        memorizing internal field names.
        """
        normalized_headers = {cls._normalize_header(header) for header in headers if header}
        missing = []
        for key, aliases in BATCH_COLUMN_ALIASES.items():
            if not any(cls._normalize_header(alias) in normalized_headers for alias in aliases):
                missing.append(BATCH_FIELD_LABELS[key])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

    @classmethod
    def _batch_row_to_form_values(cls, row):
        """Map one imported row to the exact keys expected by validation.

        中文：同一字段可能有多个表头别名，函数按别名顺序找到第一个非空值，并输出
        validate_student_input 需要的键。这样 CSV 和 XLSX 可以共用同一套校验。

        English: A field may have several header aliases. The method takes the first
        non-empty matching value and emits the keys expected by validate_student_input,
        allowing CSV and XLSX to share the same validator.
        """
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
        """Normalize a header for alias matching.

        中文：去掉大小写、空格、标点差异，只保留小写字母数字。这样 "Matric No." 与
        "matric_no" 会被视为同一个字段。

        English: Removes case, spaces, and punctuation differences by keeping only
        lowercase letters and digits. "Matric No." and "matric_no" therefore match.
        """
        return re.sub(r"[^a-z0-9]+", "", str(header).strip().lower())

    @staticmethod
    def _row_has_value(row):
        """Detect whether an imported row contains any content.

        中文：空行不应该触发“缺少字段”或“格式错误”，因此读取阶段直接跳过。

        English: Blank rows should not trigger missing-field or format errors, so
        they are skipped during reading.
        """
        return any(str(value or "").strip() for value in row.values())

    @staticmethod
    def _read_score_100(row, key):
        """Normalize legacy batch scores to the UI's 0-100 scale.

        中文：旧批量 CSV 可能使用原数据集的 0-20 分，也可能使用界面的 0-100 分。这里
        自动识别 0-20 范围并乘以 5，让旧导入流程兼容两种模板。

        English: Legacy batch CSV files may use the dataset's 0-20 scale or the UI's
        0-100 scale. Values in the 0-20 range are multiplied by 5 so both templates
        remain compatible.
        """
        value = float(row[key])
        if 0 <= value <= 20:
            return int(round(value * 5))
        return int(round(value))

    def _draw_feature_importance(self):
        """Draw the global feature-importance chart.

        中文：从模型服务取得已排序的重要性，使用水平条形图展示。Canvas 宽度会随窗口
        改变，因此每次重绘都会重新计算左边距、条宽和滚动区域。

        English: Gets sorted importances from the model service and renders a
        horizontal bar chart. Because the Canvas width changes with the window, every
        redraw recalculates margins, bar widths, and scroll region.
        """
        if not hasattr(self, "feature_canvas"):
            return
        self.feature_canvas.delete("all")
        items = self.service.feature_importances()
        canvas_width = max(self.feature_canvas.winfo_width(), 420)
        chart_width = max(canvas_width - 44, 360)
        chart_left = max((canvas_width - chart_width) // 2, 22)
        label_width = min(190, max(130, int(chart_width * 0.34)))
        bar_max_width = max(90, chart_width - label_width - 120)
        left = chart_left + label_width
        top = 16
        bar_height = 14
        gap = 7

        self.feature_canvas.create_text(chart_left, 16, text="Random Forest Feature Importance", anchor="nw", fill=COLORS["text"], font=("Times New Roman", 14, "bold"))
        self.feature_canvas.create_text(chart_left, 40, text="Higher bars indicate stronger influence in the trained model.", anchor="nw", fill=COLORS["muted"], font=("Times New Roman", 10))
        max_value = max((value for _, value in items), default=1) or 1
        for i, (label, value) in enumerate(items):
            # 中文：按最大重要性缩放所有条形，既保留相对大小，又让最大条充分占用宽度。
            # English: Scale by the largest importance to preserve ratios and fill the width.
            y = top + 48 + i * (bar_height + gap)
            bar_width = int((value / max_value) * bar_max_width)
            self.feature_canvas.create_text(left - 14, y + bar_height / 2, text=label, anchor="e", fill=COLORS["text"], font=("Times New Roman", 11))
            self._rounded_rect(self.feature_canvas, left, y, left + bar_max_width, y + bar_height, 7, fill=COLORS["bar_track"], outline="")
            self._rounded_rect(self.feature_canvas, left, y, left + bar_width, y + bar_height, 7, fill=COLORS["primary"], outline="")
            self.feature_canvas.create_text(left + bar_max_width + 16, y + bar_height / 2, text=f"{value:.1%}", anchor="w", fill=COLORS["muted"], font=("Times New Roman", 11, "bold"))
        self.feature_canvas.configure(scrollregion=(0, 0, canvas_width, top + 54 + len(items) * (bar_height + gap)))

    def _refresh_history(self):
        """Reload the history table from the database.

        中文：Treeview 不会自动同步数据库，所以这里先清空旧行，再插入最新摘要记录，并
        使用 odd/even 标签做交替底色，提升长列表可读性。

        English: Treeview does not sync with SQLite automatically. This method clears
        stale rows, inserts fresh summaries, and applies odd/even tags for readable
        zebra striping.
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
                f"{row['predicted_g3']:.1f}",
                row["prediction_result"],
                f"{row['pass_probability']:.2%}",
            )
            tag = "even" if index % 2 else "odd"
            self.history_tree.insert("", "end", values=values, tags=(tag,))

    def _delete_selected_history_single_legacy(self):
        """Legacy single-row deletion flow kept for compatibility.

        中文：当前界面使用支持多选的 _delete_selected_history。这个旧函数保留在代码中，
        是为了兼容可能仍引用单选删除逻辑的旧绑定或测试。

        English: The current UI uses _delete_selected_history for multi-selection.
        This older single-row flow remains for compatibility with any legacy binding
        or test that still calls it.
        """
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
        """Delete selected history records after user confirmation.

        中文：支持一次删除多条记录，并在确认框中展示部分 ID，避免误删。删除后数据库层
        会重新编号，界面随后刷新显示最新顺序。

        English: Allows deleting multiple selected rows and shows representative IDs
        in the confirmation dialog to prevent accidental deletion. The database layer
        renumbers IDs, and the table refreshes afterward.
        """
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
        """Open the detail window for the selected history row.

        中文：先读取完整数据库记录，再重建 StudentInput，并用当前模型重新预测。这样详情
        页能同时展示“当时保存的结果”和“当前模型的结果”；如果之后模型或数据集改变，
        两者可能不同。

        English: The full database row is loaded, converted back to StudentInput,
        and predicted again with the current model. The detail view can therefore
        show both the saved result and the current model result; they may differ if
        the model or dataset changes later.
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

        student = self._student_from_history_row(row)
        current_result = self.service.predict(student)
        self._show_detail_window(row, current_result)

    @staticmethod
    def _student_from_history_row(row):
        """Rebuild StudentInput from a full history row.

        中文：历史表保存了预测所需的全部输入字段，因此可以把一条记录恢复成模型服务
        接受的对象，用于详情页重算或迁移旧记录。

        English: The history table stores every input field required for prediction,
        so one row can be restored into the object accepted by the prediction service
        for detail recalculation or legacy migration.
        """
        return StudentInput(
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

    def _migrate_legacy_history_records(self):
        """Convert old Low/Medium/High history rows to the current Pass/Fail format.

        中文：早期版本保存三分类结果；当前界面使用二分类通过概率。启动时自动重算旧记录，
        可以让旧数据库继续在新版界面中正常显示。

        English: Earlier versions stored three-class results, while the current UI
        uses binary pass probability. Recalculating old rows on startup keeps older
        databases usable in the new interface.
        """
        for summary in self.db.list_predictions():
            needs_result_update = summary["prediction_result"] in {"Low", "Medium", "High"}
            needs_grade_update = float(summary.get("predicted_g3") or 0.0) <= 0.0
            if not needs_result_update and not needs_grade_update:
                continue
            row = self.db.get_prediction(summary["id"])
            if not row:
                continue
            student = self._student_from_history_row(row)
            result = self.service.predict(student)
            if needs_result_update:
                self.db.update_prediction_result(row["id"], result["prediction"], result["pass_probability"], result["predicted_g3"])
            else:
                self.db.update_prediction_result(
                    row["id"],
                    row["prediction_result"],
                    row["pass_probability"],
                    result["predicted_g3"],
                )

    def _show_detail_window(self, row, current_result):
        """Render saved student data and current prediction in a child window.

        中文：窗口上方显示摘要，下方列出输入字段。这样用户既能看到保存时的预测，也能
        检查当时使用的性别、年龄、成绩等输入是否正确。

        English: The child window shows a summary at the top and input fields below.
        Users can review the saved prediction and verify the gender, age, grades,
        and other inputs that produced it.
        """
        window = tk.Toplevel(self)
        window.title(f"Prediction Detail - {row['student_name']}")
        window.geometry("620x460")
        window.minsize(560, 420)

        body = ttk.Frame(window, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Prediction Detail", font=("Times New Roman", 15, "bold")).pack(anchor="w")
        probabilities = current_result["probabilities"]
        summary = (
            f"Student: {row['student_name']} ({row['matric_no']})\n"
            f"Saved: {row['timestamp']}\n"
            f"Prediction: {row['prediction_result']} | Saved pass probability: {row['pass_probability']:.2%} | "
            f"Saved predicted G3: {row['predicted_g3']:.1f}\n"
            f"Current result: {current_result['prediction']} | Current pass probability: {current_result['pass_probability']:.2%} | "
            f"Current predicted G3: {current_result['predicted_g3']:.1f}\n"
            f"Probabilities: Pass {probabilities.get('Pass', 0.0):.2%}, "
            f"Fail {probabilities.get('Fail', 0.0):.2%}"
        )
        ttk.Label(body, text=summary, justify="left").pack(anchor="w", pady=(8, 12))

        fields = (
            ("Gender", row["sex"]),
            ("Age", row["age"]),
            ("Study Time", row["study_time"]),
            ("Failures", row["failures"]),
            ("Activities", row["activities"]),
            ("Absences", row["absences"]),
            ("Previous Grade G1", row["g1"]),
            ("Midterm Grade G2", row["g2"]),
        )
        detail_frame = ttk.Frame(body)
        detail_frame.pack(fill="x", pady=(4, 0))
        for index, (label, value) in enumerate(fields):
            ttk.Label(detail_frame, text=f"{label}:", width=18).grid(row=index, column=0, sticky="w", pady=3)
            ttk.Label(detail_frame, text=value).grid(row=index, column=1, sticky="w", pady=3)

    def _export_history(self):
        """Ask for an output path and export history to CSV or styled XLSX.

        中文：文件路径由保存对话框选择，真正写文件交给 HistoryDatabase。这样 UI
        负责用户交互，数据库层负责字段、编码和 XLSX 样式。

        English: The save dialog chooses the path, while HistoryDatabase writes the
        file. The UI handles interaction, and the database layer owns fields,
        encoding, and XLSX styling.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV files", "*.csv")],
            initialfile="prediction_history.xlsx",
        )
        if not path:
            return
        if Path(path).suffix.lower() == ".csv":
            self.db.export_csv(path)
        else:
            self.db.export_xlsx(path)
        messagebox.showinfo("Exported", f"History exported to:\n{path}")


def main():
    """Create the application and enter Tkinter's event loop.

    中文：main 是包内统一入口，run_app.py 和直接执行本模块都会调用它。事件循环启动后，
    Tkinter 持续响应按钮、表格和窗口事件。

    English: main is the package entry point used by run_app.py and by direct module
    execution. After the event loop starts, Tkinter responds to buttons, tables, and
    window events.
    """
    app = StudentPerformanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PySide6 GUI Launcher for Holosoma Whole-Body Tracking Workflows.

Mirrors the logic in demo_workflow_selector.sh:
  1. LAFAN retargeting + tracking
  2. C3D  retargeting + tracking
  3. Inference (simulation / hardware)
  4. Browse files
  5. Configuration

Features:
  - Live training metrics plot (Mean Reward, Episode Length, Joint Pos Error,
    Curriculum Entropy) read from TensorBoard event files during training.
  - Configurable training options (envs, iterations, episode length, headless,
    video recording, history length, logger type, alpha init, foot tolerance).

Launch:
    python demo_scripts/workflow_launcher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, QTimer, Qt, Slot
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Optional: tbparse for reading TensorBoard events
try:
    from tbparse import SummaryReader
    HAS_TBPARSE = True
except ImportError:
    HAS_TBPARSE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LAFAN_DIR = PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan"
DEFAULT_C3D_DIR = PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/c3d"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs/WholeBodyTracking"
DEFAULT_RETARGET_DIR = "demo_results/g1/robot_only"
DEFAULT_CONVERT_DIR = "converted_res"
DEFAULT_VIDEO_DIR = "logs/videos"

ROBOTS = [("23", "G1-23DOF"), ("29", "G1-29DOF")]
ALGORITHMS = [
    ("fast-sac", "Fast SAC (default)"),
    ("ppo", "PPO"),
    ("fast-sac-no-dr", "Fast SAC (no domain rand.)"),
]
LOGGERS = [
    ("wandb-offline", "W&B Offline (default)"),
    ("wandb", "W&B Online"),
    ("tensorboard", "TensorBoard"),
]

# Metrics to plot -- same 4 panels as plot_rewards.py
PLOT_METRICS = {
    "Mean Reward": lambda t: "mean_reward" in t.lower() or t.lower() == "reward",
    "Episode Length": lambda t: "average_episode_length" in t.lower(),
    "Joint Pos Error (rad)": lambda t: "error_joint_pos" in t.lower(),
    "Curriculum Entropy": lambda t: "adaptive_timesteps_sampler_entropy" in t.lower(),
}
PLOT_COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLESHEET = """
QMainWindow { background: #1e1e2e; }
QWidget     { font-family: "Segoe UI", Arial, sans-serif; color: #cdd6f4; }
QGroupBox {
    font-weight: bold; font-size: 12px;
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 14px; padding-top: 18px;
    background: #313244;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 4px 10px; background: #45475a;
    color: #89b4fa; border-radius: 4px;
}
QPushButton {
    background: #89b4fa; color: #1e1e2e;
    border: none; border-radius: 5px;
    padding: 7px 14px; font-weight: bold; font-size: 12px;
}
QPushButton:hover   { background: #74c7ec; }
QPushButton:pressed  { background: #b4befe; }
QPushButton:disabled { background: #585b70; color: #6c7086; }
QPushButton[class="danger"]        { background: #f38ba8; }
QPushButton[class="danger"]:hover  { background: #eba0ac; }
QPushButton[class="success"]       { background: #a6e3a1; }
QPushButton[class="success"]:hover { background: #94e2d5; }
QLabel          { color: #cdd6f4; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    padding: 5px; border: 1px solid #585b70;
    border-radius: 4px; background: #45475a; color: #cdd6f4;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #45475a; color: #cdd6f4; selection-background-color: #585b70; }
QListWidget {
    background: #45475a; border: 1px solid #585b70;
    border-radius: 4px; color: #cdd6f4;
}
QListWidget::item:selected { background: #585b70; }
QTextEdit {
    background: #11111b; color: #a6e3a1;
    border: 1px solid #45475a; border-radius: 4px;
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; font-size: 11px;
}
QTabWidget::pane   { border: 1px solid #45475a; background: #1e1e2e; border-radius: 6px; }
QTabBar::tab {
    background: #313244; padding: 8px 16px; margin: 2px;
    border: 1px solid #45475a; border-bottom: none; border-radius: 4px 4px 0 0;
    color: #bac2de;
}
QTabBar::tab:selected { background: #1e1e2e; color: #89b4fa; border-bottom: 2px solid #89b4fa; }
QCheckBox          { spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #585b70; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QSplitter::handle  { background: #45475a; width: 3px; }
QScrollBar:vertical {
    background: #313244; width: 10px; margin: 0;
}
QScrollBar::handle:vertical { background: #585b70; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

BIG_BTN = """
QPushButton {{
    background: {bg}; color: #1e1e2e;
    font-size: 14px; font-weight: bold;
    border: none; border-radius: 6px; padding: 14px 28px;
}}
QPushButton:hover   {{ background: {hover}; }}
QPushButton:pressed  {{ background: {press}; }}
QPushButton:disabled {{ background: #585b70; color: #6c7086; }}
"""


# ---------------------------------------------------------------------------
# Helpers: filesystem scanning
# ---------------------------------------------------------------------------
def scan_npy(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.npy"))


def scan_c3d(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.c3d"))


def scan_checkpoints(logs_dir: Path) -> list[Path]:
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def scan_onnx(logs_dir: Path) -> list[Path]:
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("*/exported/*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)


def human_size(path: Path) -> str:
    try:
        s = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if s < 1024:
            return f"{s:.1f} {unit}"
        s /= 1024
    return f"{s:.1f} TB"


def find_latest_run_dir(logs_dir: Path) -> Path | None:
    """Return the most-recently-modified run directory under *logs_dir*."""
    if not logs_dir.is_dir():
        return None
    subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Live Training Plot Widget
# ---------------------------------------------------------------------------
class TrainingPlotWidget(QWidget):
    """Embeds a 2x2 matplotlib figure that auto-refreshes from TensorBoard logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_dir: Path | None = None
        self._last_read_size: dict[str, int] = {}  # track file sizes to detect new data

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure with dark background
        self._fig = Figure(figsize=(6, 4.5), dpi=100, facecolor="#1e1e2e")
        self._axes = self._fig.subplots(2, 2)
        self._fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92, bottom=0.08, left=0.10, right=0.96)
        for ax in self._axes.flat:
            ax.set_facecolor("#313244")
            ax.tick_params(colors="#9399b2", labelsize=7)
            ax.xaxis.label.set_color("#9399b2")
            ax.yaxis.label.set_color("#9399b2")
            ax.title.set_color("#cdd6f4")
            ax.title.set_fontsize(9)
            for spine in ax.spines.values():
                spine.set_color("#45475a")
            ax.grid(True, linestyle="--", alpha=0.3, color="#585b70")

        self._canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self._canvas)

        # Controls
        ctrl = QHBoxLayout()
        self._status_lbl = QLabel("No training data")
        self._status_lbl.setStyleSheet("color: #9399b2; font-size: 11px; padding: 2px;")
        ctrl.addWidget(self._status_lbl)
        ctrl.addStretch()
        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh_plot)
        ctrl.addWidget(refresh_btn)
        lay.addLayout(ctrl)

        # Auto-refresh timer (every 10s during training)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_plot)

        # Initial state: show placeholders
        self._draw_empty()

    def _draw_empty(self):
        titles = list(PLOT_METRICS.keys())
        for i, ax in enumerate(self._axes.flat):
            ax.clear()
            ax.set_facecolor("#313244")
            ax.set_title(titles[i] if i < len(titles) else "", fontsize=9, color="#cdd6f4")
            ax.grid(True, linestyle="--", alpha=0.3, color="#585b70")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
            ax.tick_params(colors="#9399b2", labelsize=7)
            if not HAS_TBPARSE:
                ax.text(0.5, 0.5, "tbparse not installed\npip install tbparse",
                        ha="center", va="center", color="#f38ba8", fontsize=8,
                        transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Waiting for data...",
                        ha="center", va="center", color="#585b70", fontsize=9,
                        transform=ax.transAxes)
        self._canvas.draw_idle()

    def start_monitoring(self, log_dir: Path | None = None):
        """Start polling for new TensorBoard events."""
        self._log_dir = log_dir
        self._last_read_size.clear()
        self._draw_empty()
        if HAS_TBPARSE:
            self._timer.start(10_000)  # 10s interval
            self._status_lbl.setText(f"Monitoring: {log_dir or 'latest run'}")

    def stop_monitoring(self):
        self._timer.stop()
        self._status_lbl.setText("Monitoring stopped")

    @Slot()
    def refresh_plot(self):
        if not HAS_TBPARSE:
            return

        # Resolve which directory to read
        log_dir = self._log_dir
        if log_dir is None or not log_dir.is_dir():
            log_dir = find_latest_run_dir(DEFAULT_LOGS_DIR)
        if log_dir is None or not log_dir.is_dir():
            return

        # Check if any tfevent files exist / changed
        tfevent_files = list(log_dir.glob("events.out.tfevents.*"))
        if not tfevent_files:
            # Also look one level deeper
            tfevent_files = list(log_dir.glob("*/events.out.tfevents.*"))
        if not tfevent_files:
            return

        # Skip if file sizes haven't changed (avoid re-parsing unchanged data)
        current_sizes = {str(f): f.stat().st_size for f in tfevent_files}
        if current_sizes == self._last_read_size:
            return
        self._last_read_size = current_sizes

        try:
            reader = SummaryReader(str(log_dir), pivot=False)
            df = reader.scalars
        except Exception:
            return

        if df.empty:
            return

        available_tags = df["tag"].unique().tolist()
        titles = list(PLOT_METRICS.keys())
        matchers = list(PLOT_METRICS.values())

        data_found = False
        for i, ax in enumerate(self._axes.flat):
            ax.clear()
            ax.set_facecolor("#313244")
            ax.set_title(titles[i], fontsize=9, color="#cdd6f4")
            ax.grid(True, linestyle="--", alpha=0.3, color="#585b70")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
            ax.tick_params(colors="#9399b2", labelsize=7)
            ax.set_xlabel("Iterations", fontsize=7, color="#9399b2")

            # Find matching tag
            matched = [t for t in available_tags if matchers[i](t)]
            if matched:
                tag = matched[0]
                subset = df[df["tag"] == tag].sort_values("step")
                steps = subset["step"].values
                values = subset["value"].values
                if len(steps) > 0:
                    data_found = True
                    ax.plot(steps, values, color=PLOT_COLORS[i], linewidth=1.2,
                            marker="o", markersize=1.5, alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                        color="#585b70", fontsize=9, transform=ax.transAxes)

        self._canvas.draw_idle()
        n_points = len(df)
        run_name = log_dir.name
        self._status_lbl.setText(f"Run: {run_name}  |  {n_points} data points")

    def set_log_dir(self, path: Path):
        self._log_dir = path
        self._last_read_size.clear()


# ---------------------------------------------------------------------------
# Training Config dataclass-like container returned by _make_training_group
# ---------------------------------------------------------------------------
class _TrainingWidgets:
    """Holds references to all training-config widgets for a tab."""
    __slots__ = (
        "group", "envs", "iters", "ep_len", "headless", "video_enabled",
        "video_interval", "history_length", "logger", "alpha_init",
        "foot_tolerance",
    )


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class WorkflowLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holosoma Workflow Launcher")
        self.setMinimumSize(1350, 850)
        self.setStyleSheet(STYLESHEET)

        self._process: QProcess | None = None

        # ── Central ─────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(8, 4, 8, 4)
        root_lay.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Holosoma Workflow Launcher")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa; padding: 4px;")
        hdr.addWidget(title)
        hdr.addStretch()
        root_lay.addLayout(hdr)

        # Horizontal splitter: left = tabs, right = console + plot
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        tabs = QTabWidget()
        tabs.setMinimumWidth(540)
        tabs.addTab(self._build_lafan_tab(), "LAFAN Tracking")
        tabs.addTab(self._build_c3d_tab(), "C3D Tracking")
        tabs.addTab(self._build_inference_tab(), "Inference")
        tabs.addTab(self._build_files_tab(), "Browse Files")
        tabs.addTab(self._build_settings_tab(), "Settings")
        h_splitter.addWidget(tabs)

        # Right panel: vertical splitter with plot on top, console below
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot panel
        plot_box = QGroupBox("Training Metrics (Live)")
        plot_lay = QVBoxLayout(plot_box)
        plot_lay.setContentsMargins(4, 14, 4, 4)
        self._plot_widget = TrainingPlotWidget()
        plot_lay.addWidget(self._plot_widget)
        right_splitter.addWidget(plot_box)

        # Console panel
        console_box = QGroupBox("Output Console")
        con_lay = QVBoxLayout(console_box)
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        con_lay.addWidget(self._console)
        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setProperty("class", "danger")
        clear_btn.clicked.connect(self._console.clear)
        btn_row.addWidget(clear_btn)
        stop_btn = QPushButton("Stop Process")
        stop_btn.setProperty("class", "danger")
        stop_btn.clicked.connect(self._stop_process)
        btn_row.addWidget(stop_btn)
        btn_row.addStretch()
        con_lay.addLayout(btn_row)
        right_splitter.addWidget(console_box)

        right_splitter.setSizes([420, 350])
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)

        h_splitter.addWidget(right_splitter)
        h_splitter.setSizes([540, 700])
        h_splitter.setStretchFactor(0, 2)
        h_splitter.setStretchFactor(1, 3)
        root_lay.addWidget(h_splitter, 1)

        self.statusBar().showMessage("Ready")

    # ===================================================================
    #  Console helpers
    # ===================================================================
    def _log(self, text: str, color: str = "#a6e3a1"):
        self._console.setTextColor(QColor(color))
        self._console.append(text)
        self._console.moveCursor(QTextCursor.MoveOperation.End)

    def _log_info(self, text: str):
        self._log(text, "#89b4fa")

    def _log_err(self, text: str):
        self._log(text, "#f38ba8")

    def _log_cmd(self, text: str):
        self._log(f"$ {text}", "#f9e2af")

    # ===================================================================
    #  Process runner (uses QProcess for non-blocking output)
    # ===================================================================
    def _run_script(self, script_lines: list[str], *, cwd: str | None = None,
                    start_plot: bool = False):
        """Run a bash script composed of *script_lines*."""
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Busy", "A process is already running. Stop it first.")
            return

        script = "\n".join(["set -e"] + script_lines)
        self._log_cmd(script.replace("\n", "\n$ "))

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)

        env = self._process.processEnvironment()
        if env.isEmpty():
            env = env.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)

        if cwd:
            self._process.setWorkingDirectory(cwd)
        else:
            self._process.setWorkingDirectory(str(PROJECT_ROOT))

        self._process.start("bash", ["-c", script])
        self.statusBar().showMessage("Running...")

        if start_plot:
            self._plot_widget.start_monitoring()

    @Slot()
    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        for line in data.splitlines():
            self._log(line)

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code, status):
        if exit_code == 0:
            self._log_info(f"\n=== Process finished (exit code {exit_code}) ===")
        else:
            self._log_err(f"\n=== Process failed (exit code {exit_code}) ===")
        self.statusBar().showMessage("Ready")
        # Stop live monitoring but do one final refresh
        self._plot_widget.stop_monitoring()
        self._plot_widget.refresh_plot()

    def _stop_process(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._log_err("Process killed by user.")
            self._plot_widget.stop_monitoring()

    # ===================================================================
    #  Common widget builders
    # ===================================================================
    def _make_robot_algo_group(self) -> tuple[QGroupBox, QComboBox, QComboBox]:
        grp = QGroupBox("Robot / Algorithm")
        lay = QHBoxLayout(grp)
        lay.addWidget(QLabel("Robot:"))
        robot_cb = QComboBox()
        for val, label in ROBOTS:
            robot_cb.addItem(label, val)
        lay.addWidget(robot_cb)
        lay.addSpacing(20)
        lay.addWidget(QLabel("Algorithm:"))
        algo_cb = QComboBox()
        for val, label in ALGORITHMS:
            algo_cb.addItem(label, val)
        lay.addWidget(algo_cb)
        lay.addStretch()
        return grp, robot_cb, algo_cb

    def _make_training_group(self) -> _TrainingWidgets:
        """Build an expanded training-config group box with all options."""
        tw = _TrainingWidgets()
        grp = QGroupBox("Training Configuration")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        row = 0

        # ── Row 0: Envs / Iterations / Max Episode ──
        grid.addWidget(QLabel("Num Envs:"), row, 0)
        tw.envs = QSpinBox()
        tw.envs.setRange(1, 16384)
        tw.envs.setValue(4096)
        tw.envs.setToolTip("Number of parallel environments for training")
        grid.addWidget(tw.envs, row, 1)

        grid.addWidget(QLabel("Iterations:"), row, 2)
        tw.iters = QSpinBox()
        tw.iters.setRange(100, 500000)
        tw.iters.setSingleStep(1000)
        tw.iters.setValue(50000)
        tw.iters.setToolTip("Total learning iterations")
        grid.addWidget(tw.iters, row, 3)

        grid.addWidget(QLabel("Max Episode (s):"), row, 4)
        tw.ep_len = QDoubleSpinBox()
        tw.ep_len.setRange(1.0, 60.0)
        tw.ep_len.setDecimals(1)
        tw.ep_len.setValue(6.0)
        tw.ep_len.setToolTip("Maximum episode length in seconds")
        grid.addWidget(tw.ep_len, row, 5)
        row += 1

        # ── Row 1: Alpha / History / Foot Tolerance ──
        grid.addWidget(QLabel("Alpha Init:"), row, 0)
        tw.alpha_init = QDoubleSpinBox()
        tw.alpha_init.setRange(0.001, 1.0)
        tw.alpha_init.setDecimals(3)
        tw.alpha_init.setSingleStep(0.005)
        tw.alpha_init.setValue(0.01)
        tw.alpha_init.setToolTip("SAC entropy alpha initial value (ignored for PPO)")
        grid.addWidget(tw.alpha_init, row, 1)

        grid.addWidget(QLabel("Obs History:"), row, 2)
        tw.history_length = QSpinBox()
        tw.history_length.setRange(1, 16)
        tw.history_length.setValue(1)
        tw.history_length.setToolTip("Observation history length (actor_obs)")
        grid.addWidget(tw.history_length, row, 3)

        grid.addWidget(QLabel("Foot Tolerance:"), row, 4)
        tw.foot_tolerance = QDoubleSpinBox()
        tw.foot_tolerance.setRange(0.001, 0.5)
        tw.foot_tolerance.setDecimals(3)
        tw.foot_tolerance.setSingleStep(0.005)
        tw.foot_tolerance.setValue(0.02)
        tw.foot_tolerance.setToolTip("Foot-sticking tolerance for retargeting")
        grid.addWidget(tw.foot_tolerance, row, 5)
        row += 1

        # ── Row 2: Logger / Video Interval ──
        grid.addWidget(QLabel("Logger:"), row, 0)
        tw.logger = QComboBox()
        for val, label in LOGGERS:
            tw.logger.addItem(label, val)
        tw.logger.setToolTip("Logging backend for training metrics")
        grid.addWidget(tw.logger, row, 1)

        grid.addWidget(QLabel("Video Interval:"), row, 2)
        tw.video_interval = QSpinBox()
        tw.video_interval.setRange(1, 100)
        tw.video_interval.setValue(5)
        tw.video_interval.setToolTip("Record video every N iterations")
        grid.addWidget(tw.video_interval, row, 3)

        # ── Row 2 continued: checkboxes ──
        tw.headless = QCheckBox("Headless")
        tw.headless.setChecked(True)
        tw.headless.setToolTip("Run training without a render window")
        grid.addWidget(tw.headless, row, 4)

        tw.video_enabled = QCheckBox("Record Video")
        tw.video_enabled.setChecked(True)
        tw.video_enabled.setToolTip("Enable video recording during training")
        grid.addWidget(tw.video_enabled, row, 5)

        tw.group = grp
        return tw

    # ===================================================================
    #  TAB 1 – LAFAN Retargeting + Tracking
    # ===================================================================
    def _build_lafan_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        # Robot / algo
        grp1, self._lf_robot, self._lf_algo = self._make_robot_algo_group()
        lay.addWidget(grp1)

        # Motion selection
        sel_grp = QGroupBox("LAFAN Motion Sequence")
        sel_lay = QVBoxLayout(sel_grp)
        self._lf_list = QListWidget()
        self._lf_list.setMaximumHeight(200)
        sel_lay.addWidget(self._lf_list)
        ref_btn = QPushButton("Refresh List")
        ref_btn.clicked.connect(self._refresh_lafan)
        sel_lay.addWidget(ref_btn)
        lay.addWidget(sel_grp)

        # Training config (expanded)
        self._lf_tw = self._make_training_group()
        lay.addWidget(self._lf_tw.group)

        # Ground range
        gr_grp = QGroupBox("Ground Range")
        gr_lay = QHBoxLayout(gr_grp)
        gr_lay.addWidget(QLabel("X:"))
        self._lf_gr_x = QSpinBox()
        self._lf_gr_x.setRange(-100, 100)
        self._lf_gr_x.setValue(-10)
        gr_lay.addWidget(self._lf_gr_x)
        gr_lay.addWidget(QLabel("Y:"))
        self._lf_gr_y = QSpinBox()
        self._lf_gr_y.setRange(-100, 100)
        self._lf_gr_y.setValue(10)
        gr_lay.addWidget(self._lf_gr_y)
        gr_lay.addStretch()
        lay.addWidget(gr_grp)

        # GO
        run_btn = QPushButton("START LAFAN WORKFLOW")
        run_btn.setMinimumHeight(48)
        run_btn.setStyleSheet(BIG_BTN.format(bg="#89b4fa", hover="#74c7ec", press="#b4befe"))
        run_btn.clicked.connect(self._run_lafan)
        lay.addWidget(run_btn)

        lay.addStretch()
        scroll.setWidget(w)

        self._refresh_lafan()
        return scroll

    def _refresh_lafan(self):
        self._lf_list.clear()
        tasks = scan_npy(DEFAULT_LAFAN_DIR)
        if not tasks:
            self._lf_list.addItem("(no .npy files found)")
        for t in tasks:
            item = QListWidgetItem(t)
            item.setData(Qt.ItemDataRole.UserRole, t)
            self._lf_list.addItem(item)

    def _run_lafan(self):
        item = self._lf_list.currentItem()
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            QMessageBox.warning(self, "No selection", "Select a LAFAN sequence first.")
            return

        task = item.data(Qt.ItemDataRole.UserRole)
        dof = self._lf_robot.currentData()
        algo = self._lf_algo.currentData()
        tw = self._lf_tw
        envs = tw.envs.value()
        iters = tw.iters.value()
        ep_len = tw.ep_len.value()
        headless = "True" if tw.headless.isChecked() else "False"
        video_on = tw.video_enabled.isChecked()
        vid_int = tw.video_interval.value()
        history = tw.history_length.value()
        logger = tw.logger.currentData()
        foot_tol = tw.foot_tolerance.value()
        gr_x = self._lf_gr_x.value()
        gr_y = self._lf_gr_y.value()
        stem = f"{dof}dof"
        exp = f"g1-{stem}-wbt-{algo}"
        retarget_dir = str(PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting")
        lafan_dir = str(DEFAULT_LAFAN_DIR)
        convert_out = f"{DEFAULT_CONVERT_DIR}/{task}_mj_fps50.npz"
        converted_file = f"{retarget_dir}/{DEFAULT_CONVERT_DIR}/{task}_mj_fps50.npz"

        # Build video lines
        video_lines = []
        if video_on:
            video_lines = [
                f"    --logger.video.enabled True \\",
                f"    --logger.video.interval {vid_int} \\",
                f'    --logger.video.save-dir "{DEFAULT_VIDEO_DIR}/g1_{stem}_lafan_wbt" \\',
            ]
        else:
            video_lines = [
                f"    --logger.video.enabled False \\",
            ]

        # Build history line (only if > 1)
        history_line = ""
        if history > 1:
            history_line = f"    --observation.groups.actor_obs.history-length {history} \\"

        lines = [
            f'echo "=== LAFAN Workflow: {task} (G1-{stem}) ==="',
            "",
            "# ── Step 1: Retargeting env ──",
            f'source "{PROJECT_ROOT}/scripts/source_retargeting_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_retargeting" --quiet',
            f'cd "{retarget_dir}"',
            "",
            "# ── Step 2: Retarget ──",
            f'echo "Running retargeting for {task}..."',
            f"python examples/robot_retarget.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --data_path "{lafan_dir}" \\',
            f"    --task-type robot_only \\",
            f'    --task-name "{task}" \\',
            f"    --data_format lafan \\",
            f"    --task-config.ground-range {gr_x} {gr_y} \\",
            f'    --save_dir "{DEFAULT_RETARGET_DIR}" \\',
            f"    --retargeter.foot-sticking-tolerance {foot_tol}",
            "",
            "# ── Step 3: Convert ──",
            f'echo "Converting to MuJoCo format..."',
            f"python data_conversion/convert_data_format_mj.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --input_file "./{DEFAULT_RETARGET_DIR}/{task}.npz" \\',
            f"    --output_fps 50 \\",
            f'    --output_name "{convert_out}" \\',
            f"    --data_format lafan \\",
            f'    --object_name "ground" \\',
            f"    --once",
            "",
            "# ── Step 4: IsaacSim env ──",
            f'cd "{PROJECT_ROOT}"',
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'HOLOSOMA_DEPS_DIR="${{HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}}"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma[unitree,booster]" --quiet',
            'if ! python -c "import isaaclab" 2>/dev/null; then',
            '    pip install "setuptools<81" --quiet',
            "    echo 'setuptools<81' > /tmp/hs-build-constraints.txt",
            "    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \\",
            '        pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet',
            "    rm /tmp/hs-build-constraints.txt",
            "fi",
            "",
            "# ── Step 5: Train ──",
            f'echo "Starting training ({exp})..."',
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \\",
            f"    exp:{exp} \\",
            f"    logger:{logger} \\",
            f"    --training.headless {headless} \\",
            f"    --training.num-envs {envs} \\",
            f"    --algo.config.num-learning-iterations {iters} \\",
            f"    --simulator.config.sim.max-episode-length-s {ep_len} \\",
        ]
        lines.extend(video_lines)
        if history_line:
            lines.append(history_line)
        # Last arg (no trailing backslash)
        lines.append(
            f"    --command.setup_terms.motion_command.params.motion_config.motion_file={converted_file}"
        )
        lines += [
            "",
            'echo "=== LAFAN Workflow complete! ==="',
        ]

        self._log_info(f"Starting LAFAN workflow: {task} | {exp} | envs={envs} iters={iters}")
        self._run_script(lines, start_plot=True)

    # ===================================================================
    #  TAB 2 – C3D Retargeting + Tracking
    # ===================================================================
    def _build_c3d_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        grp1, self._c3d_robot, self._c3d_algo = self._make_robot_algo_group()
        lay.addWidget(grp1)

        # C3D file
        file_grp = QGroupBox("C3D Input File")
        file_lay = QVBoxLayout(file_grp)
        self._c3d_list = QListWidget()
        self._c3d_list.setMaximumHeight(140)
        file_lay.addWidget(self._c3d_list)
        btn_row = QHBoxLayout()
        ref_btn = QPushButton("Refresh")
        ref_btn.clicked.connect(self._refresh_c3d)
        btn_row.addWidget(ref_btn)
        browse_btn = QPushButton("Browse .c3d ...")
        browse_btn.clicked.connect(self._browse_c3d)
        btn_row.addWidget(browse_btn)
        btn_row.addStretch()
        file_lay.addLayout(btn_row)
        self._c3d_path = QLineEdit()
        self._c3d_path.setPlaceholderText("Or enter full path to .c3d file")
        file_lay.addWidget(self._c3d_path)
        lay.addWidget(file_grp)

        # Marker map
        mm_grp = QGroupBox("Marker Map (for C3D prep)")
        mm_lay = QHBoxLayout(mm_grp)
        self._c3d_marker_map = QLineEdit(str(PROJECT_ROOT / "boxing_markers.json"))
        mm_lay.addWidget(self._c3d_marker_map)
        mm_btn = QPushButton("Browse...")
        mm_btn.clicked.connect(lambda: self._browse_file(self._c3d_marker_map, "JSON (*.json)"))
        mm_lay.addWidget(mm_btn)
        lay.addWidget(mm_grp)

        # Training config (expanded) – C3D defaults
        self._c3d_tw = self._make_training_group()
        self._c3d_tw.envs.setValue(2048)
        self._c3d_tw.ep_len.setValue(10.0)
        self._c3d_tw.video_interval.setValue(10)
        self._c3d_tw.history_length.setValue(4)
        lay.addWidget(self._c3d_tw.group)

        # GO
        run_btn = QPushButton("START C3D WORKFLOW")
        run_btn.setMinimumHeight(48)
        run_btn.setStyleSheet(BIG_BTN.format(bg="#89b4fa", hover="#74c7ec", press="#b4befe"))
        run_btn.clicked.connect(self._run_c3d)
        lay.addWidget(run_btn)

        lay.addStretch()
        scroll.setWidget(w)

        self._refresh_c3d()
        return scroll

    def _refresh_c3d(self):
        self._c3d_list.clear()
        files = scan_c3d(DEFAULT_C3D_DIR)
        if not files:
            self._c3d_list.addItem("(no .c3d files found)")
        for f in files:
            item = QListWidgetItem(f)
            item.setData(Qt.ItemDataRole.UserRole, str(DEFAULT_C3D_DIR / f))
            self._c3d_list.addItem(item)

    def _browse_c3d(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select C3D File", str(PROJECT_ROOT), "C3D Files (*.c3d)")
        if path:
            self._c3d_path.setText(path)

    def _browse_file(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", str(PROJECT_ROOT), filt)
        if path:
            line_edit.setText(path)

    def _run_c3d(self):
        # Resolve C3D path
        c3d_path = self._c3d_path.text().strip()
        if not c3d_path:
            item = self._c3d_list.currentItem()
            if item and item.data(Qt.ItemDataRole.UserRole):
                c3d_path = item.data(Qt.ItemDataRole.UserRole)
        if not c3d_path:
            QMessageBox.warning(self, "No file", "Select or enter a C3D file path.")
            return

        task = Path(c3d_path).stem
        dof = self._c3d_robot.currentData()
        algo = self._c3d_algo.currentData()
        tw = self._c3d_tw
        envs = tw.envs.value()
        iters = tw.iters.value()
        ep_len = tw.ep_len.value()
        headless = "True" if tw.headless.isChecked() else "False"
        video_on = tw.video_enabled.isChecked()
        vid_int = tw.video_interval.value()
        history = tw.history_length.value()
        logger = tw.logger.currentData()
        stem = f"{dof}dof"
        exp = f"g1-{stem}-wbt-{algo}"
        retarget_dir = str(PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting")
        c3d_data_dir = str(DEFAULT_C3D_DIR)
        marker_map = self._c3d_marker_map.text().strip()
        convert_out = f"{DEFAULT_CONVERT_DIR}/{task}_mj_fps50.npz"
        converted_file = f"{retarget_dir}/{DEFAULT_CONVERT_DIR}/{task}_mj_fps50.npz"

        video_lines = []
        if video_on:
            video_lines = [
                f"    --logger.video.enabled True \\",
                f"    --logger.video.interval {vid_int} \\",
                f'    --logger.video.save-dir "{DEFAULT_VIDEO_DIR}/g1_{stem}_c3d_wbt" \\',
            ]
        else:
            video_lines = [
                f"    --logger.video.enabled False \\",
            ]

        history_line = ""
        if history > 1:
            history_line = f"    --observation.groups.actor_obs.history-length {history} \\"

        lines = [
            f'echo "=== C3D Workflow: {task} (G1-{stem}) ==="',
            "",
            "# ── Step 1: Retargeting env ──",
            f'source "{PROJECT_ROOT}/scripts/source_retargeting_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_retargeting" --quiet',
            f'cd "{retarget_dir}"',
            "",
            "# Check ezc3d",
            'if ! python3 -c "import ezc3d" 2>/dev/null; then pip install ezc3d --quiet; fi',
            "",
            "# ── Step 2: Convert C3D -> NPZ ──",
            f'echo "Converting C3D to NPZ..."',
            f"python3 data_utils/prep_c3d_for_rt.py \\",
            f'    --input "{c3d_path}" \\',
            f'    --output "{c3d_data_dir}/{task}.npz" \\',
            f"    --marker-set custom \\",
            f'    --marker-map "{marker_map}" \\',
            f"    --downsample-to 100 \\",
            f"    --lowpass-hz 4.0",
            "",
            "# ── Step 3: Retarget ──",
            f'echo "Retargeting..."',
            f"python examples/robot_retarget.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --data_path "{c3d_data_dir}" \\',
            f"    --task-type robot_only \\",
            f'    --task-name "{task}" \\',
            f"    --data_format c3d \\",
            f'    --save_dir "{DEFAULT_RETARGET_DIR}/c3d"',
            "",
            "# ── Step 4: Convert ──",
            f'echo "Converting to MuJoCo format..."',
            f"python data_conversion/convert_data_format_mj.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --input_file "{DEFAULT_RETARGET_DIR}/c3d/{task}.npz" \\',
            f"    --output_fps 50 \\",
            f'    --output_name "{convert_out}" \\',
            f"    --data_format c3d \\",
            f'    --object_name "ground" \\',
            f"    --line-range 100 500 \\",
            f"    --once",
            "",
            "# ── Step 5: IsaacSim env ──",
            f'cd "{PROJECT_ROOT}"',
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'HOLOSOMA_DEPS_DIR="${{HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}}"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma[unitree,booster]" --quiet',
            'if ! python -c "import isaaclab" 2>/dev/null; then',
            '    pip install "setuptools<81" --quiet',
            "    echo 'setuptools<81' > /tmp/hs-build-constraints.txt",
            "    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \\",
            '        pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet',
            "    rm /tmp/hs-build-constraints.txt",
            "fi",
            "",
            "# ── Step 6: Train ──",
            f'echo "Starting training ({exp})..."',
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \\",
            f"    exp:{exp} \\",
            f"    logger:{logger} \\",
            f'    --logger.base-dir "{DEFAULT_LOGS_DIR.relative_to(PROJECT_ROOT)}" \\',
            f"    --training.headless {headless} \\",
            f"    --training.num-envs {envs} \\",
            f"    --algo.config.num-learning-iterations {iters} \\",
            f"    --simulator.config.sim.max-episode-length-s {ep_len} \\",
        ]
        lines.extend(video_lines)
        if history_line:
            lines.append(history_line)
        lines.append(
            f"    --command.setup_terms.motion_command.params.motion_config.motion_file={converted_file}"
        )
        lines += [
            "",
            'echo "=== C3D Workflow complete! ==="',
        ]

        self._log_info(f"Starting C3D workflow: {task} | {exp} | envs={envs} iters={iters}")
        self._run_script(lines, start_plot=True)

    # ===================================================================
    #  TAB 3 – Inference
    # ===================================================================
    def _build_inference_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        # ── Simulation Inference ──
        sim_grp = QGroupBox("Simulation Inference (IsaacSim)")
        sim_lay = QVBoxLayout(sim_grp)
        sim_lay.addWidget(QLabel("Select a trained checkpoint:"))
        self._sim_ckpt_list = QListWidget()
        self._sim_ckpt_list.setMaximumHeight(180)
        sim_lay.addWidget(self._sim_ckpt_list)
        ref_btn = QPushButton("Refresh Checkpoints")
        ref_btn.clicked.connect(self._refresh_checkpoints)
        sim_lay.addWidget(ref_btn)

        self._sim_headless = QCheckBox("Headless mode (no GUI)")
        self._sim_headless.setChecked(True)
        sim_lay.addWidget(self._sim_headless)

        sim_btn = QPushButton("RUN SIMULATION INFERENCE")
        sim_btn.setMinimumHeight(44)
        sim_btn.setStyleSheet(BIG_BTN.format(bg="#a6e3a1", hover="#94e2d5", press="#89b4fa"))
        sim_btn.clicked.connect(self._run_sim_inference)
        sim_lay.addWidget(sim_btn)
        lay.addWidget(sim_grp)

        # ── Hardware Inference ──
        hw_grp = QGroupBox("Hardware Inference (Real Robot)")
        hw_lay = QVBoxLayout(hw_grp)
        hw_lay.addWidget(QLabel("Select an ONNX model:"))
        self._hw_onnx_list = QListWidget()
        self._hw_onnx_list.setMaximumHeight(140)
        hw_lay.addWidget(self._hw_onnx_list)
        btn_row = QHBoxLayout()
        ref_onnx = QPushButton("Refresh ONNX")
        ref_onnx.clicked.connect(self._refresh_onnx)
        btn_row.addWidget(ref_onnx)
        browse_onnx = QPushButton("Browse .onnx ...")
        browse_onnx.clicked.connect(self._browse_onnx)
        btn_row.addWidget(browse_onnx)
        btn_row.addStretch()
        hw_lay.addLayout(btn_row)
        self._hw_onnx_path = QLineEdit()
        self._hw_onnx_path.setPlaceholderText("Or enter ONNX path")
        hw_lay.addWidget(self._hw_onnx_path)

        warn = QLabel("WARNING: This will control the physical G1 robot!\n"
                       "Ensure E-stop is accessible.")
        warn.setStyleSheet("color: #f38ba8; font-weight: bold; padding: 6px;")
        hw_lay.addWidget(warn)

        hw_btn = QPushButton("DEPLOY TO HARDWARE")
        hw_btn.setMinimumHeight(44)
        hw_btn.setStyleSheet(BIG_BTN.format(bg="#f38ba8", hover="#eba0ac", press="#f5c2e7"))
        hw_btn.clicked.connect(self._run_hw_inference)
        hw_lay.addWidget(hw_btn)
        lay.addWidget(hw_grp)

        # Keyboard controls info
        info_grp = QGroupBox("Hardware Keyboard Controls")
        info_lay = QVBoxLayout(info_grp)
        info = QLabel("  ]   Start policy\n  o   Stop policy\n  i   Default pose\n  s   Start motion clip")
        info.setFont(QFont("Monospace", 10))
        info_lay.addWidget(info)
        lay.addWidget(info_grp)

        lay.addStretch()
        scroll.setWidget(w)

        self._refresh_checkpoints()
        self._refresh_onnx()
        return scroll

    def _refresh_checkpoints(self):
        self._sim_ckpt_list.clear()
        ckpts = scan_checkpoints(DEFAULT_LOGS_DIR)
        if not ckpts:
            self._sim_ckpt_list.addItem("(no checkpoints found)")
        for p in ckpts:
            rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
            label = f"{rel}  ({human_size(p)})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._sim_ckpt_list.addItem(item)

    def _refresh_onnx(self):
        self._hw_onnx_list.clear()
        files = scan_onnx(DEFAULT_LOGS_DIR)
        for d in [PROJECT_ROOT / "logs", PROJECT_ROOT / "converted_res"]:
            if d.is_dir():
                for f in sorted(d.rglob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if f not in files:
                        files.append(f)
        if not files:
            self._hw_onnx_list.addItem("(no ONNX files found)")
        for p in files:
            rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
            item = QListWidgetItem(f"{rel}  ({human_size(p)})")
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._hw_onnx_list.addItem(item)

    def _browse_onnx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX", str(PROJECT_ROOT), "ONNX (*.onnx)")
        if path:
            self._hw_onnx_path.setText(path)

    def _run_sim_inference(self):
        item = self._sim_ckpt_list.currentItem()
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            QMessageBox.warning(self, "No checkpoint", "Select a checkpoint first.")
            return
        ckpt = item.data(Qt.ItemDataRole.UserRole)
        headless = "True" if self._sim_headless.isChecked() else "False"

        lines = [
            f'echo "=== Simulation Inference ==="',
            f'echo "Checkpoint: {ckpt}"',
            "",
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma[unitree,booster]" --quiet',
            "",
            f'cd "{PROJECT_ROOT}"',
            f"python src/holosoma/holosoma/eval_agent.py \\",
            f'    --checkpoint "{ckpt}" \\',
            f"    --training.export-onnx True \\",
            f"    --training.num-envs 1 \\",
            f"    --training.headless {headless}",
            "",
            'echo "=== Simulation inference complete! ==="',
        ]
        self._log_info(f"Running sim inference: {ckpt}")
        self._run_script(lines)

    def _run_hw_inference(self):
        onnx_path = self._hw_onnx_path.text().strip()
        if not onnx_path:
            item = self._hw_onnx_list.currentItem()
            if item and item.data(Qt.ItemDataRole.UserRole):
                onnx_path = item.data(Qt.ItemDataRole.UserRole)
        if not onnx_path:
            QMessageBox.warning(self, "No ONNX", "Select or enter an ONNX model path.")
            return

        reply = QMessageBox.warning(
            self, "Hardware Deployment",
            "This will control the physical G1 robot!\n"
            "Make sure E-stop is accessible.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._log_info("Hardware deployment cancelled.")
            return

        lines = [
            f'echo "=== Hardware Inference ==="',
            f'echo "ONNX: {onnx_path}"',
            "",
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_inference_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_inference" --quiet',
            "",
            f'cd "{PROJECT_ROOT}"',
            f"python src/holosoma_inference/holosoma_inference/run_policy.py \\",
            f"    inference:g1-23dof-wbt \\",
            f'    --task.model-path="{onnx_path}" \\',
            f"    --observation.groups.actor_obs.history-length=4",
            "",
            'echo "=== Hardware inference complete! ==="',
        ]
        self._log_info(f"Deploying to hardware: {onnx_path}")
        self._run_script(lines)

    # ===================================================================
    #  TAB 4 – Browse Files
    # ===================================================================
    def _build_files_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        lf_grp = QGroupBox(f"LAFAN Sequences  ({DEFAULT_LAFAN_DIR})")
        lf_lay = QVBoxLayout(lf_grp)
        self._files_lafan = QListWidget()
        self._files_lafan.setMaximumHeight(200)
        lf_lay.addWidget(self._files_lafan)
        lay.addWidget(lf_grp)

        c3d_grp = QGroupBox(f"C3D Files  ({DEFAULT_C3D_DIR})")
        c3d_lay = QVBoxLayout(c3d_grp)
        self._files_c3d = QListWidget()
        self._files_c3d.setMaximumHeight(140)
        c3d_lay.addWidget(self._files_c3d)
        lay.addWidget(c3d_grp)

        ck_grp = QGroupBox(f"Checkpoints  ({DEFAULT_LOGS_DIR})")
        ck_lay = QVBoxLayout(ck_grp)
        self._files_ckpt = QListWidget()
        self._files_ckpt.setMaximumHeight(200)
        ck_lay.addWidget(self._files_ckpt)
        lay.addWidget(ck_grp)

        ref_btn = QPushButton("Refresh All")
        ref_btn.clicked.connect(self._refresh_files_tab)
        lay.addWidget(ref_btn)

        lay.addStretch()
        scroll.setWidget(w)

        self._refresh_files_tab()
        return scroll

    def _refresh_files_tab(self):
        self._files_lafan.clear()
        for t in scan_npy(DEFAULT_LAFAN_DIR):
            p = DEFAULT_LAFAN_DIR / f"{t}.npy"
            self._files_lafan.addItem(f"{t}  ({human_size(p)})")
        if self._files_lafan.count() == 0:
            self._files_lafan.addItem("(no .npy files found)")

        self._files_c3d.clear()
        for f in scan_c3d(DEFAULT_C3D_DIR):
            p = DEFAULT_C3D_DIR / f
            self._files_c3d.addItem(f"{f}  ({human_size(p)})")
        if self._files_c3d.count() == 0:
            self._files_c3d.addItem("(no .c3d files found)")

        self._files_ckpt.clear()
        for p in scan_checkpoints(DEFAULT_LOGS_DIR):
            rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
            self._files_ckpt.addItem(f"{rel}  ({human_size(p)})")
        if self._files_ckpt.count() == 0:
            self._files_ckpt.addItem("(no checkpoints found)")

    # ===================================================================
    #  TAB 5 – Settings / Configuration
    # ===================================================================
    def _build_settings_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        dir_grp = QGroupBox("Directory Configuration")
        dir_lay = QVBoxLayout(dir_grp)
        rows = [
            ("Project Root:", str(PROJECT_ROOT)),
            ("LAFAN Data:", str(DEFAULT_LAFAN_DIR)),
            ("C3D Data:", str(DEFAULT_C3D_DIR)),
            ("Logs Dir:", str(DEFAULT_LOGS_DIR)),
            ("Retarget Output:", str(PROJECT_ROOT / DEFAULT_RETARGET_DIR)),
            ("Convert Output:", str(PROJECT_ROOT / DEFAULT_CONVERT_DIR)),
            ("Video Dir:", str(PROJECT_ROOT / DEFAULT_VIDEO_DIR)),
        ]
        for label_text, value in rows:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(120)
            row.addWidget(lbl)
            edit = QLineEdit(value)
            edit.setReadOnly(True)
            edit.setStyleSheet("background: #313244; color: #9399b2;")
            row.addWidget(edit)
            dir_lay.addLayout(row)
        dir_lay.addWidget(QLabel(
            "\nTo customise output paths, set environment variables before launching:\n"
            "  export RETARGET_OUTPUT_DIR=...\n"
            "  export CONVERT_OUTPUT_DIR=...\n"
            "  export LOGS_DIR=...\n"
            "  export VIDEO_DIR=..."
        ))
        lay.addWidget(dir_grp)

        check_grp = QGroupBox("Directory Status")
        check_lay = QVBoxLayout(check_grp)
        dirs_check = [
            ("LAFAN Data", DEFAULT_LAFAN_DIR),
            ("C3D Data", DEFAULT_C3D_DIR),
            ("Logs", DEFAULT_LOGS_DIR),
            ("Retarget Output", PROJECT_ROOT / DEFAULT_RETARGET_DIR),
            ("Convert Output", PROJECT_ROOT / DEFAULT_CONVERT_DIR),
            ("Videos", PROJECT_ROOT / DEFAULT_VIDEO_DIR),
        ]
        for name, path in dirs_check:
            exists = path.exists()
            colour = "#a6e3a1" if exists else "#f38ba8"
            symbol = "OK" if exists else "MISSING"
            lbl = QLabel(f"  [{symbol}]  {name}: {path}")
            lbl.setStyleSheet(f"color: {colour};")
            check_lay.addWidget(lbl)
        lay.addWidget(check_grp)

        # Dependencies check
        dep_grp = QGroupBox("Python Dependencies")
        dep_lay = QVBoxLayout(dep_grp)
        deps = [
            ("PySide6", True),
            ("matplotlib", True),
            ("tbparse (live plots)", HAS_TBPARSE),
        ]
        for name, ok in deps:
            colour = "#a6e3a1" if ok else "#f9e2af"
            symbol = "OK" if ok else "MISSING"
            lbl = QLabel(f"  [{symbol}]  {name}")
            lbl.setStyleSheet(f"color: {colour};")
            dep_lay.addWidget(lbl)
        if not HAS_TBPARSE:
            dep_lay.addWidget(QLabel("    Install: pip install tbparse"))
        lay.addWidget(dep_grp)

        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # ===================================================================
    #  Cleanup
    # ===================================================================
    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            reply = QMessageBox.question(
                self, "Process Running",
                "A process is still running. Kill it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._process.kill()
                self._process.waitForFinished(3000)
            else:
                event.ignore()
                return
        self._plot_widget.stop_monitoring()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = WorkflowLauncher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

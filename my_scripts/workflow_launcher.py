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
    python my_scripts/workflow_launcher.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
import subprocess
from pathlib import Path

import psutil
import numpy as np

# Suppress Matplotlib Axes3D warning (occurs if multiple versions are installed)
warnings.filterwarnings("ignore", message=".*Unable to import Axes3D.*")

import matplotlib as mpl

mpl.use("QtAgg")

# The system mpl_toolkits may be from an older distro matplotlib and is
# incompatible with the user-installed version. Override __path__ to force
# loading from the same site-packages tree as our matplotlib.
# This MUST happen before importing matplotlib.figure / matplotlib.backends,
# because those internally try to import Axes3D and register the '3d' projection.
import mpl_toolkits as _mpl_toolkits_pkg
_mpl_toolkits_pkg.__path__ = [str(Path(mpl.__file__).parent.parent / "mpl_toolkits")]

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PySide6.QtCore import QProcess, QTimer, Qt, Slot
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
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
DEFAULT_OMOMO_DIR = PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs/WholeBodyTracking"
DEFAULT_RETARGET_DIR = "demo_results/g1/robot_only"
DEFAULT_CONVERT_DIR = "converted_res/robot_only"
DEFAULT_VIDEO_DIR = "logs/videos"
PRESETS_DIR = PROJECT_ROOT / "configs" / "launcher_presets"
LAUNCHER_CONFIG_PATH = Path.home() / ".config" / "holosoma_launcher" / "config.json"

# Default values stored separately so _reset_dir_config can restore them
_DEFAULT_LAFAN_DIR = DEFAULT_LAFAN_DIR
_DEFAULT_C3D_DIR = DEFAULT_C3D_DIR
_DEFAULT_OMOMO_DIR = DEFAULT_OMOMO_DIR
_DEFAULT_LOGS_DIR = DEFAULT_LOGS_DIR
_DEFAULT_RETARGET_DIR = DEFAULT_RETARGET_DIR
_DEFAULT_CONVERT_DIR = DEFAULT_CONVERT_DIR
_DEFAULT_VIDEO_DIR = DEFAULT_VIDEO_DIR


def _apply_dir_config(cfg: dict) -> None:
    """Update module-level directory globals from a config dict."""
    global DEFAULT_LAFAN_DIR, DEFAULT_C3D_DIR, DEFAULT_OMOMO_DIR
    global DEFAULT_LOGS_DIR, DEFAULT_RETARGET_DIR, DEFAULT_CONVERT_DIR, DEFAULT_VIDEO_DIR
    if "lafan_dir" in cfg and cfg["lafan_dir"]:
        DEFAULT_LAFAN_DIR = Path(cfg["lafan_dir"])
    if "c3d_dir" in cfg and cfg["c3d_dir"]:
        DEFAULT_C3D_DIR = Path(cfg["c3d_dir"])
    if "omomo_dir" in cfg and cfg["omomo_dir"]:
        DEFAULT_OMOMO_DIR = Path(cfg["omomo_dir"])
    if "logs_dir" in cfg and cfg["logs_dir"]:
        DEFAULT_LOGS_DIR = Path(cfg["logs_dir"])
    if "retarget_dir" in cfg and cfg["retarget_dir"]:
        DEFAULT_RETARGET_DIR = cfg["retarget_dir"]
    if "convert_dir" in cfg and cfg["convert_dir"]:
        DEFAULT_CONVERT_DIR = cfg["convert_dir"]
    if "video_dir" in cfg and cfg["video_dir"]:
        DEFAULT_VIDEO_DIR = cfg["video_dir"]


def _load_launcher_config() -> None:
    """Load persisted directory config and override module defaults."""
    if LAUNCHER_CONFIG_PATH.exists():
        try:
            with open(LAUNCHER_CONFIG_PATH) as f:
                _apply_dir_config(json.load(f))
        except Exception:
            pass  # silently ignore malformed config


_load_launcher_config()

ROBOTS = [("23", "G1-23DOF"), ("29", "G1-29DOF")]
# Values are the exp: subcommand suffix appended to "g1-{N}dof-wbt".
# Empty string = PPO base preset (exp:g1-{N}dof-wbt)
# "-fast-sac"   = Fast-SAC preset (exp:g1-{N}dof-wbt-fast-sac)
ALGORITHMS = [
    ("-fast-sac", "Fast SAC (default)"),
    ("", "PPO (base)"),
]
LOGGERS = [
    ("wandb-offline", "W&B Offline (default)"),
    ("wandb", "W&B Online"),
    ("disabled", "Disabled"),
]

# Metrics to plot -- same 4 panels as plot_rewards.py
PLOT_METRICS = {
    "Mean Reward": lambda t: "mean_reward" in t.lower() or t.lower() == "reward",
    "Episode Length": lambda t: "average_episode_length" in t.lower(),
    "Joint Pos Error (rad)": lambda t: "error_joint_pos" in t.lower(),
    "Curriculum Entropy": lambda t: "adaptive_timesteps_sampler_entropy" in t.lower(),
}
PLOT_COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]

# Resource estimation constants
# Calibrated from OOM trace: 4096 envs, history=4, SAC, headless
#   actor_obs_dim=496, critic_obs_dim=256 (already include history concat)
#   OOM hit at critic_observations alloc (4.00 GiB) with 21.30 GB used already
#   => observations(7.75) + next_obs(7.75) + actions(0.36) = 15.86 GB buffer so far
#   => sim + models overhead = 21.30 - 15.86 = 5.44 GB at 4096 envs
#
# SAC replay buffer (4 tensors, buf_size=1024 default):
#   per env = 2*(actor_obs + critic_obs)*buf_size*4B
#   base dims (history=1): actor~124, critic~64  -> 0.00143 GB/env
#   scales linearly with history (obs = base_dim * history)
#
# Sim overhead: ~2 GB fixed + 0.00084 GB/env
#   at 4096 envs: 2.0 + 3.44 = 5.44 GB  ✓
VRAM_BASE_HEADLESS = 2.0    # GB fixed (IsaacSim process + neural net models)
VRAM_BASE_GUI      = 3.5    # GB (adds viewport/render overhead)
VRAM_PER_ENV_SIM   = 0.00084  # GB per env for physics/sim tensors
VRAM_ALGO_PPO      = 0.5    # GB extra for PPO rollout buffers
# SAC replay buffer: 2*(actor_obs_dim+critic_obs_dim)*buf_size*4B per env, dims scale with history
# base obs_dim (history=1): actor~124, critic~64; buf_size=1024 (default)
# = 2*(124+64)*1024*4 / 1024^3 = 0.00143 GB per env at history=1; linear with history
VRAM_SAC_BUF_PER_ENV_PER_HIST = 0.00143  # GB per env per history step

RAM_BASE           = 4.5    # GB (System + Base Isaac)
RAM_PER_ENV        = 0.0006  # GB per env

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
QProgressBar { 
    border: 1px solid #45475a; border-radius: 4px; background: #313244;
    text-align: center; height: 14px; font-weight: bold; font-size: 10px; color: #1e1e2e;
}
QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
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


def scan_omomo(directory: Path) -> list[str]:
    """Return task names (stems) of .npz files in an OMOMO data directory."""
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.npz"))


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
        self._eta_samples: list[tuple[float, int]] = []  # (wall_time, step) for ETA calc
        self._target_iters: int = 0

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

        # Folder picker row
        folder_row = QHBoxLayout()
        folder_lbl = QLabel("Log dir:")
        folder_lbl.setStyleSheet("color: #9399b2; font-size: 11px;")
        folder_lbl.setFixedWidth(52)
        folder_row.addWidget(folder_lbl)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("auto-detect latest run")
        self._dir_edit.setStyleSheet(
            "background:#313244; color:#cdd6f4; border:1px solid #45475a;"
            " border-radius:4px; padding:2px 6px; font-size:11px;"
        )
        self._dir_edit.editingFinished.connect(self._on_dir_edited)
        folder_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.setStyleSheet(
            "background:#313244; color:#cdd6f4; border:1px solid #45475a;"
            " border-radius:4px; padding:2px 6px; font-size:11px;"
        )
        browse_btn.clicked.connect(self._browse_log_dir)
        folder_row.addWidget(browse_btn)
        lay.addLayout(folder_row)

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

    def set_target_iters(self, n: int) -> None:
        self._target_iters = n
        self._eta_samples.clear()

    def start_monitoring(self, log_dir: Path | None = None):
        """Start polling for new TensorBoard events."""
        self._log_dir = log_dir
        self._last_read_size.clear()
        self._eta_samples.clear()
        self._metric_stats = {}
        self._dir_edit.setText(str(log_dir) if log_dir else "")
        self._draw_empty()
        if hasattr(self, "_scoreboard") and self._scoreboard is not None:
            self._scoreboard.clear_stats()
        if HAS_TBPARSE:
            self._timer.start(10_000)  # 10s interval
            self._status_lbl.setText(f"Monitoring: {log_dir or 'latest run'}")

    def stop_monitoring(self):
        self._timer.stop()
        self._status_lbl.setText("Monitoring stopped")

    @Slot()
    def _browse_log_dir(self):
        start = str(self._log_dir) if self._log_dir else str(DEFAULT_LOGS_DIR)
        chosen = QFileDialog.getExistingDirectory(self, "Select TensorBoard log directory", start)
        if chosen:
            self._apply_new_dir(Path(chosen))

    @Slot()
    def _on_dir_edited(self):
        text = self._dir_edit.text().strip()
        if text:
            self._apply_new_dir(Path(text))
        else:
            # Empty = revert to auto-detect
            self._log_dir = None
            self._last_read_size.clear()
            self._status_lbl.setText("Monitoring: auto-detect latest run")
            self.refresh_plot()

    def _apply_new_dir(self, path: Path):
        self._log_dir = path
        self._last_read_size.clear()
        self._dir_edit.setText(str(path))
        if HAS_TBPARSE:
            self._status_lbl.setText(f"Monitoring: {path.name}")
        self.refresh_plot()

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
                    ax.plot(steps, values, color=PLOT_COLORS[i], linewidth=1.2,
                            marker="o", markersize=1.5, alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                        color="#585b70", fontsize=9, transform=ax.transAxes)

        # Collect per-metric stats for the scoreboard
        metric_stats: dict[str, dict] = {}
        for i, (metric_name, matcher) in enumerate(PLOT_METRICS.items()):
            matched = [t for t in available_tags if matcher(t)]
            if matched:
                subset = df[df["tag"] == matched[0]].sort_values("step")
                vals = subset["value"].values
                steps_arr = subset["step"].values
                if len(vals) > 0:
                    metric_stats[metric_name] = {
                        "latest": float(vals[-1]),
                        "max": float(vals.max()),
                        "min": float(vals.min()),
                        "step": int(steps_arr[-1]),
                    }
        self._metric_stats = metric_stats

        self._canvas.draw_idle()
        n_points = len(df)
        run_name = log_dir.name

        # ETA calculation — use mean reward tag as the iteration counter
        eta_str = ""
        steps_per_sec = 0.0
        eta_sec = 0.0
        current_step = 0
        try:
            step_col = df["step"]
            current_step = int(step_col.max())
            now = time.monotonic()
            self._eta_samples.append((now, current_step))
            if len(self._eta_samples) > 10:
                self._eta_samples = self._eta_samples[-10:]
            if len(self._eta_samples) >= 2 and self._target_iters > 0:
                t0, s0 = self._eta_samples[0]
                t1, s1 = self._eta_samples[-1]
                dt = t1 - t0
                ds = s1 - s0
                if dt > 0 and ds > 0:
                    steps_per_sec = ds / dt
                    remaining = max(0, self._target_iters - current_step)
                    eta_sec = remaining / steps_per_sec
                    if eta_sec < 60:
                        eta_str = f"  |  ETA: {eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        eta_str = f"  |  ETA: {eta_sec/60:.0f}m"
                    else:
                        h = int(eta_sec // 3600)
                        m = int((eta_sec % 3600) // 60)
                        eta_str = f"  |  ETA: {h}h {m}m"
        except Exception:
            eta_str = ""

        self._status_lbl.setText(f"Run: {run_name}  |  {n_points} pts{eta_str}")

        # Push stats to scoreboard if attached
        if hasattr(self, "_scoreboard") and self._scoreboard is not None:
            self._scoreboard.update_stats(
                metric_stats, current_step, self._target_iters,
                steps_per_sec, eta_sec,
            )

    def set_log_dir(self, path: Path):
        self._apply_new_dir(path)

    def get_latest_stats(self) -> dict[str, dict]:
        """Return latest/best/step stats per metric from the last refresh, or empty."""
        return getattr(self, "_metric_stats", {})


# ---------------------------------------------------------------------------
# Training Scoreboard Widget
# ---------------------------------------------------------------------------
class TrainingScoreboard(QWidget):
    """Live scoreboard showing key training stats from TensorBoard logs."""

    _ROW_STYLE = (
        "background:#313244; border-radius:4px; padding:6px 8px; margin:1px 0;"
    )
    _VALUE_STYLE = "font-size:16px; font-weight:bold; font-family:'JetBrains Mono',monospace;"
    _LABEL_STYLE = "font-size:10px; color:#9399b2;"
    _SUBLABEL_STYLE = "font-size:9px; color:#6c7086;"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        header = QLabel("Training Scoreboard")
        header.setStyleSheet("font-weight:bold; font-size:12px; color:#cdd6f4; padding:2px;")
        lay.addWidget(header)

        # Progress row
        self._progress_lbl = QLabel("Iteration: — / —")
        self._progress_lbl.setStyleSheet("font-size:11px; color:#cdd6f4; padding:2px;")
        lay.addWidget(self._progress_lbl)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background:#313244; border:1px solid #45475a; border-radius:4px; text-align:center; font-size:9px; color:#cdd6f4; }"
            "QProgressBar::chunk { background:#89b4fa; border-radius:3px; }"
        )
        lay.addWidget(self._progress_bar)

        # Speed / ETA row
        self._speed_lbl = QLabel("Speed: —  |  ETA: —")
        self._speed_lbl.setStyleSheet("font-size:10px; color:#9399b2; padding:2px 2px 4px;")
        lay.addWidget(self._speed_lbl)

        # Metric cards
        self._cards: list[dict[str, QLabel]] = []
        for metric_name, color in zip(PLOT_METRICS, PLOT_COLORS):
            card = QWidget()
            card.setStyleSheet(self._ROW_STYLE)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(6, 4, 6, 4)
            cl.setSpacing(1)

            name_lbl = QLabel(metric_name)
            name_lbl.setStyleSheet(f"font-size:10px; font-weight:bold; color:{color};")
            cl.addWidget(name_lbl)

            val_row = QHBoxLayout()
            val_row.setSpacing(0)
            latest_lbl = QLabel("—")
            latest_lbl.setStyleSheet(f"{self._VALUE_STYLE} color:{color};")
            val_row.addWidget(latest_lbl)
            val_row.addStretch()
            cl.addLayout(val_row)

            sub_row = QHBoxLayout()
            best_lbl = QLabel("Best: —")
            best_lbl.setStyleSheet(self._SUBLABEL_STYLE)
            sub_row.addWidget(best_lbl)
            sub_row.addStretch()
            step_lbl = QLabel("Step: —")
            step_lbl.setStyleSheet(self._SUBLABEL_STYLE)
            sub_row.addWidget(step_lbl)
            cl.addLayout(sub_row)

            lay.addWidget(card)
            self._cards.append({
                "latest": latest_lbl,
                "best": best_lbl,
                "step": step_lbl,
            })

        lay.addStretch()

    def update_stats(self, stats: dict[str, dict], current_step: int,
                     target_iters: int, speed: float, eta_sec: float):
        # Progress
        if target_iters > 0:
            pct = min(100, int(100 * current_step / target_iters))
            self._progress_bar.setValue(pct)
            self._progress_lbl.setText(
                f"Iteration: {current_step:,} / {target_iters:,}  ({pct}%)"
            )
        else:
            self._progress_bar.setValue(0)
            self._progress_lbl.setText(f"Iteration: {current_step:,}")

        # Speed + ETA
        speed_s = f"{speed:.1f} it/s" if speed > 0 else "—"
        if eta_sec > 0:
            if eta_sec < 60:
                eta_s = f"{eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta_s = f"{eta_sec / 60:.0f}m"
            else:
                h = int(eta_sec // 3600)
                m = int((eta_sec % 3600) // 60)
                eta_s = f"{h}h {m}m"
        else:
            eta_s = "—"
        self._speed_lbl.setText(f"Speed: {speed_s}  |  ETA: {eta_s}")

        # Metric cards
        for card_dict, metric_name in zip(self._cards, PLOT_METRICS):
            s = stats.get(metric_name)
            if s is None:
                card_dict["latest"].setText("—")
                card_dict["best"].setText("Best: —")
                card_dict["step"].setText("Step: —")
                continue

            fmt = self._fmt_val
            card_dict["latest"].setText(fmt(s["latest"]))
            # For error metrics lower is better; for reward/length higher is better
            is_error = "error" in metric_name.lower()
            best_val = s["min"] if is_error else s["max"]
            card_dict["best"].setText(f"Best: {fmt(best_val)}")
            card_dict["step"].setText(f"Step: {s['step']:,}")

    def clear_stats(self):
        self._progress_bar.setValue(0)
        self._progress_lbl.setText("Iteration: — / —")
        self._speed_lbl.setText("Speed: —  |  ETA: —")
        for card_dict in self._cards:
            card_dict["latest"].setText("—")
            card_dict["best"].setText("Best: —")
            card_dict["step"].setText("Step: —")

    @staticmethod
    def _fmt_val(v: float) -> str:
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 0.01:
            return f"{v:.3f}"
        return f"{v:.4e}"


# ---------------------------------------------------------------------------
# LAFAN1 skeleton: 22 joints, parent indices (-1 = root)
# ---------------------------------------------------------------------------
_LAFAN_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
_LAFAN_NAMES = [
    "root", "lhip", "rhip", "belly",
    "lknee", "rknee", "spine",
    "lankle", "rankle", "chest",
    "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder", "head",
    "lshoulder", "rshoulder",
    "lelbow", "relbow",
    "lwrist", "rwrist",
]


# ---------------------------------------------------------------------------
# Motion Preview Dialog
# ---------------------------------------------------------------------------
class MotionPreviewDialog(QWidget):
    """Standalone window to preview a .npy motion file (LAFAN1 format).

    Shows a 3D skeleton plot with a frame scrubber and play/pause.
    """

    def __init__(self, npy_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Motion Preview — {npy_path.name}")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(760, 560)
        self.setStyleSheet(STYLESHEET)

        self._path = npy_path
        self._data: np.ndarray | None = None
        self._frame = 0
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Info bar
        self._info_lbl = QLabel("Loading…")
        self._info_lbl.setStyleSheet("color: #89b4fa; font-size: 11px; padding: 2px 4px;")
        lay.addWidget(self._info_lbl)

        # Main splitter: left = plot, right = metadata
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 3D matplotlib canvas
        self._fig = Figure(figsize=(5, 4), facecolor="#1e1e2e")
        self._ax: Axes3D = self._fig.add_subplot(111, projection="3d")
        
        # Set good initial view angle: 20 deg elevation, 45 deg azimuth
        self._ax.view_init(elev=20, azim=45)
        
        self._canvas = FigureCanvasQTAgg(self._fig)
        splitter.addWidget(self._canvas)

        # Metadata panel
        meta_box = QGroupBox("Sequence Info")
        meta_lay = QVBoxLayout(meta_box)
        self._meta_text = QTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setMaximumWidth(200)
        self._meta_text.setStyleSheet(
            "background:#313244; color:#cdd6f4; font-size:11px; border:none;"
        )
        meta_lay.addWidget(self._meta_text)
        splitter.addWidget(meta_box)
        splitter.setSizes([560, 200])
        lay.addWidget(splitter, 1)

        # Frame controls
        ctrl = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(60)
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn)

        self._frame_lbl = QLabel("Frame: 0 / 0")
        self._frame_lbl.setFixedWidth(100)
        ctrl.addWidget(self._frame_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self._slider)

        fps_lbl = QLabel("FPS:")
        ctrl.addWidget(fps_lbl)
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(30)
        self._fps_spin.setFixedWidth(55)
        self._fps_spin.valueChanged.connect(self._update_timer_interval)
        ctrl.addWidget(self._fps_spin)

        lay.addLayout(ctrl)

        # Load data after show
        QTimer.singleShot(50, self._load_data)

    # ------------------------------------------------------------------
    def _load_data(self):
        try:
            data = np.load(str(self._path))
        except Exception as e:
            self._info_lbl.setText(f"Error loading file: {e}")
            return

        if data.ndim != 3 or data.shape[2] != 3:
            self._info_lbl.setText(
                f"Unexpected shape {data.shape}. Expected (T, J, 3)."
            )
            return

        self._data = data
        T, J, _ = data.shape
        self._slider.setMaximum(T - 1)

        # Metadata text
        size_mb = self._path.stat().st_size / 1024 / 1024
        known = J == len(_LAFAN_NAMES)
        meta = (
            f"File: {self._path.name}\n"
            f"Size: {size_mb:.2f} MB\n\n"
            f"Frames (T): {T}\n"
            f"Joints (J): {J}\n"
            f"Shape: ({T}, {J}, 3)\n\n"
            f"Skeleton: {'LAFAN1 (22J)' if known else f'Unknown ({J}J)'}\n\n"
            f"X range: [{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}]\n"
            f"Y range: [{data[:,:,1].min():.2f}, {data[:,:,1].max():.2f}]\n"
            f"Z range: [{data[:,:,2].min():.2f}, {data[:,:,2].max():.2f}]\n"
        )
        self._meta_text.setPlainText(meta)
        self._info_lbl.setText(
            f"{self._path.name}  |  {T} frames  |  {J} joints  |  {size_mb:.1f} MB"
        )

        self._draw_frame(0)

    def _draw_frame(self, frame_idx: int):
        if self._data is None:
            return
    
        # LAFAN1 data is typically Y-UP. Matplotlib 3D is Z-UP by default.
        # We map Data(X, Y, Z) -> Plot(X, Z, Y) so the character stands upright.
        data = self._data
        J = data.shape[1]
        raw_pts = data[frame_idx]  # (J, 3) -> [x, y, z] (y is up)
        
        # Coordinate mapping for upright visualization
        pts = np.zeros_like(raw_pts)
        pts[:, 0] = raw_pts[:, 0]  # Side (X)
        pts[:, 1] = raw_pts[:, 2]  # Depth (Z)
        pts[:, 2] = raw_pts[:, 1]  # Height (Y)
    
        ax = self._ax
        # Save current view orientation to prevent resets during play
        elev, azim = ax.elev, ax.azim
        
        ax.clear()
        ax.set_facecolor("#1e1e2e")
        ax.set_xlabel("Side (X)", color="#9399b2", fontsize=7)
        ax.set_ylabel("Depth (Z)", color="#9399b2", fontsize=7)
        ax.set_zlabel("Height (Y)", color="#9399b2", fontsize=7)
        ax.tick_params(colors="#585b70", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#45475a")
        
        # Restore view orientation
        ax.view_init(elev=elev, azim=azim)
    
        # Draw a simple ground grid around the character's projection
        root_pos = pts[0]
        grid_size = 2.0
        grid_steps = 5
        x_grid = np.linspace(root_pos[0] - grid_size, root_pos[0] + grid_size, grid_steps)
        y_grid = np.linspace(root_pos[1] - grid_size, root_pos[1] + grid_size, grid_steps)
        for x in x_grid:
            ax.plot([x, x], [y_grid[0], y_grid[-1]], [0, 0], color="#313244", linewidth=0.5, alpha=0.5)
        for y in y_grid:
            ax.plot([x_grid[0], x_grid[-1]], [y, y], [0, 0], color="#313244", linewidth=0.5, alpha=0.5)
    
        # Scatter joints
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c="#89b4fa", s=25, depthshade=False, zorder=5)
    
        # Draw bones if skeleton matches
        if J == len(_LAFAN_PARENTS):
            for j, parent in enumerate(_LAFAN_PARENTS):
                if parent < 0:
                    continue
                p0, p1 = pts[parent], pts[j]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color="#a6e3a1", linewidth=2.0, alpha=0.9)
    
        # Camera following: Center on the root joint (pts[0])
        # Use a fixed span (1.2m radius) for consistent scaling across sequences
        span = 1.2
        ax.set_xlim(root_pos[0] - span, root_pos[0] + span)
        ax.set_ylim(root_pos[1] - span, root_pos[1] + span)
        # For Z (height), we show from ground to slightly above head
        ax.set_zlim(-0.1, 2 * span - 0.1)
    
        self._canvas.draw_idle()

    def _on_slider(self, value: int):
        self._frame = value
        T = 0 if self._data is None else self._data.shape[0]
        self._frame_lbl.setText(f"Frame: {value} / {T - 1}")
        self._draw_frame(value)

    def _toggle_play(self):
        self._playing = not self._playing
        self._play_btn.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._update_timer_interval()
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _next_frame(self):
        if self._data is None:
            return
        T = self._data.shape[0]
        self._frame = (self._frame + 1) % T
        self._slider.setValue(self._frame)

    def _update_timer_interval(self):
        interval = max(1, 1000 // self._fps_spin.value())
        self._play_timer.setInterval(interval)

    def closeEvent(self, event):
        self._play_timer.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Training Config dataclass-like container returned by _make_training_group
# ---------------------------------------------------------------------------
class _TrainingWidgets:
    """Holds references to all training-config widgets for a tab."""
    __slots__ = (
        "alpha_init", "envs", "ep_len", "foot_tolerance", "group",
        "headless", "history_length", "iters", "logger",
        "video_enabled", "video_interval",
        "vram_bar", "vram_lbl", "ram_bar", "ram_lbl", "health_msg",
        "robot_cb", "algo_cb", "run_btn", "preset_combo"
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
        self._gpu_used_mb = 0.0
        self._gpu_total_mb = 0.0
        self._pulse_state = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._pulse_timer.start(500)

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
        tabs.addTab(self._build_omomo_tab(), "OMOMO Tracking")
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

        # Bottom row: console + scoreboard side by side
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

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
        bottom_splitter.addWidget(console_box)

        # Scoreboard panel
        scoreboard_box = QGroupBox("Scoreboard")
        sb_lay = QVBoxLayout(scoreboard_box)
        sb_lay.setContentsMargins(4, 14, 4, 4)
        self._scoreboard = TrainingScoreboard()
        sb_lay.addWidget(self._scoreboard)
        bottom_splitter.addWidget(scoreboard_box)

        bottom_splitter.setSizes([480, 220])
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 1)

        # Link scoreboard to plot widget
        self._plot_widget._scoreboard = self._scoreboard

        right_splitter.addWidget(bottom_splitter)

        right_splitter.setSizes([420, 350])
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)

        h_splitter.addWidget(right_splitter)
        h_splitter.setSizes([540, 700])
        h_splitter.setStretchFactor(0, 2)
        h_splitter.setStretchFactor(1, 3)
        root_lay.addWidget(h_splitter, 1)

        self.statusBar().showMessage("Ready")

        # ── GPU Status Monitoring ──────────────────────────────────────
        self._gpu_lbl = QLabel("GPU VRAM: -- / -- MB")
        self._gpu_lbl.setStyleSheet("color: #f9e2af; font-family: 'JetBrains Mono', monospace; font-weight: bold; padding-right: 10px;")
        self.statusBar().addPermanentWidget(self._gpu_lbl)

        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_status)
        self._gpu_timer.start(2000)
        self._update_gpu_status()

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

    @staticmethod
    def _logs_dir_arg() -> str:
        """Return the --logger.base-dir value: relative if inside PROJECT_ROOT, else absolute."""
        p = Path(DEFAULT_LOGS_DIR)
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    @Slot()
    def _update_gpu_status(self):
        """Fetch and display GPU VRAM usage via nvidia-smi."""
        try:
            # query-gpu returns e.g. "8502, 23028"
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL).strip()
            
            # If multiple GPUs, this might return multiple lines; take the first one for now
            line = output.splitlines()[0]
            used, total = map(float, line.split(","))
            self._gpu_used_mb = used
            self._gpu_total_mb = total
            pct = (used / total) * 100
            
            # Color code based on usage
            color = "#f9e2af" # Yellow (Warning-ish)
            if pct < 50: color = "#a6e3a1" # Green
            elif pct > 90: color = "#f38ba8" # Red
            
            self._gpu_lbl.setStyleSheet(f"color: {color}; font-family: 'JetBrains Mono', monospace; font-weight: bold; padding-right: 10px;")
            self._gpu_lbl.setText(f"GPU VRAM: {used/1024:.1f} / {total/1024:.1f} GB ({pct:.1f}%)")
            
            # Also trigger resource health update for active tab
            self._update_all_resource_health()
        except Exception:
            self._gpu_lbl.setText("GPU VRAM: N/A")
            self._gpu_lbl.setStyleSheet("color: #6c7086; font-family: 'JetBrains Mono', monospace; font-weight: bold; padding-right: 10px;")
            self._gpu_used_mb = 0
            self._gpu_total_mb = 0
            self._update_all_resource_health()

    @Slot()
    def _toggle_pulse(self):
        """Toggle state for pulsing critical warnings."""
        self._pulse_state = not self._pulse_state
        self._update_all_resource_health()

    # ===================================================================
    #  Config Presets (save / load / delete)
    # ===================================================================
    def _preset_path(self, name: str) -> Path:
        return PRESETS_DIR / f"{name}.json"

    def _preset_to_dict(self, tw: _TrainingWidgets) -> dict:
        return {
            "envs":           tw.envs.value(),
            "iters":          tw.iters.value(),
            "ep_len":         tw.ep_len.value(),
            "history_length": tw.history_length.value(),
            "alpha_init":     tw.alpha_init.value(),
            "foot_tolerance": tw.foot_tolerance.value(),
            "headless":       tw.headless.isChecked(),
            "video_enabled":  tw.video_enabled.isChecked(),
            "video_interval": tw.video_interval.value(),
            "logger":         tw.logger.currentData(),
        }

    def _dict_to_tw(self, d: dict, tw: _TrainingWidgets) -> None:
        if "envs" in d:
            tw.envs.setValue(d["envs"])
        if "iters" in d:
            tw.iters.setValue(d["iters"])
        if "ep_len" in d:
            tw.ep_len.setValue(d["ep_len"])
        if "history_length" in d:
            tw.history_length.setValue(d["history_length"])
        if "alpha_init" in d:
            tw.alpha_init.setValue(d["alpha_init"])
        if "foot_tolerance" in d:
            tw.foot_tolerance.setValue(d["foot_tolerance"])
        if "headless" in d:
            tw.headless.setChecked(d["headless"])
        if "video_enabled" in d:
            tw.video_enabled.setChecked(d["video_enabled"])
        if "video_interval" in d:
            tw.video_interval.setValue(d["video_interval"])
        if "logger" in d:
            idx = tw.logger.findData(d["logger"])
            if idx >= 0:
                tw.logger.setCurrentIndex(idx)

    def _refresh_preset_combo(self, tw: _TrainingWidgets) -> None:
        tw.preset_combo.blockSignals(True)
        current = tw.preset_combo.currentText()
        tw.preset_combo.clear()
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(PRESETS_DIR.glob("*.json")):
            tw.preset_combo.addItem(f.stem)
        idx = tw.preset_combo.findText(current)
        if idx >= 0:
            tw.preset_combo.setCurrentIndex(idx)
        tw.preset_combo.blockSignals(False)

    def _refresh_all_preset_combos(self) -> None:
        for tw in [
            getattr(self, "_lf_training_widgets", None),
            getattr(self, "_c3d_training_widgets", None),
            getattr(self, "_omomo_training_widgets", None),
        ]:
            if tw:
                self._refresh_preset_combo(tw)

    def _save_preset(self, tw: _TrainingWidgets) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            text=tw.preset_combo.currentText() or "my_preset",
        )
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_")
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        self._preset_path(name).write_text(
            json.dumps(self._preset_to_dict(tw), indent=2)
        )
        self._refresh_all_preset_combos()
        # Select the just-saved preset
        for tw2 in [
            getattr(self, "_lf_training_widgets", None),
            getattr(self, "_c3d_training_widgets", None),
            getattr(self, "_omomo_training_widgets", None),
        ]:
            if tw2:
                idx = tw2.preset_combo.findText(name)
                if idx >= 0:
                    tw2.preset_combo.setCurrentIndex(idx)
        self._log_info(f"Preset saved: {name}")

    def _load_preset(self, tw: _TrainingWidgets) -> None:
        name = tw.preset_combo.currentText()
        if not name:
            QMessageBox.warning(self, "No preset", "No preset selected.")
            return
        path = self._preset_path(name)
        if not path.exists():
            QMessageBox.warning(self, "Not found", f"Preset file not found:\n{path}")
            return
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Could not parse preset:\n{e}")
            return
        self._dict_to_tw(d, tw)
        self._log_info(f"Preset loaded: {name}")

    def _delete_preset(self, tw: _TrainingWidgets) -> None:
        name = tw.preset_combo.currentText()
        if not name:
            return
        path = self._preset_path(name)
        if not path.exists():
            return
        reply = QMessageBox.question(
            self, "Delete preset",
            f"Delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            path.unlink()
            self._refresh_all_preset_combos()
            self._log_info(f"Preset deleted: {name}")

    # ===================================================================
    #  Resource Health
    # ===================================================================
    def _update_all_resource_health(self):
        """Update health for LAFAN, C3D, and OMOMO tabs."""
        if hasattr(self, "_lf_training_widgets") and self._lf_training_widgets:
            self._update_resource_health(self._lf_training_widgets)
        if hasattr(self, "_c3d_training_widgets") and self._c3d_training_widgets:
            self._update_resource_health(self._c3d_training_widgets)
        if hasattr(self, "_omomo_training_widgets") and self._omomo_training_widgets:
            self._update_resource_health(self._omomo_training_widgets)

    def _update_resource_health(self, tw: _TrainingWidgets):
        """Estimate resource usage and update UI/Run button state."""
        # Check if slots are initialized
        if not all(hasattr(tw, s) and getattr(tw, s) for s in ["envs", "headless", "algo_cb", "run_btn"]):
            return

        envs = tw.envs.value()
        is_headless = tw.headless.isChecked()
        algo_suffix = tw.algo_cb.currentData()
        is_sac = "sac" in str(algo_suffix).lower()
        history = tw.history_length.value()

        # 1. Estimate VRAM
        base_vram = VRAM_BASE_HEADLESS if is_headless else VRAM_BASE_GUI
        sim_vram = envs * VRAM_PER_ENV_SIM
        if is_sac:
            # SAC replay buffer: scales with envs and history
            algo_vram = envs * VRAM_SAC_BUF_PER_ENV_PER_HIST * history
        else:
            algo_vram = VRAM_ALGO_PPO
        est_vram_gb = base_vram + sim_vram + algo_vram

        # 2. Estimate RAM
        est_ram_gb = RAM_BASE + (envs * RAM_PER_ENV)

        # 3. Get Actual System Info
        total_vram_gb = self._gpu_total_mb / 1024.0
        
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)

        # 4. Compute Ratios against TOTAL system capacity
        vram_ratio = 0
        if total_vram_gb > 0:
            vram_ratio = (est_vram_gb / total_vram_gb) * 100
        
        ram_ratio = (est_ram_gb / total_ram_gb) * 100

        # 5. Update UI Bars
        tw.vram_bar.setValue(int(min(100, vram_ratio)))
        tw.ram_bar.setValue(int(min(100, ram_ratio)))
        tw.vram_lbl.setText(f"Est. {est_vram_gb:.1f} GB")
        tw.ram_lbl.setText(f"Est. {est_ram_gb:.1f} GB")

        # 6. Status Logic & Pulsing
        status = "Optimal configuration."
        color = "#a6e3a1" # Green
        can_run = True
        
        # Check VRAM primary
        if vram_ratio > 90 or (est_vram_gb > (total_vram_gb * 0.95) and total_vram_gb > 0):
            status = "⚠ CRITICAL: Likely to OOM. Reduce Env Count!"
            color = "#f38ba8" if not self._pulse_state else "#1e1e2e" # Pulsing Red
            can_run = False
        elif vram_ratio > 75:
            status = "⚠ Warning: High VRAM usage. Close other apps."
            color = "#f9e2af" # Yellow
        elif ram_ratio > 90 or est_ram_gb > (total_ram_gb * 0.95):
            status = "⚠ CRITICAL: Insufficient System RAM!"
            color = "#f38ba8" if not self._pulse_state else "#1e1e2e" # Pulsing Red
            can_run = False
            
        tw.health_msg.setText(status)
        tw.health_msg.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
        
        # Style the progress bars based on ratio
        bar_color = "#a6e3a1"
        if vram_ratio > 90: bar_color = "#f38ba8"
        elif vram_ratio > 75: bar_color = "#f9e2af"
        tw.vram_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {bar_color}; }}")
        
        ram_bar_color = "#a6e3a1"
        if ram_ratio > 90: ram_bar_color = "#f38ba8"
        elif ram_ratio > 75: ram_bar_color = "#f9e2af"
        tw.ram_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ram_bar_color}; }}")

        # 7. Disable Run Button if critical
        tw.run_btn.setEnabled(can_run)
        if not can_run:
            tw.run_btn.setToolTip("Resource check failed: Reduce Env Count or use Headless mode.")
        else:
            tw.run_btn.setToolTip("")

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
        row += 1

        # ── Row 3: Resource Health (Capacity Bars) ──
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background: #45475a;")
        grid.addWidget(line, row, 0, 1, 6)
        row += 1

        grid.addWidget(QLabel("VRAM Capacity:"), row, 0)
        tw.vram_bar = QProgressBar()
        tw.vram_bar.setRange(0, 100)
        grid.addWidget(tw.vram_bar, row, 1, 1, 2)
        tw.vram_lbl = QLabel("Est. -- GB")
        grid.addWidget(tw.vram_lbl, row, 3)

        grid.addWidget(QLabel("RAM Capacity:"), row, 4)
        tw.ram_bar = QProgressBar()
        tw.ram_bar.setRange(0, 100)
        grid.addWidget(tw.ram_bar, row, 5)
        tw.ram_lbl = QLabel("Est. -- GB")
        grid.addWidget(tw.ram_lbl, row, 6)
        row += 1

        tw.health_msg = QLabel("Calculating resource feasibility...")
        tw.health_msg.setStyleSheet("font-style: italic; color: #9399b2; font-size: 11px;")
        grid.addWidget(tw.health_msg, row, 0, 1, 6)

        # Connect signals for auto-update
        tw.envs.valueChanged.connect(self._update_all_resource_health)
        tw.headless.stateChanged.connect(self._update_all_resource_health)

        # ── Preset save/load row ──
        row += 1
        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Preset:")
        preset_lbl.setStyleSheet("color: #9399b2; font-size: 11px;")
        preset_row.addWidget(preset_lbl)
        tw.preset_combo = QComboBox()
        tw.preset_combo.setMinimumWidth(160)
        tw.preset_combo.setToolTip("Saved training configuration presets")
        preset_row.addWidget(tw.preset_combo)
        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(60)
        load_btn.setToolTip("Load selected preset into the fields above")
        load_btn.clicked.connect(lambda: self._load_preset(tw))
        preset_row.addWidget(load_btn)
        save_btn = QPushButton("Save…")
        save_btn.setFixedWidth(60)
        save_btn.setToolTip("Save current settings as a named preset")
        save_btn.clicked.connect(lambda: self._save_preset(tw))
        preset_row.addWidget(save_btn)
        del_btn = QPushButton("Delete")
        del_btn.setFixedWidth(60)
        del_btn.setToolTip("Delete the selected preset")
        del_btn.clicked.connect(lambda: self._delete_preset(tw))
        preset_row.addWidget(del_btn)
        preset_row.addStretch()
        grid.addLayout(preset_row, row, 0, 1, 7)
        self._refresh_preset_combo(tw)

        tw.group = grp
        return grp, tw

    # ===================================================================
    #  TAB 1 - LAFAN Retargeting + Tracking
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
        self._lf_list.setMaximumHeight(180)
        self._lf_list.currentItemChanged.connect(self._on_lafan_selection_changed)
        sel_lay.addWidget(self._lf_list)
        lf_btn_row = QHBoxLayout()
        ref_btn = QPushButton("Refresh List")
        ref_btn.clicked.connect(self._refresh_lafan)
        lf_btn_row.addWidget(ref_btn)
        self._lf_preview_btn = QPushButton("Preview Motion")
        self._lf_preview_btn.setEnabled(False)
        self._lf_preview_btn.clicked.connect(self._preview_lafan)
        lf_btn_row.addWidget(self._lf_preview_btn)
        lf_btn_row.addStretch()
        sel_lay.addLayout(lf_btn_row)
        # Sequence info label
        self._lf_seq_info = QLabel("Select a sequence to see frame info")
        self._lf_seq_info.setStyleSheet("color: #9399b2; font-size: 11px; padding: 2px 4px;")
        sel_lay.addWidget(self._lf_seq_info)
        lay.addWidget(sel_grp)

        # Frame range
        fr_grp = QGroupBox("Frame Range (applied at convert step)")
        fr_lay = QHBoxLayout(fr_grp)
        self._lf_use_range = QCheckBox("Enable frame range")
        self._lf_use_range.setChecked(False)
        self._lf_use_range.toggled.connect(self._on_lf_range_toggled)
        fr_lay.addWidget(self._lf_use_range)
        fr_lay.addSpacing(12)
        fr_lay.addWidget(QLabel("Start:"))
        self._lf_fr_start = QSpinBox()
        self._lf_fr_start.setRange(0, 999999)
        self._lf_fr_start.setValue(0)
        self._lf_fr_start.setEnabled(False)
        fr_lay.addWidget(self._lf_fr_start)
        fr_lay.addWidget(QLabel("End:"))
        self._lf_fr_end = QSpinBox()
        self._lf_fr_end.setRange(1, 999999)
        self._lf_fr_end.setValue(500)
        self._lf_fr_end.setEnabled(False)
        fr_lay.addWidget(self._lf_fr_end)
        fr_lay.addStretch()
        lay.addWidget(fr_grp)

        # Training config (expanded)
        grp_lf, self._lf_training_widgets = self._make_training_group()
        self._lf_training_widgets.robot_cb = self._lf_robot
        self._lf_training_widgets.algo_cb = self._lf_algo
        # Note: self._lf_algo is also connected in _build_lafan_tab's robot/algo section
        lay.addWidget(grp_lf)

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
        self._lf_run_btn = QPushButton("🚀 START LAFAN WORKFLOW")
        self._lf_run_btn.setMinimumHeight(48)
        self._lf_run_btn.setStyleSheet(BIG_BTN.format(bg="#a6e3a1", hover="#94e2d5", press="#89dceb"))
        self._lf_run_btn.clicked.connect(self._run_lafan)
        self._lf_training_widgets.run_btn = self._lf_run_btn
        lay.addWidget(self._lf_run_btn)

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
            npy = DEFAULT_LAFAN_DIR / f"{t}.npy"
            try:
                arr = np.load(str(npy), mmap_mode="r")
                T, J, _ = arr.shape
                label = f"{t}  [{T} frames, {J} joints, {human_size(npy)}]"
            except Exception:
                label = t
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, t)
            self._lf_list.addItem(item)

    def _on_lafan_selection_changed(self, current: QListWidgetItem | None, _prev):
        if current is None or current.data(Qt.ItemDataRole.UserRole) is None:
            self._lf_preview_btn.setEnabled(False)
            self._lf_seq_info.setText("Select a sequence to see frame info")
            return
        task = current.data(Qt.ItemDataRole.UserRole)
        npy = DEFAULT_LAFAN_DIR / f"{task}.npy"
        self._lf_preview_btn.setEnabled(npy.exists())
        try:
            arr = np.load(str(npy), mmap_mode="r")
            T, J, _ = arr.shape
            self._lf_seq_info.setText(
                f"{task}  |  {T} frames  |  {J} joints  |  {human_size(npy)}"
            )
            # Auto-populate frame range end to the actual max frames by default
            self._lf_fr_end.setMaximum(T - 1)
            self._lf_fr_end.setValue(T - 1)
        except Exception as e:
            self._lf_seq_info.setText(f"Could not read file: {e}")

    def _preview_lafan(self):
        item = self._lf_list.currentItem()
        if item is None:
            return
        task = item.data(Qt.ItemDataRole.UserRole)
        npy = DEFAULT_LAFAN_DIR / f"{task}.npy"
        if not npy.exists():
            QMessageBox.warning(self, "Not Found", f"File not found:\n{npy}")
            return
        dlg = MotionPreviewDialog(npy, parent=self)
        dlg.show()

    def _on_lf_range_toggled(self, checked: bool):
        self._lf_fr_start.setEnabled(checked)
        self._lf_fr_end.setEnabled(checked)

    def _on_c3d_range_toggled(self, checked: bool):
        self._c3d_fr_start.setEnabled(checked)
        self._c3d_fr_end.setEnabled(checked)

    def _run_lafan(self):
        item = self._lf_list.currentItem()
        if item is None or item.data(Qt.ItemDataRole.UserRole) is None:
            QMessageBox.warning(self, "No selection", "Select a LAFAN sequence first.")
            return

        task = item.data(Qt.ItemDataRole.UserRole)
        dof = self._lf_robot.currentData()
        algo = self._lf_algo.currentData()
        tw = self._lf_training_widgets
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
        exp = f"g1-{stem}-wbt{algo}"
        retarget_dir = str(PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting")
        lafan_dir = str(DEFAULT_LAFAN_DIR)
        retarget_out = PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting" / DEFAULT_RETARGET_DIR / "lafan" / f"{task}.npz"
        convert_out = f"{DEFAULT_CONVERT_DIR}/lafan/{task}_mj_fps50.npz"
        converted_file = f"{retarget_dir}/{DEFAULT_CONVERT_DIR}/lafan/{task}_mj_fps50.npz"

        # ── Overwrite check ───────────────────────────────────────────
        skip_retarget = False
        if retarget_out.exists():
            reply = QMessageBox.question(
                self, "Retargeted file exists",
                f"Retargeted output already exists:\n{retarget_out}\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            skip_retarget = (reply == QMessageBox.StandardButton.No)

        # ── Frame range ───────────────────────────────────────────────
        use_range = self._lf_use_range.isChecked()
        fr_start = self._lf_fr_start.value()
        fr_end = self._lf_fr_end.value()

        # Build video lines
        if video_on:
            video_lines = [
                "    --logger.video.enabled True \\",
                f"    --logger.video.interval {vid_int} \\",
                f'    --logger.video.save-dir "{DEFAULT_VIDEO_DIR}/g1_{stem}_lafan_wbt" \\',
            ]
        else:
            video_lines = [
                "    --logger.video.enabled False \\",
            ]

        history_line = ""
        if history > 1:
            history_line = f"    --observation.groups.actor_obs.history-length {history} \\"

        # ── Retarget step (conditionally skipped) ────────────────────
        if skip_retarget:
            retarget_lines = [
                f'echo "Skipping retargeting — using existing: {retarget_out}"',
            ]
        else:
            retarget_lines = [
                f'echo "Running retargeting for {task}..."',
                "python examples/robot_retarget.py \\",
                f"    --robot-config.robot-dof {dof} \\",
                f'    --data_path "{lafan_dir}" \\',
                "    --task-type robot_only \\",
                f'    --task-name "{task}" \\',
                "    --data_format lafan \\",
                f"    --task-config.ground-range {gr_x} {gr_y} \\",
                f'    --save_dir "{DEFAULT_RETARGET_DIR}/lafan" \\',
                f"    --retargeter.foot-sticking-tolerance {foot_tol}",
            ]
            if use_range:
                retarget_lines.insert(-1, f"    --line-range {fr_start} {fr_end} \\")

        # ── Convert step (with optional line-range) ───────────────────
        convert_range_arg = f"    --line-range {fr_start} {fr_end} \\" if use_range else ""
        range_suffix = f" (frames {fr_start}-{fr_end})" if use_range else ""
        convert_lines = [
            f'echo "Converting to MuJoCo format{range_suffix}..."',
            "python data_conversion/convert_data_format_mj.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --input_file "./{DEFAULT_RETARGET_DIR}/lafan/{task}.npz" \\',
            "    --output_fps 50 \\",
            f'    --output_name "{convert_out}" \\',
            "    --data_format lafan \\",
            '    --object_name "ground" \\',
        ]
        if convert_range_arg:
            convert_lines.append(convert_range_arg)
        convert_lines.append("    --once \\")
        if headless == "True":
            convert_lines.append("    --headless")
        else:
            convert_lines.append("    --no-headless")

        lines = [
            f'echo "=== LAFAN Workflow: {task} (G1-{stem}) ==="',
            "",
            "# ── Step 1: Retargeting env ──",
            f'source "{PROJECT_ROOT}/scripts/source_retargeting_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_retargeting" --quiet',
            f'cd "{retarget_dir}"',
            "",
            "# ── Step 2: Retarget ──",
            *retarget_lines,
            "",
            "# ── Step 3: Convert ──",
            *convert_lines,
            "",
            "# ── Step 4: IsaacSim env ──",
            f'cd "{PROJECT_ROOT}"',
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'HOLOSOMA_DEPS_DIR="${{HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}}"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma" --quiet',
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
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \\",
            f"    exp:{exp} \\",
            f"    logger:{logger} \\",
            f'    --logger.base-dir "{self._logs_dir_arg()}" \\',
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
        self._plot_widget.set_target_iters(iters)
        self._run_script(lines, start_plot=True)

    # ===================================================================
    #  TAB 2 - C3D Retargeting + Tracking
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

        # Frame range
        fr_grp = QGroupBox("Frame Range (applied at convert step)")
        fr_lay = QHBoxLayout(fr_grp)
        self._c3d_use_range = QCheckBox("Enable frame range")
        self._c3d_use_range.setChecked(True)  # C3D default on (script used 100-500)
        self._c3d_use_range.toggled.connect(self._on_c3d_range_toggled)
        fr_lay.addWidget(self._c3d_use_range)
        fr_lay.addSpacing(12)
        fr_lay.addWidget(QLabel("Start:"))
        self._c3d_fr_start = QSpinBox()
        self._c3d_fr_start.setRange(0, 999999)
        self._c3d_fr_start.setValue(100)
        fr_lay.addWidget(self._c3d_fr_start)
        fr_lay.addWidget(QLabel("End:"))
        self._c3d_fr_end = QSpinBox()
        self._c3d_fr_end.setRange(1, 999999)
        self._c3d_fr_end.setValue(500)
        fr_lay.addWidget(self._c3d_fr_end)
        fr_lay.addStretch()
        lay.addWidget(fr_grp)

        # Training config (expanded) - C3D defaults
        grp_c3d, self._c3d_training_widgets = self._make_training_group()
        self._c3d_training_widgets.robot_cb = self._c3d_robot
        self._c3d_training_widgets.algo_cb = self._c3d_algo
        self._c3d_training_widgets.envs.setValue(2048)
        self._c3d_training_widgets.ep_len.setValue(10.0)
        self._c3d_training_widgets.video_interval.setValue(10)
        self._c3d_training_widgets.history_length.setValue(4)
        lay.addWidget(grp_c3d)

        # GO
        self._c3d_run_btn = QPushButton("🚀 START C3D WORKFLOW")
        self._c3d_run_btn.setMinimumHeight(48)
        self._c3d_run_btn.setStyleSheet(BIG_BTN.format(bg="#a6e3a1", hover="#94e2d5", press="#89dceb"))
        self._c3d_run_btn.clicked.connect(self._run_c3d)
        self._c3d_training_widgets.run_btn = self._c3d_run_btn
        lay.addWidget(self._c3d_run_btn)

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
        tw = self._c3d_training_widgets
        envs = tw.envs.value()
        iters = tw.iters.value()
        ep_len = tw.ep_len.value()
        headless = "True" if tw.headless.isChecked() else "False"
        video_on = tw.video_enabled.isChecked()
        vid_int = tw.video_interval.value()
        history = tw.history_length.value()
        logger = tw.logger.currentData()
        stem = f"{dof}dof"
        # algo is the exp suffix (e.g. "-fast-sac" or "" for PPO base)
        exp = f"g1-{stem}-wbt{algo}"
        retarget_dir = str(PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting")
        c3d_data_dir = str(DEFAULT_C3D_DIR)
        marker_map = self._c3d_marker_map.text().strip()
        convert_out = f"{DEFAULT_CONVERT_DIR}/c3d/{task}_mj_fps50.npz"
        converted_file = f"{retarget_dir}/{DEFAULT_CONVERT_DIR}/c3d/{task}_mj_fps50.npz"

        # Overwrite check for retarget output
        retarget_out = (
            PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting"
            / DEFAULT_RETARGET_DIR / "c3d" / f"{task}.npz"
        )
        skip_retarget = False
        if retarget_out.exists():
            reply = QMessageBox.question(
                self, "Retargeted file exists",
                f"Retargeted output already exists:\n{retarget_out}\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            skip_retarget = (reply == QMessageBox.StandardButton.No)

        # Frame range
        use_range = self._c3d_use_range.isChecked()
        fr_start = self._c3d_fr_start.value()
        fr_end = self._c3d_fr_end.value()

        video_lines = []
        if video_on:
            video_lines = [
                "    --logger.video.enabled True \\",
                f"    --logger.video.interval {vid_int} \\",
                f'    --logger.video.save-dir "{DEFAULT_VIDEO_DIR}/g1_{stem}_c3d_wbt" \\',
            ]
        else:
            video_lines = [
                "    --logger.video.enabled False \\",
            ]

        history_line = ""
        if history > 1:
            history_line = f"    --observation.groups.actor_obs.history-length {history} \\"

        range_suffix = f" (frames {fr_start}-{fr_end})" if use_range else ""
        convert_range_arg = f"    --line-range {fr_start} {fr_end} \\" if use_range else ""

        retarget_lines: list[str] = []
        if skip_retarget:
            retarget_lines = [
                f'echo "Skipping retargeting (using existing: {retarget_out})"',
            ]
        else:
            retarget_lines = [
                "# ── Step 2: Convert C3D -> NPZ ──",
                'echo "Converting C3D to NPZ..."',
                "python3 data_utils/prep_c3d_for_rt.py \\",
                f'    --input "{c3d_path}" \\',
                f'    --output "{c3d_data_dir}/{task}.npz" \\',
                "    --marker-set custom \\",
                f'    --marker-map "{marker_map}" \\',
                "    --downsample-to 100 \\",
                "    --lowpass-hz 4.0",
                "",
                "# ── Step 3: Retarget ──",
                'echo "Retargeting..."',
                "python examples/robot_retarget.py \\",
                f"    --robot-config.robot-dof {dof} \\",
                f'    --data_path "{c3d_data_dir}" \\',
                "    --task-type robot_only \\",
                f'    --task-name "{task}" \\',
                "    --data_format c3d \\",
                f'    --save_dir "{DEFAULT_RETARGET_DIR}/c3d"',
            ]
            if use_range:
                retarget_lines.insert(-1, f"    --line-range {fr_start} {fr_end} \\")

        convert_lines = [
            "# ── Step 4: Convert ──",
            f'echo "Converting to MuJoCo format{range_suffix}..."',
            "python data_conversion/convert_data_format_mj.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --input_file "{DEFAULT_RETARGET_DIR}/c3d/{task}.npz" \\',
            "    --output_fps 50 \\",
            f'    --output_name "{convert_out}" \\',
            "    --data_format c3d \\",
            '    --object_name "ground" \\',
        ]
        if convert_range_arg:
            convert_lines.append(convert_range_arg)
        convert_lines.append("    --once \\")
        if headless == "True":
            convert_lines.append("    --headless")
        else:
            convert_lines.append("    --no-headless")

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
        ]
        lines.extend(retarget_lines)
        lines.extend(convert_lines)
        lines += [
            "",
            "# ── Step 5: IsaacSim env ──",
            f'cd "{PROJECT_ROOT}"',
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'HOLOSOMA_DEPS_DIR="${{HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}}"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma" --quiet',
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
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \\",
            f"    exp:{exp} \\",
            f"    logger:{logger} \\",
            f'    --logger.base-dir "{self._logs_dir_arg()}" \\',
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
        self._plot_widget.set_target_iters(iters)
        self._run_script(lines, start_plot=True)

    # ===================================================================
    #  TAB 3 - OMOMO (SMPLH) Retargeting + Tracking
    # ===================================================================
    def _build_omomo_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        grp1, self._omomo_robot, self._omomo_algo = self._make_robot_algo_group()
        lay.addWidget(grp1)

        # Data directory
        dir_grp = QGroupBox("OMOMO Data Directory")
        dir_lay = QHBoxLayout(dir_grp)
        self._omomo_data_dir = QLineEdit(str(DEFAULT_OMOMO_DIR))
        self._omomo_data_dir.setPlaceholderText("Path to OMOMO_new folder")
        dir_lay.addWidget(self._omomo_data_dir)
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._browse_omomo_dir)
        dir_lay.addWidget(browse_dir_btn)
        lay.addWidget(dir_grp)

        # Task/sequence selection
        seq_grp = QGroupBox("Motion Sequence")
        seq_lay = QVBoxLayout(seq_grp)
        self._omomo_list = QListWidget()
        self._omomo_list.setMaximumHeight(150)
        seq_lay.addWidget(self._omomo_list)
        btn_row = QHBoxLayout()
        ref_btn = QPushButton("Refresh List")
        ref_btn.clicked.connect(self._refresh_omomo)
        btn_row.addWidget(ref_btn)
        btn_row.addStretch()
        seq_lay.addLayout(btn_row)
        self._omomo_task_edit = QLineEdit()
        self._omomo_task_edit.setPlaceholderText("Or enter task name manually (e.g. sub3_largebox_003)")
        seq_lay.addWidget(self._omomo_task_edit)
        self._omomo_list.currentItemChanged.connect(self._on_omomo_selection_changed)
        lay.addWidget(seq_grp)

        # Frame range
        fr_grp = QGroupBox("Frame Range (applied at convert step)")
        fr_lay = QHBoxLayout(fr_grp)
        self._omomo_use_range = QCheckBox("Enable frame range")
        self._omomo_use_range.setChecked(False)
        self._omomo_use_range.toggled.connect(self._on_omomo_range_toggled)
        fr_lay.addWidget(self._omomo_use_range)
        fr_lay.addSpacing(12)
        fr_lay.addWidget(QLabel("Start:"))
        self._omomo_fr_start = QSpinBox()
        self._omomo_fr_start.setRange(0, 999999)
        self._omomo_fr_start.setValue(0)
        self._omomo_fr_start.setEnabled(False)
        fr_lay.addWidget(self._omomo_fr_start)
        fr_lay.addWidget(QLabel("End:"))
        self._omomo_fr_end = QSpinBox()
        self._omomo_fr_end.setRange(1, 999999)
        self._omomo_fr_end.setValue(500)
        self._omomo_fr_end.setEnabled(False)
        fr_lay.addWidget(self._omomo_fr_end)
        fr_lay.addStretch()
        lay.addWidget(fr_grp)

        # Training config
        grp_omomo, self._omomo_training_widgets = self._make_training_group()
        self._omomo_training_widgets.robot_cb = self._omomo_robot
        self._omomo_training_widgets.algo_cb = self._omomo_algo
        lay.addWidget(grp_omomo)

        # GO
        self._omomo_run_btn = QPushButton("🚀 START OMOMO WORKFLOW")
        self._omomo_run_btn.setMinimumHeight(48)
        self._omomo_run_btn.setStyleSheet(BIG_BTN.format(bg="#a6e3a1", hover="#94e2d5", press="#89dceb"))
        self._omomo_run_btn.clicked.connect(self._run_omomo)
        self._omomo_training_widgets.run_btn = self._omomo_run_btn
        lay.addWidget(self._omomo_run_btn)

        lay.addStretch()
        scroll.setWidget(w)

        self._refresh_omomo()
        return scroll

    def _browse_omomo_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select OMOMO Data Directory", str(PROJECT_ROOT))
        if path:
            self._omomo_data_dir.setText(path)
            self._refresh_omomo()

    def _refresh_omomo(self):
        self._omomo_list.clear()
        d = Path(self._omomo_data_dir.text().strip())
        tasks = scan_omomo(d)
        if not tasks:
            self._omomo_list.addItem("(no .npz files found)")
        for t in tasks:
            item = QListWidgetItem(t)
            item.setData(Qt.ItemDataRole.UserRole, t)
            self._omomo_list.addItem(item)

    def _on_omomo_selection_changed(self, current: QListWidgetItem | None, _prev):
        if current is None or current.data(Qt.ItemDataRole.UserRole) is None:
            return
        self._omomo_task_edit.setText(current.data(Qt.ItemDataRole.UserRole))

    def _on_omomo_range_toggled(self, checked: bool):
        self._omomo_fr_start.setEnabled(checked)
        self._omomo_fr_end.setEnabled(checked)

    def _run_omomo(self):
        # Resolve task name
        task = self._omomo_task_edit.text().strip()
        if not task:
            item = self._omomo_list.currentItem()
            if item and item.data(Qt.ItemDataRole.UserRole):
                task = item.data(Qt.ItemDataRole.UserRole)
        if not task:
            QMessageBox.warning(self, "No sequence", "Select or enter an OMOMO task name.")
            return

        dof = self._omomo_robot.currentData()
        algo = self._omomo_algo.currentData()
        tw = self._omomo_training_widgets
        envs = tw.envs.value()
        iters = tw.iters.value()
        ep_len = tw.ep_len.value()
        headless = "True" if tw.headless.isChecked() else "False"
        video_on = tw.video_enabled.isChecked()
        vid_int = tw.video_interval.value()
        history = tw.history_length.value()
        logger = tw.logger.currentData()
        stem = f"{dof}dof"
        exp = f"g1-{stem}-wbt{algo}"
        retarget_dir = str(PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting")
        omomo_data_dir = self._omomo_data_dir.text().strip()
        retarget_out = (
            PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting"
            / DEFAULT_RETARGET_DIR / "omomo" / f"{task}.npz"
        )
        convert_out = f"{DEFAULT_CONVERT_DIR}/omomo/{task}_mj_fps50.npz"
        converted_file = f"{retarget_dir}/{DEFAULT_CONVERT_DIR}/omomo/{task}_mj_fps50.npz"

        # Overwrite check
        skip_retarget = False
        if retarget_out.exists():
            reply = QMessageBox.question(
                self, "Retargeted file exists",
                f"Retargeted output already exists:\n{retarget_out}\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            skip_retarget = (reply == QMessageBox.StandardButton.No)

        # Frame range
        use_range = self._omomo_use_range.isChecked()
        fr_start = self._omomo_fr_start.value()
        fr_end = self._omomo_fr_end.value()

        video_lines: list[str] = []
        if video_on:
            video_lines = [
                "    --logger.video.enabled True \\",
                f"    --logger.video.interval {vid_int} \\",
                f'    --logger.video.save-dir "{DEFAULT_VIDEO_DIR}/g1_{stem}_omomo_wbt" \\',
            ]
        else:
            video_lines = [
                "    --logger.video.enabled False \\",
            ]

        history_line = ""
        if history > 1:
            history_line = f"    --observation.groups.actor_obs.history-length {history} \\"

        range_suffix = f" (frames {fr_start}-{fr_end})" if use_range else ""
        convert_range_arg = f"    --line-range {fr_start} {fr_end} \\" if use_range else ""

        if skip_retarget:
            retarget_lines: list[str] = [
                f'echo "Skipping retargeting — using existing: {retarget_out}"',
            ]
        else:
            retarget_lines = [
                f'echo "Retargeting {task} (SMPLH/OMOMO)..."',
                "python examples/robot_retarget.py \\",
                f"    --robot-config.robot-dof {dof} \\",
                f'    --data_path "{omomo_data_dir}" \\',
                "    --task-type robot_only \\",
                f'    --task-name "{task}" \\',
                "    --data_format smplh \\",
                f'    --save_dir "{DEFAULT_RETARGET_DIR}/omomo"',
            ]
            if use_range:
                retarget_lines.insert(-1, f"    --line-range {fr_start} {fr_end} \\")

        convert_lines = [
            f'echo "Converting to MuJoCo format{range_suffix}..."',
            "python data_conversion/convert_data_format_mj.py \\",
            f"    --robot-config.robot-dof {dof} \\",
            f'    --input_file "{DEFAULT_RETARGET_DIR}/omomo/{task}.npz" \\',
            "    --output_fps 50 \\",
            f'    --output_name "{convert_out}" \\',
            "    --data_format smplh \\",
            '    --object_name "ground" \\',
        ]
        if convert_range_arg:
            convert_lines.append(convert_range_arg)
        convert_lines.append("    --once \\")
        if headless == "True":
            convert_lines.append("    --headless")
        else:
            convert_lines.append("    --no-headless")

        lines = [
            f'echo "=== OMOMO Workflow: {task} (G1-{stem}) ==="',
            "",
            "# ── Step 1: Retargeting env ──",
            f'source "{PROJECT_ROOT}/scripts/source_retargeting_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_retargeting" --quiet',
            f'cd "{retarget_dir}"',
            "",
            "# ── Step 2: Retarget ──",
            *retarget_lines,
            "",
            "# ── Step 3: Convert ──",
            *convert_lines,
            "",
            "# ── Step 4: IsaacSim env ──",
            f'cd "{PROJECT_ROOT}"',
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'HOLOSOMA_DEPS_DIR="${{HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}}"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma" --quiet',
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
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \\",
            f"    exp:{exp} \\",
            f"    logger:{logger} \\",
            f'    --logger.base-dir "{self._logs_dir_arg()}" \\',
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
            'echo "=== OMOMO Workflow complete! ==="',
        ]

        self._log_info(f"Starting OMOMO workflow: {task} | {exp} | envs={envs} iters={iters}")
        self._plot_widget.set_target_iters(iters)
        self._run_script(lines, start_plot=True)

    # ===================================================================
    #  TAB 4 - Inference
    # ===================================================================
    def _build_inference_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        # ── Shared Inference Config ──
        cfg_grp = QGroupBox("Inference Configuration")
        cfg_grid = QGridLayout(cfg_grp)
        cfg_grid.setColumnStretch(1, 1)
        cfg_grid.setColumnStretch(3, 1)

        row = 0
        cfg_grid.addWidget(QLabel("Robot Config:"), row, 0)
        self._inf_robot_cb = QComboBox()
        self._inf_robot_cb.addItem("G1-23DOF WBT", "g1-23dof-wbt")
        self._inf_robot_cb.addItem("G1-29DOF WBT", "g1-29dof-wbt")
        self._inf_robot_cb.addItem("G1-29DOF Loco", "g1-29dof-loco")
        cfg_grid.addWidget(self._inf_robot_cb, row, 1)

        cfg_grid.addWidget(QLabel("Obs History:"), row, 2)
        self._inf_history = QSpinBox()
        self._inf_history.setRange(1, 16)
        self._inf_history.setValue(1)
        self._inf_history.setToolTip(
            "Must match training value (--observation.groups.actor_obs.history-length).\n"
            "Check your training config or run log."
        )
        cfg_grid.addWidget(self._inf_history, row, 3)

        row += 1
        cfg_grid.addWidget(QLabel("RL Rate (Hz):"), row, 0)
        self._inf_rl_rate = QSpinBox()
        self._inf_rl_rate.setRange(10, 200)
        self._inf_rl_rate.setValue(50)
        self._inf_rl_rate.setToolTip("Policy inference rate in Hz (task.rl-rate)")
        cfg_grid.addWidget(self._inf_rl_rate, row, 1)

        cfg_grid.addWidget(QLabel("Action Scale:"), row, 2)
        self._inf_action_scale = QDoubleSpinBox()
        self._inf_action_scale.setRange(0.01, 2.0)
        self._inf_action_scale.setSingleStep(0.05)
        self._inf_action_scale.setDecimals(3)
        self._inf_action_scale.setValue(0.25)
        self._inf_action_scale.setToolTip("Policy action scale (task.policy-action-scale)")
        cfg_grid.addWidget(self._inf_action_scale, row, 3)

        row += 1
        cfg_grid.addWidget(QLabel("Input:"), row, 0)
        self._inf_input_cb = QComboBox()
        self._inf_input_cb.addItem("Keyboard", "keyboard")
        self._inf_input_cb.addItem("Joystick / Gamepad", "joystick")
        self._inf_input_cb.setToolTip("Velocity + state input source (task.velocity-input / task.state-input)")
        cfg_grid.addWidget(self._inf_input_cb, row, 1)

        cfg_grid.addWidget(QLabel("Network Interface:"), row, 2)
        self._inf_interface = QLineEdit("auto")
        self._inf_interface.setToolTip(
            "DDS network interface (task.interface).\n"
            "Use 'auto' to detect, or set explicitly e.g. 'eth0'."
        )
        cfg_grid.addWidget(self._inf_interface, row, 3)

        row += 1
        self._inf_scale_by_effort = QCheckBox("Scale actions by effort/Kp")
        self._inf_scale_by_effort.setChecked(True)
        self._inf_scale_by_effort.setToolTip(
            "task.action-scales-by-effort-limit-over-p-gain\n"
            "Enable for WBT models trained with action_scales_by_effort_limit_over_p_gain=True."
        )
        cfg_grid.addWidget(self._inf_scale_by_effort, row, 0, 1, 2)

        self._inf_use_phase = QCheckBox("Use gait phase obs")
        self._inf_use_phase.setChecked(True)
        self._inf_use_phase.setToolTip("task.use-phase — enable for locomotion policies, disable for WBT-only.")
        cfg_grid.addWidget(self._inf_use_phase, row, 2, 1, 2)

        lay.addWidget(cfg_grp)

        # ── Simulation Inference ──
        sim_grp = QGroupBox("Simulation Inference (IsaacSim) — Export ONNX")
        sim_lay = QVBoxLayout(sim_grp)
        sim_lay.addWidget(QLabel("Select a trained checkpoint:"))
        self._sim_ckpt_list = QListWidget()
        self._sim_ckpt_list.setMaximumHeight(150)
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
        self._hw_onnx_list.setMaximumHeight(120)
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
        self._hw_onnx_path.setPlaceholderText("Or enter ONNX path manually")
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
        history = self._inf_history.value()

        lines = [
            'echo "=== Simulation Inference ==="',
            f'echo "Checkpoint: {ckpt}"',
            "",
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_isaacsim_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma" --quiet',
            "",
            f'cd "{PROJECT_ROOT}"',
            "python src/holosoma/holosoma/eval_agent.py \\",
            f'    --checkpoint "{ckpt}" \\',
            "    --training.export-onnx True \\",
            "    --training.num-envs 1 \\",
            f"    --training.headless {headless} \\",
            f"    --observation.groups.actor_obs.history-length {history}",
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

        robot_cfg = self._inf_robot_cb.currentData()
        history = self._inf_history.value()
        rl_rate = self._inf_rl_rate.value()
        action_scale = self._inf_action_scale.value()
        input_src = self._inf_input_cb.currentData()
        interface = self._inf_interface.text().strip() or "auto"
        scale_by_effort = "True" if self._inf_scale_by_effort.isChecked() else "False"
        use_phase = "True" if self._inf_use_phase.isChecked() else "False"

        lines = [
            'echo "=== Hardware Inference ==="',
            f'echo "ONNX: {onnx_path}"',
            "",
            "unset CONDA_ENV_NAME",
            f'source "{PROJECT_ROOT}/scripts/source_inference_setup.sh"',
            f'pip install -e "{PROJECT_ROOT}/src/holosoma_inference" --quiet',
            "",
            f'cd "{PROJECT_ROOT}"',
            "python src/holosoma_inference/holosoma_inference/run_policy.py \\",
            f"    inference:{robot_cfg} \\",
            f'    --task.model-path="{onnx_path}" \\',
            f"    --observation.groups.actor_obs.history-length={history} \\",
            f"    --task.rl-rate={rl_rate} \\",
            f"    --task.policy-action-scale={action_scale} \\",
            f"    --task.action-scales-by-effort-limit-over-p-gain={scale_by_effort} \\",
            f"    --task.use-phase={use_phase} \\",
            f"    --task.velocity-input={input_src} \\",
            f"    --task.state-input={input_src} \\",
            f"    --task.interface={interface}",
            "",
            'echo "=== Hardware inference complete! ==="',
        ]
        self._log_info(f"Deploying to hardware: {onnx_path}")
        self._run_script(lines)

    # ===================================================================
    #  TAB 4 - Browse Files
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
    #  TAB 5 - Settings / Configuration
    # ===================================================================
    def _build_settings_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        lay = QVBoxLayout(w)

        # ── Editable Directory Configuration ────────────────────────────
        dir_grp = QGroupBox("Directory Configuration")
        dir_lay = QGridLayout(dir_grp)
        dir_lay.setColumnStretch(1, 1)

        # (label, instance attr, current value, show browse button)
        dir_fields = [
            ("LAFAN Data:",      "_settings_lafan_edit",    str(DEFAULT_LAFAN_DIR),  True),
            ("C3D Data:",        "_settings_c3d_edit",     str(DEFAULT_C3D_DIR),    True),
            ("OMOMO Data:",      "_settings_omomo_edit",   str(DEFAULT_OMOMO_DIR),  True),
            ("Logs Dir:",        "_settings_logs_edit",    str(DEFAULT_LOGS_DIR),   True),
            ("Retarget Output:", "_settings_retarget_edit",str(DEFAULT_RETARGET_DIR), False),
            ("Convert Output:",  "_settings_convert_edit", str(DEFAULT_CONVERT_DIR), False),
            ("Video Dir:",       "_settings_video_edit",   str(DEFAULT_VIDEO_DIR),  False),
        ]
        for row_idx, (label_text, attr, default, has_browse) in enumerate(dir_fields):
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(130)
            dir_lay.addWidget(lbl, row_idx, 0)
            edit = QLineEdit(default)
            setattr(self, attr, edit)
            dir_lay.addWidget(edit, row_idx, 1)
            if has_browse:
                btn = QPushButton("Browse…")
                btn.setFixedWidth(80)
                btn.clicked.connect(lambda _, e=edit: self._browse_settings_dir(e))
                dir_lay.addWidget(btn, row_idx, 2)

        note = QLabel(
            "Project Root: " + str(PROJECT_ROOT) + "\n\n"
            "Retarget Output and Convert Output are relative to Project Root.\n"
            "Click Save & Apply to update all tabs immediately."
        )
        note.setStyleSheet("color: #9399b2; font-size: 11px;")
        note.setWordWrap(True)
        dir_lay.addWidget(note, len(dir_fields), 0, 1, 3)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e; font-weight:bold;"
            "padding:6px 16px; border-radius:4px;"
        )
        save_btn.clicked.connect(self._save_dir_config)
        btn_row.addWidget(save_btn)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_dir_config)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        dir_lay.addLayout(btn_row, len(dir_fields) + 1, 0, 1, 3)
        lay.addWidget(dir_grp)

        # ── Directory Status (dynamic) ───────────────────────────────────
        check_grp = QGroupBox("Directory Status")
        check_lay = QVBoxLayout(check_grp)
        self._settings_status_labels: list[tuple[str, QLabel]] = []
        status_names = [
            "LAFAN Data", "C3D Data", "OMOMO Data",
            "Logs", "Retarget Output", "Convert Output", "Videos",
        ]
        for name in status_names:
            lbl = QLabel()
            check_lay.addWidget(lbl)
            self._settings_status_labels.append((name, lbl))
        lay.addWidget(check_grp)
        self._update_dir_status()

        # ── Python Dependencies ──────────────────────────────────────────
        dep_grp = QGroupBox("Python Dependencies")
        dep_lay = QVBoxLayout(dep_grp)
        for name, ok in [("PySide6", True), ("matplotlib", True), ("tbparse (live plots)", HAS_TBPARSE)]:
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

    @Slot()
    def _browse_settings_dir(self, edit: QLineEdit):
        start = edit.text().strip() or str(PROJECT_ROOT)
        chosen = QFileDialog.getExistingDirectory(self, "Select Directory", start)
        if chosen:
            edit.setText(chosen)

    @Slot()
    def _save_dir_config(self):
        cfg = {
            "lafan_dir":    self._settings_lafan_edit.text().strip(),
            "c3d_dir":      self._settings_c3d_edit.text().strip(),
            "omomo_dir":    self._settings_omomo_edit.text().strip(),
            "logs_dir":     self._settings_logs_edit.text().strip(),
            "retarget_dir": self._settings_retarget_edit.text().strip(),
            "convert_dir":  self._settings_convert_edit.text().strip(),
            "video_dir":    self._settings_video_edit.text().strip(),
        }
        _apply_dir_config(cfg)
        try:
            LAUNCHER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LAUNCHER_CONFIG_PATH, "w") as fh:
                json.dump(cfg, fh, indent=2)
        except Exception as exc:
            self._log(f"[Settings] Could not save config: {exc}", "#f38ba8")

        # Sync OMOMO tab's data-dir widget
        self._omomo_data_dir.setText(str(DEFAULT_OMOMO_DIR))

        # Refresh all file lists and status
        self._refresh_lafan()
        self._refresh_c3d()
        self._refresh_omomo()
        self._refresh_files_tab()
        self._update_dir_status()
        self.statusBar().showMessage("Directories saved and applied.", 5000)
        self._log("[Settings] Directories saved and applied.", "#a6e3a1")

    @Slot()
    def _reset_dir_config(self):
        self._settings_lafan_edit.setText(str(_DEFAULT_LAFAN_DIR))
        self._settings_c3d_edit.setText(str(_DEFAULT_C3D_DIR))
        self._settings_omomo_edit.setText(str(_DEFAULT_OMOMO_DIR))
        self._settings_logs_edit.setText(str(_DEFAULT_LOGS_DIR))
        self._settings_retarget_edit.setText(str(_DEFAULT_RETARGET_DIR))
        self._settings_convert_edit.setText(str(_DEFAULT_CONVERT_DIR))
        self._settings_video_edit.setText(str(_DEFAULT_VIDEO_DIR))

    def _update_dir_status(self):
        paths = [
            DEFAULT_LAFAN_DIR,
            DEFAULT_C3D_DIR,
            DEFAULT_OMOMO_DIR,
            DEFAULT_LOGS_DIR,
            PROJECT_ROOT / DEFAULT_RETARGET_DIR,
            PROJECT_ROOT / DEFAULT_CONVERT_DIR,
            PROJECT_ROOT / DEFAULT_VIDEO_DIR,
        ]
        for (name, lbl), path in zip(self._settings_status_labels, paths):
            exists = Path(path).exists()
            colour = "#a6e3a1" if exists else "#f38ba8"
            symbol = "OK" if exists else "MISSING"
            lbl.setText(f"  [{symbol}]  {name}: {path}")
            lbl.setStyleSheet(f"color: {colour};")

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
def _apply_dark_palette(app: QApplication) -> None:
    """Apply a dark QPalette so native widgets (QMessageBox, tooltips) match the theme."""
    pal = QPalette()
    # Window / panel backgrounds
    pal.setColor(QPalette.ColorRole.Window,          QColor("#1e1e2e"))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.Base,            QColor("#313244"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#45475a"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#313244"))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor("#cdd6f4"))
    # Text
    pal.setColor(QPalette.ColorRole.Text,            QColor("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.BrightText,      QColor("#f38ba8"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6c7086"))
    # Buttons
    pal.setColor(QPalette.ColorRole.Button,          QColor("#45475a"))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor("#cdd6f4"))
    # Highlight (selection)
    pal.setColor(QPalette.ColorRole.Highlight,       QColor("#89b4fa"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#1e1e2e"))
    # Disabled text
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor("#6c7086"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6c7086"))
    app.setPalette(pal)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_dark_palette(app)
    win = WorkflowLauncher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

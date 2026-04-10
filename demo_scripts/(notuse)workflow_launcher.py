#!/usr/bin/env python3
"""
PySide GUI Launcher for Holosoma Training Workflows

This application provides a graphical interface for selecting and running
different workflow options including:
- LAFAN Retargeting + Tracking
- C3D Retargeting + Tracking
- Browse Available Files
- Inference (Simulation)
- Inference (Hardware)
"""

import sys
import os
import subprocess
import shlex
from pathlib import Path
import json

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTabWidget,
        QLabel,
        QPushButton,
        QLineEdit,
        QComboBox,
        QTextEdit,
        QGroupBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QMessageBox,
        QFileDialog,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QFrame,
        QScrollArea,
        QStackedWidget,
        QProgressBar,
        QSizePolicy,
        QInputDialog,
    )
    from PySide6.QtCore import Qt, QSize, Signal, QThread, Slot
    from PySide6.QtGui import QFont, QCursor, QPalette, QColor
except ImportError:
    print("PySide6 not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"])
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTabWidget,
        QLabel,
        QPushButton,
        QLineEdit,
        QComboBox,
        QTextEdit,
        QGroupBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QMessageBox,
        QFileDialog,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QSizePolicy,
        QInputDialog,
    )
    from PySide6.QtCore import Qt, QSize, Signal, QThread, Slot
    from PySide6.QtGui import QFont, QCursor


class WorkflowThread(QThread):
    """Thread for running workflow commands."""

    output_ready = Signal(str)
    error_ready = Signal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            process = subprocess.Popen(
                self.command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            # Stream stdout line by line
            for line in process.stdout:
                line = line.rstrip('\n')
                if line:
                    self.output_ready.emit(line)
            # After stdout closes, read any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                for line in stderr_output.splitlines():
                    if line.strip():
                        self.error_ready.emit(f"STDERR: {line}")
            process.wait()
            if process.returncode != 0:
                self.error_ready.emit(f"Process exited with code {process.returncode}")
        except Exception as e:
            self.error_ready.emit(f"Error: {str(e)}")


class WorkflowLauncher(QMainWindow):
    """Main window for the workflow launcher application."""

    # Command signals for output
    output_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs" / "WholeBodyTracking"
        self.retarget_output_dir = "demo_results/g1/robot_only"
        self.convert_output_dir = "converted_res"
        self.video_dir = "logs/videos"
        self.lafan_data_dir = "src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan"
        self.c3d_data_dir = "src/holosoma_retargeting/holosoma_retargeting/demo_data/c3d"

        # Default training config
        self.num_envs = 4096
        self.num_iterations = 50000
        self.max_episode_length = 6.0
        self.alpha_init = 0.01
        self.ground_range = "-10 10"

        self.robots = ["23dof", "29dof"]
        self.algorithms = ["fast-sac", "fast-sac-no-dr", "ppo"]

        # Motion sequences cache
        self.lafan_tasks = []
        self.c3d_files = []
        self.checkpoint_files = []

        self._setup_ui()
        self._connect_signals()
        self._refresh_files()
        self._check_directories()

    def _check_directories(self):
        """Check and report on important directories."""
        dirs_to_check = [
            ("LAFAN Data", self.project_root / self.lafan_data_dir),
            ("C3D Data", self.project_root / self.c3d_data_dir),
            ("Logs", self.logs_dir),
            ("Converted Res", self.project_root / self.convert_output_dir),
            ("Retarget Output", self.project_root / self.retarget_output_dir),
            ("Videos", self.project_root / self.video_dir),
        ]

        for name, path in dirs_to_check:
            exists = path.exists()
            self.output_ready.emit(f"[{name}] {'✓ Found' if exists else '✗ Not found'}")

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Holosoma Workflow Launcher")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 12px;
                background-color: #fff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 12px;
                background-color: #e8f0fe;
                color: #1a73e8;
            }
            QPushButton {
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3367d6;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QLabel {
                color: #333;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: #fff;
                border-radius: 5px;
            }
            QTabWidget::tab-bar {
                background: #f0f0f0;
            }
            QTabBar::tab {
                background: #e8e8e8;
                padding: 10px;
                margin: 2px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 3px 3px 0 0;
            }
            QTabBar::tab:selected {
                background: #fff;
            }
            QTableWidget {
                background-color: #fff;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QProgressBar {
                height: 10px;
                border: none;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4285f4;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 2, 4, 2)
        title = QLabel("🤖 Holosoma Workflow Launcher")
        title.setFont(QFont("Helvetica", 16, QFont.Bold))
        title.setStyleSheet("color: #1a73e8; padding: 4px;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(version_label)
        layout.addLayout(header_layout)

        # Main splitter with tabs and console
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Tab widget for workflow selection
        tab_widget = QTabWidget()
        tab_widget.setMinimumWidth(500)

        # Main workflow tab
        main_tab = self._create_main_tab()
        tab_widget.addTab(main_tab, "Training Workflows")

        # Inference tab
        inference_tab = self._create_inference_tab()
        tab_widget.addTab(inference_tab, "Inference")

        # Files browser tab
        files_tab = self._create_files_tab()
        tab_widget.addTab(files_tab, "Browse Files")

        # Settings tab
        settings_tab = self._create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")

        splitter.addWidget(tab_widget)

        # Right side - Output console
        console_group = QGroupBox("📋 Output Console")
        console_layout = QVBoxLayout(console_group)
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setFont(QFont("Monospace", 9))
        console_layout.addWidget(self.output_console)

        # Console controls
        console_controls = QHBoxLayout()

        clear_btn = QPushButton("🗑️ Clear Console")
        clear_btn.clicked.connect(self._clear_console)
        console_controls.addWidget(clear_btn)

        copy_btn = QPushButton("📋 Copy Output")
        copy_btn.clicked.connect(self._copy_output)
        console_controls.addWidget(copy_btn)

        console_controls.addStretch()
        console_layout.addLayout(console_controls)

        splitter.addWidget(console_group)

        # Give more space to tabs (left) than console (right)
        splitter.setSizes([700, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # Splitter should fill all remaining vertical space
        layout.addWidget(splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _create_main_tab(self):
        """Create the main workflow selection tab."""
        # Use a scroll area so content doesn't clip on smaller windows
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Robot DOF selection
        robot_group = QGroupBox("🤖 Robot Configuration")
        robot_layout = QVBoxLayout(robot_group)

        robot_label = QLabel("Robot DOF:")
        self.robot_dof_combo = QComboBox()
        for dof in self.robots:
            dof_display = f"G1-{dof}DOF"
            self.robot_dof_combo.addItem(dof_display, dof)
        self.robot_dof_combo.setCurrentIndex(0)
        robot_layout.addWidget(robot_label)
        robot_layout.addWidget(self.robot_dof_combo)

        algo_label = QLabel("Training Algorithm:")
        self.training_algo_combo = QComboBox()
        for algo in self.algorithms:
            algo_display = algo.replace("-", " ").title()
            self.training_algo_combo.addItem(algo_display, algo)
        self.training_algo_combo.setCurrentIndex(0)
        robot_layout.addWidget(algo_label)
        robot_layout.addWidget(self.training_algo_combo)

        robot_layout.addStretch()
        layout.addWidget(robot_group)

        # Data format selection
        data_group = QGroupBox("📊 Data Source")
        data_layout = QVBoxLayout(data_group)

        data_format_label = QLabel("Data Format:")
        self.data_format_combo = QComboBox()
        self.data_format_combo.addItem("📁 LAFAN", "lafan")
        self.data_format_combo.addItem("📁 C3D", "c3d")
        self.data_format_combo.addItem("📁 Other", "other")
        self.data_format_combo.setCurrentIndex(0)
        data_layout.addWidget(data_format_label)
        data_layout.addWidget(self.data_format_combo)

        # LAFAN specific options
        lafan_group = QGroupBox("📁 LAFAN Options (only for LAFAN format)")
        lafan_layout = QVBoxLayout(lafan_group)

        # Motion selection
        self.motion_select_btn = QPushButton("📂 Select Motion Sequence...")
        self.motion_select_btn.clicked.connect(self._select_lafan_motion)
        lafan_layout.addWidget(self.motion_select_btn)

        # Motion path field
        self.motion_path_edit = QLineEdit()
        self.motion_path_edit.setPlaceholderText("Motion path will be displayed after selection...")
        lafan_layout.addWidget(self.motion_path_edit)

        # Custom ground range
        ground_range_label = QLabel("Ground Range (X, Y):")
        self.ground_range_edit = QLineEdit(self.ground_range)
        lafan_layout.addWidget(ground_range_label)
        lafan_layout.addWidget(self.ground_range_edit)

        lafan_layout.addStretch()
        data_layout.addWidget(lafan_group)
        layout.addWidget(data_group)

        # C3D specific options
        c3d_group = QGroupBox("📁 C3D Options (only for C3D format)")
        c3d_layout = QVBoxLayout(c3d_group)

        self.c3d_path_edit = QLineEdit()
        self.c3d_path_edit.setPlaceholderText("Place .c3d file in C3D directory or enter full path")
        c3d_layout.addWidget(self.c3d_path_edit)

        c3d_layout.addStretch()

        data_layout.addWidget(c3d_group)

        # Training configuration
        train_group = QGroupBox("⚙️ Training Configuration")
        train_layout = QVBoxLayout(train_group)

        envs_label = QLabel("Number of Envs:")
        self.num_envs_spin = QSpinBox()
        self.num_envs_spin.setRange(1, 8192)
        self.num_envs_spin.setValue(self.num_envs)
        train_layout.addWidget(envs_label)
        train_layout.addWidget(self.num_envs_spin)

        iterations_label = QLabel("Number of Iterations:")
        self.num_iterations_spin = QSpinBox()
        self.num_iterations_spin.setRange(1000, 200000)
        self.num_iterations_spin.setValue(self.num_iterations)
        train_layout.addWidget(iterations_label)
        train_layout.addWidget(self.num_iterations_spin)

        max_len_label = QLabel("Max Episode Length (seconds):")
        self.max_episode_length_spin = QDoubleSpinBox()
        self.max_episode_length_spin.setRange(1.0, 30.0)
        self.max_episode_length_spin.setValue(self.max_episode_length)
        self.max_episode_length_spin.setDecimals(1)
        train_layout.addWidget(max_len_label)
        train_layout.addWidget(self.max_episode_length_spin)

        train_layout.addStretch()
        layout.addWidget(train_group)

        # Output folder configuration
        output_group = QGroupBox("📁 Output Folders")
        output_layout = QVBoxLayout(output_group)

        retarget_path_label = QLabel("Retarget Output Directory:")
        self.retarget_path_edit = QLineEdit(self.retarget_output_dir)
        self.retarget_path_edit.textChanged.connect(self._on_retarget_path_changed)
        output_layout.addWidget(retarget_path_label)
        output_layout.addWidget(self.retarget_path_edit)

        convert_path_label = QLabel("Convert Output Directory:")
        self.convert_path_edit = QLineEdit(self.convert_output_dir)
        self.convert_path_edit.textChanged.connect(self._on_convert_path_changed)
        output_layout.addWidget(convert_path_label)
        output_layout.addWidget(self.convert_path_edit)

        video_path_label = QLabel("Video Output Directory:")
        self.video_path_edit = QLineEdit(self.video_dir)
        output_layout.addWidget(video_path_label)
        output_layout.addWidget(self.video_path_edit)

        output_layout.addStretch()
        layout.addWidget(output_group)

        # Run button
        run_button = QPushButton("🚀 START WORKFLOW")
        run_button.setMinimumHeight(50)
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #9aa0a6;
            }
        """)
        run_button.clicked.connect(self._run_workflow)
        layout.addWidget(run_button)

        # Status label
        self.status_label = QLabel("Ready to start workflow")
        self.status_label.setStyleSheet("color: #333; font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)

        scroll_area.setWidget(widget)
        return scroll_area

    def _create_inference_tab(self):
        """Create the inference tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Inference type selection
        inference_group = QGroupBox("🔍 Inference Type")
        inference_layout = QVBoxLayout(inference_group)

        self.inference_type_combo = QComboBox()
        self.inference_type_combo.addItem("🖥️ Simulation", "simulation")
        self.inference_type_combo.addItem("🤖 Real Robot Hardware", "hardware")
        self.inference_type_combo.addItem("📂 Load ONNX Model", "onnx")
        self.inference_type_combo.setCurrentIndex(0)
        inference_layout.addWidget(self.inference_type_combo)

        # Checkpoint/ONNX file path
        self.checkpoint_label = QLabel("Checkpoint/ONNX Model Path:")
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Select or enter checkpoint/ONNX path...")
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(self._browse_checkpoint)
        inference_layout.addWidget(self.checkpoint_label)
        inference_layout.addWidget(self.checkpoint_edit)
        inference_layout.addWidget(browse_btn)

        # Hardware specific settings
        self.hardware_group = QGroupBox("🤖 Hardware Settings (only for Real Robot)")
        self.hardware_group.setVisible(False)
        hardware_layout = QVBoxLayout(self.hardware_group)

        ros2_group = QGroupBox("ROS2 Configuration (optional)")
        ros2_layout = QVBoxLayout(ros2_group)

        ros2_check = QCheckBox("Use ROS2 for hardware inference")
        ros2_check.setChecked(False)
        ros2_layout.addWidget(ros2_check)

        ros2_namespace_edit = QLineEdit()
        ros2_namespace_edit.setPlaceholderText("ros2 namespace (leave empty for default)")
        ros2_layout.addWidget(ros2_namespace_edit)

        hardware_layout.addWidget(ros2_group)

        inference_layout.addWidget(self.hardware_group)
        layout.addWidget(inference_group)

        # Start inference button
        start_inference_btn = QPushButton("▶️ START INFERENCE")
        start_inference_btn.setMinimumHeight(50)
        start_inference_btn.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #2b8a40;
            }
            QPushButton:pressed {
                background-color: #1e6b30;
            }
            QPushButton:disabled {
                background-color: #9aa0a6;
            }
        """)
        start_inference_btn.clicked.connect(self._start_inference)
        layout.addWidget(start_inference_btn)

        # Status label
        self.inference_status_label = QLabel("Ready to start inference")
        self.inference_status_label.setStyleSheet("color: #333; font-weight: bold; padding: 10px;")
        layout.addWidget(self.inference_status_label)

        return widget

    def _create_files_tab(self):
        """Create the files browser tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Files browser
        files_group = QGroupBox("📂 Files Browser")
        files_layout = QVBoxLayout(files_group)

        # Tab for LAFAN sequences
        self.files_tab_widget = QTabWidget()

        # LAFAN sequences
        self.lafan_table = QTableWidget()
        self.lafan_table.setColumnCount(2)
        self.lafan_table.setHorizontalHeaderLabels(["Name", "Path"])
        self.lafan_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.lafan_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lafan_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # C3D files
        self.c3d_table = QTableWidget()
        self.c3d_table.setColumnCount(2)
        self.c3d_table.setHorizontalHeaderLabels(["Name", "Path"])
        self.c3d_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.c3d_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.c3d_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Checkpoint files
        self.checkpoint_table = QTableWidget()
        self.checkpoint_table.setColumnCount(3)
        self.checkpoint_table.setHorizontalHeaderLabels(["Name", "Path", "Size"])
        self.checkpoint_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.checkpoint_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.checkpoint_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.checkpoint_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.files_tab_widget.addTab(self.lafan_table, "LAFAN Sequences (.npy)")
        self.files_tab_widget.addTab(self.c3d_table, "C3D Files (.c3d)")
        self.files_tab_widget.addTab(self.checkpoint_table, "Checkpoints (.pt)")

        files_layout.addWidget(self.files_tab_widget)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8;
                color: white;
                font-size: 12px;
                padding: 5px 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_files)
        files_layout.addWidget(refresh_btn)

        layout.addWidget(files_group)

        return widget

    def _create_settings_tab(self):
        """Create the settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        settings_group = QGroupBox("⚙️ Application Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Version info
        version_label = QLabel(f"Version: 1.0.0\nProject: {self.project_root.name}")
        version_label.setStyleSheet("color: #666; font-style: italic;")
        settings_layout.addWidget(version_label)

        # About button
        about_btn = QPushButton("ℹ️ About")
        about_btn.clicked.connect(self._show_about)
        settings_layout.addWidget(about_btn)

        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # Quick actions
        actions_group = QGroupBox("🔧 Quick Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Create output directories
        create_dirs_btn = QPushButton("📁 Create Output Directories")
        create_dirs_btn.clicked.connect(self._create_output_dirs)
        create_dirs_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffb300;
                color: black;
                font-size: 12px;
                padding: 5px 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e6a200;
            }
        """)
        actions_layout.addWidget(create_dirs_btn)

        # Copy config to home
        copy_home_btn = QPushButton("📋 Copy Config to Home Directory")
        copy_home_btn.clicked.connect(self._copy_config_to_home)
        copy_home_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffb300;
                color: black;
                font-size: 12px;
                padding: 5px 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e6a200;
            }
        """)
        actions_layout.addWidget(copy_home_btn)

        actions_layout.addStretch()
        layout.addWidget(actions_group)

        return widget

    def _connect_signals(self):
        """Connect signals and slots."""
        # Format change - enable/disable appropriate fields
        self.data_format_combo.currentIndexChanged.connect(self._on_format_changed)

        # Path changes update status
        self.retarget_path_edit.textChanged.connect(self._on_retarget_path_changed)
        self.convert_path_edit.textChanged.connect(self._on_convert_path_changed)

    def _on_format_changed(self, index):
        """Handle data format change."""
        format_type = self.data_format_combo.currentData()
        self.motion_path_edit.setVisible(format_type == "lafan")
        self.motion_path_edit.setEnabled(format_type == "lafan")
        self.c3d_path_edit.setVisible(format_type == "c3d")
        self.c3d_path_edit.setEnabled(format_type == "c3d")

    def _on_retarget_path_changed(self, text):
        """Handle retarget path change."""
        self.status_label.setText(f"Retarget output directory: {text}")

    def _on_convert_path_changed(self, text):
        """Handle convert path change."""
        self.status_label.setText(f"Convert output directory: {text}")

    def _select_lafan_motion(self):
        """Select a LAFAN motion sequence."""
        motion_dir = self.project_root / self.lafan_data_dir
        if not motion_dir.exists():
            QMessageBox.warning(self, "Directory Not Found",
                                f"LAFAN directory not found:\n{motion_dir}")
            return

        files = sorted(motion_dir.glob("*.npy"))
        if not files:
            QMessageBox.information(self, "No Files",
                                   f"No .npy files found in:\n{motion_dir}")
            return

        # Build display list: show stem names for readability
        items = [f.stem for f in files]
        selected, ok = QInputDialog.getItem(
            self, "Select Motion Sequence",
            f"Available LAFAN sequences in {motion_dir.name}/:",
            items, 0, False
        )
        if ok and selected:
            # Find the matching file and set full path
            match = next((f for f in files if f.stem == selected), None)
            if match:
                self.motion_path_edit.setText(str(match))
                self.output_ready.emit(f"Selected motion: {match.name}")

    def _select_c3d_file(self):
        """Select a C3D file."""
        c3d_dir = self.project_root / self.c3d_data_dir
        if not c3d_dir.exists():
            QMessageBox.warning(self, "Directory Not Found",
                                f"C3D directory not found:\n{c3d_dir}")
            return

        files = sorted(c3d_dir.glob("*.c3d"))
        if not files:
            QMessageBox.information(self, "No Files",
                                   f"No .c3d files found in:\n{c3d_dir}")
            return

        items = [f.stem for f in files]
        selected, ok = QInputDialog.getItem(
            self, "Select C3D File",
            f"Available C3D files in {c3d_dir.name}/:",
            items, 0, False
        )
        if ok and selected:
            match = next((f for f in files if f.stem == selected), None)
            if match:
                self.c3d_path_edit.setText(str(match))
                self.output_ready.emit(f"Selected C3D file: {match.name}")

    def _browse_checkpoint(self):
        """Browse for checkpoint or ONNX files."""
        files = []

        # Look for checkpoint files
        pt_files = list(self.logs_dir.glob("**/*model_*.pt"))
        if pt_files:
            files.extend(pt_files)

        # Look for ONNX files
        onnx_files = list(self.project_root.glob("**/*.onnx"))
        if onnx_files:
            files.extend(onnx_files)

        # Look in converted_res
        converted_files = list(self.project_root / self.convert_output_dir.glob("**/*.pt"))
        if converted_files:
            files.extend(converted_files)

        if files:
            # Display in console
            self.output_ready.emit("\n=== Available Checkpoints/Models ===")
            for i, f in enumerate(files):
                size = f.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.2f} MB"
                self.output_ready.emit(f"[{i}] {f.name} ({size_str})")
            self.output_ready.emit(f"[{len(files)}] (new)")
            self.checkpoint_edit.setText(files[0].as_posix())
        else:
            self.output_ready.emit("\nNo checkpoint or ONNX files found.\n")
            self.output_ready.emit("Train a model first using Training Workflows tab.")

    def _start_inference(self):
        """Start inference workflow."""
        inference_type = self.inference_type_combo.currentData()

        if not inference_type:
            self.output_ready.emit("\nError: Please select an inference type.\n")
            return

        if inference_type == "hardware" and not self.checkpoint_edit.text().strip():
            self.output_ready.emit("\nError: Please provide an ONNX model path for hardware inference.\n")
            return

        checkpoint_path = self.checkpoint_edit.text().strip()

        # Build command
        if inference_type == "simulation":
            if not checkpoint_path:
                self.output_ready.emit("\nError: Please provide a checkpoint path.\n")
                return

            cmd = f'cd {self.project_root} && python src/holosoma/holosoma/eval_agent.py --checkpoint "{checkpoint_path}" --training.export-onnx True --training.num-envs 1 --training.headless True'
            self.output_ready.emit("\nStarting simulation inference...\n")
            self.output_ready.emit(f"Command: {cmd}\n")
            self._run_command(cmd)

        elif inference_type == "onnx":
            if not checkpoint_path:
                self.output_ready.emit("\nError: Please provide an ONNX model path.\n")
                return

            cmd = f'cd {self.project_root} && python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-23dof-wbt --task.model-path="{checkpoint_path}" --observation.groups.actor_obs.history-length=4'
            self.output_ready.emit("\nStarting ONNX inference...\n")
            self.output_ready.emit(f"Command: {cmd}\n")
            self._run_command(cmd)

        elif inference_type == "hardware":
            onnx_path = self.checkpoint_edit.text().strip()
            if not onnx_path:
                self.output_ready.emit("\nError: Please provide an ONNX model path.\n")
                return

            cmd = f'cd {self.project_root} && python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-23dof-wbt --task.model-path="{onnx_path}" --observation.groups.actor_obs.history-length=4'
            self.output_ready.emit("\n⚠️ WARNING: This will control the physical G1 robot!\n")
            self.output_ready.emit("Make sure the robot is in a safe position and E-stop is accessible.\n")
            self.output_ready.emit("\nStarting hardware inference...\n")
            self.output_ready.emit(f"Command: {cmd}\n")
            self._run_command(cmd)

    def _refresh_files(self):
        """Refresh files in the browser tabs."""
        # Refresh LAFAN files
        lafan_dir = self.project_root / self.lafan_data_dir
        if lafan_dir.exists():
            files = list(lafan_dir.glob("*.npy"))
            self.lafan_table.setRowCount(len(files))
            for i, f in enumerate(files):
                self.lafan_table.setItem(i, 0, QTableWidgetItem(f.stem))
                self.lafan_table.setItem(i, 1, QTableWidgetItem(f.as_posix()))

        # Refresh C3D files
        c3d_dir = self.project_root / self.c3d_data_dir
        if c3d_dir.exists():
            files = list(c3d_dir.glob("*.c3d"))
            self.c3d_table.setRowCount(len(files))
            for i, f in enumerate(files):
                self.c3d_table.setItem(i, 0, QTableWidgetItem(f.stem))
                self.c3d_table.setItem(i, 1, QTableWidgetItem(f.as_posix()))

        # Refresh checkpoint files
        pt_files = list(self.logs_dir.glob("**/*model_*.pt"))
        onnx_files = list(self.project_root.glob("**/*.onnx"))
        all_files = pt_files + onnx_files

        self.checkpoint_table.setRowCount(len(all_files))
        for i, f in enumerate(all_files):
            size = f.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.2f} MB"
            self.checkpoint_table.setItem(i, 0, QTableWidgetItem(f.stem))
            self.checkpoint_table.setItem(i, 1, QTableWidgetItem(f.as_posix()))
            self.checkpoint_table.setItem(i, 2, QTableWidgetItem(size_str))

        self.output_ready.emit("\nFiles refreshed.\n")

    def _run_command(self, cmd):
        """Run a command in a thread."""
        self._run_command_thread(cmd)

    def _run_command_thread(self, cmd):
        """Run a command in a thread."""
        self.output_ready.emit(f"\n{'=' * 60}\n")
        self.output_ready.emit(f"Starting workflow...\n{'=' * 60}\n")
        self.output_ready.emit(f"Command: {cmd}\n")
        self.output_ready.emit(f"\nOutput will appear below...\n{'=' * 60}\n")

        # Store as instance attribute so the thread isn't garbage collected
        self._worker_thread = WorkflowThread(cmd)
        self._worker_thread.output_ready.connect(self.output_ready.emit)
        self._worker_thread.error_ready.connect(self.output_ready.emit)
        self._worker_thread.finished.connect(self._on_workflow_finished)
        self._worker_thread.start()

        self.status_label.setText("Running workflow...")

    def _on_workflow_finished(self):
        """Handle workflow thread completion."""
        self.status_label.setText("Workflow finished")
        self.output_ready.emit(f"\n{'=' * 60}")
        self.output_ready.emit("Workflow completed.")
        self.output_ready.emit(f"{'=' * 60}\n")

    def _run_workflow(self):
        """Run the selected workflow."""
        data_format = self.data_format_combo.currentData()
        robot_dof = self.robot_dof_combo.currentData()
        training_algo = self.training_algo_combo.currentData()

        self.output_ready.emit(f"\n{'=' * 60}\n")
        self.output_ready.emit(f"Selected Configuration:\n")
        self.output_ready.emit(f"  Data Format:    {data_format}")
        self.output_ready.emit(f"  Robot DOF:      {robot_dof}")
        self.output_ready.emit(f"  Training Algo:  {training_algo}")
        self.output_ready.emit(f"  Num Envs:       {self.num_envs_spin.value()}")
        self.output_ready.emit(f"  Iterations:     {self.num_iterations_spin.value()}")
        self.output_ready.emit(f"  Max Episode:    {self.max_episode_length_spin.value()}s\n")

        if data_format == "lafan":
            motion_path_raw = self.motion_path_edit.text().strip()
            ground_range = self.ground_range_edit.text().strip()

            if not motion_path_raw:
                self.output_ready.emit("\nError: Please select a motion sequence.\n")
                return

            # Extract just the task name (stem) from full path or filename
            motion_name = Path(motion_path_raw).stem

            cmd = f"cd {self.project_root} && bash demo_scripts/demo_g1_{robot_dof}_lafan_wb_tracking.sh"
            self.output_ready.emit(f"\nLAFAN Workflow Command:\n")
            self.output_ready.emit(f"  python src/holosoma/holosoma/train_agent.py \\")
            self.output_ready.emit(f"    exp:g1-{robot_dof}-wbt-{training_algo} \\")
            self.output_ready.emit(f"    logger:wandb-offline \\")
            self.output_ready.emit(f"    --training.headless True \\")
            self.output_ready.emit(f"    --training.num-envs {self.num_envs_spin.value()} \\")
            self.output_ready.emit(f"    --algo.config.num-learning-iterations {self.num_iterations_spin.value()} \\")
            self.output_ready.emit(
                f"    --simulator.config.sim.max-episode-length-s {self.max_episode_length_spin.value()} \\"
            )
            self.output_ready.emit(f"    --logger.video.enabled True \\")
            self.output_ready.emit(f"    --command.setup_terms.motion_command.params.motion_config.motion_file=...\\n")

            # Build and run retargeting command
            self.output_ready.emit(f"\nStarting retargeting workflow for '{motion_name}'...\n")

            retarget_cmd = (
                f"cd {self.project_root}/src/holosoma_retargeting/holosoma_retargeting && "
                f"python examples/robot_retarget.py "
                f"--robot-config.robot-dof {robot_dof} "
                f"--data_path demo_data/lafan "
                f"--task-type robot_only "
                f'--task-name "{motion_name}" '
                f"--data_format lafan "
                f"--task-config.ground-range {ground_range} "
                f"--save_dir demo_results/g1/robot_only "
                f"--retargeter.foot-sticking-tolerance 0.02"
            )

            convert_cmd = (
                f"python data_conversion/convert_data_format_mj.py "
                f"--robot-config.robot-dof {robot_dof} "
                f"--input_file demo_results/g1/robot_only/{motion_name}.npz "
                f"--output_fps 50 "
                f"--output_name converted_res/robot_only/{motion_name}_mj_fps50.npz "
                f"--data_format lafan "
                f'--object_name "ground" '
                f"--once"
            )

            train_cmd = (
                f"cd {self.project_root} && "
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py "
                f"exp:g1-{robot_dof}-wbt-{training_algo} "
                f"logger:wandb-offline "
                f"--training.headless True "
                f"--training.num-envs {self.num_envs_spin.value()} "
                f"--algo.config.num-learning-iterations {self.num_iterations_spin.value()} "
                f"--simulator.config.sim.max-episode-length-s {self.max_episode_length_spin.value()} "
                f"--logger.video.enabled True "
                f"--logger.video.interval 5 "
                f"--logger.video.save-dir logs/videos/g1_{robot_dof}_lafan_wbt "
            )

            self.output_ready.emit(f"Step 1: Retargeting...\n")
            self.output_ready.emit(f"{retarget_cmd}\n")
            self._run_command(retarget_cmd)

        elif data_format == "c3d":
            c3d_path = self.c3d_path_edit.text().strip()

            if not c3d_path:
                self.output_ready.emit("\nError: Please provide a C3D file path.\n")
                return

            self.output_ready.emit(f"\nC3D Workflow starting with file: {c3d_path}\n")

            cmd = f"cd {self.project_root} && bash demo_scripts/demo_g1_{robot_dof}_c3d_wb_tracking.sh"
            self._run_command(cmd)

        else:
            self.output_ready.emit(f"\nOther data format workflow (not fully implemented yet).\n")

    def _clear_console(self):
        """Clear the console."""
        self.output_console.clear()
        self.output_ready.emit("\nConsole cleared.\n")

    def _copy_output(self):
        """Copy console output to clipboard."""
        self.output_console.textCursor().selectAll()
        QApplication.clipboard().setText(self.output_console.toPlainText())
        self.output_ready.emit("\nOutput copied to clipboard.\n")

    def _show_about(self):
        """Show about dialog."""
        about_text = (
            "Holosoma Workflow Launcher v1.0.0\n\n"
            "A GUI launcher for the Holosoma robot learning framework.\n\n"
            "Features:\n"
            "• Retargeting workflows for LAFAN and C3D data\n"
            "• Whole-body tracking training\n"
            "• Simulation inference\n"
            "• Hardware deployment\n"
            "• File browsing\n\n"
            f"Project Directory: {self.project_root}\n\n"
            "Licensed under MIT License.\n"
        )

        QMessageBox.about(self, "About Holosoma Launcher", about_text)

    def _create_output_dirs(self):
        """Create output directories."""
        dirs_to_create = [
            self.retarget_output_dir,
            self.convert_output_dir,
            self.video_dir,
        ]

        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                self.output_ready.emit(f"Created directory: {full_path}\n")
            except FileExistsError:
                self.output_ready.emit(f"Directory exists: {full_path}\n")
            except Exception as e:
                self.output_ready.emit(f"Error creating {full_path}: {e}\n")

        self.output_ready.emit("\nAll directories ready.\n")

    def _copy_config_to_home(self):
        """Copy configuration to home directory."""
        config_content = f"""# ~/.holosoma_workflow_config.sh
# Holosoma Workflow Configuration
# Edit this file to customize your workflow settings

export ROBOT_DOF={self.robot_dof_combo.currentData()}
export TRAINING_ALGO={self.training_algo_combo.currentData()}
export NUM_ENVS={self.num_envs_spin.value()}
export NUM_ITERATIONS={self.num_iterations_spin.value()}
export MAX_EPISODE_LENGTH={self.max_episode_length_spin.value()}
export GROUND_RANGE="{self.ground_range_edit.text()}"
export RETARGET_OUTPUT_DIR="{self.retarget_path_edit.text()}"
export CONVERT_OUTPUT_DIR="{self.convert_path_edit.text()}"
export VIDEO_DIR="{self.video_path_edit.text()}"
"""

        home_config_path = Path.home() / ".holosoma_workflow_config.sh"
        try:
            with open(home_config_path, "w") as f:
                f.write(config_content)
            self.output_ready.emit(f"\nConfiguration copied to: {home_config_path}\n")
            self.output_ready.emit("Edit this file to persist your settings.\n")
        except Exception as e:
            self.output_ready.emit(f"Error copying config: {e}\n")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create launcher
    launcher = WorkflowLauncher()
    launcher.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Interactive EEG Viewer Module
Provides a scrollable, scalable viewer for RAW_EEG and FILTERED_EEG data
with proper metadata display and channel labeling.
"""

import warnings
import time
import numpy as np
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QDoubleSpinBox, QPushButton, QScrollBar, QFrame, QGroupBox,
    QComboBox, QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QGridLayout, QSizePolicy, QToolBar,
    QAction, QStatusBar, QMainWindow, QApplication, QHeaderView,
    QScrollArea, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from core_data_structures import EEGSignal, DataType
from scipy.signal import butter, sosfiltfilt

from pathlib import Path
import file_loaders
from topo_window import TopoHeatmapWindow

# Fallback - same mapping as gui/constants.py
EGI_TO_1020_MAPPING = {
    'Fp1': ['E22', 'E23', 'E26', 'E27'], 'Fp2': ['E9', 'E10', 'E3', 'E2'],
    'F7': ['E33', 'E34'], 'F3': ['E24', 'E20', 'E28', 'E19'],
    'Fz': ['E11', 'E4', 'E12', 'E5'], 'F4': ['E118', 'E124', 'E117', 'E123'],
    'F8': ['E1', 'E121', 'E2', 'E122'],
    'FT9': ['E43', 'E48'], 'FT10': ['E119', 'E120'],
    'T7': ['E45', 'E46'], 'C3': ['E36', 'E30', 'E37', 'E31'],
    'Cz': ['E55', 'Cz', 'E129'], 'C4': ['E104', 'E105', 'E111', 'E110'],
    'T8': ['E108', 'E109'],
    'TP9': ['E56', 'E63'], 'TP10': ['E99', 'E100'],
    'P7': ['E58', 'E59', 'E51', 'E52'],
    'P3': ['E52', 'E53', 'E60', 'E61'], 'Pz': ['E62', 'E72', 'E67'],
    'P4': ['E85', 'E86', 'E92', 'E93'], 'P8': ['E91', 'E96', 'E97'],
    'O1': ['E70', 'E71', 'E66', 'E65'], 'Oz': ['E75', 'E82', 'E76'],
    'O2': ['E83', 'E84', 'E89', 'E90'],
}
def is_egi_format(channel_names):
    if not channel_names:
        return False
    egi_count = sum(1 for ch in channel_names
                   if str(ch).startswith('E') and str(ch)[1:].isdigit())
    return egi_count >= len(channel_names) * 0.5


class EEGViewerWidget(QWidget):
    """
    Interactive EEG viewer with scrolling, scaling, and metadata display.

    Features:
    - Adjustable X-scale (time window)
    - Adjustable Y-scale (amplitude)
    - Scrollable X-axis (time navigation)
    - Channel labels from metadata
    - Metadata panel showing all signal information
    - Support for both 2D (continuous) and 3D (epoched) data
    """
    # Emitted after every plot update so that linked windows (e.g. topo heatmap)
    # can refresh themselves.  No payload — listeners pull what they need from
    # the viewer's public attributes.
    view_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eeg_signal: Optional[EEGSignal] = None
        self.current_time_start = 0.0  # Current view start time in seconds
        self.time_window = 10.0  # Visible time window in seconds
        self.amplitude_scale = 1.0  # Y-scale multiplier (direct µV scaling)
        self.channel_spacing = 1.0  # Spacing multiplier between channels
        self.selected_epoch = 0  # For 3D data
        self.show_grid = True
        self.visible_channels: List[int] = []  # Indices of visible channels
        
        # FIXED: Use constant base spacing for consistent amplitude display
        # 100 µV spacing means amplitude_scale of 1.0 shows 100µV peaks clearly
        self.base_channel_spacing = 100.0  # Base spacing in µV units
        
        # 10-20 conversion mode
        self.convert_to_1020 = False
        self.visible_1020_channels = None  # None = show all; list = show selected
        
        # Highpass filter settings
        self.highpass_enabled = False
        self.highpass_freq = 0.5  # Default highpass cutoff in Hz
        self._highpass_cache = {}  # Cache filtered data to avoid recomputing
        self._highpass_cache_key = None  # Key to invalidate cache
        
        # Auto-scroll settings
        self.auto_scroll_active = False
        self.auto_scroll_speed = 20.0  # seconds per second (20x real-time default)
        self.auto_scroll_timer = QTimer(self)
        self.auto_scroll_timer.timeout.connect(self._auto_scroll_step)
        self.auto_scroll_interval = 50  # ms between updates
        self._last_scroll_time = None  # For frame-rate independent scrolling
        
        # Color scheme for channels
        self.channel_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)

        # Toolbar spans full width
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # Matplotlib figure and canvas — takes all available vertical space
        self.figure = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.figure.patch.set_facecolor('#f8f8f8')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(400, 300)
        main_layout.addWidget(self.canvas, stretch=1)

        # Time scrollbar
        scroll_layout = QHBoxLayout()
        scroll_layout.addWidget(QLabel("Time:"))
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.setValue(0)
        self.time_scrollbar.valueChanged.connect(self._on_scroll_changed)
        scroll_layout.addWidget(self.time_scrollbar, stretch=1)
        self.time_label = QLabel("0.00 – 10.00 s")
        self.time_label.setMinimumWidth(130)
        scroll_layout.addWidget(self.time_label)
        main_layout.addLayout(scroll_layout)

        # Control panel — two-row layout at the bottom
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("No data loaded — open a file to begin")
        main_layout.addWidget(self.status_bar)
    
    def _create_toolbar(self) -> QToolBar:
        """Create toolbar with quick actions."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Open file action
        open_action = QAction("📂 Open File", self)
        open_action.triggered.connect(self._open_file)
        toolbar.addAction(open_action)
        
        # Export action
        export_action = QAction("💾 Export Plot", self)
        export_action.triggered.connect(self._export_plot)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # Zoom actions
        zoom_in_action = QAction("🔍+ Zoom In", self)
        zoom_in_action.triggered.connect(self._zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("🔍- Zoom Out", self)
        zoom_out_action.triggered.connect(self._zoom_out)
        toolbar.addAction(zoom_out_action)
        
        toolbar.addSeparator()
        
        # Navigation
        go_start_action = QAction("⏮ Start", self)
        go_start_action.triggered.connect(self._go_to_start)
        toolbar.addAction(go_start_action)
        
        go_end_action = QAction("⏭ End", self)
        go_end_action.triggered.connect(self._go_to_end)
        toolbar.addAction(go_end_action)
        
        toolbar.addSeparator()
        
        # Auto scale
        auto_scale_action = QAction("📐 Auto Scale", self)
        auto_scale_action.triggered.connect(self._auto_scale_amplitude)
        toolbar.addAction(auto_scale_action)
        
        # Reset view
        reset_action = QAction("🔄 Reset View", self)
        reset_action.triggered.connect(self._reset_view)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()

        # Channel selector
        channel_action = QAction("📊 Channels", self)
        channel_action.setToolTip("Select which channels to display")
        channel_action.triggered.connect(self._open_channel_selector)
        toolbar.addAction(channel_action)

        toolbar.addSeparator()

        # Analysis tool windows (previously in the right-side button panel)
        signal_info_action = QAction("📋 Signal Info", self)
        signal_info_action.setToolTip("Open signal metadata and channel information")
        signal_info_action.triggered.connect(self._open_signal_info_window)
        toolbar.addAction(signal_info_action)

        fft_action = QAction("🌊 FFT View", self)
        fft_action.setToolTip("Open FFT spectrogram view in a new window")
        fft_action.triggered.connect(self._open_fft_window)
        toolbar.addAction(fft_action)

        filter_action = QAction("🔬 Filter View", self)
        filter_action.setToolTip("Open multi-band filtering view in a new window")
        filter_action.triggered.connect(self._open_filtering_window)
        toolbar.addAction(filter_action)

        topo_action = QAction("🧠 Topo Map", self)
        topo_action.setToolTip(
            "Open topographic heatmap — updates in real time as you scroll.\n"
            "Supports EGI 128-ch and standard 10-20 montages.\n"
            "EGI data can be converted to 10-20 inside the topo window."
        )
        topo_action.triggered.connect(self._open_topo_window)
        toolbar.addAction(topo_action)

        return toolbar
    
    def _create_control_panel(self) -> QFrame:
        """Create the two-row control panel at the bottom of the viewer."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(6, 5, 6, 5)
        outer.setSpacing(4)

        # ── Row 1: Time window, amplitude, spacing, epoch ───────────────
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # Time Window
        x_group = QGroupBox("Time Window")
        x_layout = QHBoxLayout(x_group)
        x_layout.setContentsMargins(8, 6, 8, 6)
        x_layout.setSpacing(4)
        x_layout.addWidget(QLabel("Window (s):"))
        self.time_window_spin = QDoubleSpinBox()
        self.time_window_spin.setRange(0.1, 300.0)
        self.time_window_spin.setValue(10.0)
        self.time_window_spin.setSingleStep(1.0)
        self.time_window_spin.setDecimals(1)
        self.time_window_spin.setMinimumWidth(72)
        self.time_window_spin.setFocusPolicy(Qt.StrongFocus)
        self.time_window_spin.valueChanged.connect(self._on_time_window_changed)
        x_layout.addWidget(self.time_window_spin)
        for preset in [1, 5, 10, 30, 60]:
            btn = QPushButton(f"{preset}s")
            btn.setMinimumWidth(36)
            btn.setMaximumWidth(46)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda checked, t=preset: self._set_time_window(t))
            x_layout.addWidget(btn)
        row1.addWidget(x_group)

        # Amplitude Scale
        y_group = QGroupBox("Amplitude Scale")
        y_layout = QHBoxLayout(y_group)
        y_layout.setContentsMargins(8, 6, 8, 6)
        y_layout.setSpacing(4)
        y_layout.addWidget(QLabel("Scale:"))
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.01, 200.0)
        self.amplitude_spin.setValue(1.0)
        self.amplitude_spin.setSingleStep(1.0)
        self.amplitude_spin.setDecimals(2)
        self.amplitude_spin.setMinimumWidth(72)
        self.amplitude_spin.valueChanged.connect(self._on_amplitude_changed)
        y_layout.addWidget(self.amplitude_spin)
        amp_down = QPushButton("½×")
        amp_down.setMinimumWidth(38)
        amp_down.setToolTip("Halve amplitude scale")
        amp_down.clicked.connect(lambda: self._adjust_amplitude(0.5))
        y_layout.addWidget(amp_down)
        amp_up = QPushButton("2×")
        amp_up.setMinimumWidth(38)
        amp_up.setToolTip("Double amplitude scale")
        amp_up.clicked.connect(lambda: self._adjust_amplitude(2.0))
        y_layout.addWidget(amp_up)
        row1.addWidget(y_group)

        # Channel Spacing
        spacing_group = QGroupBox("Ch. Spacing")
        spacing_layout = QHBoxLayout(spacing_group)
        spacing_layout.setContentsMargins(8, 6, 8, 6)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 10.0)
        self.spacing_spin.setValue(1.0)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setDecimals(1)
        self.spacing_spin.setMinimumWidth(64)
        self.spacing_spin.valueChanged.connect(self._on_spacing_changed)
        spacing_layout.addWidget(self.spacing_spin)
        row1.addWidget(spacing_group)

        # Epoch selector (hidden unless 3D data is loaded)
        self.epoch_group = QGroupBox("Epoch")
        epoch_layout = QHBoxLayout(self.epoch_group)
        epoch_layout.setContentsMargins(8, 6, 8, 6)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(0, 0)
        self.epoch_spin.setMinimumWidth(58)
        self.epoch_spin.valueChanged.connect(self._on_epoch_changed)
        epoch_layout.addWidget(self.epoch_spin)
        self.epoch_label = QLabel("/ 0")
        epoch_layout.addWidget(self.epoch_label)
        self.epoch_group.setVisible(False)
        row1.addWidget(self.epoch_group)

        row1.addStretch()
        outer.addLayout(row1)

        # ── Row 2: Playback, highpass filter, display toggles ───────────
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        # Auto Scroll
        scroll_group = QGroupBox("Auto Scroll")
        scroll_layout = QHBoxLayout(scroll_group)
        scroll_layout.setContentsMargins(8, 6, 8, 6)
        scroll_layout.setSpacing(4)
        self.auto_scroll_btn = QPushButton("▶  Play")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setMinimumWidth(80)
        self.auto_scroll_btn.clicked.connect(self._toggle_auto_scroll)
        scroll_layout.addWidget(self.auto_scroll_btn)
        scroll_layout.addWidget(QLabel("Speed:"))
        self.scroll_speed_spin = QDoubleSpinBox()
        self.scroll_speed_spin.setRange(0.1, 40.0)
        self.scroll_speed_spin.setValue(20.0)
        self.scroll_speed_spin.setSingleStep(5.0)
        self.scroll_speed_spin.setSuffix("×")
        self.scroll_speed_spin.setMinimumWidth(72)
        self.scroll_speed_spin.setToolTip("Playback speed (1× = real-time)")
        self.scroll_speed_spin.valueChanged.connect(self._on_scroll_speed_changed)
        scroll_layout.addWidget(self.scroll_speed_spin)
        row2.addWidget(scroll_group)

        # Highpass Filter
        hp_group = QGroupBox("Highpass Filter")
        hp_layout = QHBoxLayout(hp_group)
        hp_layout.setContentsMargins(8, 6, 8, 6)
        hp_layout.setSpacing(4)
        self.highpass_check = QCheckBox("Enable")
        self.highpass_check.setToolTip("Apply highpass filter to remove DC drift and align traces to baseline")
        self.highpass_check.stateChanged.connect(self._on_highpass_changed)
        hp_layout.addWidget(self.highpass_check)
        hp_layout.addWidget(QLabel("Cutoff:"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0.01, 30.0)
        self.highpass_spin.setValue(0.5)
        self.highpass_spin.setSingleStep(0.1)
        self.highpass_spin.setDecimals(2)
        self.highpass_spin.setSuffix(" Hz")
        self.highpass_spin.setMinimumWidth(84)
        self.highpass_spin.setToolTip("Highpass cutoff frequency in Hz")
        self.highpass_spin.valueChanged.connect(self._on_highpass_freq_changed)
        hp_layout.addWidget(self.highpass_spin)
        row2.addWidget(hp_group)

        # Display toggles
        display_group = QGroupBox("Display")
        display_layout = QHBoxLayout(display_group)
        display_layout.setContentsMargins(12, 6, 12, 6)
        display_layout.setSpacing(16)
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._on_grid_changed)
        display_layout.addWidget(self.grid_check)
        self.convert_1020_check = QCheckBox("10-20 Format")
        self.convert_1020_check.setToolTip("Convert EGI channels to standard 10-20 format by averaging mapped electrodes")
        self.convert_1020_check.stateChanged.connect(self._on_convert_1020_changed)
        display_layout.addWidget(self.convert_1020_check)
        row2.addWidget(display_group)

        row2.addStretch()
        outer.addLayout(row2)

        return frame
    
    def _create_button_panel(self) -> QFrame:
        """Create the right-side button group panel."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setMinimumWidth(160)
        frame.setMaximumWidth(220)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(12)

        title = QLabel("Tools")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        btn_style = (
            "QPushButton { padding: 8px 4px; font-size: 12px; border-radius: 6px; "
            "background-color: #e8eaf6; border: 1px solid #9fa8da; }"
            "QPushButton:hover { background-color: #c5cae9; }"
            "QPushButton:pressed { background-color: #9fa8da; }"
        )

        signal_info_btn = QPushButton("📋 Signal\nInformation")
        signal_info_btn.setToolTip("Open signal information in a new window")
        signal_info_btn.setStyleSheet(btn_style)
        signal_info_btn.setMinimumHeight(55)
        signal_info_btn.clicked.connect(self._open_signal_info_window)
        layout.addWidget(signal_info_btn)

        fft_btn = QPushButton("📊 FFT Over\nTime")
        fft_btn.setToolTip("Open FFT spectrogram view in a new window")
        fft_btn.setStyleSheet(btn_style)
        fft_btn.setMinimumHeight(55)
        fft_btn.clicked.connect(self._open_fft_window)
        layout.addWidget(fft_btn)

        filter_btn = QPushButton("🔬 Filtering\nView")
        filter_btn.setToolTip("Open multi-band filtered view in a new window")
        filter_btn.setStyleSheet(btn_style)
        filter_btn.setMinimumHeight(55)
        filter_btn.clicked.connect(self._open_filtering_window)
        layout.addWidget(filter_btn)

        layout.addStretch()
        return frame

    def _open_signal_info_window(self):
        """Open signal information in a new window."""
        if self.eeg_signal is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No EEG data loaded yet.")
            return
        win = SignalInfoWindow(self.eeg_signal, parent=self)
        win.show()
        # Keep reference so it's not garbage collected
        if not hasattr(self, '_open_windows'):
            self._open_windows = []
        self._open_windows.append(win)
        win.destroyed.connect(lambda: self._open_windows.remove(win) if win in self._open_windows else None)

    def _open_fft_window(self):
        """Open FFT over time view in a new window."""
        if self.eeg_signal is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No EEG data loaded yet.")
            return
        win = FFTOverTimeWindow(self.eeg_signal, parent=self)
        win.show()
        if not hasattr(self, '_open_windows'):
            self._open_windows = []
        self._open_windows.append(win)
        win.destroyed.connect(lambda: self._open_windows.remove(win) if win in self._open_windows else None)

    def _open_filtering_window(self):
        """Open filtering view in a new window."""
        if self.eeg_signal is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No EEG data loaded yet.")
            return
        win = FilteringWindow(self.eeg_signal, parent=self)
        win.show()
        if not hasattr(self, '_open_windows'):
            self._open_windows = []
        self._open_windows.append(win)
        win.destroyed.connect(lambda: self._open_windows.remove(win) if win in self._open_windows else None)

    def _open_topo_window(self):
        """Open topographic heatmap window.

        The window listens to this viewer's view_changed signal so it redraws
        automatically as the user scrolls or changes the time window.
        Data is not required to open — the topo window will show a placeholder
        until data is loaded.
        """
        if not hasattr(self, '_open_windows'):
            self._open_windows = []

        win = TopoHeatmapWindow(viewer=self, parent=None)
        win.show()
        # Connect so the topo window updates whenever the main view changes
        self.view_changed.connect(win.on_viewer_changed)
        self._open_windows.append(win)

        def _on_topo_closed():
            try:
                self.view_changed.disconnect(win.on_viewer_changed)
            except Exception:
                pass
            if win in self._open_windows:
                self._open_windows.remove(win)

        win.destroyed.connect(_on_topo_closed)

    def _create_metadata_panel(self) -> QFrame:
        """Create the metadata display panel."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setMinimumWidth(250)
        layout = QVBoxLayout(frame)
        
        # Title
        title = QLabel("📋 Signal Information")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Basic info table
        info_group = QGroupBox("Basic Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.setAlternatingRowColors(True)
        info_layout.addWidget(self.info_table)
        
        layout.addWidget(info_group)
        
        # Metadata table
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)
        
        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(2)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metadata_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metadata_table.setAlternatingRowColors(True)
        meta_layout.addWidget(self.metadata_table)
        
        layout.addWidget(meta_group)
        
        # Channel list
        channel_group = QGroupBox("Channels")
        channel_layout = QVBoxLayout(channel_group)
        
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(3)
        self.channel_table.setHorizontalHeaderLabels(["#", "Name", "Range (µV)"])
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        self.channel_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.channel_table.setAlternatingRowColors(True)
        self.channel_table.setMaximumHeight(200)
        channel_layout.addWidget(self.channel_table)
        
        layout.addWidget(channel_group)
        
        layout.addStretch()
        
        return frame
    
    def set_eeg_signal(self, eeg_signal: EEGSignal, title: str = "EEG Viewer"):
        """
        Load an EEG signal into the viewer.
        
        Parameters
        ----------
        eeg_signal : EEGSignal
            The EEG signal to display
        title : str
            Title for the viewer window
        """
        self.eeg_signal = eeg_signal
        self.current_time_start = 0.0
        
        # Handle 3D data (epoched)
        if eeg_signal.data.ndim == 3:
            n_epochs, n_channels, n_samples = eeg_signal.data.shape
            self.epoch_group.setVisible(True)
            self.epoch_spin.setRange(0, n_epochs - 1)
            self.epoch_spin.setValue(0)
            self.epoch_label.setText(f"/ {n_epochs - 1}")
            self.selected_epoch = 0
            total_duration = n_samples / eeg_signal.sampling_rate
        else:
            self.epoch_group.setVisible(False)
            n_channels, n_samples = eeg_signal.data.shape
            total_duration = n_samples / eeg_signal.sampling_rate
        
        # Initialize visible channels (all by default, or first 32 if too many)
        if n_channels > 64:
            self.visible_channels = list(range(32))  # Show first 32 channels by default
        else:
            self.visible_channels = list(range(n_channels))  # Show all
        
        # Update scrollbar
        self._update_scrollbar()
        
        # Set reasonable time window
        self.time_window = min(10.0, total_duration)
        self.time_window_spin.blockSignals(True)
        self.time_window_spin.setValue(self.time_window)
        self.time_window_spin.blockSignals(False)
        
        # Auto scale amplitude
        self._auto_scale_amplitude()
        
        # Update metadata display
        self._update_metadata_display()
        
        # Update channel table
        self._update_channel_table()
        
        # Draw the plot
        self._update_plot()
        
        # Update status
        data_type_str = eeg_signal.data_type.value if eeg_signal.data_type else "unknown"
        self.status_bar.showMessage(
            f"Loaded: {n_channels} channels, {total_duration:.2f}s duration, "
            f"{eeg_signal.sampling_rate:.1f} Hz, Type: {data_type_str}"
        )
    
    def _update_scrollbar(self):
        """Update scrollbar range based on data."""
        if self.eeg_signal is None:
            return
        
        if self.eeg_signal.data.ndim == 3:
            n_samples = self.eeg_signal.data.shape[2]
        else:
            n_samples = self.eeg_signal.data.shape[1]
        
        total_duration = n_samples / self.eeg_signal.sampling_rate
        max_scroll = max(0, total_duration - self.time_window)
        
        # Convert to integer steps (0.1s resolution)
        self.time_scrollbar.setMaximum(int(max_scroll * 10))
        self.time_scrollbar.setPageStep(int(self.time_window * 10))
    
    def _update_metadata_display(self):
        """Update the metadata tables.

        These tables live inside _create_metadata_panel(), which is only
        instantiated on demand (legacy path) — not in the main _init_ui().
        Guard every access so calling this method is a safe no-op when the
        panel has not been built.
        """
        if self.eeg_signal is None:
            return
        if not hasattr(self, 'info_table'):
            return
        
        eeg = self.eeg_signal
        
        # Basic info
        if eeg.data.ndim == 3:
            n_epochs, n_channels, n_samples = eeg.data.shape
            shape_str = f"{n_epochs} epochs × {n_channels} ch × {n_samples} samples"
        else:
            n_channels, n_samples = eeg.data.shape
            shape_str = f"{n_channels} ch × {n_samples} samples"
        
        basic_info = [
            ("Data Type", eeg.data_type.value if eeg.data_type else "N/A"),
            ("Shape", shape_str),
            ("Sampling Rate", f"{eeg.sampling_rate:.1f} Hz"),
            ("Duration", f"{n_samples / eeg.sampling_rate:.2f} s"),
            ("Channels", str(n_channels)),
            ("Data dtype", str(eeg.data.dtype)),
        ]
        
        self.info_table.setRowCount(len(basic_info))
        for i, (key, value) in enumerate(basic_info):
            self.info_table.setItem(i, 0, QTableWidgetItem(key))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # Metadata
        metadata = eeg.metadata or {}
        self.metadata_table.setRowCount(len(metadata))
        
        for i, (key, value) in enumerate(sorted(metadata.items())):
            self.metadata_table.setItem(i, 0, QTableWidgetItem(str(key)))
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            self.metadata_table.setItem(i, 1, QTableWidgetItem(value_str))
    
    def _update_channel_table(self):
        """Update channel information table.

        When 10-20 conversion is active, shows the converted 10-20 electrode
        names and their averaged amplitude ranges instead of the raw EGI channels.
        Guard against the table not existing (legacy panel not built).
        """
        if self.eeg_signal is None:
            return
        if not hasattr(self, 'channel_table'):
            return
        
        eeg = self.eeg_signal
        
        if eeg.data.ndim == 3:
            data = eeg.data[self.selected_epoch]
        else:
            data = eeg.data
        
        n_channels = data.shape[0]
        channel_names = eeg.channel_names or [f"Ch{i+1}" for i in range(n_channels)]
        
        # If 10-20 conversion is active, show converted channels
        if self.convert_to_1020 and is_egi_format(channel_names):
            self._update_channel_table_1020(data, channel_names)
            return
        
        self.channel_table.setRowCount(min(n_channels, 64))  # Limit display
        
        for i in range(min(n_channels, 64)):
            ch_data = data[i]
            ch_min, ch_max = np.min(ch_data), np.max(ch_data)
            
            self.channel_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.channel_table.setItem(i, 1, QTableWidgetItem(channel_names[i] if i < len(channel_names) else f"Ch{i+1}"))
            self.channel_table.setItem(i, 2, QTableWidgetItem(f"{ch_min:.1f} to {ch_max:.1f}"))
    
    def _update_channel_table_1020(self, data, channel_names):
        """Update channel table showing 10-20 converted electrodes."""
        # Build name->index lookup
        name_to_idx = {}
        for idx, name in enumerate(channel_names):
            name_to_idx[str(name)] = idx
            name_to_idx[str(name).upper()] = idx
        
        standard_order = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FT9', 'FT10',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'Oz', 'O2',
        ]
        
        rows = []
        for electrode_1020 in standard_order:
            if electrode_1020 not in EGI_TO_1020_MAPPING:
                continue
            egi_channels = EGI_TO_1020_MAPPING[electrode_1020]
            matched_indices = []
            for egi_ch in egi_channels:
                if egi_ch in name_to_idx:
                    matched_indices.append(name_to_idx[egi_ch])
                elif egi_ch.upper() in name_to_idx:
                    matched_indices.append(name_to_idx[egi_ch.upper()])
            
            if matched_indices:
                avg_data = np.mean(data[matched_indices], axis=0)
                ch_min, ch_max = np.min(avg_data), np.max(avg_data)
                src = ', '.join(egi_channels[:3])
                if len(egi_channels) > 3:
                    src += '...'
                rows.append((electrode_1020, f"{ch_min:.1f} to {ch_max:.1f}", src))
        
        self.channel_table.setRowCount(len(rows))
        for i, (name, range_str, src) in enumerate(rows):
            self.channel_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.channel_table.setItem(i, 1, QTableWidgetItem(f"{name} ({src})"))
            self.channel_table.setItem(i, 2, QTableWidgetItem(range_str))
    
    def _apply_highpass(self, data_array, sfreq):
        """Apply highpass filter to display data.
        
        Uses a 4th-order Butterworth highpass filter applied with sosfiltfilt
        (zero-phase filtering) to remove DC drift and align traces to baseline.
        This only affects the display - the original signal is not modified.
        
        Args:
            data_array: numpy array of shape (n_channels, n_samples)
            sfreq: sampling rate in Hz
            
        Returns:
            Filtered data array
        """
        if data_array.shape[1] < 20:
            # Not enough samples for reliable filtering
            return data_array
        
        nyquist = sfreq / 2.0
        if self.highpass_freq >= nyquist:
            return data_array
        
        try:
            # Design Butterworth highpass filter (4th order)
            sos = butter(4, self.highpass_freq / nyquist, btype='highpass', output='sos')
            
            # Apply zero-phase filtering to each channel
            filtered = np.zeros_like(data_array)
            for i in range(data_array.shape[0]):
                # Pad to avoid edge artifacts (reflect padding built into sosfiltfilt)
                filtered[i] = sosfiltfilt(sos, data_array[i])
            return filtered
        except Exception:
            # If filtering fails, return original data
            return data_array
    
    def _convert_data_to_1020(self, data, channel_names, start_sample, end_sample,
                              sfreq, downsample_factor, time_array):
        """Convert EGI channel data to 10-20 format by averaging mapped electrodes.
        
        Uses the same EGI_TO_1020_MAPPING as the results section to ensure consistency.
        For each 10-20 electrode, averages the time-series of the corresponding EGI channels.
        
        Args:
            data: full data array (n_channels, n_samples)
            channel_names: list of channel name strings
            start_sample: view start sample
            end_sample: view end sample
            sfreq: sampling rate
            downsample_factor: downsampling factor for display
            time_array: time axis array
            
        Returns:
            tuple: (plot_data, plot_names, plot_colors)
        """
        # Build a name->index lookup
        name_to_idx = {}
        for idx, name in enumerate(channel_names):
            name_to_idx[str(name)] = idx
            name_to_idx[str(name).upper()] = idx
        
        # Standard 10-20 order (anterior to posterior)
        standard_order = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FT9', 'FT10',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'Oz', 'O2',
        ]
        
        # 10-20 region colors
        region_colors = {
            'Fp1': '#228B22', 'Fp2': '#228B22',
            'F7': '#2E8B57', 'F3': '#2E8B57', 'Fz': '#2E8B57', 'F4': '#2E8B57', 'F8': '#2E8B57',
            'FT9': '#FF8C00', 'FT10': '#FF8C00',
            'T7': '#FF8C00', 'C3': '#4B0082', 'Cz': '#4B0082', 'C4': '#4B0082', 'T8': '#FF8C00',
            'TP9': '#FF8C00', 'TP10': '#FF8C00',
            'P7': '#9400D3', 'P3': '#9400D3', 'Pz': '#9400D3', 'P4': '#9400D3', 'P8': '#9400D3',
            'O1': '#191970', 'Oz': '#191970', 'O2': '#191970',
        }
        
        plot_names = []
        plot_data_list = []
        plot_colors = []
        
        # Filter to visible 10-20 channels if a selection exists
        visible_1020_set = set(self.visible_1020_channels) if self.visible_1020_channels else None
        
        for electrode_1020 in standard_order:
            if electrode_1020 not in EGI_TO_1020_MAPPING:
                continue
            
            # Skip if not in visible selection
            if visible_1020_set is not None and electrode_1020 not in visible_1020_set:
                continue
            
            egi_channels = EGI_TO_1020_MAPPING[electrode_1020]
            
            # Find matching channel indices
            matched_indices = []
            for egi_ch in egi_channels:
                if egi_ch in name_to_idx:
                    matched_indices.append(name_to_idx[egi_ch])
                elif egi_ch.upper() in name_to_idx:
                    matched_indices.append(name_to_idx[egi_ch.upper()])
            
            if not matched_indices:
                continue
            
            # Average the matched channels' time-series data
            if downsample_factor > 1:
                n_points = len(time_array)
                avg_data = np.zeros(n_points)
                for ch_idx in matched_indices:
                    ch_raw = data[ch_idx, start_sample:end_sample]
                    n_chunks = len(ch_raw) // downsample_factor
                    if n_chunks > 0:
                        reshaped = ch_raw[:n_chunks * downsample_factor].reshape(n_chunks, downsample_factor)
                        ds = np.mean(reshaped, axis=1)
                        avg_data[:n_chunks] += ds
                avg_data /= len(matched_indices)
            else:
                channel_data = data[matched_indices, start_sample:end_sample]
                avg_data = np.mean(channel_data, axis=0)
            
            plot_data_list.append(avg_data)
            plot_names.append(electrode_1020)
            
            # Color by region
            color_hex = region_colors.get(electrode_1020, '#333333')
            # Convert hex to RGBA tuple for matplotlib
            r, g, b = int(color_hex[1:3], 16)/255, int(color_hex[3:5], 16)/255, int(color_hex[5:7], 16)/255
            plot_colors.append((r, g, b, 0.9))
        
        if plot_data_list:
            plot_data = np.array(plot_data_list)
        else:
            # Fallback if no channels matched
            plot_data = np.zeros((1, len(time_array)))
            plot_names = ['No match']
            plot_colors = [(0.5, 0.5, 0.5, 1.0)]
        
        return plot_data, plot_names, plot_colors
    
    def _update_plot(self):
        """Update the matplotlib plot.
        
        FIXED: Uses consistent amplitude scaling independent of channel count.
        Amplitude scale directly multiplies the signal values in µV.
        """
        if self.eeg_signal is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        eeg = self.eeg_signal
        
        # Get data for current epoch if 3D
        if eeg.data.ndim == 3:
            data = eeg.data[self.selected_epoch]
            total_channels = eeg.data.shape[1]
        else:
            data = eeg.data
            total_channels = eeg.data.shape[0]
        
        n_channels_total, n_samples = data.shape
        sfreq = eeg.sampling_rate
        
        # Get visible channels (or all if not set)
        if not self.visible_channels:
            self.visible_channels = list(range(total_channels))
        
        # Filter to valid indices
        visible_indices = [i for i in self.visible_channels if 0 <= i < total_channels]
        if not visible_indices:
            visible_indices = [0]  # Always show at least one channel
        
        n_visible = len(visible_indices)
        
        # Calculate sample range for current view
        start_sample = int(self.current_time_start * sfreq)
        end_sample = int((self.current_time_start + self.time_window) * sfreq)
        end_sample = min(end_sample, n_samples)
        
        if start_sample >= end_sample:
            start_sample = max(0, end_sample - int(sfreq))
        
        n_view_samples = end_sample - start_sample
        
        # OPTIMIZATION: Downsample if too many points
        # Target ~2000 points per channel for smooth display without lag
        max_points = 2000
        downsample_factor = 1
        if n_view_samples > max_points:
            downsample_factor = n_view_samples // max_points
        
        # Time axis (downsampled)
        if downsample_factor > 1:
            time = np.arange(start_sample, end_sample, downsample_factor) / sfreq
        else:
            time = np.arange(start_sample, end_sample) / sfreq
        
        # Get channel names - use actual metadata names
        channel_names = eeg.channel_names if eeg.channel_names else [f"Ch{i+1}" for i in range(total_channels)]
        
        # FIXED: Use constant channel spacing that doesn't depend on data amplitude
        channel_offset = self.base_channel_spacing * self.channel_spacing
        
        # --- 10-20 CONVERSION ---
        # If enabled, average EGI channels into 10-20 positions
        if self.convert_to_1020 and is_egi_format(channel_names):
            plot_data, plot_names, plot_colors = self._convert_data_to_1020(
                data, channel_names, start_sample, end_sample, sfreq, downsample_factor, time)
        else:
            # Standard mode: extract visible channels
            plot_names = []
            for ch_idx in visible_indices:
                plot_names.append(channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx+1}")
            
            if downsample_factor > 1:
                plot_data = np.zeros((n_visible, len(time)))
                for plot_idx, ch_idx in enumerate(visible_indices):
                    ch_raw = data[ch_idx, start_sample:end_sample]
                    n_chunks = len(ch_raw) // downsample_factor
                    if n_chunks > 0:
                        reshaped = ch_raw[:n_chunks * downsample_factor].reshape(n_chunks, downsample_factor)
                        plot_data[plot_idx, :n_chunks] = np.mean(reshaped, axis=1)
            else:
                plot_data = data[visible_indices, start_sample:end_sample].copy()
            
            plot_colors = [self.channel_colors[ch_idx % len(self.channel_colors)] for ch_idx in visible_indices]
        
        n_plot = len(plot_names)
        
        # --- HIGHPASS FILTER ---
        # Apply to display data (doesn't modify original signal)
        if self.highpass_enabled and self.highpass_freq > 0 and sfreq > 0:
            plot_data = self._apply_highpass(plot_data, sfreq)
        
        # Plot each channel
        y_positions = []
        y_labels = []
        
        for plot_idx in range(n_plot):
            center_y = (n_plot - 1 - plot_idx) * channel_offset
            y_positions.append(center_y)
            y_labels.append(plot_names[plot_idx])
            
            ch_data = plot_data[plot_idx].copy()
            ch_data = ch_data - np.mean(ch_data)  # Center around zero
            ch_data = ch_data * self.amplitude_scale  # Scale amplitude
            ch_data = ch_data + center_y  # Offset to channel position
            
            color = plot_colors[plot_idx] if plot_idx < len(plot_colors) else self.channel_colors[plot_idx % len(self.channel_colors)]
            ax.plot(time[:len(ch_data)], ch_data, linewidth=0.5, color=color, alpha=0.9, 
                   solid_capstyle='butt', solid_joinstyle='miter')
        
        # Set axis limits - fixed based on channel positions, not amplitude
        y_margin = channel_offset * 0.5
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-y_margin, (n_plot - 1) * channel_offset + y_margin)
        
        # Y-axis: channel labels at center positions
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        
        # X-axis
        ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
        
        # Title with metadata
        title_parts = []
        if eeg.metadata:
            if 'filename' in eeg.metadata:
                title_parts.append(eeg.metadata['filename'])
            if 'processing_step' in eeg.metadata:
                title_parts.append(f"Step: {eeg.metadata['processing_step']}")
            if 'original_format' in eeg.metadata:
                title_parts.append(f"Format: {eeg.metadata['original_format']}")
        
        # Add channel/mode info to title
        if self.convert_to_1020 and is_egi_format(channel_names):
            title_parts.append(f"10-20 Format ({n_plot} channels)")
        elif n_visible < total_channels:
            title_parts.append(f"Showing {n_visible}/{total_channels} channels")
        
        if self.highpass_enabled:
            title_parts.append(f"HP: {self.highpass_freq:.2f} Hz")
        
        data_type_str = eeg.data_type.value if eeg.data_type else ""
        title = f"EEG Data ({data_type_str})"
        if title_parts:
            title += f" - {', '.join(title_parts)}"
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Grid
        if self.show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', which='major')
            ax.grid(True, alpha=0.15, linestyle=':', which='minor')
            ax.minorticks_on()
        
        # Add scale bar showing actual µV value (use channel_offset for reference)
        self._add_scale_bar(ax, channel_offset)
        
        # Draw canvas (constrained_layout handles spacing automatically)
        self.canvas.draw()

        # Update time label
        self.time_label.setText(f"{self.current_time_start:.2f} - {self.current_time_start + self.time_window:.2f} s")

        # Notify linked windows (e.g. topo heatmap) that the view has changed
        self.view_changed.emit()
    
    def _add_scale_bar(self, ax, channel_offset):
        """Add amplitude scale bar to the plot showing actual µV values.
        
        FIXED: With consistent channel spacing, the scale bar correctly represents
        µV values based on the amplitude_scale.
        
        Args:
            ax: matplotlib axis
            channel_offset: the visual spacing between channels in µV units
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Scale bar position (bottom right)
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.02
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.1
        
        # With fixed channel spacing:
        # - Signal values are multiplied by amplitude_scale for display
        # - So a displayed height H corresponds to H / amplitude_scale µV
        # - We want a scale bar that's about 20% of channel spacing visually
        
        target_visual_height = channel_offset * 0.3  # Visual height on plot
        
        # What µV does this represent in the original data?
        actual_uv = target_visual_height / self.amplitude_scale
        
        # Round to a nice number
        if actual_uv > 0:
            magnitude = 10 ** np.floor(np.log10(max(actual_uv, 0.001)))
        else:
            magnitude = 1
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        scale_uv = magnitude
        for nv in nice_values:
            if nv * magnitude >= actual_uv * 0.5:
                scale_uv = nv * magnitude
                break
        
        # Calculate visual height of the scale bar
        bar_height = scale_uv * self.amplitude_scale
        
        # Draw scale bar
        ax.plot([x_pos, x_pos], [y_pos, y_pos + bar_height], 
                color='black', linewidth=2, solid_capstyle='butt')
        
        # Format label
        if scale_uv >= 1:
            label = f"{int(scale_uv)} µV"
        else:
            label = f"{scale_uv:.1f} µV"
            
        ax.text(x_pos - (xlim[1] - xlim[0]) * 0.01, y_pos + bar_height / 2,
                label, fontsize=8, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    def _update_plot_fast(self):
        """Fast plot update for auto-scroll - updates data without full redraw.
        
        OPTIMIZATION: Uses blitting-like approach - only updates line data
        instead of clearing and redrawing everything. Falls back to _update_plot
        if cached state is invalid or when 10-20 conversion is active.
        """
        if self.eeg_signal is None:
            return
        
        # Fall back to full redraw when 10-20 conversion is active
        # (different channel count/layout than raw data)
        eeg = self.eeg_signal
        channel_names = eeg.channel_names if eeg.channel_names else []
        if self.convert_to_1020 and is_egi_format(channel_names):
            self._update_plot()
            return
        
        # Check if we have a valid cached plot state
        ax = self.figure.axes[0] if self.figure.axes else None
        if ax is None or not ax.lines:
            self._update_plot()
            return
        
        # Get data for current epoch if 3D
        if eeg.data.ndim == 3:
            data = eeg.data[self.selected_epoch]
            total_channels = eeg.data.shape[1]
        else:
            data = eeg.data
            total_channels = eeg.data.shape[0]
        
        n_channels_total, n_samples = data.shape
        sfreq = eeg.sampling_rate
        
        # Get visible channels
        if not self.visible_channels:
            self.visible_channels = list(range(total_channels))
        visible_indices = [i for i in self.visible_channels if 0 <= i < total_channels]
        if not visible_indices:
            visible_indices = [0]
        n_visible = len(visible_indices)
        
        # Check if channel count changed - need full redraw
        if len(ax.lines) != n_visible:
            self._update_plot()
            return
        
        # Calculate sample range
        start_sample = int(self.current_time_start * sfreq)
        end_sample = int((self.current_time_start + self.time_window) * sfreq)
        end_sample = min(end_sample, n_samples)
        if start_sample >= end_sample:
            start_sample = max(0, end_sample - int(sfreq))
        
        n_view_samples = end_sample - start_sample
        
        # Downsample
        max_points = 2000
        downsample_factor = 1
        if n_view_samples > max_points:
            downsample_factor = n_view_samples // max_points
        
        # Time axis
        if downsample_factor > 1:
            time = np.arange(start_sample, end_sample, downsample_factor) / sfreq
        else:
            time = np.arange(start_sample, end_sample) / sfreq
        
        # FIXED: Use consistent fixed channel spacing
        channel_offset = self.base_channel_spacing * self.channel_spacing
        
        # Extract all visible channel data
        if downsample_factor > 1:
            all_ch_data = np.zeros((n_visible, len(time)))
            for plot_idx, ch_idx in enumerate(visible_indices):
                ch_raw = data[ch_idx, start_sample:end_sample]
                n_chunks = len(ch_raw) // downsample_factor
                if n_chunks > 0:
                    reshaped = ch_raw[:n_chunks * downsample_factor].reshape(n_chunks, downsample_factor)
                    all_ch_data[plot_idx, :n_chunks] = np.mean(reshaped, axis=1)
        else:
            all_ch_data = data[visible_indices, start_sample:end_sample].copy()
        
        # Apply highpass filter if enabled
        if self.highpass_enabled and self.highpass_freq > 0 and sfreq > 0:
            all_ch_data = self._apply_highpass(all_ch_data, sfreq)
        
        # Update each line's data
        for plot_idx in range(n_visible):
            center_y = (n_visible - 1 - plot_idx) * channel_offset
            
            ch_data = all_ch_data[plot_idx].copy()
            ch_data = ch_data - np.mean(ch_data)
            ch_data = ch_data * self.amplitude_scale + center_y
            
            line = ax.lines[plot_idx]
            line.set_xdata(time[:len(ch_data)])
            line.set_ydata(ch_data)
        
        # Update axis limits
        y_margin = channel_offset * 0.5
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-y_margin, (n_visible - 1) * channel_offset + y_margin)
        
        # Use draw_idle for non-blocking render
        self.canvas.draw_idle()

        # Update time label
        self.time_label.setText(f"{self.current_time_start:.2f} - {self.current_time_start + self.time_window:.2f} s")

        # Notify linked windows (topo heatmap, etc.)
        self.view_changed.emit()

    # ========================
    # Event Handlers
    # ========================
    
    def _on_scroll_changed(self, value):
        """Handle scrollbar value change."""
        self.current_time_start = value / 10.0  # Convert from 0.1s steps
        self._update_plot()
    
    def _on_time_window_changed(self, value):
        """Handle time window change."""
        self.time_window = value
        self._update_scrollbar()
        self._update_plot()
    
    def _on_amplitude_changed(self, value):
        """Handle amplitude scale change."""
        self.amplitude_scale = value
        self._update_plot()
    
    def _on_spacing_changed(self, value):
        """Handle channel spacing change."""
        self.channel_spacing = value
        self._update_plot()
    
    def _on_epoch_changed(self, value):
        """Handle epoch selection change."""
        self.selected_epoch = value
        self._update_channel_table()
        self._update_plot()
    
    def _on_grid_changed(self, state):
        """Handle grid toggle."""
        self.show_grid = (state == Qt.Checked)
        self._update_plot()
    
    def _on_scroll_speed_changed(self, value):
        """Handle auto-scroll speed change."""
        self.auto_scroll_speed = value
    
    def _on_convert_1020_changed(self, state):
        """Handle 10-20 format conversion toggle.
        
        When enabled, converts EGI channels to 10-20 format by averaging
        the corresponding EGI electrodes for each 10-20 position.
        Also updates the channel information table to reflect the conversion.
        """
        self.convert_to_1020 = (state == Qt.Checked)
        
        if self.eeg_signal is not None:
            if self.convert_to_1020:
                # Check if this looks like EGI format
                channel_names = self.eeg_signal.channel_names or []
                egi_count = sum(1 for ch in channel_names 
                               if str(ch).startswith('E') and str(ch)[1:].isdigit())
                
                if egi_count < len(channel_names) * 0.3:
                    self.status_bar.showMessage(
                        "Warning: Data doesn't appear to be EGI format", 5000)
                else:
                    self.status_bar.showMessage(
                        f"Converting {egi_count} EGI channels to 10-20 format", 3000)
            
            self._update_channel_table()
            self._update_plot()
    
    def _on_highpass_changed(self, state):
        """Handle highpass filter toggle."""
        self.highpass_enabled = (state == Qt.Checked)
        self._highpass_cache = {}  # Clear cache
        self._highpass_cache_key = None
        if self.eeg_signal is not None:
            if self.highpass_enabled:
                self.status_bar.showMessage(
                    f"Highpass filter enabled: {self.highpass_freq:.2f} Hz", 3000)
            else:
                self.status_bar.showMessage("Highpass filter disabled", 2000)
            self._update_plot()
    
    def _on_highpass_freq_changed(self, value):
        """Handle highpass frequency change."""
        self.highpass_freq = value
        self._highpass_cache = {}  # Clear cache when frequency changes
        self._highpass_cache_key = None
        if self.highpass_enabled and self.eeg_signal is not None:
            self.status_bar.showMessage(f"Highpass: {value:.2f} Hz", 2000)
            self._update_plot()
    
    def _toggle_auto_scroll(self, checked):
        """Toggle auto-scroll on/off."""
        if checked:
            self._start_auto_scroll()
        else:
            self._stop_auto_scroll()
    
    def _start_auto_scroll(self):
        """Start auto-scrolling."""
        if self.eeg_signal is None:
            self.auto_scroll_btn.setChecked(False)
            return
        
        self.auto_scroll_active = True
        self.auto_scroll_btn.setText("⏸ Pause")
        self._last_scroll_time = None  # Reset timing
        self.auto_scroll_timer.start(self.auto_scroll_interval)
    
    def _stop_auto_scroll(self):
        """Stop auto-scrolling."""
        self.auto_scroll_active = False
        self.auto_scroll_btn.setText("▶ Play")
        self.auto_scroll_btn.setChecked(False)
        self.auto_scroll_timer.stop()
        self._last_scroll_time = None
    
    def _auto_scroll_step(self):
        """Perform one step of auto-scrolling.
        
        FIXED: Uses real elapsed time for frame-rate independent scrolling.
        This ensures consistent speed regardless of rendering time.
        """
        import time
        
        if self.eeg_signal is None:
            self._stop_auto_scroll()
            return
        
        # Calculate actual elapsed time since last frame
        current_time = time.time()
        if self._last_scroll_time is None:
            elapsed_sec = self.auto_scroll_interval / 1000.0
        else:
            elapsed_sec = current_time - self._last_scroll_time
        self._last_scroll_time = current_time
        
        # Time increment = speed * actual elapsed time
        time_increment = self.auto_scroll_speed * elapsed_sec
        
        # Get max time
        if self.eeg_signal.data.ndim == 3:
            n_samples = self.eeg_signal.data.shape[2]
        else:
            n_samples = self.eeg_signal.data.shape[1]
        total_duration = n_samples / self.eeg_signal.sampling_rate
        max_start = total_duration - self.time_window
        
        # Update position
        new_start = self.current_time_start + time_increment
        
        if new_start >= max_start:
            # Reached end, stop auto-scroll
            self.current_time_start = max(0, max_start)
            self._stop_auto_scroll()
        else:
            self.current_time_start = new_start
        
        # Update scrollbar (without triggering its event)
        self.time_scrollbar.blockSignals(True)
        self.time_scrollbar.setValue(int(self.current_time_start * 10))
        self.time_scrollbar.blockSignals(False)
        
        # OPTIMIZATION: Use draw_idle for non-blocking render during scroll
        self._update_plot_fast()
    
    def keyPressEvent(self, event):
        """Handle keyboard navigation."""
        if self.eeg_signal is None:
            super().keyPressEvent(event)
            return
        
        # Calculate step size (10% of time window, or 1 second minimum)
        step = max(1.0, self.time_window * 0.1)
        
        # Get max time
        if self.eeg_signal.data.ndim == 3:
            n_samples = self.eeg_signal.data.shape[2]
        else:
            n_samples = self.eeg_signal.data.shape[1]
        total_duration = n_samples / self.eeg_signal.sampling_rate
        max_start = total_duration - self.time_window
        
        if event.key() == Qt.Key_Left:
            # Move backward in time
            new_start = max(0, self.current_time_start - step)
            self.current_time_start = new_start
            self.time_scrollbar.blockSignals(True)
            self.time_scrollbar.setValue(int(new_start * 10))
            self.time_scrollbar.blockSignals(False)
            self._update_plot()
            
        elif event.key() == Qt.Key_Right:
            # Move forward in time
            new_start = min(max_start, self.current_time_start + step)
            self.current_time_start = max(0, new_start)
            self.time_scrollbar.blockSignals(True)
            self.time_scrollbar.setValue(int(self.current_time_start * 10))
            self.time_scrollbar.blockSignals(False)
            self._update_plot()
            
        elif event.key() == Qt.Key_Home:
            # Go to start
            self._go_to_start()
            
        elif event.key() == Qt.Key_End:
            # Go to end
            self._go_to_end()
            
        elif event.key() == Qt.Key_Space:
            # Toggle auto-scroll
            self.auto_scroll_btn.toggle()
            self._toggle_auto_scroll(self.auto_scroll_btn.isChecked())
            
        elif event.key() == Qt.Key_Up:
            # Increase amplitude
            self._adjust_amplitude(1.5)
            
        elif event.key() == Qt.Key_Down:
            # Decrease amplitude
            self._adjust_amplitude(0.67)
            
        else:
            super().keyPressEvent(event)
    
    def _set_time_window(self, seconds):
        """Set time window to specific value."""
        self.time_window_spin.setValue(seconds)
    
    def _adjust_amplitude(self, factor):
        """Adjust amplitude by factor."""
        new_value = self.amplitude_spin.value() * factor
        self.amplitude_spin.setValue(new_value)
    
    def _zoom_in(self):
        """Zoom in (reduce time window)."""
        self.time_window_spin.setValue(self.time_window / 2)
    
    def _zoom_out(self):
        """Zoom out (increase time window)."""
        self.time_window_spin.setValue(self.time_window * 2)
    
    def _go_to_start(self):
        """Go to start of recording."""
        self.time_scrollbar.setValue(0)
    
    def _go_to_end(self):
        """Go to end of recording."""
        self.time_scrollbar.setValue(self.time_scrollbar.maximum())
    
    def _auto_scale_amplitude(self):
        """Automatically scale amplitude to show signals clearly.
        
        FIXED: With constant channel spacing, amplitude_scale directly controls
        how many µV fit in the channel spacing. We want signals to fill ~60%
        of the channel spacing for clear visibility.
        """
        if self.eeg_signal is None:
            return
        
        eeg = self.eeg_signal
        
        # Get current view data
        if eeg.data.ndim == 3:
            data = eeg.data[self.selected_epoch]
        else:
            data = eeg.data
        
        sfreq = eeg.sampling_rate
        start_sample = int(self.current_time_start * sfreq)
        end_sample = int((self.current_time_start + self.time_window) * sfreq)
        end_sample = min(end_sample, data.shape[1])
        
        if start_sample >= end_sample:
            start_sample = max(0, end_sample - int(sfreq))
        
        # Get visible channels
        visible_indices = [i for i in self.visible_channels if 0 <= i < data.shape[0]]
        if not visible_indices:
            visible_indices = list(range(min(data.shape[0], 16)))
        
        # Calculate typical signal range across visible channels
        visible_data = data[visible_indices, start_sample:end_sample]
        
        if visible_data.size > 0:
            # Center each channel
            centered_data = visible_data - np.mean(visible_data, axis=1, keepdims=True)
            
            # Get peak-to-peak range (95th percentile to ignore artifacts)
            q95 = np.percentile(np.abs(centered_data), 95)
            
            # Also get median for reference
            median_amp = np.median(np.abs(centered_data))
            
            # Use peak amplitude for scaling (larger of q95 or median*3)
            signal_amplitude = max(q95, median_amp * 3)
            
            if signal_amplitude > 0:
                # With fixed channel spacing of base_channel_spacing µV,
                # we want signal_amplitude * amplitude_scale to be ~60% of spacing
                target_visual = self.base_channel_spacing * self.channel_spacing * 0.6
                
                optimal_scale = target_visual / signal_amplitude
                
                # Clamp to reasonable range (0.1x to 200x)
                optimal_scale = max(0.1, min(200.0, optimal_scale))
                
                self.amplitude_scale = optimal_scale
                self.amplitude_spin.blockSignals(True)
                self.amplitude_spin.setValue(optimal_scale)
                self.amplitude_spin.blockSignals(False)
                
                self.status_bar.showMessage(
                    f"Auto-scaled: {optimal_scale:.1f}x (signal ~{signal_amplitude:.1f}µV)")
        
        self._update_plot()
    
    def _reset_view(self):
        """Reset all view parameters to defaults and auto-scale."""
        self.current_time_start = 0.0
        self.time_window = 10.0
        self.channel_spacing = 1.0
        
        # Block signals to prevent cascading updates
        self.time_window_spin.blockSignals(True)
        self.spacing_spin.blockSignals(True)
        self.time_scrollbar.blockSignals(True)
        
        self.time_window_spin.setValue(10.0)
        self.spacing_spin.setValue(1.0)
        self.time_scrollbar.setValue(0)
        
        self.time_window_spin.blockSignals(False)
        self.spacing_spin.blockSignals(False)
        self.time_scrollbar.blockSignals(False)
        
        # Reset to show all channels (or first 32 if too many)
        if self.eeg_signal is not None:
            if self.eeg_signal.data.ndim == 3:
                n_channels = self.eeg_signal.data.shape[1]
            else:
                n_channels = self.eeg_signal.data.shape[0]
            
            if n_channels > 64:
                self.visible_channels = list(range(32))
            else:
                self.visible_channels = list(range(n_channels))
            
            # Auto-scale amplitude for best view
            self._auto_scale_amplitude()
        else:
            self.amplitude_scale = 1.0
            self.amplitude_spin.setValue(1.0)
            self._update_plot()
    
    def _open_channel_selector(self):
        """Open dialog to select which channels to display.
        
        When 10-20 conversion is active, shows a simplified selector
        for the converted 10-20 electrodes instead of raw EGI channels.
        """
        if self.eeg_signal is None:
            return
        
        eeg = self.eeg_signal
        channel_names = eeg.channel_names or []
        
        # If 10-20 mode is active with EGI data, show 10-20 channel selector
        if self.convert_to_1020 and is_egi_format(channel_names):
            self._open_1020_channel_selector()
            return
        
        if eeg.data.ndim == 3:
            n_channels = eeg.data.shape[1]
        else:
            n_channels = eeg.data.shape[0]
        
        channel_names = eeg.channel_names or [f"Ch{i+1}" for i in range(n_channels)]
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Channels to Display")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(dialog)
        
        # Info label
        info_label = QLabel(f"Select channels to display ({len(self.visible_channels)}/{n_channels} currently visible)")
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # Quick selection buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_first_btn = QPushButton("First 16")
        select_every_nth_btn = QPushButton("Every Nth...")
        
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addWidget(select_first_btn)
        btn_layout.addWidget(select_every_nth_btn)
        layout.addLayout(btn_layout)
        
        # Scroll area with checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        checkboxes = []
        n_cols = 4  # Number of columns
        
        for i in range(n_channels):
            cb = QCheckBox(f"{i}: {channel_names[i]}")
            cb.setChecked(i in self.visible_channels)
            checkboxes.append(cb)
            row = i // n_cols
            col = i % n_cols
            scroll_layout.addWidget(cb, row, col)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Connect quick selection buttons
        def select_all():
            for cb in checkboxes:
                cb.setChecked(True)
            info_label.setText(f"Select channels to display ({n_channels}/{n_channels} currently visible)")
        
        def select_none():
            for cb in checkboxes:
                cb.setChecked(False)
            info_label.setText(f"Select channels to display (0/{n_channels} currently visible)")
        
        def select_first_n():
            for i, cb in enumerate(checkboxes):
                cb.setChecked(i < 16)
            info_label.setText(f"Select channels to display ({min(16, n_channels)}/{n_channels} currently visible)")
        
        def select_every_nth():
            n, ok = QInputDialog.getInt(dialog, "Every Nth Channel", 
                                        "Show every Nth channel:", 
                                        value=4, min=2, max=n_channels)
            if ok:
                for i, cb in enumerate(checkboxes):
                    cb.setChecked(i % n == 0)
                selected_count = len([i for i in range(n_channels) if i % n == 0])
                info_label.setText(f"Select channels to display ({selected_count}/{n_channels} currently visible)")
        
        select_all_btn.clicked.connect(select_all)
        select_none_btn.clicked.connect(select_none)
        select_first_btn.clicked.connect(select_first_n)
        select_every_nth_btn.clicked.connect(select_every_nth)
        
        # Update count when checkboxes change
        def update_count():
            count = sum(1 for cb in checkboxes if cb.isChecked())
            info_label.setText(f"Select channels to display ({count}/{n_channels} currently visible)")
        
        for cb in checkboxes:
            cb.stateChanged.connect(update_count)
        
        # Range selection helper
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        range_start = QSpinBox()
        range_start.setRange(0, n_channels - 1)
        range_layout.addWidget(range_start)
        range_layout.addWidget(QLabel("to"))
        range_end = QSpinBox()
        range_end.setRange(0, n_channels - 1)
        range_end.setValue(min(15, n_channels - 1))
        range_layout.addWidget(range_end)
        range_btn = QPushButton("Select Range")
        range_layout.addWidget(range_btn)
        range_layout.addStretch()
        
        def select_range():
            start = range_start.value()
            end = range_end.value()
            for i, cb in enumerate(checkboxes):
                cb.setChecked(start <= i <= end)
            update_count()
        
        range_btn.clicked.connect(select_range)
        layout.addLayout(range_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Update visible channels
            self.visible_channels = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
            
            if not self.visible_channels:
                # At least show one channel
                self.visible_channels = [0]
            
            # Update display
            self._update_plot()
            self._update_channel_table()
            
            # Update status
            self.status_bar.showMessage(
                f"Displaying {len(self.visible_channels)} of {n_channels} channels"
            )
    
    def _open_1020_channel_selector(self):
        """Open a channel selector for 10-20 converted electrodes."""
        channel_names = self.eeg_signal.channel_names or []
        
        # Build name->index lookup to find which 10-20 electrodes are available
        name_to_idx = {}
        for idx, name in enumerate(channel_names):
            name_to_idx[str(name)] = idx
            name_to_idx[str(name).upper()] = idx
        
        standard_order = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FT9', 'FT10',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'Oz', 'O2',
        ]
        
        # Find available 10-20 electrodes
        available = []
        for electrode in standard_order:
            if electrode not in EGI_TO_1020_MAPPING:
                continue
            egi_channels = EGI_TO_1020_MAPPING[electrode]
            has_match = any(
                egi_ch in name_to_idx or egi_ch.upper() in name_to_idx
                for egi_ch in egi_channels
            )
            if has_match:
                available.append(electrode)
        
        if not available:
            self.status_bar.showMessage("No matching 10-20 channels found", 3000)
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select 10-20 Channels to Display")
        dialog.setMinimumWidth(350)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        n_available = len(available)
        visible_set = set(self.visible_1020_channels) if self.visible_1020_channels else set(available)
        info_label = QLabel(f"Select 10-20 channels ({len(visible_set)}/{n_available} visible)")
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # Quick buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        layout.addLayout(btn_layout)
        
        # Region group buttons
        region_layout = QHBoxLayout()
        for region_name, region_chs in [
            ('Frontal', ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']),
            ('Central', ['C3', 'Cz', 'C4']),
            ('Temporal', ['FT9', 'FT10', 'T7', 'T8', 'TP9', 'TP10']),
            ('Parietal', ['P7', 'P3', 'Pz', 'P4', 'P8']),
            ('Occipital', ['O1', 'Oz', 'O2']),
        ]:
            btn = QPushButton(region_name)
            btn.setMaximumWidth(70)
            region_layout.addWidget(btn)
        layout.addLayout(region_layout)
        
        # Checkboxes for each 10-20 electrode
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        checkboxes = {}
        n_cols = 4
        for i, electrode in enumerate(available):
            egi_chs = EGI_TO_1020_MAPPING[electrode]
            src_str = '+'.join(egi_chs[:2])
            if len(egi_chs) > 2:
                src_str += f'+{len(egi_chs)-2}more'
            cb = QCheckBox(f"{electrode} ({src_str})")
            cb.setChecked(electrode in visible_set)
            checkboxes[electrode] = cb
            scroll_layout.addWidget(cb, i // n_cols, i % n_cols)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Connect quick buttons
        def select_all():
            for cb in checkboxes.values():
                cb.setChecked(True)
        
        def select_none():
            for cb in checkboxes.values():
                cb.setChecked(False)
        
        select_all_btn.clicked.connect(select_all)
        select_none_btn.clicked.connect(select_none)
        
        # Connect region buttons
        region_defs = [
            ('Frontal', ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']),
            ('Central', ['C3', 'Cz', 'C4']),
            ('Temporal', ['FT9', 'FT10', 'T7', 'T8', 'TP9', 'TP10']),
            ('Parietal', ['P7', 'P3', 'Pz', 'P4', 'P8']),
            ('Occipital', ['O1', 'Oz', 'O2']),
        ]
        for idx, (region_name, region_chs) in enumerate(region_defs):
            btn = region_layout.itemAt(idx).widget()
            def make_toggle(chs=region_chs):
                def toggle():
                    # Toggle: if all selected, deselect; otherwise select all
                    all_checked = all(checkboxes[ch].isChecked() for ch in chs if ch in checkboxes)
                    for ch in chs:
                        if ch in checkboxes:
                            checkboxes[ch].setChecked(not all_checked)
                return toggle
            btn.clicked.connect(make_toggle())
        
        # Update count
        def update_count():
            count = sum(1 for cb in checkboxes.values() if cb.isChecked())
            info_label.setText(f"Select 10-20 channels ({count}/{n_available} visible)")
        
        for cb in checkboxes.values():
            cb.stateChanged.connect(update_count)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            selected = [name for name, cb in checkboxes.items() if cb.isChecked()]
            if not selected:
                selected = available[:1]  # At least one channel
            
            self.visible_1020_channels = selected
            self._update_plot()
            self._update_channel_table()
            self.status_bar.showMessage(
                f"Displaying {len(selected)} of {n_available} 10-20 channels"
            )

    def _open_file(self):
        """Open a file dialog and load an EEG file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import file_loaders
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG File", "",
            "EEG Files (*.edf *.bdf *.mff *.nsx *.ns2 *.ncs *.rhd *.dat);;All Files (*)",
            options=options
        )
        
        if not file_path:
            return
            
        fp = Path(file_path)
        self.status_bar.showMessage(f"Loading {fp.name}...")
        QApplication.processEvents()
        
        loaders = [
            file_loaders.EDFLoader(),
            file_loaders.MFFLoader(),
            file_loaders.NSXLoader(),
            file_loaders.NCSLoader(),
            file_loaders.RHDLoader(),
            file_loaders.DATLoader()
        ]
        
        for loader in loaders:
            if loader.can_load(fp):
                try:
                    result = loader.load(fp)
                    if isinstance(result, tuple):
                        signal, success, err = result
                        if not success:
                            QMessageBox.critical(self, "Error", f"Failed to load file:\n{err}")
                            self.status_bar.showMessage("Error loading file")
                            return
                        self.set_eeg_signal(signal, title=fp.name)
                    else:
                        self.set_eeg_signal(result, title=fp.name)
                    return
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error while loading {fp.name}:\n{str(e)}")
                    self.status_bar.showMessage("Error loading file")
                    return
                    
        QMessageBox.warning(self, "Unsupported Format", f"No loader available for format: {fp.suffix}")
        self.status_bar.showMessage("Unsupported file format")

    def _export_plot(self):
        """Export the current plot to an image file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        if self.eeg_signal is None:
            QMessageBox.warning(self, "No Data", "There is no data to export.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "eeg_trace_export.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf);;SVG Vector (*.svg)",
            options=options
        )
        
        if file_path:
            try:
                # Save the matplotlib figure to the given file
                self.figure.savefig(file_path, bbox_inches='tight', dpi=300)
                self.status_bar.showMessage(f"Successfully exported plot to {file_path}", 5000)
                QMessageBox.information(self, "Export Successful", f"Plot successfully saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save plot:\n{str(e)}")
                self.status_bar.showMessage("Export failed")

class SignalInfoWindow(QWidget):
    """Standalone window that shows signal information (metadata) for an EEGSignal."""

    def __init__(self, eeg_signal: EEGSignal, parent=None):
        super().__init__(parent)
        self.eeg_signal = eeg_signal
        self.setWindowTitle("Signal Information")
        self.setMinimumSize(500, 500)
        self.resize(600, 600)
        # Make it a real independent window
        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("📋 Signal Information")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        # Basic info
        info_group = QGroupBox("Basic Information")
        info_layout = QVBoxLayout(info_group)
        info_table = QTableWidget()
        info_table.setColumnCount(2)
        info_table.setHorizontalHeaderLabels(["Property", "Value"])
        info_table.horizontalHeader().setStretchLastSection(True)
        info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        info_table.setAlternatingRowColors(True)

        eeg = eeg_signal
        if eeg.data.ndim == 3:
            n_epochs, n_channels, n_samples = eeg.data.shape
            shape_str = f"{n_epochs} epochs × {n_channels} ch × {n_samples} samples"
        else:
            n_channels, n_samples = eeg.data.shape
            shape_str = f"{n_channels} ch × {n_samples} samples"

        basic_info = [
            ("Data Type", eeg.data_type.value if eeg.data_type else "N/A"),
            ("Shape", shape_str),
            ("Sampling Rate", f"{eeg.sampling_rate:.1f} Hz"),
            ("Duration", f"{n_samples / eeg.sampling_rate:.2f} s"),
            ("Channels", str(n_channels)),
            ("Data dtype", str(eeg.data.dtype)),
        ]
        info_table.setRowCount(len(basic_info))
        for i, (k, v) in enumerate(basic_info):
            info_table.setItem(i, 0, QTableWidgetItem(k))
            info_table.setItem(i, 1, QTableWidgetItem(str(v)))
        info_layout.addWidget(info_table)
        layout.addWidget(info_group)

        # Metadata
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_table = QTableWidget()
        meta_table.setColumnCount(2)
        meta_table.setHorizontalHeaderLabels(["Key", "Value"])
        meta_table.horizontalHeader().setStretchLastSection(True)
        meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        meta_table.setEditTriggers(QTableWidget.NoEditTriggers)
        meta_table.setAlternatingRowColors(True)
        metadata = eeg.metadata or {}
        meta_table.setRowCount(len(metadata))
        for i, (k, v) in enumerate(sorted(metadata.items())):
            meta_table.setItem(i, 0, QTableWidgetItem(str(k)))
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            meta_table.setItem(i, 1, QTableWidgetItem(val_str))
        meta_layout.addWidget(meta_table)
        layout.addWidget(meta_group)

        # Channels
        ch_group = QGroupBox("Channels")
        ch_layout = QVBoxLayout(ch_group)
        ch_table = QTableWidget()
        ch_table.setColumnCount(3)
        ch_table.setHorizontalHeaderLabels(["#", "Name", "Range (µV)"])
        ch_table.horizontalHeader().setStretchLastSection(True)
        ch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ch_table.setAlternatingRowColors(True)
        data = eeg.data[0] if eeg.data.ndim == 3 else eeg.data
        ch_names = eeg.channel_names or [f"Ch{i+1}" for i in range(data.shape[0])]
        ch_table.setRowCount(min(data.shape[0], 128))
        for i in range(min(data.shape[0], 128)):
            ch_min, ch_max = np.min(data[i]), np.max(data[i])
            ch_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            ch_table.setItem(i, 1, QTableWidgetItem(ch_names[i] if i < len(ch_names) else f"Ch{i+1}"))
            ch_table.setItem(i, 2, QTableWidgetItem(f"{ch_min:.1f} to {ch_max:.1f}"))
        ch_layout.addWidget(ch_table)
        layout.addWidget(ch_group)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


# ---------------------------------------------------------------------------
# Shared single-channel control mixin
# ---------------------------------------------------------------------------

class _SingleChannelControlsMixin:
    """
    Mixin that provides a re-usable single-channel selector + playback/scale
    controls panel, identical in behaviour to the main EEGViewerWidget controls.

    Subclasses must call  _build_shared_controls(eeg_signal)  and will receive:
        self._eeg         – the EEGSignal
        self._ch_idx      – currently selected channel index (int)
        self._time_start  – current view-start time (float, seconds)
        self._time_win    – visible time window (float, seconds)
        self._amp_scale   – amplitude scale multiplier
        self._hp_enabled  – highpass filter enabled (bool)
        self._hp_freq     – highpass cut-off (Hz)
        self._auto_timer  – QTimer for auto-scroll
        self._scroll_speed– playback speed (x real-time)
    And will have connected slot  _on_controls_changed()  that subclasses must implement.
    """

    def _build_shared_controls(self, eeg_signal: EEGSignal, include_scrollbar: bool = True) -> QWidget:
        """Build and return the shared controls widget. Stores state on self."""
        self._eeg = eeg_signal
        self._ch_idx = 0
        self._time_start = 0.0
        self._time_win = 10.0
        self._amp_scale = 1.0
        self._hp_enabled = False
        self._hp_freq = 0.5
        self._scroll_speed = 20.0
        self._auto_scroll_active = False
        self._last_scroll_time_mono = None
        self._base_spacing = 100.0

        outer = QWidget()
        vbox = QVBoxLayout(outer)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(6)

        # ── Channel selector ──
        ch_group = QGroupBox("Channel")
        ch_row = QHBoxLayout(ch_group)
        ch_names = eeg_signal.channel_names or [
            f"Ch{i+1}" for i in range(self._n_channels())
        ]
        self._ch_combo = QComboBox()
        for i, n in enumerate(ch_names):
            self._ch_combo.addItem(f"{i}: {n}", i)
        self._ch_combo.currentIndexChanged.connect(self._on_channel_changed)
        ch_row.addWidget(self._ch_combo)
        vbox.addWidget(ch_group)

        # ── Time window ──
        tw_group = QGroupBox("Time Window")
        tw_vbox = QVBoxLayout(tw_group)
        tw_vbox.setContentsMargins(4, 4, 4, 4)
        tw_vbox.setSpacing(3)
        tw_top = QHBoxLayout()
        tw_top.addWidget(QLabel("Window:"))
        self._tw_spin = QDoubleSpinBox()
        self._tw_spin.setRange(0.1, 300.0)
        self._tw_spin.setValue(self._time_win)
        self._tw_spin.setSingleStep(1.0)
        self._tw_spin.setDecimals(1)
        self._tw_spin.setSuffix(" s")
        self._tw_spin.valueChanged.connect(self._on_tw_changed)
        tw_top.addWidget(self._tw_spin)
        tw_vbox.addLayout(tw_top)
        tw_btns = QHBoxLayout()
        tw_btns.setSpacing(2)
        for t in [1, 5, 10, 30]:
            b = QPushButton(f"{t}s")
            b.setMaximumWidth(38)
            b.setAutoDefault(False)
            b.setDefault(False)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _, v=t: self._tw_spin.setValue(v))
            tw_btns.addWidget(b)
        tw_vbox.addLayout(tw_btns)
        vbox.addWidget(tw_group)

        # ── Time scroll bar ── (created here; placed in controls if include_scrollbar=True,
        #                        otherwise caller adds self._time_sb / self._time_lbl below canvas)
        self._time_sb = QScrollBar(Qt.Horizontal)
        self._time_sb.setMinimum(0)
        self._time_sb.setMaximum(1000)
        self._time_sb.valueChanged.connect(self._on_sb_changed)
        self._time_lbl = QLabel("0.00 – 10.00 s")
        self._time_lbl.setMinimumWidth(120)
        if include_scrollbar:
            sb_row = QHBoxLayout()
            sb_row.addWidget(QLabel("Pos:"))
            sb_row.addWidget(self._time_sb, stretch=1)
            sb_row.addWidget(self._time_lbl)
            vbox.addLayout(sb_row)

        # ── Amplitude ──
        amp_group = QGroupBox("Amplitude Scale")
        amp_row = QHBoxLayout(amp_group)
        amp_row.addWidget(QLabel("Scale:"))
        self._amp_spin = QDoubleSpinBox()
        self._amp_spin.setRange(0.01, 200.0)
        self._amp_spin.setValue(self._amp_scale)
        self._amp_spin.setSingleStep(0.5)
        self._amp_spin.setDecimals(2)
        self._amp_spin.valueChanged.connect(self._on_amp_changed)
        amp_row.addWidget(self._amp_spin)
        amp_dn = QPushButton("−")
        amp_dn.setMaximumWidth(28)
        amp_dn.clicked.connect(lambda: self._amp_spin.setValue(self._amp_spin.value() * 0.5))
        amp_row.addWidget(amp_dn)
        amp_up = QPushButton("+")
        amp_up.setMaximumWidth(28)
        amp_up.clicked.connect(lambda: self._amp_spin.setValue(self._amp_spin.value() * 2.0))
        amp_row.addWidget(amp_up)
        vbox.addWidget(amp_group)

        # ── Zoom ──
        zoom_row = QHBoxLayout()
        zi = QPushButton("🔍+ Zoom In")
        zi.clicked.connect(lambda: self._tw_spin.setValue(self._time_win / 2))
        zoom_row.addWidget(zi)
        zo = QPushButton("🔍− Zoom Out")
        zo.clicked.connect(lambda: self._tw_spin.setValue(self._time_win * 2))
        zoom_row.addWidget(zo)
        vbox.addLayout(zoom_row)

        # ── Auto-scroll ──
        sc_group = QGroupBox("Auto Scroll")
        sc_row = QHBoxLayout(sc_group)
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setMaximumWidth(70)
        self._play_btn.clicked.connect(self._on_play_clicked)
        sc_row.addWidget(self._play_btn)
        sc_row.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 40.0)
        self._speed_spin.setValue(self._scroll_speed)
        self._speed_spin.setSingleStep(5.0)
        self._speed_spin.setSuffix("x")
        self._speed_spin.valueChanged.connect(lambda v: setattr(self, '_scroll_speed', v))
        sc_row.addWidget(self._speed_spin)
        vbox.addWidget(sc_group)

        # ── Highpass filter ──
        hp_group = QGroupBox("Highpass Filter")
        hp_row = QHBoxLayout(hp_group)
        self._hp_check = QCheckBox("Enable")
        self._hp_check.stateChanged.connect(self._on_hp_changed)
        hp_row.addWidget(self._hp_check)
        hp_row.addWidget(QLabel("Hz:"))
        self._hp_spin = QDoubleSpinBox()
        self._hp_spin.setRange(0.01, 30.0)
        self._hp_spin.setValue(self._hp_freq)
        self._hp_spin.setSingleStep(0.1)
        self._hp_spin.setDecimals(2)
        self._hp_spin.valueChanged.connect(self._on_hp_freq_changed)
        hp_row.addWidget(self._hp_spin)
        vbox.addWidget(hp_group)

        if include_scrollbar:
            vbox.addStretch()

        # ── Auto-scroll timer ──
        self._auto_timer = QTimer()
        self._auto_timer.setInterval(50)
        self._auto_timer.timeout.connect(self._auto_scroll_tick)

        self._update_sb_range()
        return outer

    # ── helpers ──
    def _n_channels(self):
        eeg = self._eeg
        if eeg.data.ndim == 3:
            return eeg.data.shape[1]
        return eeg.data.shape[0]

    def _n_samples(self):
        eeg = self._eeg
        if eeg.data.ndim == 3:
            return eeg.data.shape[2]
        return eeg.data.shape[1]

    def _get_channel_data(self, ch_idx=None):
        """Return 1-D array for the selected channel (full recording)."""
        if ch_idx is None:
            ch_idx = self._ch_idx
        eeg = self._eeg
        data = eeg.data[0] if eeg.data.ndim == 3 else eeg.data
        return data[ch_idx]

    def _apply_hp(self, signal_1d, sfreq):
        if not self._hp_enabled or len(signal_1d) < 20:
            return signal_1d
        nyq = sfreq / 2.0
        if self._hp_freq >= nyq:
            return signal_1d
        try:
            sos = butter(4, self._hp_freq / nyq, btype='highpass', output='sos')
            return sosfiltfilt(sos, signal_1d)
        except Exception:
            return signal_1d

    def _apply_bp(self, signal_1d, sfreq, lo, hi):
        """Bandpass filter a 1-D signal."""
        nyq = sfreq / 2.0
        lo_n = max(0.001, lo / nyq)
        hi_n = min(0.999, hi / nyq)
        if lo_n >= hi_n or len(signal_1d) < 20:
            return signal_1d
        try:
            sos = butter(4, [lo_n, hi_n], btype='bandpass', output='sos')
            return sosfiltfilt(sos, signal_1d)
        except Exception:
            return signal_1d

    def _update_sb_range(self):
        sfreq = self._eeg.sampling_rate
        total = self._n_samples() / sfreq
        max_scroll = max(0.0, total - self._time_win)
        self._time_sb.setMaximum(int(max_scroll * 10))
        self._time_sb.setPageStep(int(self._time_win * 10))

    def _update_time_label(self):
        self._time_lbl.setText(
            f"{self._time_start:.2f} – {self._time_start + self._time_win:.2f} s"
        )

    # ── slots ──
    def _on_channel_changed(self, idx):
        self._ch_idx = self._ch_combo.itemData(idx)
        self._on_controls_changed()

    def _on_tw_changed(self, val):
        self._time_win = val
        self._update_sb_range()
        self._on_controls_changed()

    def _on_sb_changed(self, val):
        self._time_start = val / 10.0
        self._update_time_label()
        self._on_controls_changed()

    def _on_amp_changed(self, val):
        self._amp_scale = val
        self._on_controls_changed()

    def _on_hp_changed(self, state):
        self._hp_enabled = (state == Qt.Checked)
        self._on_controls_changed()

    def _on_hp_freq_changed(self, val):
        self._hp_freq = val
        if self._hp_enabled:
            self._on_controls_changed()

    def _on_play_clicked(self, checked):
        if checked:
            self._auto_scroll_active = True
            self._play_btn.setText("⏸ Pause")
            self._last_scroll_time_mono = None
            self._auto_timer.start()
        else:
            self._auto_scroll_active = False
            self._play_btn.setText("▶ Play")
            self._play_btn.setChecked(False)
            self._auto_timer.stop()

    def _auto_scroll_tick(self):
        import time as _time
        now = _time.monotonic()
        if self._last_scroll_time_mono is None:
            elapsed = 0.05
        else:
            elapsed = now - self._last_scroll_time_mono
        self._last_scroll_time_mono = now

        sfreq = self._eeg.sampling_rate
        total = self._n_samples() / sfreq
        max_start = total - self._time_win
        new_start = self._time_start + self._scroll_speed * elapsed

        if new_start >= max_start:
            self._time_start = max(0.0, max_start)
            self._on_play_clicked(False)
        else:
            self._time_start = new_start

        self._time_sb.blockSignals(True)
        self._time_sb.setValue(int(self._time_start * 10))
        self._time_sb.blockSignals(False)
        self._update_time_label()
        self._on_controls_changed()

    def _on_controls_changed(self):
        """Override in subclass to redraw."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FFT Over Time window
# ---------------------------------------------------------------------------

class FFTOverTimeWindow(_SingleChannelControlsMixin, QWidget):
    """
    New window: top = EEG trace of the selected channel,
    bottom = FFT spectrogram time-locked to the trace.
    Controls are identical to the main viewer (single-channel).
    """

    def __init__(self, eeg_signal: EEGSignal, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("FFT Over Time")
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)

        # Build shared controls WITHOUT embedded scrollbar — we put it below the canvas
        controls_widget = self._build_shared_controls(eeg_signal, include_scrollbar=False)

        # ── FFT frequency range controls (appended to controls widget) ──
        cw_vbox = controls_widget.layout()
        fft_grp = QGroupBox("FFT Frequency Range")
        fft_grp.setToolTip("Sets the visible frequency band in the spectrogram")
        fft_grid = QGridLayout(fft_grp)
        fft_grid.addWidget(QLabel("Min (Hz):"), 0, 0)
        self._fft_fmin_spin = QDoubleSpinBox()
        self._fft_fmin_spin.setRange(0.0, 990.0)
        self._fft_fmin_spin.setValue(0.0)
        self._fft_fmin_spin.setSingleStep(1.0)
        self._fft_fmin_spin.setDecimals(1)
        self._fft_fmin_spin.valueChanged.connect(self._on_controls_changed)
        fft_grid.addWidget(self._fft_fmin_spin, 0, 1)
        fft_grid.addWidget(QLabel("Max (Hz):"), 1, 0)
        self._fft_fmax_spin = QDoubleSpinBox()
        self._fft_fmax_spin.setRange(1.0, 1000.0)
        self._fft_fmax_spin.setValue(100.0)
        self._fft_fmax_spin.setSingleStep(5.0)
        self._fft_fmax_spin.setDecimals(1)
        self._fft_fmax_spin.valueChanged.connect(self._on_controls_changed)
        fft_grid.addWidget(self._fft_fmax_spin, 1, 1)
        cw_vbox.addWidget(fft_grp)
        cw_vbox.addStretch()

        # ── Matplotlib figures ──
        self._figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self._figure.patch.set_facecolor('#f8f8f8')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Main layout: plot+scrollbar on left, controls on right
        main = QHBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)

        # Left side: canvas with time scrollbar pinned below it
        plot_side = QVBoxLayout()
        plot_side.setSpacing(2)
        plot_side.addWidget(self._canvas, stretch=1)
        sb_row = QHBoxLayout()
        sb_row.addWidget(QLabel("Position:"))
        sb_row.addWidget(self._time_sb, stretch=1)
        sb_row.addWidget(self._time_lbl)
        plot_side.addLayout(sb_row)
        main.addLayout(plot_side, stretch=1)

        # Right side: controls in a scroll area so nothing gets clipped
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(260)
        ctrl_scroll.setWidget(controls_widget)
        main.addWidget(ctrl_scroll)

        self._on_controls_changed()

    def _on_controls_changed(self):
        self._draw()

    def _draw(self):
        eeg = self._eeg
        sfreq = eeg.sampling_rate
        raw = self._get_channel_data()

        # Apply highpass to full signal first
        raw = self._apply_hp(raw, sfreq)

        n_samples = len(raw)
        start_s = int(self._time_start * sfreq)
        end_s = int((self._time_start + self._time_win) * sfreq)
        end_s = min(end_s, n_samples)
        if start_s >= end_s:
            start_s = max(0, end_s - int(sfreq))

        view = raw[start_s:end_s]
        time_axis = np.arange(start_s, min(end_s, start_s + len(view))) / sfreq

        self._figure.clear()
        ax_eeg = self._figure.add_subplot(2, 1, 1)
        ax_spec = self._figure.add_subplot(2, 1, 2)

        # ── EEG trace ──
        display = view - np.mean(view)
        display = display * self._amp_scale
        ax_eeg.plot(time_axis[:len(display)], display, linewidth=0.7, color='#1565C0')
        ax_eeg.set_xlim(time_axis[0], time_axis[-1])
        ax_eeg.set_ylabel("Amplitude (µV)")
        ax_eeg.set_title(
            f"EEG — {self._ch_combo.currentText().split(': ')[-1]}  "
            f"[{self._time_start:.2f} – {self._time_start + self._time_win:.2f} s]"
        )
        ax_eeg.grid(True, alpha=0.3, linestyle='--')

        # ── Spectrogram ──
        n_fft = min(256, len(view) // 4)
        n_fft = max(n_fft, 32)
        overlap = n_fft * 3 // 4
        try:
            from matplotlib.mlab import specgram
            Pxx, freqs, t_spec = specgram(
                view, NFFT=n_fft, Fs=sfreq,
                noverlap=overlap, detrend='mean'
            )
            # Limit to user-defined frequency range
            _fmin = self._fft_fmin_spin.value()
            _fmax = self._fft_fmax_spin.value()
            if _fmax <= _fmin:
                _fmax = _fmin + 1.0
            freq_mask = (freqs >= _fmin) & (freqs <= _fmax)
            Pxx = Pxx[freq_mask, :]
            freqs = freqs[freq_mask]
            # dB scale, clip to avoid log(0)
            Pxx_db = 10 * np.log10(np.maximum(Pxx, 1e-12))
            t_abs = t_spec + self._time_start
            im = ax_spec.pcolormesh(
                t_abs, freqs, Pxx_db,
                cmap='viridis', shading='gouraud'
            )
            self._figure.colorbar(im, ax=ax_spec, label='Power (dB)')
            ax_spec.set_xlim(time_axis[0], time_axis[-1])
            ax_spec.set_ylim(_fmin, _fmax)
            ax_spec.set_ylabel("Frequency (Hz)")
            ax_spec.set_xlabel("Time (s)")
            ax_spec.set_title(f"FFT Spectrogram  [{_fmin:.0f}–{_fmax:.0f} Hz]")
        except Exception as e:
            ax_spec.text(0.5, 0.5, f"Spectrogram error:\n{e}",
                         ha='center', va='center', transform=ax_spec.transAxes)

        self._canvas.draw_idle()
        self._update_time_label()


# ---------------------------------------------------------------------------
# Filtering window
# ---------------------------------------------------------------------------

class FilteringWindow(_SingleChannelControlsMixin, QWidget):
    """
    New window: three vertically stacked panels for a single channel —
      top:    raw (or highpass-filtered) trace
      middle: bandpass trace (default theta 4–8 Hz), user-adjustable
      bottom: bandpass trace (default gamma 30–80 Hz), user-adjustable
    Each of the two filtered panels has its own lo/hi spinboxes on the right.
    """

    def __init__(self, eeg_signal: EEGSignal, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("Filtering View")
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(950, 700)
        self.resize(1150, 780)

        # Band-specific state
        self._band1_lo = 4.0
        self._band1_hi = 8.0   # theta
        self._band2_lo = 30.0
        self._band2_hi = 80.0  # gamma

        # Build shared controls WITHOUT embedded scrollbar — we put it below the canvas
        controls_widget = self._build_shared_controls(eeg_signal, include_scrollbar=False)
        controls_widget.layout().addStretch()

        # ── Matplotlib figure ──
        self._figure = Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        self._figure.patch.set_facecolor('#f8f8f8')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Right-side panel: shared controls + band controls
        right_panel = QWidget()
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(8)

        right_vbox.addWidget(controls_widget)

        # Band 1 box
        b1_box = QGroupBox("Middle Band (default: Theta)")
        b1_grid = QGridLayout(b1_box)
        b1_grid.addWidget(QLabel("Low cut (Hz):"), 0, 0)
        self._b1_lo_spin = QDoubleSpinBox()
        self._b1_lo_spin.setRange(0.5, 200.0)
        self._b1_lo_spin.setValue(self._band1_lo)
        self._b1_lo_spin.setSingleStep(0.5)
        self._b1_lo_spin.setDecimals(1)
        self._b1_lo_spin.valueChanged.connect(self._on_b1_lo)
        b1_grid.addWidget(self._b1_lo_spin, 0, 1)
        b1_grid.addWidget(QLabel("High cut (Hz):"), 1, 0)
        self._b1_hi_spin = QDoubleSpinBox()
        self._b1_hi_spin.setRange(1.0, 500.0)
        self._b1_hi_spin.setValue(self._band1_hi)
        self._b1_hi_spin.setSingleStep(0.5)
        self._b1_hi_spin.setDecimals(1)
        self._b1_hi_spin.valueChanged.connect(self._on_b1_hi)
        b1_grid.addWidget(self._b1_hi_spin, 1, 1)
        right_vbox.addWidget(b1_box)

        # Band 2 box
        b2_box = QGroupBox("Bottom Band (default: Gamma)")
        b2_grid = QGridLayout(b2_box)
        b2_grid.addWidget(QLabel("Low cut (Hz):"), 0, 0)
        self._b2_lo_spin = QDoubleSpinBox()
        self._b2_lo_spin.setRange(0.5, 200.0)
        self._b2_lo_spin.setValue(self._band2_lo)
        self._b2_lo_spin.setSingleStep(0.5)
        self._b2_lo_spin.setDecimals(1)
        self._b2_lo_spin.valueChanged.connect(self._on_b2_lo)
        b2_grid.addWidget(self._b2_lo_spin, 0, 1)
        b2_grid.addWidget(QLabel("High cut (Hz):"), 1, 0)
        self._b2_hi_spin = QDoubleSpinBox()
        self._b2_hi_spin.setRange(1.0, 500.0)
        self._b2_hi_spin.setValue(self._band2_hi)
        self._b2_hi_spin.setSingleStep(0.5)
        self._b2_hi_spin.setDecimals(1)
        self._b2_hi_spin.valueChanged.connect(self._on_b2_hi)
        b2_grid.addWidget(self._b2_hi_spin, 1, 1)
        right_vbox.addWidget(b2_box)

        # Scroll area for right panel
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(280)
        ctrl_scroll.setWidget(right_panel)

        main = QHBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)

        # Left side: canvas with time scrollbar pinned below it
        plot_side = QVBoxLayout()
        plot_side.setSpacing(2)
        plot_side.addWidget(self._canvas, stretch=1)
        sb_row = QHBoxLayout()
        sb_row.addWidget(QLabel("Position:"))
        sb_row.addWidget(self._time_sb, stretch=1)
        sb_row.addWidget(self._time_lbl)
        plot_side.addLayout(sb_row)
        main.addLayout(plot_side, stretch=1)

        main.addWidget(ctrl_scroll)

        self._on_controls_changed()

    # ── band spinbox slots ──
    def _on_b1_lo(self, v):
        self._band1_lo = v
        self._draw()

    def _on_b1_hi(self, v):
        self._band1_hi = v
        self._draw()

    def _on_b2_lo(self, v):
        self._band2_lo = v
        self._draw()

    def _on_b2_hi(self, v):
        self._band2_hi = v
        self._draw()

    def _on_controls_changed(self):
        self._draw()

    def _draw(self):
        eeg = self._eeg
        sfreq = eeg.sampling_rate
        raw_full = self._get_channel_data()

        # Apply highpass to full signal first (display only)
        filtered_full = self._apply_hp(raw_full, sfreq)

        n_samples = len(filtered_full)
        start_s = int(self._time_start * sfreq)
        end_s = int((self._time_start + self._time_win) * sfreq)
        end_s = min(end_s, n_samples)
        if start_s >= end_s:
            start_s = max(0, end_s - int(sfreq))

        time_axis = np.arange(start_s, min(end_s, n_samples)) / sfreq

        # Slice for display
        raw_view = filtered_full[start_s:end_s]
        t = time_axis[:len(raw_view)]

        # Bandpass on the *full* signal then slice (avoids edge artefacts)
        bp1_full = self._apply_bp(filtered_full, sfreq, self._band1_lo, self._band1_hi)
        bp2_full = self._apply_bp(filtered_full, sfreq, self._band2_lo, self._band2_hi)
        bp1_view = bp1_full[start_s:end_s]
        bp2_view = bp2_full[start_s:end_s]

        def _scale(sig):
            s = sig - np.mean(sig)
            return s * self._amp_scale

        ch_label = self._ch_combo.currentText().split(': ')[-1]
        time_range = f"{self._time_start:.2f} – {self._time_start + self._time_win:.2f} s"

        self._figure.clear()
        ax1 = self._figure.add_subplot(3, 1, 1)
        ax2 = self._figure.add_subplot(3, 1, 2)
        ax3 = self._figure.add_subplot(3, 1, 3)

        # ── Top: raw (or highpassed) trace ──
        ax1.plot(t, _scale(raw_view), linewidth=0.7, color='#1565C0')
        ax1.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
        ax1.set_ylabel("µV")
        hp_label = f"HP {self._hp_freq:.2f} Hz" if self._hp_enabled else "No HP"
        ax1.set_title(f"Raw Trace — {ch_label}  [{time_range}]  ({hp_label})")
        ax1.grid(True, alpha=0.3, linestyle='--')

        # ── Middle: band 1 ──
        ax2.plot(t, _scale(bp1_view), linewidth=0.7, color='#2E7D32')
        ax2.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
        ax2.set_ylabel("µV")
        ax2.set_title(
            f"Bandpass  {self._band1_lo:.1f} – {self._band1_hi:.1f} Hz"
            f"  (default: Theta)"
        )
        ax2.grid(True, alpha=0.3, linestyle='--')

        # ── Bottom: band 2 ──
        ax3.plot(t, _scale(bp2_view), linewidth=0.7, color='#6A1B9A')
        ax3.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
        ax3.set_ylabel("µV")
        ax3.set_xlabel("Time (s)")
        ax3.set_title(
            f"Bandpass  {self._band2_lo:.1f} – {self._band2_hi:.1f} Hz"
            f"  (default: Gamma)"
        )
        ax3.grid(True, alpha=0.3, linestyle='--')

        self._canvas.draw_idle()
        self._update_time_label()


class EEGViewerDialog(QDialog):
    """
    Standalone dialog window for the EEG viewer.
    Can be opened from Preview button or intermediate step results.
    """
    
    def __init__(self, eeg_signal: Optional[EEGSignal] = None, title: str = "EEG Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create viewer widget
        self.viewer = EEGViewerWidget(self)
        layout.addWidget(self.viewer)
        
        # Load data
        if eeg_signal is not None:
            self.viewer.set_eeg_signal(eeg_signal, title)
    
    @staticmethod
    def show_eeg(eeg_signal: Optional[EEGSignal] = None, title: str = "EEG Viewer", parent=None):
        """
        Static method to quickly show EEG data in a viewer dialog.
        
        Parameters
        ----------
        eeg_signal : EEGSignal
            The EEG signal to display
        title : str
            Window title
        parent : QWidget, optional
            Parent widget
        
        Returns
        -------
        EEGViewerDialog
            The dialog instance
        """
        dialog = EEGViewerDialog(eeg_signal, title, parent)
        dialog.show()
        return dialog


def show_eeg_viewer(eeg_signal: Optional[EEGSignal] = None, title: str = "EEG Viewer", parent=None):
    """
    Convenience function to display EEG data in the interactive viewer.
    
    Parameters
    ----------
    eeg_signal : EEGSignal
        The EEG signal to display
    title : str
        Window title
    parent : QWidget, optional
        Parent widget
    
    Returns
    -------
    EEGViewerDialog
        The dialog instance (keep reference to prevent garbage collection)
    """
    return EEGViewerDialog.show_eeg(eeg_signal, title, parent)


# For testing
if __name__ == "__main__":
    import sys
    from core_data_structures import EEGSignal, DataType
    
    app = QApplication(sys.argv)
    
    viewer = EEGViewerDialog()
    viewer.show()
    
    sys.exit(app.exec_())

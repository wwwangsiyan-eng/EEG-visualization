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
        
        # Create splitter for plot and metadata panel
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Plot and controls
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        plot_layout.addWidget(toolbar)
        
        # Matplotlib figure and canvas
        # Use constrained_layout to avoid tight_layout warnings and prevent resizing issues
        self.figure = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.figure.patch.set_facecolor('#f8f8f8')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(400, 300)  # Prevent canvas from becoming too small
        plot_layout.addWidget(self.canvas, stretch=1)
        
        # Time scrollbar (horizontal)
        scroll_layout = QHBoxLayout()
        scroll_layout.addWidget(QLabel("Time:"))
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.setValue(0)
        self.time_scrollbar.valueChanged.connect(self._on_scroll_changed)
        scroll_layout.addWidget(self.time_scrollbar, stretch=1)
        
        self.time_label = QLabel("0.00 - 10.00 s")
        self.time_label.setMinimumWidth(120)
        scroll_layout.addWidget(self.time_label)
        plot_layout.addLayout(scroll_layout)
        
        splitter.addWidget(plot_widget)
        
        # Right side: Metadata panel
        metadata_panel = self._create_metadata_panel()
        splitter.addWidget(metadata_panel)
        
        # Set splitter sizes (70% plot, 30% metadata)
        splitter.setSizes([700, 300])
        # Set stretch factors to maintain proportions during resize
        splitter.setStretchFactor(0, 7)  # Plot widget gets more stretch
        splitter.setStretchFactor(1, 3)  # Metadata panel gets less stretch
        
        main_layout.addWidget(splitter, stretch=1)
        
        # Control panel at the bottom (spanning full width)
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("No data loaded")
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
        channel_action = QAction("📊 Select Channels", self)
        channel_action.triggered.connect(self._open_channel_selector)
        toolbar.addAction(channel_action)
        
        return toolbar
    
    def _create_control_panel(self) -> QFrame:
        """Create the control panel with scale adjustments."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # X-Scale (Time Window)
        x_group = QGroupBox("Time Window")
        x_layout = QHBoxLayout(x_group)
        
        x_layout.addWidget(QLabel("Window (s):"))
        self.time_window_spin = QDoubleSpinBox()
        self.time_window_spin.setRange(0.1, 300.0)
        self.time_window_spin.setValue(10.0)
        self.time_window_spin.setSingleStep(1.0)
        self.time_window_spin.setDecimals(1)
        self.time_window_spin.setFocusPolicy(Qt.StrongFocus)  # Require explicit click to focus
        self.time_window_spin.valueChanged.connect(self._on_time_window_changed)
        x_layout.addWidget(self.time_window_spin)
        
        # Quick time presets
        for preset in [1, 5, 10, 30, 60]:
            btn = QPushButton(f"{preset}s")
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, t=preset: self._set_time_window(t))
            x_layout.addWidget(btn)
        
        layout.addWidget(x_group)
        
        # Y-Scale (Amplitude)
        y_group = QGroupBox("Amplitude Scale")
        y_layout = QHBoxLayout(y_group)
        
        y_layout.addWidget(QLabel("Scale:"))
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.01, 200.0)
        self.amplitude_spin.setValue(1.0)
        self.amplitude_spin.setSingleStep(1.0)
        self.amplitude_spin.setDecimals(2)
        self.amplitude_spin.valueChanged.connect(self._on_amplitude_changed)
        y_layout.addWidget(self.amplitude_spin)
        
        # Amplitude presets
        amp_down = QPushButton("−")
        amp_down.setMaximumWidth(30)
        amp_down.clicked.connect(lambda: self._adjust_amplitude(0.5))
        y_layout.addWidget(amp_down)
        
        amp_up = QPushButton("+")
        amp_up.setMaximumWidth(30)
        amp_up.clicked.connect(lambda: self._adjust_amplitude(2.0))
        y_layout.addWidget(amp_up)
        
        layout.addWidget(y_group)
        
        # Channel Spacing
        spacing_group = QGroupBox("Channel Spacing")
        spacing_layout = QHBoxLayout(spacing_group)
        
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 10.0)
        self.spacing_spin.setValue(1.0)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setDecimals(1)
        self.spacing_spin.valueChanged.connect(self._on_spacing_changed)
        spacing_layout.addWidget(self.spacing_spin)
        
        layout.addWidget(spacing_group)
        
        # Epoch selector (for 3D data)
        self.epoch_group = QGroupBox("Epoch")
        epoch_layout = QHBoxLayout(self.epoch_group)
        
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(0, 0)
        self.epoch_spin.valueChanged.connect(self._on_epoch_changed)
        epoch_layout.addWidget(self.epoch_spin)
        
        self.epoch_label = QLabel("/ 0")
        epoch_layout.addWidget(self.epoch_label)
        
        self.epoch_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.epoch_group)
        
        # Grid toggle
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._on_grid_changed)
        layout.addWidget(self.grid_check)
        
        # Auto-scroll controls
        scroll_group = QGroupBox("Auto Scroll")
        scroll_layout = QHBoxLayout(scroll_group)
        
        self.auto_scroll_btn = QPushButton("▶ Play")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setMaximumWidth(70)
        self.auto_scroll_btn.clicked.connect(self._toggle_auto_scroll)
        scroll_layout.addWidget(self.auto_scroll_btn)
        
        scroll_layout.addWidget(QLabel("Speed:"))
        self.scroll_speed_spin = QDoubleSpinBox()
        self.scroll_speed_spin.setRange(0.1, 40.0)
        self.scroll_speed_spin.setValue(20.0)
        self.scroll_speed_spin.setSingleStep(5.0)
        self.scroll_speed_spin.setSuffix("x")
        self.scroll_speed_spin.setToolTip("Playback speed (1x = real-time)")
        self.scroll_speed_spin.valueChanged.connect(self._on_scroll_speed_changed)
        scroll_layout.addWidget(self.scroll_speed_spin)
        
        layout.addWidget(scroll_group)
        
        # Highpass filter controls
        hp_group = QGroupBox("Highpass Filter")
        hp_layout = QHBoxLayout(hp_group)
        
        self.highpass_check = QCheckBox("Enable")
        self.highpass_check.setToolTip("Apply highpass filter to remove DC drift and align traces to baseline")
        self.highpass_check.stateChanged.connect(self._on_highpass_changed)
        hp_layout.addWidget(self.highpass_check)
        
        hp_layout.addWidget(QLabel("Hz:"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0.01, 30.0)
        self.highpass_spin.setValue(0.5)
        self.highpass_spin.setSingleStep(0.1)
        self.highpass_spin.setDecimals(2)
        self.highpass_spin.setToolTip("Highpass cutoff frequency in Hz")
        self.highpass_spin.valueChanged.connect(self._on_highpass_freq_changed)
        hp_layout.addWidget(self.highpass_spin)
        
        layout.addWidget(hp_group)
        
        # 10-20 Conversion toggle
        self.convert_1020_check = QCheckBox("10-20 Format")
        self.convert_1020_check.setToolTip("Convert EGI channels to standard 10-20 format by averaging mapped electrodes")
        self.convert_1020_check.stateChanged.connect(self._on_convert_1020_changed)
        layout.addWidget(self.convert_1020_check)
        
        layout.addStretch()
        
        return frame
    
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
        """Update the metadata tables."""
        if self.eeg_signal is None:
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
        """
        if self.eeg_signal is None:
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

"""
Topographic Heatmap Window — EEG Viewer
========================================
Floating window that shows a continuously-updated scalp heatmap of the
segment currently displayed in the main EEG viewer.

Position data and drawing logic adapted (without modification) from:
  new_python_eeg_processing_organized/visualization/eeg_visualization.py
  new_python_eeg_processing_organized/visualization/egi_utils.py
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QComboBox, QCheckBox, QPushButton,
    QScrollArea, QSizePolicy, QGridLayout, QSpinBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import griddata

# ─────────────────────────────────────────────────────────────────────────────
# EGI GSN-HydroCel-128 electrode positions
# Source: MNE-Python 1.10.2, azimuthal equidistant projection from top of head.
# Nose up (+y), right ear (+x).  Head circle = r=1.0; peripheral electrodes
# (E48, E49, E113, E119, E125-E128) have r>1 and are clipped to the head rim.
# ─────────────────────────────────────────────────────────────────────────────
_EGI_128_POS = {
    "E1":   ( 0.652760,  0.622590), "E2":   ( 0.506357,  0.668487),
    "E3":   ( 0.335729,  0.710543), "E4":   ( 0.234869,  0.645194),
    "E5":   ( 0.114561,  0.511723), "E6":   ( 0.000000,  0.362171),
    "E7":   (-0.089721,  0.195024), "E8":   ( 0.447531,  0.859286),
    "E9":   ( 0.254788,  0.872302), "E10":  ( 0.160123,  0.809780),
    "E11":  ( 0.000000,  0.716418), "E12":  (-0.114561,  0.511723),
    "E13":  (-0.182511,  0.319608), "E14":  ( 0.133202,  1.008561),
    "E15":  ( 0.000000,  0.883187), "E16":  ( 0.000000,  0.846608),
    "E17":  ( 0.000000,  1.049393), "E18":  (-0.160123,  0.809780),
    "E19":  (-0.234869,  0.645194), "E20":  (-0.300613,  0.468326),
    "E21":  (-0.133202,  1.008561), "E22":  (-0.254788,  0.872302),
    "E23":  (-0.335729,  0.710543), "E24":  (-0.366669,  0.550931),
    "E25":  (-0.447531,  0.859286), "E26":  (-0.506357,  0.668487),
    "E27":  (-0.486639,  0.512548), "E28":  (-0.446552,  0.392222),
    "E29":  (-0.365159,  0.274741), "E30":  (-0.274427,  0.147385),
    "E31":  (-0.142526,  0.032072), "E32":  (-0.652760,  0.622590),
    "E33":  (-0.614659,  0.417983), "E34":  (-0.580109,  0.295175),
    "E35":  (-0.507622,  0.176329), "E36":  (-0.416795,  0.090540),
    "E37":  (-0.289238, -0.034720), "E38":  (-0.776816,  0.420249),
    "E39":  (-0.692844,  0.100055), "E40":  (-0.625053,  0.044140),
    "E41":  (-0.546852, -0.005201), "E42":  (-0.456983, -0.111891),
    "E43":  (-0.936766,  0.311630), "E44":  (-0.811396,  0.142601),
    "E45":  (-0.686410, -0.154081), "E46":  (-0.611696, -0.148371),
    "E47":  (-0.537399, -0.184039), "E48":  (-1.228032,  0.355047),
    "E49":  (-0.999812,  0.015481), "E50":  (-0.621810, -0.343568),
    "E51":  (-0.542707, -0.338868), "E52":  (-0.457124, -0.290580),
    "E53":  (-0.315045, -0.229279), "E54":  (-0.166763, -0.169842),
    "E55":  ( 0.000000, -0.071518), "E56":  (-0.881499, -0.388419),
    "E57":  (-0.712724, -0.404651), "E58":  (-0.546539, -0.492992),
    "E59":  (-0.427466, -0.479172), "E60":  (-0.319234, -0.405200),
    "E61":  (-0.175697, -0.335792), "E62":  ( 0.000000, -0.434031),
    "E63":  (-0.737705, -0.629879), "E64":  (-0.573814, -0.620812),
    "E65":  (-0.418764, -0.626814), "E66":  (-0.295228, -0.576126),
    "E67":  (-0.143344, -0.507132), "E68":  (-0.487764, -0.851511),
    "E69":  (-0.369798, -0.792633), "E70":  (-0.246240, -0.742937),
    "E71":  (-0.114460, -0.635061), "E72":  ( 0.000000, -0.553661),
    "E73":  (-0.242290, -0.965114), "E74":  (-0.116109, -0.865730),
    "E75":  ( 0.000000, -0.766385), "E76":  ( 0.114460, -0.635061),
    "E77":  ( 0.143344, -0.507132), "E78":  ( 0.175697, -0.335792),
    "E79":  ( 0.166763, -0.169842), "E80":  ( 0.142526,  0.032072),
    "E81":  ( 0.000000, -0.986854), "E82":  ( 0.116109, -0.865730),
    "E83":  ( 0.246240, -0.742937), "E84":  ( 0.295228, -0.576126),
    "E85":  ( 0.319234, -0.405200), "E86":  ( 0.315045, -0.229279),
    "E87":  ( 0.289238, -0.034720), "E88":  ( 0.242290, -0.965114),
    "E89":  ( 0.369798, -0.792633), "E90":  ( 0.418764, -0.626814),
    "E91":  ( 0.427466, -0.479172), "E92":  ( 0.457124, -0.290580),
    "E93":  ( 0.456983, -0.111891), "E94":  ( 0.487764, -0.851511),
    "E95":  ( 0.573814, -0.620812), "E96":  ( 0.546539, -0.492992),
    "E97":  ( 0.542707, -0.338868), "E98":  ( 0.537399, -0.184039),
    "E99":  ( 0.737705, -0.629879), "E100": ( 0.712724, -0.404651),
    "E101": ( 0.621810, -0.343568), "E102": ( 0.611696, -0.148371),
    "E103": ( 0.546852, -0.005201), "E104": ( 0.416795,  0.090540),
    "E105": ( 0.274427,  0.147385), "E106": ( 0.089721,  0.195024),
    "E107": ( 0.881499, -0.388419), "E108": ( 0.686410, -0.154081),
    "E109": ( 0.625053,  0.044140), "E110": ( 0.507622,  0.176329),
    "E111": ( 0.365159,  0.274741), "E112": ( 0.182511,  0.319608),
    "E113": ( 0.999812,  0.015481), "E114": ( 0.811396,  0.142601),
    "E115": ( 0.692844,  0.100055), "E116": ( 0.580109,  0.295175),
    "E117": ( 0.446552,  0.392222), "E118": ( 0.300613,  0.468326),
    "E119": ( 1.228032,  0.355047), "E120": ( 0.936766,  0.311630),
    "E121": ( 0.776816,  0.420249), "E122": ( 0.614659,  0.417983),
    "E123": ( 0.486639,  0.512548), "E124": ( 0.366669,  0.550931),
    "E125": ( 0.807715,  0.573992), "E126": ( 0.636050,  1.062706),
    "E127": (-0.636050,  1.062706), "E128": (-0.807715,  0.573992),
    "E129": ( 0.000000,  0.000000),   # vertex reference / Cz
    # Extended positions for E130+ (256-channel nets)
    "E130": (-0.15,  0.12), "E131": ( 0.15,  0.12),
    "E132": (-0.15, -0.12), "E133": ( 0.15, -0.12),
    "E134": (-0.28,  0.28), "E135": ( 0.28,  0.28),
    "E136": (-0.28, -0.28), "E137": ( 0.28, -0.28),
    "E138": ( 0.0,   0.28), "E139": ( 0.0,  -0.28),
    "E140": (-0.42,  0.0 ), "E141": ( 0.42,  0.0 ),
    "E142": (-0.56,  0.14), "E143": ( 0.56,  0.14),
    "E144": (-0.56, -0.14), "E145": ( 0.56, -0.14),
    "E146": (-0.70,  0.0 ), "E147": ( 0.70,  0.0 ),
    "E148": (-0.42,  0.42), "E149": ( 0.42,  0.42),
    "E150": (-0.42, -0.42), "E151": ( 0.42, -0.42),
    "E152": ( 0.0,   0.56), "E153": ( 0.0,  -0.56),
}

# ─────────────────────────────────────────────────────────────────────────────
# Standard 10-20 electrode positions (unit circle, nose up)
# Source: eeg_visualization.py / egi_utils.py
# ─────────────────────────────────────────────────────────────────────────────
_POS_1020 = {
    # Frontal
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95), 'Fpz': (0.0,  0.95),
    'AF3': (-0.38, 0.83), 'AF4': (0.38, 0.83), 'AFz': (0.0,  0.85),
    'AF7': (-0.59, 0.81), 'AF8': (0.59, 0.81),
    'F7':  (-0.81, 0.59), 'F8':  (0.81, 0.59),
    'F3':  (-0.55, 0.67), 'F4':  (0.55, 0.67), 'Fz':  (0.0,  0.71),
    'F1':  (-0.28, 0.69), 'F2':  (0.28, 0.69),
    'F5':  (-0.68, 0.63), 'F6':  (0.68, 0.63),
    # Fronto-central
    'FT7': (-0.90, 0.35), 'FT8': (0.90, 0.35),
    'FT9': (-1.00, 0.31), 'FT10':(1.00, 0.31),
    'FC1': (-0.28, 0.40), 'FC2': (0.28, 0.40), 'FCz': (0.0,  0.43),
    'FC3': (-0.55, 0.38), 'FC4': (0.55, 0.38),
    'FC5': (-0.76, 0.36), 'FC6': (0.76, 0.36),
    # Temporal / Central
    'T3':  (-0.95, 0.0 ), 'T4':  (0.95, 0.0 ),
    'T7':  (-0.95, 0.0 ), 'T8':  (0.95, 0.0 ),
    'C1':  (-0.28, 0.0 ), 'C2':  (0.28, 0.0 ), 'Cz':  (0.0,  0.0 ),
    'C3':  (-0.55, 0.0 ), 'C4':  (0.55, 0.0 ),
    'C5':  (-0.76, 0.0 ), 'C6':  (0.76, 0.0 ),
    # Centro-parietal
    'TP7': (-0.90,-0.35), 'TP8': (0.90,-0.35),
    'TP9': (-1.00,-0.31), 'TP10':(1.00,-0.31),
    'CP1': (-0.28,-0.40), 'CP2': (0.28,-0.40), 'CPz': (0.0, -0.43),
    'CP3': (-0.55,-0.38), 'CP4': (0.55,-0.38),
    'CP5': (-0.76,-0.36), 'CP6': (0.76,-0.36),
    # Parietal
    'T5':  (-0.81,-0.59), 'T6':  (0.81,-0.59),
    'P7':  (-0.81,-0.59), 'P8':  (0.81,-0.59),
    'P3':  (-0.55,-0.67), 'P4':  (0.55,-0.67), 'Pz':  (0.0, -0.71),
    'P1':  (-0.28,-0.69), 'P2':  (0.28,-0.69),
    'P5':  (-0.68,-0.63), 'P6':  (0.68,-0.63),
    'P9':  (-0.90,-0.65), 'P10': (0.90,-0.65),
    # Parieto-occipital
    'PO3': (-0.38,-0.83), 'PO4': (0.38,-0.83), 'POz': (0.0, -0.85),
    'PO7': (-0.59,-0.81), 'PO8': (0.59,-0.81),
    # Occipital
    'O1':  (-0.31,-0.95), 'O2':  (0.31,-0.95), 'Oz':  (0.0, -0.95),
    # Mastoids / Reference
    'A1':  (-1.05,-0.10), 'A2':  (1.05,-0.10),
    'M1':  (-1.05,-0.15), 'M2':  (1.05,-0.15),
    'Iz':  (0.0, -1.05),  'Nz':  (0.0,  1.05),
}

# ─────────────────────────────────────────────────────────────────────────────
# EGI → 10-20 electrode mapping
# Source: egi_utils.py  (GSN-HydroCel-128 standard, verified against MNE montage)
# Each 10-20 position is obtained by averaging its 2-3 nearest EGI electrodes.
# ─────────────────────────────────────────────────────────────────────────────
_EGI_TO_1020 = {
    # Frontal Pole
    'Fp1': ['E25', 'E22', 'E21'],
    'Fp2': ['E8',  'E9',  'E14'],
    'Fpz': ['E15', 'E14', 'E21'],
    # Frontal
    'F7':  ['E33', 'E32', 'E38'],
    'F3':  ['E24', 'E28', 'E27'],
    'Fz':  ['E5',  'E12', 'E11'],
    'F4':  ['E124','E123','E117'],
    'F8':  ['E122','E1',  'E121'],
    # Fronto-central
    'FC3': ['E35', 'E29', 'E28'],
    'FCz': ['E6',  'E106','E7' ],
    'FC4': ['E110','E111','E117'],
    # Temporal
    'T7':  ['E45', 'E39', 'E50'],
    'T8':  ['E108','E115','E101'],
    # Central
    'C3':  ['E42', 'E36', 'E41'],
    'Cz':  ['E55', 'E80', 'E31'],
    'C4':  ['E93', 'E103','E104'],
    # Centro-parietal
    'CP3': ['E52', 'E53', 'E42'],
    'CPz': ['E55', 'E79', 'E54'],
    'CP4': ['E92', 'E93', 'E86'],
    # Posterior temporal
    'T5':  ['E58', 'E65', 'E50'],
    'T6':  ['E96', 'E90', 'E101'],
    # Parietal
    'P3':  ['E60', 'E59', 'E66'],
    'Pz':  ['E62', 'E72', 'E77'],
    'P4':  ['E85', 'E91', 'E84'],
    'P7':  ['E58', 'E65', 'E64'],
    'P8':  ['E96', 'E90', 'E95'],
    # Parieto-occipital
    'PO3': ['E66', 'E71', 'E67'],
    'POz': ['E72', 'E76', 'E71'],
    'PO4': ['E84', 'E76', 'E77'],
    # Occipital
    'O1':  ['E70', 'E75', 'E71'],
    'Oz':  ['E75', 'E76', 'E71'],
    'O2':  ['E83', 'E75', 'E76'],
}

# Non-EEG channel prefixes/patterns to skip when looking for electrode positions
_NON_EEG_PREFIXES = (
    'EMG', 'ECG', 'EKG', 'EOG', 'VEOG', 'HEOG', 'STI', 'STIM',
    'TRIG', 'TRIGGER', 'REF', 'GND', 'GROUND', 'VREF', 'AUX',
    'EXT', 'MISC', 'TEMP', 'GSR', 'RESP', 'PULSE', 'HR', 'SPO2',
    'ACC', 'ACCEL', 'GYRO', 'STATUS', 'FLAG', 'MARK', 'EVENT',
    'BLANK', 'EMPTY', 'UNUSED',
)
_NON_EEG_PATTERNS = ('VERTICAL', 'HORIZONTAL', 'MASTOID', 'CHIN', 'JAW', 'NECK')
_NON_EEG_EXACT = {'A1', 'A2', 'M1', 'M2'}


def _is_non_eeg(name: str) -> bool:
    n = str(name).upper().strip()
    if n in _NON_EEG_EXACT:
        return True
    if n.isdigit():
        return True
    for p in _NON_EEG_PREFIXES:
        if n.startswith(p):
            return True
    for p in _NON_EEG_PATTERNS:
        if p in n:
            return True
    return False


def _is_egi_format(channel_names) -> bool:
    if not channel_names:
        return False
    count = sum(
        1 for ch in channel_names
        if str(ch).upper().startswith('E') and str(ch)[1:].isdigit()
    )
    return count >= max(1, len(channel_names) * 0.5)


def _bandpass(data_1d: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    nyq = sfreq / 2.0
    lo_n = max(1e-4, lo / nyq)
    hi_n = min(0.9999, hi / nyq)
    if lo_n >= hi_n or len(data_1d) < 20:
        return data_1d
    try:
        sos = butter(4, [lo_n, hi_n], btype='bandpass', output='sos')
        return sosfiltfilt(sos, data_1d)
    except Exception:
        return data_1d


def _compute_metric(data: np.ndarray, sfreq: float,
                    metric: str, freq_lo: float, freq_hi: float) -> np.ndarray:
    """
    Compute one scalar per channel from (n_channels, n_samples) array.
    Returns shape (n_channels,).

    The frequency band [freq_lo, freq_hi] is applied as a bandpass pre-filter
    for ALL metrics so that changing the band always updates the plot.
    For 'band_power' the filtered signal power is returned directly.
    """
    n_ch = data.shape[0]
    out = np.zeros(n_ch, dtype=float)
    # Bandpass is applied when freq range is valid (> nyquist guard handled in _bandpass)
    use_filter = (freq_lo > 0.0 and freq_hi > freq_lo)
    for i in range(n_ch):
        ch = data[i].astype(float)
        ch -= ch.mean()                          # remove DC
        if use_filter:
            ch = _bandpass(ch, sfreq, freq_lo, freq_hi)
        if metric == 'mean_amplitude':
            out[i] = float(np.mean(np.abs(ch)))
        elif metric == 'rms':
            out[i] = float(np.sqrt(np.mean(ch ** 2)))
        elif metric == 'variance':
            out[i] = float(np.var(ch))
        elif metric == 'band_power':
            out[i] = float(np.mean(ch ** 2))    # already filtered above
        elif metric == 'raw_mean':
            out[i] = float(np.mean(ch))
        else:
            out[i] = float(np.mean(np.abs(ch)))
    return out


def _draw_topo_heatmap(figure: Figure, values: np.ndarray, channel_names,
                       title: str = 'Topographic Heatmap',
                       cmap: str = 'RdBu_r',
                       vmin=None, vmax=None,
                       resolution: int = 100,
                       show_sensors: bool = True,
                       show_head: bool = True,
                       show_colorbar: bool = True,
                       interp: str = 'cubic',
                       colorbar_label: str = 'Value'):
    """
    Render a topographic heatmap onto *figure*.

    Adapted from plot_topographic_heatmap() in
    new_python_eeg_processing_organized/visualization/eeg_visualization.py.

    Parameters
    ----------
    figure      : matplotlib Figure to draw on (cleared first).
    values      : 1-D array, one scalar per channel.
    channel_names : list of channel name strings.
    """
    import matplotlib.patches as mpatches

    figure.clear()
    ax = figure.add_subplot(111)
    head_radius = 1.0

    # ── Match channel names to positions ─────────────────────────────────────
    x_pos, y_pos, valid_vals, valid_names = [], [], [], []

    for i, name in enumerate(channel_names):
        if _is_non_eeg(name):
            continue
        clean = str(name).upper().strip()
        pos = None

        # EGI positions take priority
        if clean in _EGI_128_POS:
            pos = _EGI_128_POS[clean]
        else:
            # Try 10-20 positions (case-insensitive)
            for std, p in _POS_1020.items():
                if std.upper() == clean:
                    pos = p
                    break

        if pos is not None:
            x_pos.append(float(pos[0]))
            y_pos.append(float(pos[1]))
            valid_vals.append(float(values[i]))
            valid_names.append(name)

    x_pos  = np.array(x_pos,  dtype=float)
    y_pos  = np.array(y_pos,  dtype=float)
    valid_vals = np.array(valid_vals, dtype=float)

    if len(x_pos) < 3:
        ax.text(0.5, 0.5,
                f'Not enough matched positions\n'
                f'({len(x_pos)} / {len(channel_names)} channels found)\n\n'
                f'Check channel names are EGI (E1-E128) or 10-20 format.',
                ha='center', va='center', transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', alpha=0.9))
        ax.axis('off')
        # No tight_layout here: figure uses constrained_layout=True which handles
        # spacing automatically; calling tight_layout on a constrained figure
        # that already has a colorbar causes a matplotlib RuntimeError.
        return

    # ── Clip peripheral EGI electrodes to the head circle ────────────────────
    r = np.sqrt(x_pos ** 2 + y_pos ** 2)
    outside = r > head_radius
    if outside.any():
        x_pos[outside] = x_pos[outside] / r[outside] * head_radius
        y_pos[outside] = y_pos[outside] / r[outside] * head_radius

    # ── Interpolation grid ────────────────────────────────────────────────────
    xi = np.linspace(-head_radius, head_radius, resolution)
    yi = np.linspace(-head_radius, head_radius, resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        Zi = griddata((x_pos, y_pos), valid_vals, (Xi, Yi), method=interp)
    except Exception:
        Zi = griddata((x_pos, y_pos), valid_vals, (Xi, Yi), method='linear')

    # Fill NaN with nearest-neighbour
    Zi_nn = griddata((x_pos, y_pos), valid_vals, (Xi, Yi), method='nearest')
    Zi = np.where(np.isnan(Zi), Zi_nn, Zi)

    # ── Colour limits ─────────────────────────────────────────────────────────
    if vmin is None:
        vmin = float(np.nanmin(valid_vals))
    if vmax is None:
        vmax = float(np.nanmax(valid_vals))
    if vmin == vmax:
        vmax = vmin + 1e-6

    # Use TwoSlopeNorm for diverging colourmaps when data straddles zero
    norm = None
    if ('RdBu' in cmap or 'seismic' in cmap.lower() or
            'bwr' in cmap.lower() or 'coolwarm' in cmap.lower()):
        if vmin < 0 < vmax:
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, norm=norm,
                       vmin=(None if norm else vmin),
                       vmax=(None if norm else vmax),
                       shading='gouraud', rasterized=True)

    # Smooth circular clip (avoids staircase artefact at boundary)
    clip_circle = mpatches.Circle((0, 0), head_radius,
                                  transform=ax.transData,
                                  facecolor='none', edgecolor='none')
    ax.add_patch(clip_circle)
    im.set_clip_path(clip_circle)

    # Contour lines, also clipped
    try:
        cs = ax.contour(Xi, Yi, Zi,
                        levels=np.linspace(vmin, vmax, 8),
                        colors='black', alpha=0.3, linewidths=0.5)
        try:
            cs.set_clip_path(clip_circle)
        except AttributeError:
            for col in cs.collections:
                col.set_clip_path(clip_circle)
    except Exception:
        pass

    # ── Head outline ──────────────────────────────────────────────────────────
    if show_head:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta),
                'k-', linewidth=2.0, zorder=4)
        # Nose (triangle pointing up)
        nw, nh = 0.12, 0.15
        ax.plot([0, -nw, 0, nw, 0],
                [head_radius,
                 head_radius + nh * 0.5,
                 head_radius + nh,
                 head_radius + nh * 0.5,
                 head_radius], 'k-', linewidth=1.5, zorder=4)
        # Ears (left and right arcs)
        ew, eh = 0.08, 0.20
        et = np.linspace(-np.pi / 3, np.pi / 3, 30)
        ax.plot(-head_radius - ew / 2 + ew * np.cos(et),
                eh * np.sin(et), 'k-', linewidth=1.5, zorder=4)
        ax.plot( head_radius + ew / 2 - ew * np.cos(et),
                eh * np.sin(et), 'k-', linewidth=1.5, zorder=4)

    # ── Electrode dots + labels ───────────────────────────────────────────────
    if show_sensors and len(x_pos) > 0:
        ax.scatter(x_pos, y_pos, s=20, c='black', marker='o',
                   zorder=6, alpha=0.75, linewidths=0)
        n = len(valid_names)
        fs = 3.5 if n > 100 else (4.5 if n > 64 else (5.5 if n > 32 else 7.0))
        for x, y, nm in zip(x_pos, y_pos, valid_names):
            dx, dy = x * 0.09, y * 0.09
            ax.annotate(nm, (x + dx, y + dy),
                        ha=('left' if x >= 0 else 'right'), va='center',
                        fontsize=fs, alpha=0.85, zorder=7)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    if show_colorbar:
        cbar = figure.colorbar(im, ax=ax, shrink=0.80, pad=0.05)
        cbar.set_label(colorbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.17, 1.28)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    # constrained_layout=True (set on figure init) handles spacing; do NOT call
    # tight_layout — it raises RuntimeError once a colorbar exists.


# ─────────────────────────────────────────────────────────────────────────────
# TopoHeatmapWindow
# ─────────────────────────────────────────────────────────────────────────────

class TopoHeatmapWindow(QWidget):
    """
    Standalone window that shows a topographic heatmap for the currently
    visible segment of the main EEGViewerWidget.

    • Updates automatically in real time whenever the main window scrolls,
      changes its time window, or changes its channel selection.
    • Supports both EGI 128-channel and standard 10-20 montages.
    • For EGI data, offers one-click conversion to 10-20 format.
    • Own controls: metric, frequency band, colourmap, resolution, clim.
    """

    # Metric index → internal key
    _METRIC_KEYS = [
        'mean_amplitude',   # 0
        'rms',              # 1
        'variance',         # 2
        'band_power',       # 3
        'raw_mean',         # 4
    ]
    _METRIC_LABELS = {
        'mean_amplitude': 'Mean |µV|',
        'rms':            'RMS (µV)',
        'variance':       'Variance (µV²)',
        'band_power':     'Band Power',
        'raw_mean':       'Mean (µV)',
    }
    _BAND_PRESETS = [
        ('δ',  0.5,  4.0),
        ('θ',  4.0,  8.0),
        ('α',  8.0, 13.0),
        ('β', 13.0, 30.0),
        ('γ', 30.0, 80.0),
    ]

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topographic Heatmap")
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(720, 580)
        self.resize(960, 680)

        self._viewer = viewer

        # Debounce: only redraw at most every 150 ms during rapid scrolling
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._draw)

        self._build_ui()
        self._draw()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)

        # Left: canvas
        self._figure = Figure(figsize=(6, 5.5), dpi=100, constrained_layout=True)
        self._figure.patch.set_facecolor('#f8f8f8')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self._canvas, stretch=1)

        # Right: settings in a scroll area
        ctrl = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(248)
        scroll.setWidget(ctrl)
        main.addWidget(scroll)

    def _build_controls(self) -> QWidget:
        outer = QWidget()
        vbox = QVBoxLayout(outer)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(8)

        # ── Metric ────────────────────────────────────────────────────────────
        g = QGroupBox("Metric")
        gl = QVBoxLayout(g)
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([
            "Mean Amplitude (|µV|)",
            "RMS Amplitude (µV)",
            "Variance (µV²)",
            "Band Power",
            "Raw Mean (µV)",
        ])
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        gl.addWidget(self._metric_combo)
        vbox.addWidget(g)

        # ── Frequency band ────────────────────────────────────────────────────
        self._freq_group = QGroupBox("Frequency Band Filter")
        self._freq_group.setToolTip(
            "Bandpass filter applied BEFORE computing any metric.\n"
            "Set Low = 0.1 and High = 500 to effectively disable filtering.")
        fg = QGridLayout(self._freq_group)

        fg.addWidget(QLabel("Low (Hz):"), 0, 0)
        self._freq_lo = QDoubleSpinBox()
        self._freq_lo.setRange(0.1, 200.0)
        self._freq_lo.setValue(8.0)
        self._freq_lo.setSingleStep(0.5)
        self._freq_lo.setDecimals(1)
        self._freq_lo.valueChanged.connect(self._schedule)
        fg.addWidget(self._freq_lo, 0, 1)

        fg.addWidget(QLabel("High (Hz):"), 1, 0)
        self._freq_hi = QDoubleSpinBox()
        self._freq_hi.setRange(0.5, 500.0)
        self._freq_hi.setValue(13.0)
        self._freq_hi.setSingleStep(0.5)
        self._freq_hi.setDecimals(1)
        self._freq_hi.valueChanged.connect(self._schedule)
        fg.addWidget(self._freq_hi, 1, 1)

        preset_row = QHBoxLayout()
        for name, lo, hi in self._BAND_PRESETS:
            btn = QPushButton(name)
            btn.setMaximumWidth(36)
            btn.setToolTip(f"{name}: {lo}–{hi} Hz")
            btn.clicked.connect(lambda _, l=lo, h=hi: self._set_band(l, h))
            preset_row.addWidget(btn)
        fg.addLayout(preset_row, 2, 0, 1, 2)

        self._freq_group.setEnabled(True)   # always accessible; used by Band Power metric
        self._freq_group.setToolTip("Frequency range used when 'Band Power' metric is selected")
        vbox.addWidget(self._freq_group)

        # ── Format ────────────────────────────────────────────────────────────
        g = QGroupBox("Channel Format")
        gl = QVBoxLayout(g)
        self._fmt_label = QLabel("Detected: —")
        self._fmt_label.setStyleSheet("font-size: 10px; color: #444;")
        gl.addWidget(self._fmt_label)
        self._convert_check = QCheckBox("Convert EGI → 10-20")
        self._convert_check.setToolTip(
            "Average EGI electrodes into standard 10-20 positions.\n"
            "Available only when EGI format is detected."
        )
        self._convert_check.setEnabled(False)
        self._convert_check.stateChanged.connect(self._schedule)
        gl.addWidget(self._convert_check)
        vbox.addWidget(g)

        # ── Colourmap ─────────────────────────────────────────────────────────
        g = QGroupBox("Colormap")
        gl = QVBoxLayout(g)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            'RdBu_r', 'viridis', 'hot', 'jet', 'plasma',
            'inferno', 'magma', 'seismic', 'coolwarm', 'bwr',
        ])
        self._cmap_combo.currentIndexChanged.connect(self._schedule)
        gl.addWidget(self._cmap_combo)
        vbox.addWidget(g)

        # ── Display options ───────────────────────────────────────────────────
        g = QGroupBox("Display")
        gl = QGridLayout(g)

        gl.addWidget(QLabel("Resolution:"), 0, 0)
        self._res_spin = QSpinBox()
        self._res_spin.setRange(30, 300)
        self._res_spin.setValue(100)
        self._res_spin.setSingleStep(10)
        self._res_spin.setSuffix(" px")
        self._res_spin.valueChanged.connect(self._schedule)
        gl.addWidget(self._res_spin, 0, 1)

        self._sensors_check = QCheckBox("Show electrodes")
        self._sensors_check.setChecked(True)
        self._sensors_check.stateChanged.connect(self._schedule)
        gl.addWidget(self._sensors_check, 1, 0, 1, 2)

        self._head_check = QCheckBox("Show head outline")
        self._head_check.setChecked(True)
        self._head_check.stateChanged.connect(self._schedule)
        gl.addWidget(self._head_check, 2, 0, 1, 2)

        vbox.addWidget(g)

        # ── Colour limits ─────────────────────────────────────────────────────
        g = QGroupBox("Color Limits")
        gl = QGridLayout(g)

        self._auto_clim = QCheckBox("Auto (symmetric)")
        self._auto_clim.setChecked(True)
        self._auto_clim.stateChanged.connect(self._on_auto_clim)
        gl.addWidget(self._auto_clim, 0, 0, 1, 2)

        gl.addWidget(QLabel("Min:"), 1, 0)
        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e7, 1e7)
        self._vmin_spin.setValue(-100.0)
        self._vmin_spin.setDecimals(3)
        self._vmin_spin.setEnabled(False)
        self._vmin_spin.valueChanged.connect(self._schedule)
        gl.addWidget(self._vmin_spin, 1, 1)

        gl.addWidget(QLabel("Max:"), 2, 0)
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e7, 1e7)
        self._vmax_spin.setValue(100.0)
        self._vmax_spin.setDecimals(3)
        self._vmax_spin.setEnabled(False)
        self._vmax_spin.valueChanged.connect(self._schedule)
        gl.addWidget(self._vmax_spin, 2, 1)

        vbox.addWidget(g)

        # ── Buttons ───────────────────────────────────────────────────────────
        refresh_btn = QPushButton("🔄  Refresh Now")
        refresh_btn.clicked.connect(self._draw)
        vbox.addWidget(refresh_btn)

        export_btn = QPushButton("💾  Export Image")
        export_btn.clicked.connect(self._export)
        vbox.addWidget(export_btn)

        vbox.addStretch()
        return outer

    # ── Public slot called by the main viewer ─────────────────────────────────

    def on_viewer_changed(self):
        """Called by EEGViewerWidget.view_changed signal."""
        self._schedule()

    # ── Internal slots ────────────────────────────────────────────────────────

    def _schedule(self):
        self._timer.start()  # restart 150 ms debounce

    def _on_metric_changed(self, idx):
        # freq band group is always enabled; only used when Band Power (idx 3) is active
        self._schedule()

    def _on_auto_clim(self, state):
        manual = (state != Qt.Checked)
        self._vmin_spin.setEnabled(manual)
        self._vmax_spin.setEnabled(manual)
        self._schedule()

    def _set_band(self, lo: float, hi: float):
        self._freq_lo.blockSignals(True)
        self._freq_hi.blockSignals(True)
        self._freq_lo.setValue(lo)
        self._freq_hi.setValue(hi)
        self._freq_lo.blockSignals(False)
        self._freq_hi.blockSignals(False)
        self._schedule()

    # ── Core drawing ─────────────────────────────────────────────────────────

    def _draw(self):
        viewer = self._viewer
        if viewer is None or viewer.eeg_signal is None:
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "No EEG data loaded.\nOpen a file in the main viewer.",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=13, color='gray',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
            ax.axis('off')
            self._canvas.draw_idle()
            return

        eeg = viewer.eeg_signal
        sfreq = eeg.sampling_rate

        # ── Extract current segment from the viewer ────────────────────────
        if eeg.data.ndim == 3:
            data_full = eeg.data[viewer.selected_epoch]
        else:
            data_full = eeg.data

        n_ch, n_samples = data_full.shape
        t0   = viewer.current_time_start
        twin = viewer.time_window
        s0 = max(0, int(t0 * sfreq))
        s1 = min(n_samples, int((t0 + twin) * sfreq))
        if s1 <= s0:
            s1 = min(n_samples, s0 + max(1, int(sfreq)))

        seg = data_full[:, s0:s1]           # (n_ch, n_samples_in_window)

        channel_names = list(eeg.channel_names or [f'Ch{i+1}' for i in range(n_ch)])
        is_egi = _is_egi_format(channel_names)

        # Update format label and EGI→10-20 checkbox availability
        self._fmt_label.setText(
            f"Detected: {'EGI (E1…E128)' if is_egi else '10-20 / Other'}"
        )
        self._convert_check.setEnabled(is_egi)
        if not is_egi:
            self._convert_check.blockSignals(True)
            self._convert_check.setChecked(False)
            self._convert_check.blockSignals(False)

        # ── Read settings ──────────────────────────────────────────────────
        metric_idx = self._metric_combo.currentIndex()
        metric     = self._METRIC_KEYS[metric_idx]
        freq_lo    = self._freq_lo.value()
        freq_hi    = self._freq_hi.value()
        cmap       = self._cmap_combo.currentText()
        resolution = self._res_spin.value()
        show_snsr  = self._sensors_check.isChecked()
        show_head  = self._head_check.isChecked()
        auto_clim  = self._auto_clim.isChecked()
        vmin       = None if auto_clim else self._vmin_spin.value()
        vmax       = None if auto_clim else self._vmax_spin.value()
        convert    = self._convert_check.isChecked() and is_egi

        # ── Compute per-channel metric ─────────────────────────────────────
        all_values = _compute_metric(seg, sfreq, metric, freq_lo, freq_hi)

        if convert:
            # Average EGI channels into 10-20 electrode positions
            name_to_idx = {str(ch).upper(): i for i, ch in enumerate(channel_names)}
            plot_vals  = []
            plot_names = []
            for elec, egi_chs in _EGI_TO_1020.items():
                matched = [
                    all_values[name_to_idx[ch.upper()]]
                    for ch in egi_chs
                    if ch.upper() in name_to_idx
                ]
                if matched:
                    plot_vals.append(float(np.mean(matched)))
                    plot_names.append(elec)
            plot_vals  = np.array(plot_vals, dtype=float)
        else:
            plot_vals  = all_values
            plot_names = channel_names

        # ── Build colourbar label and title ───────────────────────────────
        cb_label = self._METRIC_LABELS.get(metric, 'Value')
        if metric == 'band_power':
            cb_label = f'Band Power {freq_lo:.1f}–{freq_hi:.1f} Hz (µV²)'

        fmt_tag = ' [10-20]' if convert else (' [EGI]' if is_egi else '')
        t_range = f'{t0:.2f}–{t0 + twin:.2f} s'
        title   = f'Topo{fmt_tag} | {cb_label} | {t_range}'

        # ── Draw ──────────────────────────────────────────────────────────
        _draw_topo_heatmap(
            self._figure, plot_vals, plot_names,
            title=title, cmap=cmap, vmin=vmin, vmax=vmax,
            resolution=resolution, show_sensors=show_snsr,
            show_head=show_head, show_colorbar=True,
            interp='cubic', colorbar_label=cb_label,
        )
        self._canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Topograph", "topograph.png",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)"
        )
        if path:
            try:
                self._figure.savefig(path, dpi=200, bbox_inches='tight')
                QMessageBox.information(self, "Saved", f"Topograph saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

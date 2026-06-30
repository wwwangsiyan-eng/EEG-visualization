# User Manual

## 1. Installation

### Windows
1. Open **Command Prompt** or **PowerShell**.
2. Navigate to the folder containing the viewer:
   ```cmd
   cd path\to\EEG_viewer
   ```
3. (Optional) Create and activate a virtual environment:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
   *Note: If you receive an error saying `pip is not recognized as an internal or external command`, you can easily install/enable it by running:*
   ```cmd
   python -m ensurepip --upgrade
   ```
   *Alternatively, you may need to re-run the Python installer from python.org and ensure you check both the **"Add Python to PATH"** and **"pip"** boxes during installation.*

5. Launch the viewer:
   ```cmd
   python eeg_viewer.py
   ```

### Mac
1. Open **Terminal**.
2. Navigate to the folder containing the viewer:
   ```bash
   cd path/to/EEG_viewer
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
5. Launch the viewer:
   ```bash
   python3 eeg_viewer.py
   ```

---

## 2. Using the Interface

### Supported File Formats
You can load the following raw EEG & electrophysiology data formats using the viewer:
* **European Data Format** (`.edf`)
* **BioSemi Data Format** (`.bdf`)
* **EGI / Philips** (`.mff`)
* **BCI2000 & Curry** (`.dat`)
* **Blackrock** (`.nsx`, `.ns2`)
* **Neuralynx** (`.ncs`)
* **Intan** (`.rhd`)

### Top Toolbar

#### File
* **📂 Open File** — Opens a file browser so you can select and load an EEG dataset.
* **💾 Export Plot** — Saves a screenshot of the currently visible traces. Supported formats: `.png`, `.jpg`, `.pdf` (vector), `.svg` (vector).

#### Navigation & Scale
* **🔍+ Zoom In** — Shrinks the visible time window so you can see finer time-domain detail.
* **🔍- Zoom Out** — Expands the visible time window to fit more of the recording on screen.
* **⏮ Start** — Jumps the view back to the very beginning of the recording.
* **⏭ End** — Jumps the view to the very end of the recording.
* **📐 Auto Scale** — Automatically recalculates the vertical amplitude scale so traces fit the window without clipping.
* **🔄 Reset View** — Restores the time window and amplitude scaling to their default states and shows all channels.
* **📊 Channels** — Opens a checklist where you can show or hide individual channels. Includes quick-select options (All, None, First 16, Every Nth, range selection).

#### Analysis Windows
Each of these opens a separate, independent window that you can keep open alongside the main viewer.

* **📋 Signal Info** — Displays a detailed table of signal metadata: data type, shape, sampling rate, duration, and per-channel amplitude ranges.
* **🌊 FFT View** — Shows the selected channel's EEG trace alongside a real-time FFT spectrogram (frequency content over time), time-locked to the current view position. Use the **FFT Frequency Range** controls (Low / High Hz) to zoom in on a specific frequency range in the spectrogram.
* **🔬 Filter View** — Displays three stacked panels for a single channel: the raw (or highpass-filtered) trace, a user-adjustable bandpass trace (default: Theta 4–8 Hz), and a second bandpass trace (default: Gamma 30–80 Hz).
* **🧠 Topo Map** — Opens a topographic heatmap window showing the spatial distribution of EEG activity across the scalp. Updates automatically in real time as you scroll the main viewer. See [Topographic Heatmap Window](#topographic-heatmap-window) below.

### Bottom Control Panel

The control panel is split into two rows.

#### Row 1 — Navigation & Scale

* **Time Window** — Set the length of the visible time window in seconds. Quick-preset buttons (**1s, 5s, 10s, 30s, 60s**) let you snap to common durations instantly.
* **Amplitude Scale** — Adjust the vertical stretch of the waveforms. Use the spinbox for fine control or the **½×** and **2×** buttons to halve or double the scale quickly.
* **Ch. Spacing** — Increase or decrease the vertical gap between adjacent channels to prevent traces from overlapping.
* **Epoch** *(appears only with 3D epoched data)* — Cycle through individual recording epochs.

#### Row 2 — Playback, Filters & Display

* **Auto Scroll** — Click **▶ Play** to smoothly animate through the recording in real time. Adjust the **Speed** box (e.g., `20×`) to play faster or slower than real-time. Press again (or press **Space**) to pause.
* **Highpass Filter** — Check **Enable** and set a cutoff frequency to apply a live zero-phase highpass filter. This removes slow DC drift and baseline wander without modifying the original file.
* **Show Grid** — Toggles the background reference grid on and off.
* **10-20 Format** — For high-density EGI recordings, this averages electrode clusters to produce traces at standard 10-20 positions (Fp1, Fp2, F3, Fz … O1, Oz, O2). Does nothing for non-EGI data.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Step backward / forward by 10 % of the current window |
| `Home` | Jump to the start of the recording |
| `End` | Jump to the end of the recording |
| `Space` | Toggle auto-scroll play / pause |
| `↑` / `↓` | Increase / decrease amplitude scale (×1.5 / ×0.67) |

### Bottom Scrollbar
Drag the horizontal scrollbar directly below the plot to navigate through the timeline manually. The timestamp label on the right shows the exact start and end time of the current view.

### Closing the App
Click the standard close button (`×`) at the top corner of the application window.

---

## 3. Topographic Heatmap Window

Click **🧠 Topo Map** in the toolbar to open the topographic heatmap. It renders a 2-D head map showing how a chosen scalar metric is distributed across all electrodes, and updates automatically whenever you scroll or change the time window in the main viewer.

### Metric
Choose what each electrode's colour represents:

| Metric | Description |
|--------|-------------|
| Mean Amplitude (|µV|) | Average absolute deflection from DC baseline |
| RMS Amplitude (µV) | Root-mean-square amplitude |
| Variance (µV²) | Signal variance |
| Band Power | Mean squared amplitude (power) in the selected band |
| Raw Mean (µV) | Arithmetic mean (signed; useful for DC offset maps) |

### Frequency Band Filter
The **Low Hz** and **High Hz** spinboxes define a bandpass pre-filter applied **before** computing any metric. Changing the band always updates the map, regardless of which metric is selected — so you can view, for example, "alpha-band RMS" or "theta-band variance". Quick-preset buttons (δ, θ, α, β, γ) snap to standard clinical bands.

To view broadband (unfiltered) activity, set Low = 0.1 Hz and High to the Nyquist limit (half your sampling rate).

### Colormap
Choose from a set of perceptual and diverging colormaps. **RdBu_r** and **seismic** work well for signed metrics (Raw Mean). **viridis**, **hot**, and **plasma** suit unsigned power metrics.

### EGI / 10-20 Format
For high-density EGI 128-channel recordings the window shows all 128 electrodes by default. Check **EGI → 10-20** to collapse the map to 24 standard 10-20 positions (each averaged over the surrounding EGI cluster). Uncheck to return to the full 128-channel view.

### Interpolation Grid
The **Resolution** spinbox (pixels) controls the spatial interpolation grid size. Higher values give smoother maps but take slightly longer to render.

### Colour Limits
By default, the colour scale is fitted automatically to the data range. Uncheck **Auto** to enter manual **Min** and **Max** values — useful for comparing maps across different time segments on the same scale.

### Save Figure
Click **💾 Save Figure** to export the current heatmap as PDF (vector), SVG (vector), or PNG. PDF and SVG formats are fully editable in Adobe Illustrator or Inkscape.

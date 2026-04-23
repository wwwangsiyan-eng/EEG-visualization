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
* **EGI / Phillips** (`.mff`)
* **BCI2000 & Curry** (`.dat`)
* **Blackrock / Neuralynx / Intan** (`.nsx`, `.ns2`, `.ncs`, `.rhd`)

### Top Toolbar Buttons
* **📂 Open File**: Opens a file browser window allowing you to select and load an EEG dataset.
* **💾 Export Plot**: Opens a "Save As" menu giving you the option to save a screenshot of the currently visible traces. You can export as `.png`, `.jpg`, `.pdf` (Vectorized), or `.svg` (Vectorized).
* **🔍+ Zoom In**: Decreases the visible time window to enlarge the waveforms horizontally, letting you see finer time-domain details.
* **🔍- Zoom Out**: Increases the visible time window so more of the recording fits on the screen at once.
* **⏮ Start**: Instantly jumps the timeline view back to the very beginning of the recording.
* **⏭ End**: Instantly jumps the timeline view to the very end of the file.
* **📐 Auto Scale**: Automatically recalculates and adjusts the vertical amplitude scale so the traces perfectly fit your window without clipping.
* **🔄 Reset View**: Restores both the time window and the amplitude scaling to their original default states.
* **📊 Select Channels**: Opens a separate checklist window where you can specifically choose which channels to show or hide.

### Bottom Control Panel
* **Time Window (s)**: Allows you to manually input the exact length of time displayed on the screen. You can also click the quick presets (**1s, 5s, 10s, 30s, 60s**) to snap to a specific window size.
* **Amplitude Scale**: Use the spinbox or the **− / +** buttons to stretch or shrink the vertical height of the brainwaves.
* **Channel Spacing**: Increases or decreases the vertical distance separating adjacent channels from one another to prevent lines from overlapping.
* **Epoch Box**: *(Only appears if you load structured 3D epoched data)* Allows you to cycle through different recording epochs.
* **Grid**: Toggles the background graphing grid on and off.
* **Auto Scroll**: Click **▶ Play** to automatically move through the timeline smoothly. Modify the **Speed** box (e.g., 20.0x) to make the scrolling faster or slower compared to real-time.
* **Highpass Filter**: Check **Enable** and set the frequency (Hz) to apply a live filter. This removes slow baseline drifts (DC offset) and straightens out floating traces automatically without altering the raw file.
* **10-20 Format**: If a high-density EGI rig was used during recording, checking this evaluates mapped clusters and automatically averages them to mimic standard 10-20 recording locations.

### Other Useful Regions
* **Bottom Scrollbar**: Drag the horizontal scrollbar located just below the plotting area to manually navigate through the timeline.
* **Right Metadata Panel**: Shows information about the current file such as sampling rate, duration, hardware layout, and a localized chart showing live microvolt (µV) ranges for the visible channels.
* **Closing the App**: Simply click the standard close button (the 'X') at the top corner of the application window.
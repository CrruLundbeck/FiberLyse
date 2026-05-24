# FiberLyse

**FiberLyse** is a desktop program for analyzing fiber photometry CSV files.

It is made for users who want a visual way to inspect and process fiber photometry recordings without writing code. You load your CSV files, press **Run analysis**, and FiberLyse shows the raw signal, artifact removal, fitting, `ΔF/F`, `zF`, smoothed traces, frequency analysis, and AUC results.

The main file is:

```text
FiberLyseV19.py
```

---

## What FiberLyse does

FiberLyse can:

- open up to **8 CSV files** at once
- analyze multiple signal channels such as `G0`, `G1`, `G2`
- separate isosbestic and excitatory signals
- remove sharp artifacts
- optionally fill removed artifact points with interpolation
- calculate `ΔF/F`
- calculate global or baseline-based `zF`
- smooth traces for easier viewing
- show frequency-band plots
- calculate AUC for selected time intervals
- export graphs as `.png`, `.svg`, or `.pdf`
- export plotted data to Excel `.xlsx`
- edit graph titles, labels, colors, axes, and legends inside the program

---

## Installation

### 1. Install Python

FiberLyse needs Python 3.

Python 3.9 or newer is recommended.

Download Python here:

```text
https://www.python.org/downloads/
```

On Windows, make sure to tick:

```text
Add Python to PATH
```

during installation.

---

### 2. Install the required packages

Open a terminal, command prompt, or PowerShell in the same folder as `FiberLyseV19.py`.

Run:

```bash
pip install numpy pandas matplotlib scipy
```

On Windows, this may work better:

```bash
py -m pip install numpy pandas matplotlib scipy
```

On macOS or Linux, this may work better:

```bash
python3 -m pip install numpy pandas matplotlib scipy
```

---

## Required packages

FiberLyse uses these Python packages:

| Package | Used for |
|---|---|
| `numpy` | calculations |
| `pandas` | reading CSV files |
| `matplotlib` | making graphs |
| `scipy` | statistics and frequency filtering |
| `tkinter` | opening the desktop window |

`tkinter` usually comes with Python.

On some Linux systems, it may need to be installed separately:

```bash
sudo apt install python3-tk
```

---

## Starting FiberLyse

Run:

```bash
python FiberLyseV19.py
```

On Windows, use this if the command above does not work:

```bash
py FiberLyseV19.py
```

On macOS or Linux, use this if needed:

```bash
python3 FiberLyseV19.py
```

FiberLyse should open as a desktop window.

---

## CSV file format

Your CSV file must contain these columns:

| Column | Meaning |
|---|---|
| `SystemTimestamp` | time information |
| `LedState` | tells FiberLyse what type of signal each row is |
| `G0`, `G1`, etc. | signal channels |

FiberLyse expects these `LedState` values:

| LedState | Meaning |
|---|---|
| `1` | isosbestic/reference signal |
| `2` | excitatory/signal channel |
| `7` | optional start marker |

Example:

```text
SystemTimestamp,LedState,G0,G1
123456.00,7,0,0
123456.05,1,100.2,98.7
123456.10,2,130.4,125.1
```

Each `G` column is analyzed as its own channel.

---

## Basic use

### 1. Open the program

Start FiberLyse by running:

```bash
python FiberLyseV19.py
```

---

### 2. Add CSV files

Click:

```text
Add CSV(s)...
```

Choose one or more CSV files.

You can load up to **8 files** at the same time.

---

### 3. Check the settings

At the top of the window, you can change the main analysis settings.

For a first run, the default settings are usually a good starting point.

Important settings:

| Setting | What it means |
|---|---|
| `Enable artifact remover` | removes sudden sharp signal jumps |
| `Factor` | controls how sensitive artifact detection is |
| `Pad` | removes extra points around artifacts |
| `Require shared artifacts` | only removes artifacts found in both signals |
| `Acq FPS` | recording frame rate |
| `Smooth win` | amount of smoothing |
| `Linear interpolate holes` | fills removed artifact points |

---

### 4. Run the analysis

Click:

```text
Run analysis
```

FiberLyse will process the selected files.

After analysis, use the **File** dropdown to choose which loaded file to view.

Each `G` channel appears as its own tab.

---

## Main plot tabs

After running the analysis, FiberLyse shows several tabs for each channel.

### Raw

Shows the original isosbestic and excitatory signals.

Use this tab first to check that the file loaded correctly.

---

### Slope normality

Shows slope information from the raw signal.

This can help you see sudden jumps or unusual changes in the recording.

---

### Artifact remover

Shows which points were removed as artifacts.

In this tab:

- grey lines are the raw signal
- red points are detected artifacts
- orange sections are interpolated/fill-in sections
- cleaned lines show the signal after artifact removal

---

### Fit

Shows the fitted isosbestic signal compared with the excitatory signal.

FiberLyse uses this fit to calculate `ΔF/F`.

You can drag across the plot to choose a new fitting window.

---

### ΔF/F and zF

This tab shows the main processed signal.

You can choose between:

| Option | Meaning |
|---|---|
| `ΔF/F` | standard delta F over F |
| `zF (global, GUI)` | z-score using the whole trace |
| `zF - interval based` | z-score using a chosen baseline interval |

For interval-based `zF`, enter the baseline start and end time, then click:

```text
Apply interval
```

---

### Smoothed trace

Shows a smoothed version of the processed signal.

This is useful for viewing the signal more clearly.

The amount of smoothing is controlled by:

```text
Smooth win
```

---

### Frequency analysis

Shows different frequency bands of the `ΔF/F` signal.

This can be useful for checking slower and faster signal components.

---

## AUC calculation

Each graph has a button called:

```text
AUC interval...
```

Use this to calculate the area under the curve for a chosen interval.

You enter:

- start time
- end time
- baseline value

FiberLyse reports:

- signed AUC
- absolute AUC
- positive AUC
- negative AUC
- coverage
- mean minus baseline
- number of points used

You can also open the AUC tool with:

```text
Ctrl + U
```

---

## Exporting results

Each graph has export buttons.

### Save graph

Click:

```text
Save this graph...
```

You can save the graph as:

```text
.png
.svg
.pdf
```

### Export data

Click:

```text
Export data (Excel)...
```

This saves the data behind the current graph as an Excel file.

---

## Editing graphs

FiberLyse lets you make simple figure edits directly inside the program.

| Action | How to do it |
|---|---|
| Rename a title or axis label | double-click the text |
| Rename a legend label | double-click the legend text |
| Change a line color | right-click the line or legend item |
| Add a vertical time marker | press `Ctrl + I` |
| Change axis range or tick spacing | press `Ctrl + K` |
| Move or resize the legend | drag the legend or press `Ctrl + L` |
| Show file number mapping | press `Ctrl + J` |

---

## Recommended workflow

A typical workflow is:

1. Open FiberLyse.
2. Add your CSV file or files.
3. Click **Run analysis**.
4. Check the **Raw** tab.
5. Check the **Artifact remover** tab.
6. Check the **Fit** tab.
7. View `ΔF/F` or `zF`.
8. Use the smoothed trace for easier viewing.
9. Use **AUC interval...** for selected time windows.
10. Export graphs or Excel data.

---


- the final `ΔF/F` or `zF` trace

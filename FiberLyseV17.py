from __future__ import annotations

import sys
import os
import re
import argparse
import threading
import zipfile
from xml.sax.saxutils import escape as _xml_escape
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
from statistics import NormalDist

# =============================================================================
# Fiberlyse – Photometry analysis GUI (single-file + batch mode)
# =============================================================================
#
# This program is a *graphical* tool (Tkinter + Matplotlib) for analyzing fiber
# photometry CSV exports that contain:
#
#   - A timestamp column:       SystemTimestamp
#   - A LED state column:       LedState
#   - One or more signal cols:  G0, G1, G2, ...  (each typically corresponds to a mouse)
#
# The acquisition alternates between:
#   - ISO_STATE (isosbestic channel)
#   - EXC_STATE (excitatory channel)
#
# For every G* column the GUI:
#   1) Splits the raw data into ISO and EXC samples using LedState.
#   2) Optionally detects movement/LED artifacts using the *derivative* (MAD-based).
#      Artifacts are removed by inserting NaNs ("holes").
#   3) Optionally linearly interpolates those holes (for the *display/fit* pipeline).
#      IMPORTANT: Frequency analysis always uses the *non-interpolated* pipeline.
#   4) Aligns ISO samples to the EXC timestamps (no time interpolation – just mapping).
#   5) Fits:  EXC ≈ a * ISO + b  (within a user-selected fit window).
#   6) Computes ΔF/F and two z-score options (global and interval-based).
#
# The GUI contains:
#   - Per-mouse tabs: Raw, Slope normality, Artifact remover, Fit, Normalization,
#     Normalization (smoothed), and Frequency analysis (ΔF/F only).
#   - Batch mode (NEW): load up to 8 CSV files at once and:
#       * Compare selected mice by overlaying their *normalized* traces.
#       * Average selected mice (mean ± SEM) from their *normalized* traces.
#     These batch plots always use the currently selected “Normalization view”.
#
# Interactive editing (already present):
#   - Double-click titles / axis labels / legend text to rename.
#   - Right-click a legend entry or plot element to recolor it.
#
# Time markers (NEW):
#   - Press Ctrl+I to open a small dialog.
#   - Type a time in seconds and press Enter/OK to add a vertical marker line.
#   - To remove the *last* marker line: press Ctrl+Backspace in that dialog
#     (this matches the requested "Ctrl+I + Backspace" workflow).
#   - Up to 4 markers are supported; each uses a different line style.
#
# NOTE:
#   This file is intentionally very “verbose” and includes many comments so that
#   someone without Python experience can follow what happens step-by-step.
# =============================================================================
# TABLE OF CONTENTS - QUICK NAVIGATION GUIDE
# =============================================================================
# Use this table to jump to specific sections of the code.
# In VS Code: Press Ctrl+G to open "Go to Line" dialog, type line number
#
# SECTION                              | LINE  | WHAT'S THERE
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS & SETUP                      | 81    | Required libraries (tkinter, matplotlib, numpy)
# CONFIGURATION & CONSTANTS            | 112   | Hardware settings, defaults, frequency bands
# CALCULATION MAP GUIDE                | 228   | Reference for where plot calculations happen
# HELPER FUNCTIONS                     | 533   | Utility functions (find_g_columns, subsample, etc.)
#
# ┌─────── CALCULATION FUNCTIONS ──────────────────────────────────────────┐
# │ Functions that perform the mathematical calculations                    │
# ├─────────────────────────────────────────────────────────────────────────┤
# Artifact Detection                   | 555   | detect_artifacts_by_derivative()
# Slope Analysis & Normality Testing   | 638   | compute_slopes(), qq_plot_points(), normality tests
# Smoothing & Filtering                | 1008  | smooth_like_batch(), Butterworth, FFT
# Z-Score Normalization                | 1202  | zscore_global_gui(), zscore_interval_based()
# │
# └─────────────────────────────────────────────────────────────────────────┘
#
# DATA STRUCTURES                      | 1303  | ChannelResult class (holds all computed data)
#
# ┌─────── CORE ALGORITHM ─────────────────────────────────────────────────┐
# │ Main orchestrator functions that control the analysis workflow         │
# ├─────────────────────────────────────────────────────────────────────────┤
# Linear Fit & ΔF/F Computation        | 1419  | _compute_fit_and_downstream() [CORE CALCULATION]
# Fit Orchestrator                     | 1484  | recompute_fit_and_downstream()
# Normalization Computation            | 1381  | recompute_normalizations()
# Interpolation Control                | 1542  | set_interpolation_mode()
# Artifact Pipeline                    | 1556  | recompute_artifact_pipeline_inplace()
# MAIN ANALYSIS PIPELINE               | 1637  | analyze_csv() [Entry point: loads CSV, does all processing]
# │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────── GUI COMPONENTS (Display & User Interaction) ──────────────────────┐
# │ Classes that build and manage the visible windows and controls           │
# ├─────────────────────────────────────────────────────────────────────────┤
# PlotTabTk CLASS                      | 1813  | Single matplotlib figure with toolbar
#   ├─ Initialization                  | 1813  | __init__: creates figure, canvas, toolbar
#   ├─ Font Size Controls              | 2165  | set_axis_label_fontsize(), set_graph_title_fontsize()
#   ├─ Export Functions                | 2225  | export_excel(): save data to Excel workbook
#   └─ Drawing Helpers                 | 2275  | apply_user_overrides(), redraw(), time markers
#
# ChannelTabsTk CLASS                  | 2971  | 7 analysis tabs for each mouse
#   ├─ Initialization                  | 2971  | __init__: creates 7 tabs (Raw, Slope, Artifacts, etc.)
#   ├─ Tab Management                  | 3023  | _on_inner_tab_changed(): switch between 7 tabs
#   ├─ Drawing Functions:
#   │   ├─ Raw Signal Tab              | 3355  | _draw_raw(): shows exc_raw, iso_raw
#   │   ├─ Slope Normality Tab         | 3370  | _draw_slope_normality(): histogram + Q-Q plots
#   │   ├─ Artifact Map Tab            | 3450  | _draw_artifact(): shows flagged samples in red
#   │   ├─ Fit Tab                     | 3540  | _draw_fit_and_attach_selector(): linear regression
#   │   ├─ Normalization Tab           | 3610  | _draw_norm(): ΔF/F or z-score
#   │   ├─ Normalized+Smoothed Tab     | 3625  | _draw_norm_smooth(): with noise reduction
#   │   └─ Frequency Analysis Tab      | 3675  | _draw_frequency(): 6 frequency bands [FFT/Butterworth]
#   ├─ Data Export                     | 3755  | _export_raw(), _export_artifact(), etc. [7 methods]
#   └─ Interactive Editing             | 3905  | update_fit_window(), drag-to-select fit region
#
# BatchCompareTk CLASS                 | 3960  | Compare plot: overlay multiple mice
#   ├─ Initialization                  | 3960  | __init__: creates comparison figure
#   └─ Refresh & Redraw                | 4005  | refresh_plot(): updates when norm mode changes
#
# BatchAverageTk CLASS                 | 4160  | Average plot: mean ± SEM of multiple mice
#   ├─ Initialization                  | 4160  | __init__: creates average figure
#   └─ Refresh & Redraw                | 4205  | refresh_plot(): updates with error bands
# │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────── MAIN APPLICATION (MainAppTk) ───────────────────────────────────┐
# │ Master window: top-level GUI with controls and tab management          │
# ├─────────────────────────────────────────────────────────────────────────┤
# MainAppTk CLASS (Main Window)        | 4527  | Master application window
#   ├─ Initialization & Layout         | 4527  | __init__: creates entire UI
#   │   ├─ File Selection Row          | 4691  | [Add CSV] [Clear] [Run Analysis] buttons
#   │   ├─ Artifact Controls Row       | 4774  | Artifact factor, pad, shared, FPS, interpolation
#   │   ├─ Normalization Controls      | 4855  | Normalization dropdown, interval, font sizes
#   │   └─ Outer Tabs                  | 4925  | Main notebook (32+ mouse tabs + Batch tabs)
#   ├─ File Management                 | 4989  | add_csvs(): dialog to select CSV files
#   ├─ Analysis Execution              | 5052  | run_analysis(): main workflow (background thread)
#   │   ├─ Completion Handler          | 5082  | on_analysis_finished(): build tabs after analysis
#   │   └─ Error Handler               | 5109  | on_analysis_failed(): show error dialog
#   ├─ Tab Building                    | 5118  | build_tabs(): lazy-load mouse tabs on demand
#   ├─ Lazy Loading                    | 5185  | _on_outer_tab_changed(): create widgets when tab clicked
#   ├─ Normalization Controls          | 5232  | update_norm_controls_visibility(): show/hide interval fields
#   ├─ Font Size Application           | 5260  | apply_*_fontsize(): update plot text sizes
#   └─ Hotkeys/Interaction             | 5345  | Ctrl+I (time markers), Ctrl+J (file mapping)
#
# UTILITY FUNCTIONS                    | 4494  | _basename_no_ext(), _unique_aliases(), helpers
# │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────── ENTRY POINTS ───────────────────────────────────────────────────┐
# │ Functions that start the application                                    │
# ├─────────────────────────────────────────────────────────────────────────┤
# parse_cli()                          | 5792  | Parse command-line arguments (--autorun, --csv, etc.)
# main()                               | 5805  | Application startup: Tkinter event loop begins
# │
# └─────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================
# HOW TO USE THIS TABLE
# =============================================================================
# 1. Find the function/class you want to understand in the list above
# 2. Note the LINE number in the first column
# 3. In VS Code:
#    • Press Ctrl+G to open "Go to Line" dialog
#    • Type the line number
#    • Press Enter to jump to that location
# 4. You'll land at the start of that section
# =============================================================================

# ---------------------- Tkinter import with clear error ----------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except Exception as e:
    raise SystemExit(
        "Tkinter is not available in this Python environment.\n\n"
        "On Windows, Tkinter usually comes with the official python.org installer.\n"
        "If your company Python build excludes Tkinter, you’ll need IT to install a Python "
        "distribution that includes Tk/Tcl.\n\n"
        f"Original error: {e}"
    )

# ---------------------- Matplotlib backend (TkAgg) ----------------------
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colors as mcolors
from matplotlib.font_manager import FontProperties

# SciPy (optional): stats for normality + signal for bandpass/smoothing
try:
    from scipy import stats  # type: ignore
    _HAVE_SCIPY_STATS = True
except Exception:
    stats = None  # type: ignore
    _HAVE_SCIPY_STATS = False

try:
    from scipy.signal import butter, sosfiltfilt, filtfilt  # type: ignore
    _HAVE_SCIPY_SIGNAL = True
except Exception:
    butter = None  # type: ignore
    sosfiltfilt = None  # type: ignore
    filtfilt = None  # type: ignore
    _HAVE_SCIPY_SIGNAL = False


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
# This section defines all the fixed settings for the fiber photometry analyzer.
# These are the "knobs" that control how the data is processed.
# =============================================================================

# ---- Hardware & Data States ----
# The fiber photometry system alternates between acquiring signals from two channels:
#   ISO_STATE (1): Isosbestic LED is on (reference signal, less sensitive to activity)
#   EXC_STATE (2): Excitatory LED is on (activity-dependent signal)
ISO_STATE = 1
EXC_STATE = 2

# ---- Data Analysis Settings ----
# These control which time windows and methods are used for fitting and filtering

# Time window for fitting the relationship: EXC ≈ a * ISO + b (in seconds)
# Default range covers the whole typical recording (0 to ~2 hours)
DEFAULT_FIT_WINDOWS = [(0.0, 6500.0)]

# How to align ISO samples to EXC times when they don't match exactly
# "nearest" = use the closest ISO time value
# "prev" = use the last ISO sample before the EXC time
# "next" = use the first ISO sample after the EXC time
DEFAULT_ALIGN_MODE = "nearest"

# Maximum number of data points to include in the slope normality test
# (Large files may exceed Shapiro-Wilk limits; this prevents crashes)
DEFAULT_SLOPE_TEST_MAX_N = 5000

# ---- Acquisition Rate (Camera FPS) ----
# The camera captures frames at this rate (Hz). The system then *alternates*
# between ISO and EXC channels, so the effective rate per channel is FPS/2
DEFAULT_ACQ_FPS_HZ = 40.0

# Butterworth filter order (used for low-pass filtering before normalization)
DEFAULT_BUTTER_ORDER = 2

# ---- Smoothing Window ----
# Number of samples to average together when user enables trace smoothing
# Larger = more smoothing (visual clarity but less detail)
DEFAULT_SMOOTH_WINDOW = 20

# ---- Z-Score Normalization (Interval-Based) ----
# When the user selects "interval-based z-score", the baseline is calculated
# from a specific time window. These are the default start/end times (seconds)
DEFAULT_ZF_INTERVAL_START_S = 0.0
DEFAULT_ZF_INTERVAL_END_S = 6500.0

# ---- Data Interpolation ----
# After removing artifact "holes" from the data:
#   If True: linearly fill the gaps (makes traces smooth, but adds synthetic data)
#   If False: leave gaps as NaN (preserves only real data, but traces look choppy)
# NOTE: Frequency analysis always uses the NON-interpolated data (pure recordings)
DEFAULT_USE_LINEAR_INTERP = True


# ---- Font Size Defaults (for display clarity) ----
# These control text size in the plots. Users can override all of these from the GUI.

def _fontsize_points(v: Any, fallback: float) -> float:
    """Convert matplotlib font size (can be string like 'medium' or number) to points."""
    try:
        return float(FontProperties(size=v).get_size_in_points())
    except Exception:
        try:
            return float(v)
        except Exception:
            return float(fallback)

# Default font sizes (read from matplotlib's current theme)
DEFAULT_AXIS_LABEL_FONTSIZE = _fontsize_points(
    matplotlib.rcParams.get("axes.labelsize", 10), 
    fallback=10.0
)
DEFAULT_GRAPH_TITLE_FONTSIZE = _fontsize_points(
    matplotlib.rcParams.get("axes.titlesize", 12), 
    fallback=12.0
)
DEFAULT_TICK_LABEL_FONTSIZE = _fontsize_points(
    matplotlib.rcParams.get("xtick.labelsize", matplotlib.rcParams.get("ytick.labelsize", 8)),
    fallback=8.0,
)

# ---- Normalization Methods ----
# The user can choose how to normalize the signal. These are the three options
# available in the dropdown menu on the main window.
NORMALIZATION_OPTION_DF_OVER_F = "ΔF/F"
NORMALIZATION_OPTION_ZF_GLOBAL = "zF (global, GUI)"
NORMALIZATION_OPTION_ZF_INTERVAL = "zF - interval based"
NORMALIZATION_ALL_OPTIONS = [
    NORMALIZATION_OPTION_DF_OVER_F, 
    NORMALIZATION_OPTION_ZF_GLOBAL, 
    NORMALIZATION_OPTION_ZF_INTERVAL
]

# Shorter aliases for use throughout the code
NORM_DFF = NORMALIZATION_OPTION_DF_OVER_F
NORM_ZF_GLOBAL = NORMALIZATION_OPTION_ZF_GLOBAL
NORM_ZF_INTERVAL = NORMALIZATION_OPTION_ZF_INTERVAL
NORM_CHOICES = NORMALIZATION_ALL_OPTIONS

# ---- Frequency Analysis Bands ----
# When analyzing frequency content, the data is split into these frequency ranges (Hz)
# These are standard ranges used in neuroscience studies
FREQ_BANDS: List[Tuple[float, float]] = [
    (5.0, 10.0),
    (2.5, 5.0),
    (1.25, 2.5),
    (0.6, 1.25),
    (0.3, 0.6),
    (0.15, 0.3),
]


# =============================================================================
# CALCULATION MAP: WHERE TO FIND EACH PLOT'S CALCULATIONS
# =============================================================================
# Use this reference to understand data flow from CSV → calculation → plot
#
# QUICK NAVIGATION FOR NON-PROGRAMMERS:
# If you want to understand where the numbers come from for a specific plot, search below
# for "[TAB X: name]" to see what calculations are involved.
#
# LEGEND:
# - "CALC:" = where the calculation happens (search for the function name)
# - "PLOT:" = where results are displayed on screen
# - "INPUT:" = what raw data or settings are used
# =============================================================================

CALCULATION_MAP_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                        PLOT-BY-PLOT CALCULATION GUIDE                          ║
╚════════════════════════════════════════════════════════════════════════════════╝

[TAB 1: RAW SIGNAL]
─────────────────────────────────────────────────────────────────────────────────
CALC:   analyze_csv() function
        - Loads CSV with Pandas library
        - Separates ISO (LedState=6) and EXC (LedState=7) samples by timestamp
        - Stores as res.iso_raw, res.exc_raw and times as res.t_iso, res.t_exc
        
PLOT:   _draw_raw() function
        - Plots res.exc_raw vs res.t_exc (excitatory signal)
        - Plots res.iso_raw vs res.t_iso (isosbestic reference signal)
        - No processing on display: shows raw signal directly from CSV

INPUT:  CSV columns: SystemTimestamp, LedState, G0-G7 (signal columns)


[TAB 2: SLOPE NORMALITY]
─────────────────────────────────────────────────────────────────────────────────
CALC:   compute_slopes() function
        - Computes dy/dt (rate of change) for every time step
        - Ignores missing data (NaN/inf) and zero time steps
        - Called separately IN _draw_slope_normality() for both ISO and EXC
        
        normality_summary() function
        - If SciPy available: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling tests
        - Computes mean, standard deviation, skewness, kurtosis
        - Formula: Shapiro p-value tests if slope values come from a normal distribution
        
        qq_plot_points() function
        - Creates Q-Q plot: how data deviates from theoretical normal distribution
        - X-axis: what normal distribution would predict
        - Y-axis: ordered actual slope values
        
PLOT:   _draw_slope_normality() function
        - 4-panel figure:
          * Top-left: Histogram of ISO slopes (distribution shape)
          * Top-right: Histogram of EXC slopes
          * Bottom-left: Q-Q plot for ISO slopes (straight line = normal)
          * Bottom-right: Q-Q plot for EXC slopes
        - Text box shows normality test results (if SciPy installed)

INPUT:  res.iso_raw, res.exc_raw (RAW signal - before artifact removal)
        res.t_iso, res.t_exc (timestamps)
        (Uses raw data, NOT cleaned data!)


[TAB 3: ARTIFACT MAP]
─────────────────────────────────────────────────────────────────────────────────
CALC:   detect_artifacts_by_derivative() function  ← CORE CALCULATION
        
        ALGORITHM STEP-BY-STEP:
        1. Compute dy/dt (signal derivative = rate of change)
        2. Find center value: median(slopes)
        3. Compute spread: MAD = Median Absolute Deviation = 1.4826 × median(|slopes - center|)
        4. Flag suspicious samples: |slope - center| > factor × MAD  (where factor ≈ 11.9)
        5. Pad artifacts: mark ±pad neighbors also as artifacts
        6. Return boolean array: True = artifact, False = real data
        
        (Runs separately on ISO and EXC signals)
        
        shared_artifacts_by_time() function (if require_shared=True)
        - Only flags samples where BOTH ISO AND EXC channels show artifacts
        - Reduces false positives from single-channel noise
        
PLOT:   _draw_artifact() function
        - Shows exc_clean with shaded RED regions where art_exc=True
        - Shows iso_clean with shaded RED regions where art_iso=True
        - Statistics box shows: total samples, # artifacts, % removed
        - Also shows max consecutive artifact length

INPUT:  res.iso_raw, res.exc_raw (RAW signal)
        res.t_iso, res.t_exc (timestamps for derivative calculation)
        User settings: 
          - artifact_factor (~11.9): sensitivity (higher = fewer artifacts)
          - pad (~1): how many neighbors to mark
          - require_shared (True): only if both channels flag the same spot


[TAB 4: FIT (LINEAR REGRESSION)]
─────────────────────────────────────────────────────────────────────────────────
CALC:   _compute_fit_and_downstream() function  ← CORE CALCULATION
        
        MATHEMATICAL STEPS:
        1. Create fit mask: mark samples WITHIN user-selected time windows
        2. Linear regression: Find a, b such that EXC ≈ a×ISO_on_EXC + b
           (minimizes sum of squared residuals)
        3. Compute R² (R-squared): measure of fit quality
           R² = 1 - (residual_variance / total_variance)
           R² = 1.0 = perfect fit, R² = 0.5 = mediocre, R² = 0 = no fit
        4. Calculate fitted values: fitted = a×ISO_on_EXC + b
        5. Calculate residual: residual = EXC - fitted (the "cleaned" signal)
        6. Calculate ΔF/F = residual / fitted (normalized neuronal signal)
        
        IMPORTANT: Uses ACTIVE pipeline (interpolated if user enabled interpolation)
        Also computes _nointerp version (for frequency analysis which needs pure data)
        
        Called from: recompute_fit_and_downstream() function
        
PLOT:   _draw_fit_and_attach_selector() function
        - Thick line: exc_clean (excitatory fluorescence after artifact removal)
        - Thin line: fitted_iso_on_exc (predicted baseline from isosbestic)
        - Light shaded region: marks the time window used for fitting
        - Info box (bottom-right): shows slope (a), intercept (b), R²
        - INTERACTIVE: Click and drag to select a new fit window (triggers recalculation)
        
INPUT:  res.exc_clean (excitatory signal after artifacts removed)
        res.iso_on_exc (isosbestic aligned to excitatory times)
        res.t_exc (timestamps)
        res.windows (time windows for fitting - user can drag to change)


[TAB 5: NORMALIZATION (ΔF/F or Z-Score)]
─────────────────────────────────────────────────────────────────────────────────
CALC:   recompute_normalizations() function
        
        Computes THREE normalization options:
        
        1. ΔF/F = (EXC - fitted_ISO) / fitted_ISO
           - Formula: (residual / fitted)
           - Interpretation: percent change from baseline
           - Most common in neuroscience
           - Baseline = isosbestic signal (motion artifact)
        
        2. zF (global) = (ΔF/F - mean_all) / std_all
           - Computes z-score across ENTIRE recording
           - Removes units: values centered at 0, std=1
           - Good for comparing recordings with different baselines
        
        3. zF (interval-based) = (ΔF/F - mean_window) / std_window
           - Computes z-score using ONLY a user-selected time window
           - Good for: "How different is this stimulus from the baseline period?"
           - Window = time when animal is at rest/baseline
        
        Formula for z-score: z = (x - mean) / std_dev
        
PLOT:   _draw_norm() function
        - Single line plot of time vs. selected normalization
        - User dropdown selector (top of window) picks which normalization to show
        - X-axis: Time (seconds)
        - Y-axis: Normalized signal (units depend on selection)

INPUT:  res.dFF (ΔF/F values, computed by _compute_fit_and_downstream)
        res.t_exc (timestamps)
        User settings: 
          - Normalization dropdown: dF/F, zF global, or zF interval
          - Interval start/end times (if interval-based selected)


[TAB 6: NORMALIZED + SMOOTHED]
─────────────────────────────────────────────────────────────────────────────────
CALC:   Same normalization as Tab 5 (dFF, zF_global, or zF_interval)
        
        smooth_like_batch() function
        - Moving average filter: average each sample with ±(window/2) neighbors
        - Formula: y_smooth[i] = mean(y[i-w/2:i+w/2])
        - Window size = smooth_window (default 20 samples)
        - Cached in res._smooth_cache for speed (recalculates only when settings change)
        - Re-caches whennormalization view changes
        
PLOT:   _draw_norm_smooth() function
        - Thin line: original normalized signal (same as Tab 5)
        - Thick line: smoothed version
        - Smoothing removes high-frequency noise but loses fine temporal detail
        - User control: "Smooth win (samples)" spinner (top controls)

INPUT:  Same as Tab 5
        User setting: Smooth window size (20 samples = default)


[TAB 7: FREQUENCY ANALYSIS]
─────────────────────────────────────────────────────────────────────────────────
CALC:   6 frequency bands (defined by FREQ_BANDS constant)
        Band ranges: 0.15-0.3 Hz, 0.3-0.6 Hz, 0.6-1.25 Hz, 1.25-2.5 Hz, 2.5-5 Hz, 5-10 Hz
        
        For each frequency band:
        
        METHOD 1 - Butterworth Filter (if scipy.signal available):
        - Create bandpass filter: scipy.signal.butter()
        - Apply: sosfiltfilt() (zero-phase filtering, no lag)
        - Handles NaN "holes" gracefully (filters piecewise around them)
        - Result: signal_filtered with power in that frequency band
        
        METHOD 2 - FFT (fallback, used only if scipy unavailable):
        - Compute: np.fft.rfft(signal)  (Fast Fourier Transform)
        - Extract magnitude in frequency band
        - Remove DC component (subtract mean first)
        - LIMITATION: This method FAILS if data has NaN holes
        
        ✓ ALL frequency analysis uses dFF_nointerp (NEVER interpolated)
          This preserves data integrity; holes remain as NaN
        
        Power is normalized by sampling rate (eff_fs_hz) for fair comparison
        
PLOT:   _draw_frequency() function
        - 6 subplots (one per frequency band)
        - X-axis: Time (seconds)
        - Y-axis: Power (filtered magnitude or FFT amplitude)
        - Title shows: channel, method used, sampling rate
        
        [CRITICAL NOTE]: This is the only tab that explicitly preserves NaN holes
                         Other tabs use interpolated data for smooth display

INPUT:  res.dFF_nointerp (NON-INTERPOLATED ΔF/F with NaN gaps)
        res.t_exc (timestamps)
        res.eff_fs_hz (effective sampling rate = acq_fps / 2)
        FREQ_BANDS (6 frequency ranges defined at top of code)


╔════════════════════════════════════════════════════════════════════════════════╗
║                    DATA FLOW DIAGRAM (SIMPLIFIED)                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

CSV File (SystemTimestamp, LedState, G0-G7)
  ↓
  analyze_csv()   ← Main entry point
  ├─→ Load CSV with Pandas
  ├─→ Separate by LedState: ISO vs EXC
  │
  ├─→ detect_artifacts_by_derivative()  [MAD method]
  │   └─→ art_iso, art_exc  (boolean: which samples are artifacts)
  │
  ├─→ Remove artifact samples (create "holes" = NaN)
  │   ├─→ iso_clean_holes, exc_clean_holes
  │
  ├─→ Optionally: linear_interpolate_by_time()
  │   ├─→ iso_clean_interp, exc_clean_interp (fill the holes)
  │
  ├─→ align_iso_to_exc_no_interp()  (match ISO times to EXC times)
  │   └─→ iso_on_exc_holes, iso_on_exc_interp
  │
  ├─→ _compute_fit_and_downstream()  [LINEAR REGRESSION]
  │   └─→ EXC ≈ a × ISO + b
  │       ├─→ fitted (predicted values)
  │       ├─→ dF = EXC - fitted  (residual)
  │       └─→ dFF = dF / fitted  (ΔF/F = normalized neuronal signal)
  │
  ├─→ recompute_normalizations()
  │   ├─→ zF_global = (dFF - mean_all) / std_all
  │   └─→ zF_interval = (dFF - mean_window) / std_window
  │
  └─→ Store in ChannelResult object (contains ALL computed values)


╔════════════════════════════════════════════════════════════════════════════════╗
║                    FUNCTION REFERENCE (Search For These)                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

To find calculations for a specific purpose, search for:

Artifact Detection:
  →  detect_artifacts_by_derivative()

Slope Calculation:
  →  compute_slopes()

Linear Fitting (core ΔF/F calculation):
  →  _compute_fit_and_downstream()

Normalizations:
  →  recompute_normalizations()
  →  zscore_global_gui()
  →  zscore_interval_based()

Smoothing:
  →  smooth_like_batch()

Frequency Analysis:
  →  butter() and sosfiltfilt()  from scipy.signal
  →  np.fft.rfft()  for FFT fallback

Plotting (display of calculated data):
  →  _draw_raw()
  →  _draw_slope_normality()
  →  _draw_artifact()
  →  _draw_fit_and_attach_selector()
  →  _draw_norm()
  →  _draw_norm_smooth()
  →  _draw_frequency()

Main Analysis:
  →  analyze_csv()      (orchestrates all calculations)
"""


# =============================================================================

def find_g_columns(columns) -> List[str]:
    """
    Find and return signal columns from the CSV data.
    
    CSV files contain many columns. This function identifies the ones we care about:
    columns that start with 'G' (G0, G1, G2, etc.) - these are the mouse signal 
    recordings. It removes the system columns like timestamp and LED state.
    
    Args:
        columns: List of column names from the CSV file
        
    Returns:
        List of signal column names (G0, G1, etc.) in the order they appear in CSV
    """
    g_cols = [c for c in columns if re.match(r"^G\d*$", str(c)) or str(c).startswith("G")]
    ordered = [c for c in columns if c in g_cols]
    for drop in ["FrameCounter", "LedState", "SystemTimestamp", "ComputerTimestamp"]:
        if drop in ordered:
            ordered.remove(drop)
    return ordered


# ========================================================================================================
# CALCULATION FUNCTIONS - ARTIFACT DETECTION
# ========================================================================================================
# Used by: "Artifact Map" plot tab
# Calculation method: Median Absolute Deviation (MAD) on signal derivatives
# Where it's called: analyze_csv() line ~1290
# What it computes: Boolean array marking samples flagged as artifacts

def detect_artifacts_by_derivative(
    t: np.ndarray,
    y: np.ndarray,
    factor: float,
    method: str = "mad",  # MAD ONLY in this build
    pad: int = 1,
) -> np.ndarray:
    """
    Identify artifact points (movement, LED flicker, etc.) by rapid signal changes.
    
    CALCULATION OVERVIEW:
    - Computes dy/dt (signal derivative) to detect sudden jumps
    - Calculates median slope value and MAD (Median Absolute Deviation)
    - Flags slopes deviating by >factor*MAD from median as artifacts
    - Pads artifacts to capture surrounding samples
    
    The principle: signals from a fiber photometry setup should change relatively slowly.
    Sudden jumps usually indicate an artifact (mouse moved, equipment glitched, etc.).
    This function calculates the derivative (dy/dt), measures how "extreme" each
    point is compared to the baseline (using MAD: Median Absolute Deviation), and
    marks extreme points as artifacts.
    
    Args:
        t: Timestamps (1D array in seconds)
        y: Signal values (1D array, same length as t)
        factor: Sensitivity multiplier (higher = fewer artifacts marked, ~11.9 default)
        method: Always "mad" (Median Absolute Deviation) - other methods not supported
        pad: How many neighbors around each artifact to also mark (reduces artifacts)
        
    Returns:
        Boolean array (True = artifact, False = real data) same length as y
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 3:
        return np.zeros_like(y, dtype=bool)

    tv = t[valid]
    yv = y[valid]
    dt = np.diff(tv)
    dy = np.diff(yv)

    dt_safe = np.where(dt == 0, np.nan, dt)
    slopes = dy / dt_safe
    slopes_valid = slopes[np.isfinite(slopes)]
    if slopes_valid.size < 3:
        return np.zeros_like(y, dtype=bool)

    method = (method or "mad").strip().lower()
    if method != "mad":
        raise ValueError("Only 'mad' artifact method is supported (sd removed).")

    center = float(np.median(slopes_valid))
    mad = float(np.median(np.abs(slopes_valid - center)))
    scale = float(1.4826 * mad)

    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(y, dtype=bool)

    bad_seg = np.abs(slopes - center) > (factor * scale)

    art_valid = np.zeros_like(yv, dtype=bool)
    bad_idx = np.where(bad_seg)[0]
    for k in bad_idx:
        i0 = max(0, k - pad)
        i1 = min(len(art_valid) - 1, k + 1 + pad)
        art_valid[i0:i1 + 1] = True

    art = np.zeros_like(y, dtype=bool)
    art[np.where(valid)[0]] = art_valid
    return art


# ========================================================================================================
# CALCULATION FUNCTIONS - SLOPE COMPUTATION
# ========================================================================================================
# Used by: "Slope Normality" plot tab
# Calculation method: Simple dy/dt calculation on RAW data
# Where it's called: _draw_slope_normality() line ~2950
# What it computes: Array of slopes (rate of change) for normality testing

def compute_slopes(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute dy/dt slopes from a (t, y) series, ignoring NaNs/infs and zero dt."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 3:
        return np.array([], dtype=float)

    tv = t[valid]
    yv = y[valid]
    dt = np.diff(tv)
    dy = np.diff(yv)

    dt = np.where(dt == 0, np.nan, dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dy / dt

    slopes = slopes[np.isfinite(slopes)]
    return slopes


def subsample_even(x: np.ndarray, max_n: int) -> np.ndarray:
    """Deterministic, evenly-spaced subsampling."""
    x = np.asarray(x, dtype=float)
    if max_n is None or max_n <= 0 or x.size <= max_n:
        return x
    idx = np.linspace(0, x.size - 1, max_n, dtype=int)
    return x[idx]


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def normality_summary(
    x: np.ndarray,
    alpha: float = 0.05,
    max_n: int = DEFAULT_SLOPE_TEST_MAX_N,
) -> dict:
    """Run normality checks on x (after finite filtering)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    out = {
        "n": int(x.size),
        "n_test": int(min(x.size, max_n if (max_n is not None and max_n > 0) else x.size)),
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "mad": np.nan,
        "shapiro_p": np.nan,
        "dagostino_p": np.nan,
        "anderson_stat": np.nan,
        "anderson_cv5": np.nan,
        "anderson_ok": None,
        "verdict": "n/a",
        "note": "",
    }

    if x.size < 3:
        out["note"] = "Need ≥3 finite slopes"
        return out

    x_test = subsample_even(x, max_n=max_n)

    out["mean"] = float(np.mean(x_test))
    out["std"] = float(np.std(x_test, ddof=1)) if x_test.size >= 2 else np.nan
    out["median"] = float(np.median(x_test))
    out["mad"] = float(np.median(np.abs(x_test - out["median"])))

    if not _HAVE_SCIPY_STATS:
        out["note"] = "SciPy not installed: skipping formal tests"
        out["verdict"] = "n/a"
        return out

    try:
        if x_test.size >= 3:
            _, p = stats.shapiro(x_test)
            out["shapiro_p"] = float(p)
    except Exception as e:
        out["note"] += f"Shapiro failed: {e}. "

    try:
        if x_test.size >= 8:
            _, p = stats.normaltest(x_test)
            out["dagostino_p"] = float(p)
    except Exception as e:
        out["note"] += f"K² failed: {e}. "

    try:
        ad = stats.anderson(x_test, dist="norm")
        out["anderson_stat"] = float(ad.statistic)

        cv5 = np.nan
        for sl, cv in zip(ad.significance_level, ad.critical_values):
            if float(sl) == 5.0:
                cv5 = float(cv)
                break
        if not np.isfinite(cv5) and len(ad.critical_values) > 0:
            i = int(np.argmin(np.abs(np.asarray(ad.significance_level, dtype=float) - 5.0)))
            cv5 = float(ad.critical_values[i])

        out["anderson_cv5"] = cv5
        out["anderson_ok"] = bool(np.isfinite(cv5) and ad.statistic < cv5)
    except Exception as e:
        out["note"] += f"AD failed: {e}. "

    rejects = []
    if np.isfinite(out["shapiro_p"]):
        rejects.append(out["shapiro_p"] <= alpha)
    if np.isfinite(out["dagostino_p"]):
        rejects.append(out["dagostino_p"] <= alpha)
    if out["anderson_ok"] is not None:
        rejects.append(not out["anderson_ok"])

    out["verdict"] = "n/a" if not rejects else ("REJECT" if any(rejects) else "OK")
    return out


def qq_plot_points(
    x: np.ndarray,
    max_n: int = DEFAULT_SLOPE_TEST_MAX_N,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return Q-Q plot data for x against a Normal distribution."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float), np.nan, np.nan, np.nan

    xq = subsample_even(x, max_n=max_n)

    if _HAVE_SCIPY_STATS:
        (theor, ordered), (slope, intercept, r) = stats.probplot(xq, dist="norm")
        return np.asarray(theor), np.asarray(ordered), float(slope), float(intercept), float(r)

    ordered = np.sort(xq)
    n = ordered.size
    p = (np.arange(1, n + 1) - 0.5) / n
    theor = np.array([NormalDist().inv_cdf(float(pi)) for pi in p], dtype=float)

    slope, intercept = np.polyfit(theor, ordered, 1)
    r = float(np.corrcoef(theor, ordered)[0, 1]) if n >= 2 else np.nan
    return theor, ordered, float(slope), float(intercept), r


def shared_artifacts_by_time(
    t_iso: np.ndarray,
    art_iso: np.ndarray,
    t_exc: np.ndarray,
    art_exc: np.ndarray,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only artifacts present in BOTH channels within a time tolerance."""
    t_iso = np.asarray(t_iso, dtype=float)
    t_exc = np.asarray(t_exc, dtype=float)
    art_iso = np.asarray(art_iso, dtype=bool)
    art_exc = np.asarray(art_exc, dtype=bool)

    if not art_iso.any() or not art_exc.any():
        return np.zeros_like(art_iso, dtype=bool), np.zeros_like(art_exc, dtype=bool)

    if tol is None:
        dt_iso = np.median(np.diff(t_iso)) if t_iso.size > 1 else np.inf
        dt_exc = np.median(np.diff(t_exc)) if t_exc.size > 1 else np.inf
        base_dt = min(dt_iso, dt_exc)
        tol = 0.5 * base_dt if np.isfinite(base_dt) else 0.0

    iso_idx = np.where(art_iso)[0]
    exc_idx = np.where(art_exc)[0]
    iso_times = t_iso[iso_idx]
    exc_times = t_exc[exc_idx]

    iso_shared = np.zeros_like(art_iso, dtype=bool)
    exc_shared = np.zeros_like(art_exc, dtype=bool)

    for ei, te in zip(exc_idx, exc_times):
        if np.any(np.abs(iso_times - te) <= tol):
            exc_shared[ei] = True

    for ii, ti in zip(iso_idx, iso_times):
        if np.any(np.abs(exc_times - ti) <= tol):
            iso_shared[ii] = True

    return iso_shared, exc_shared


def remove_with_holes(y: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    """Remove artifacts by setting them to NaN (holes)."""
    y = np.asarray(y, dtype=float).copy()
    y[np.asarray(artifact_mask, dtype=bool)] = np.nan
    return y


def linear_interpolate_by_time(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by linear interpolation using the time axis.
    Applied AFTER artifact removal if the user enables it.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    out = y.copy()

    valid = np.isfinite(t) & np.isfinite(out)
    if valid.sum() < 2:
        return out

    # np.interp assumes t is increasing. SystemTimestamp should be monotonic.
    # Endpoints are filled by constant extrapolation (np.interp behavior).
    out = np.interp(t, t[valid], out[valid]).astype(float)
    return out


def align_iso_to_exc_no_interp(
    t_iso: np.ndarray,
    y_iso: np.ndarray,
    t_exc: np.ndarray,
    mode: str = "nearest",  # "prev", "next", "nearest"
) -> np.ndarray:
    """Map iso samples onto excitatory timestamps WITHOUT interpolation."""
    t_iso = np.asarray(t_iso, dtype=float)
    y_iso = np.asarray(y_iso, dtype=float)
    t_exc = np.asarray(t_exc, dtype=float)

    if t_iso.size == 0:
        return np.full_like(t_exc, np.nan, dtype=float)

    idx = np.searchsorted(t_iso, t_exc, side="left")

    mode = (mode or "nearest").strip().lower()
    if mode == "prev":
        j = idx - 1
    elif mode == "next":
        j = idx
    elif mode == "nearest":
        j_prev = np.clip(idx - 1, 0, len(t_iso) - 1)
        j_next = np.clip(idx, 0, len(t_iso) - 1)
        d_prev = np.abs(t_exc - t_iso[j_prev])
        d_next = np.abs(t_exc - t_iso[j_next])
        j = np.where(d_next < d_prev, j_next, j_prev)
    else:
        raise ValueError("align mode must be 'prev', 'next', or 'nearest'")

    j = np.clip(j, 0, len(t_iso) - 1)
    return y_iso[j]


def fit_linear(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a*x + b."""
    X = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(a), float(b)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ---------------------- Smoothing (batch-style moving average + filtfilt) ----------------------
# NOTE ABOUT PERFORMANCE
# ----------------------
# In earlier versions, smoothing used pandas (Series.interpolate + rolling).
# That is convenient but can become noticeably slow when you have many mice/files.
#
# The helpers below do the same *practical* job using NumPy only:
#   1) Fill NaN "holes" by linear interpolation (so filtfilt / averaging can run)
#   2) Apply the batch-style smoothing:
#        - Prefer SciPy filtfilt(b=ones(W)/W, a=1) when available (same as batch script)
#        - Otherwise fall back to a fast centered moving average computed with prefix sums
#   3) Restore NaNs so you still see holes if you are in "holes" mode
#
# This is dramatically faster than pandas for typical fiber-photometry traces.


def _fill_nans_linear_1d(x: np.ndarray) -> np.ndarray:
    """
    Fill NaNs/inf values in a 1D array by linear interpolation, similar to:
        pd.Series(x).interpolate(limit_direction="both")

    Practical behaviour:
      - If there are NaNs in the middle: linearly interpolate between neighbouring finite points.
      - If there are NaNs at the start or end: fill them with the nearest finite value
        (NumPy's np.interp does constant extrapolation at the ends).

    If the array is all-NaN it is returned unchanged.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()

    out = x.copy()
    finite = np.isfinite(out)

    # If everything is missing, we cannot interpolate
    if not np.any(finite):
        return out

    # If only a single finite point exists, fill everything with that value
    idx = np.where(finite)[0]
    if idx.size == 1:
        out[~finite] = out[idx[0]]
        return out

    # Interpolate across the whole vector (this also fills endpoints)
    xp = idx.astype(float)
    fp = out[idx].astype(float)
    x_all = np.arange(out.size, dtype=float)

    out = np.interp(x_all, xp, fp).astype(float)
    return out


def _moving_average_centered_fast(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Fast centered moving average (like pandas rolling(window=W, center=True, min_periods=1).mean()).

    We compute the mean for each index using prefix sums, which is O(n) even for large windows.

    Edge handling:
      - Near the edges, the window is effectively smaller (min_periods=1 behaviour),
        so the average uses the samples that exist.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    w = int(max(1, window_size))
    if w == 1:
        return x.copy()

    # For "center=True", pandas uses an asymmetric split for even windows.
    # This matches that practical behaviour:
    left = (w - 1) // 2
    right = w // 2

    idx = np.arange(n, dtype=int)
    start = idx - left
    end = idx + right

    # Clip to valid bounds
    start = np.clip(start, 0, n - 1)
    end = np.clip(end, 0, n - 1)

    # Prefix sum so sum(start..end) is O(1)
    prefix = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    sums = prefix[end + 1] - prefix[start]
    counts = (end - start + 1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = sums / counts

    return np.asarray(y, dtype=float)


def smooth_like_batch(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply the same smoothing style as the batch script:
      b = ones(W)/W; a=1; filtfilt(b, a, x)

    Practical notes:
      - This smoothing is intended for display / comparison plots.
      - If the signal contains NaN "holes" (artifact remover holes mode), we:
          1) Fill NaNs by linear interpolation so the filter can run
          2) Apply the filter / moving average
          3) Put NaNs back where the holes were, so the user still sees the gaps

    Performance:
      - Uses NumPy interpolation (fast) instead of pandas.
      - Uses SciPy filtfilt when available; otherwise uses a fast centered moving average.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    try:
        w = int(window_size)
    except Exception:
        w = DEFAULT_SMOOTH_WINDOW
    w = max(1, w)

    if w == 1:
        return x.copy()

    # Remember where data is missing so we can restore holes afterwards
    nan_mask = ~np.isfinite(x)

    # Fill NaNs/inf so filtering works
    x_filled = _fill_nans_linear_1d(x)

    # SciPy filtfilt (preferred, matches batch script)
    padlen = 3 * (w - 1)
    if _HAVE_SCIPY_SIGNAL and filtfilt is not None and n > padlen:
        b = np.ones(w, dtype=float) / float(w)
        a = 1.0
        try:
            y = filtfilt(b, a, x_filled)
        except Exception:
            # If filtfilt fails (very short signals etc.), fall back to moving average
            y = _moving_average_centered_fast(x_filled, w)
    else:
        # Pure NumPy fallback (still centered and fast)
        y = _moving_average_centered_fast(x_filled, w)

    y = np.asarray(y, dtype=float)

    # Restore NaN holes so the user can still see missing segments
    y[nan_mask] = np.nan
    return y


# ---------------------- Frequency helpers (NO interpolation across holes) ----------------------
def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) inclusive for contiguous True segments.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    runs: List[Tuple[int, int]] = []
    s = idx[0]
    p = idx[0]
    for k in idx[1:]:
        if k == p + 1:
            p = k
        else:
            runs.append((s, p))
            s = p = k
    runs.append((s, p))
    return runs


def bandpass_butterworth_segmentwise_no_interp(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = DEFAULT_BUTTER_ORDER,
) -> np.ndarray:
    """
    Bandpass Butterworth applied per continuous finite segment.
    - DOES NOT interpolate across NaN holes.
    - Leaves NaNs where data is missing.
    Requires SciPy signal.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    if not _HAVE_SCIPY_SIGNAL or butter is None or sosfiltfilt is None:
        return out

    if not np.isfinite(fs) or fs <= 0:
        return out

    nyq = fs / 2.0
    high_hz = min(float(high_hz), nyq - 1e-6)

    if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:
        return out

    low = float(low_hz) / nyq
    high = float(high_hz) / nyq

    try:
        sos = butter(int(order), [low, high], btype="band", output="sos")
    except Exception:
        return out

    finite = np.isfinite(x)
    for i0, i1 in _contiguous_true_runs(finite):
        seg = x[i0:i1 + 1]
        if seg.size < 8:
            continue
        try:
            yseg = sosfiltfilt(sos, seg)
        except Exception:
            try:
                yseg = sosfiltfilt(sos, seg, padlen=0)
            except Exception:
                continue
        out[i0:i1 + 1] = yseg

    return out


def bandpass_fft_no_interp(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
) -> np.ndarray:
    """
    FFT bandpass that requires a fully finite signal.
    This avoids interpolation, but cannot handle NaN holes.
    """
    x = np.asarray(x, dtype=float)

    if not np.isfinite(fs) or fs <= 0:
        return np.full_like(x, np.nan, dtype=float)

    if not np.all(np.isfinite(x)):
        return np.full_like(x, np.nan, dtype=float)

    nyq = fs / 2.0
    high_hz = min(float(high_hz), nyq - 1e-6)
    if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:
        return np.full_like(x, np.nan, dtype=float)

    n = x.size
    if n < 4:
        return np.full_like(x, np.nan, dtype=float)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    Xf = np.where(mask, X, 0.0)
    y = np.fft.irfft(Xf, n=n)
    return np.asarray(y, dtype=float)


def estimate_fs_from_t(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    med = float(np.median(dt))
    if med <= 0 or not np.isfinite(med):
        return np.nan
    return float(1.0 / med)


# ---------------------- Z-score methods ----------------------
def _nanstd_safe(x: np.ndarray, ddof: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    try:
        return float(np.std(x, ddof=int(ddof)))
    except Exception:
        return np.nan


def zscore_global_gui(dff: np.ndarray, ddof: int = 0) -> np.ndarray:
    dff = np.asarray(dff, dtype=float)
    mu = float(np.nanmean(dff))
    sigma = _nanstd_safe(dff, ddof=ddof)
    if np.isfinite(sigma) and sigma > 1e-12:
        return (dff - mu) / sigma
    return np.full_like(dff, np.nan, dtype=float)


def zscore_interval_based(
    dff: np.ndarray,
    t: np.ndarray,
    start_s: float,
    end_s: float,
    ddof: int = 1,
) -> np.ndarray:
    """
    "Interval based" z-score:
      - compute mean/std from ΔF/F only inside [start_s, end_s]
      - apply those stats to the whole trace
    """
    dff = np.asarray(dff, dtype=float)
    t = np.asarray(t, dtype=float)

    a = float(min(start_s, end_s))
    b = float(max(start_s, end_s))

    mask = (t >= a) & (t <= b) & np.isfinite(dff)
    if mask.sum() < 2:
        mask = np.isfinite(dff)

    mu = float(np.nanmean(dff[mask])) if mask.any() else np.nan
    sigma = _nanstd_safe(dff[mask], ddof=ddof) if mask.any() else np.nan

    if np.isfinite(sigma) and sigma > 1e-12:
        return (dff - mu) / sigma
    return np.full_like(dff, np.nan, dtype=float)



# ---------------------- Normalization helpers for plotting (fast, cached) ----------------------
# These functions are shared by:
#   - Individual mouse normalization plots (ChannelTabsTk)
#   - Batch compare plot (BatchCompareTk)
#   - Batch average plot (BatchAverageTk)
#
# They return the *ACTIVE normalization series* (ΔF/F or zF), and provide a small
# cache for the smoothed version so the GUI does not re-smooth the same signal
# again and again when the user switches tabs.


def get_norm_array(res: "ChannelResult", norm_mode: str) -> np.ndarray:
    """Return the normalization array for the current mode (ΔF/F or zF)."""
    if norm_mode == NORM_DFF:
        return np.asarray(res.dFF, dtype=float)
    if norm_mode == NORM_ZF_GLOBAL:
        return np.asarray(res.zF_global, dtype=float)
    if norm_mode == NORM_ZF_INTERVAL:
        return np.asarray(res.zF_interval, dtype=float)
    # Fallback (should not normally happen)
    return np.asarray(res.dFF, dtype=float)


def get_smoothed_norm_array(res: "ChannelResult", norm_mode: str, window_size: int) -> np.ndarray:
    """
    Return a cached smoothed version of the selected normalization series.

    Why caching matters:
      - Smoothing can be expensive (especially for long traces and many mice)
      - Users often switch between tabs / groups without changing the smoothing window
      - With caching, the smoothing is computed once per (mode, window) and re-used
    """
    # Make sure window is a sane integer
    try:
        w = int(window_size)
    except Exception:
        w = DEFAULT_SMOOTH_WINDOW
    w = max(1, w)

    # Create cache dict if it doesn't exist yet
    cache = getattr(res, "_smooth_cache", None)
    if cache is None or not isinstance(cache, dict):
        cache = {}
        setattr(res, "_smooth_cache", cache)

    key = (str(norm_mode), int(w))
    if key in cache:
        return cache[key]

    y = get_norm_array(res, norm_mode)
    y_s = smooth_like_batch(y, window_size=w)

    # Keep the cache from growing forever if the user tries many windows
    if len(cache) > 12:
        cache.clear()

    cache[key] = y_s
    return y_s


# ---------------------- Data container ----------------------
@dataclass
class ChannelResult:
    gcol: str

    t_iso: np.ndarray
    t_exc: np.ndarray

    iso_raw: np.ndarray
    exc_raw: np.ndarray

    art_iso: np.ndarray
    art_exc: np.ndarray

    # Cleaned signals with holes
    iso_clean_holes: np.ndarray
    exc_clean_holes: np.ndarray

    # Linearly interpolated versions (filled)
    iso_clean_interp: np.ndarray
    exc_clean_interp: np.ndarray

    # Iso aligned onto excit timestamps (holes vs interpolated)
    iso_on_exc_holes: np.ndarray
    iso_on_exc_interp: np.ndarray

    # Current mode
    use_interpolation: bool

    # Active signals used for fit/ΔF/F/zF display (either holes or interpolated)
    iso_clean: np.ndarray
    exc_clean: np.ndarray
    iso_on_exc: np.ndarray

    # Fit windows (seconds)
    windows: List[Tuple[float, float]]

    # sampling rate info for frequency analysis
    acq_fps_hz: float
    eff_fs_hz: float

    # Fit + downstream for ACTIVE pipeline
    slope: float
    intercept: float
    r2: float
    fitted_iso_on_exc: np.ndarray
    residual: np.ndarray
    dF: np.ndarray
    dFF: np.ndarray

    # Fit + downstream for NO-INTERPOLATION pipeline (for frequency analysis)
    slope_nointerp: float
    intercept_nointerp: float
    r2_nointerp: float
    fitted_iso_on_exc_nointerp: np.ndarray
    residual_nointerp: np.ndarray
    dF_nointerp: np.ndarray
    dFF_nointerp: np.ndarray

    # Display smoothing parameter
    smooth_window: int

    # interval-based zF parameters
    zf_interval_start_s: float
    zf_interval_end_s: float

    # z-score variants computed from ACTIVE ΔF/F
    zF_global: np.ndarray
    zF_interval: np.ndarray


# ========================================================================================================
# CALCULATION FUNCTIONS - NORMALIZATION (Z-SCORES)
# ========================================================================================================
# Used by: "Normalization" and "Normalization + Smoothed" plot tabs
# Calculation methods: Z-score (global) and Z-score (interval-based)
# Where it's called: From recompute_fit_and_downstream() and apply_normalization() in MainAppTk
# What it computes: Standardized signal values (zero mean, unit variance)

def recompute_normalizations(res: ChannelResult) -> None:
    """
    Compute both z-score normalization variants from the already-computed ΔF/F trace.
    
    CALCULATIONS:
    1. zF_global: Z-score across entire recording (uses all ΔF/F values)
    2. zF_interval: Z-score across a user-selected time window only
    
    Formula: z = (x - mean) / std_dev
    Note: These use the ACTIVE pipeline (interpolated or holes, depending on user toggle)
    """
    # Compute z-score variants from the *ACTIVE* ΔF/F trace.
    res.zF_global = zscore_global_gui(res.dFF, ddof=0)
    res.zF_interval = zscore_interval_based(
        res.dFF,
        res.t_exc,
        start_s=res.zf_interval_start_s,
        end_s=res.zf_interval_end_s,
        ddof=1,
    )

    # IMPORTANT FOR SPEED:
    # If the normalization values changed, any cached "smoothed normalization" arrays are now outdated.
    cache = getattr(res, "_smooth_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    else:
        setattr(res, "_smooth_cache", {})


# ========================================================================================================
# CALCULATION FUNCTIONS - LINEAR FIT & DELTA F/F COMPUTATION (CORE)
# ========================================================================================================
# Used by: "Raw", "Fit", "Normalization", "Normalization+Smoothed", "Frequency" tabs
# Calculation method: Linear least-squares fit of exc = a*iso_on_exc + b
# Where it's called: recompute_fit_and_downstream() on line ~1120
# What it computes: Fitted curve, residuals, and ΔF/F = residual/fitted

def _compute_fit_and_downstream(
    t_exc: np.ndarray,
    exc: np.ndarray,
    iso_on_exc: np.ndarray,
    windows: List[Tuple[float, float]],
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    THE CORE CALCULATION - Fit isosbestic activity to explain excitatory fluorescence.
    
    STEP-BY-STEP:
    1. Create mask of samples within fit windows (user-selected in "Fit" tab)
    2. Perform linear regression: exc = a*iso_on_exc + b
    3. Compute R² (goodness of fit)
    4. Calculate residual = exc - fitted (variation NOT explained by ISO)
    5. Calculate ΔF/F = residual / fitted (normalized by baseline)
    
    FIBER PHOTOMETRY PRINCIPLE:
    - Excitatory (EXC) fluorescence = signal from neurons + motion artifact
    - Isosbestic (ISO) fluorescence = motion artifact only (same wavelength)
    - ΔF/F = (EXC - a*ISO - b) / (a*ISO + b) = neuronal signal with motion removed
    
    Returns:
        a: slope (scaling factor for ISO)
        b: intercept (baseline)
        r2: model fit quality (0-1, closer to 1 is better)
        fitted: predicted EXC values (a*ISO + b)
        residual: EXC - fitted (the "cleaned" signal)
        dF: alias for residual
        dFF: ΔF/F = residual / fitted (most common normalization)
    """
    t_exc = np.asarray(t_exc, dtype=float)
    exc = np.asarray(exc, dtype=float)
    iso_on_exc = np.asarray(iso_on_exc, dtype=float)

    fit_mask = np.zeros_like(t_exc, dtype=bool)
    for a, b in windows:
        fit_mask |= (t_exc >= float(min(a, b))) & (t_exc <= float(max(a, b)))

    finite = np.isfinite(exc) & np.isfinite(iso_on_exc)
    fit_idx = fit_mask & finite
    if fit_idx.sum() < 2:
        fit_idx = finite

    if fit_idx.sum() >= 2:
        a_fit, b_fit = fit_linear(exc[fit_idx], iso_on_exc[fit_idx])
        fitted = a_fit * iso_on_exc + b_fit
        r2 = r2_score(exc[fit_idx], fitted[fit_idx])
    else:
        a_fit, b_fit, r2 = np.nan, np.nan, np.nan
        fitted = np.full_like(iso_on_exc, np.nan, dtype=float)

    residual = exc - fitted
    dF = residual

    with np.errstate(divide="ignore", invalid="ignore"):
        dFF = np.divide(
            dF,
            fitted,
            out=np.full_like(dF, np.nan),
            where=np.isfinite(fitted) & (np.abs(fitted) > 1e-12),
        )

    return float(a_fit), float(b_fit), float(r2), fitted, residual, dF, dFF


def recompute_fit_and_downstream(res: ChannelResult) -> None:
    """
    ORCHESTRATOR FUNCTION:
    Recomputes all downstream calculations after any changes to the fit window or data pipeline.
    
    WHICH PIPELINE IS USED:
    - ACTIVE pipeline: Uses interpolated or holes version based on res.use_interpolation flag
    - NO-INTERP pipeline: Always uses holes version (for frequency analysis stability)
    
    WHAT GETS UPDATED IN res OBJECT:
    - slope, intercept, r2: Linear fit parameters (ACTIVE pipeline)
    - fitted_iso_on_exc: Predicted values (ACTIVE pipeline)
    - dF, dFF: Residual and ΔF/F (ACTIVE pipeline)
    - slope_nointerp, intercept_nointerp, r2_nointerp: Fit for holes (NO-INTERP)
    - dF_nointerp, dFF_nointerp: For frequency analysis (NO-INTERP, never interpolated)
    - zF_global, zF_interval: Z-score normalizations
    - _data_version: Incremented counter (tells cache when to refresh)
    """
    # ACTIVE pipeline
    a, b, r2, fitted, resid, dF, dFF = _compute_fit_and_downstream(
        res.t_exc, res.exc_clean, res.iso_on_exc, res.windows
    )
    res.slope = a
    res.intercept = b
    res.r2 = r2
    res.fitted_iso_on_exc = fitted
    res.residual = resid
    res.dF = dF
    res.dFF = dFF

    # NO-INTERPOLATION pipeline (for frequency analysis)
    a2, b2, r22, fitted2, resid2, dF2, dFF2 = _compute_fit_and_downstream(
        res.t_exc, res.exc_clean_holes, res.iso_on_exc_holes, res.windows
    )
    res.slope_nointerp = a2
    res.intercept_nointerp = b2
    res.r2_nointerp = r22
    res.fitted_iso_on_exc_nointerp = fitted2
    res.residual_nointerp = resid2
    res.dF_nointerp = dF2
    res.dFF_nointerp = dFF2

    recompute_normalizations(res)

    # Data version counter:
    # Every time we recompute the fit (or artifact pipeline), ΔF/F changes.
    # Some tabs (especially Frequency analysis) can cache heavy computations.
    # This counter lets them know when they must refresh.
    res._data_version = int(getattr(res, "_data_version", 0)) + 1

    # Clear any cached frequency-band results because the base ΔF/F changed.
    fcache = getattr(res, "_freq_cache", None)
    if isinstance(fcache, dict):
        fcache.clear()
    else:
        setattr(res, "_freq_cache", {})


def set_interpolation_mode(res: ChannelResult, use_interpolation: bool) -> None:
    res.use_interpolation = bool(use_interpolation)
    if res.use_interpolation:
        res.iso_clean = res.iso_clean_interp
        res.exc_clean = res.exc_clean_interp
        res.iso_on_exc = res.iso_on_exc_interp
    else:
        res.iso_clean = res.iso_clean_holes
        res.exc_clean = res.exc_clean_holes
        res.iso_on_exc = res.iso_on_exc_holes

    recompute_fit_and_downstream(res)


def recompute_artifact_pipeline_inplace(
    res: ChannelResult,
    artifact_enabled: bool,
    artifact_factor: float,
    artifact_pad: int,
    require_shared: bool,
    align_mode: str,
    use_linear_interp: bool,
) -> None:
    """
    Recompute artifact masks + cleaned signals + alignment from the RAW arrays already stored in res.
    Does NOT re-read the CSV and does NOT rebuild channels; it only updates res.* arrays and downstream.
    """
    # 1) Compute artifact masks (or disable)
    if artifact_enabled:
        art_iso_raw = detect_artifacts_by_derivative(
            res.t_iso, res.iso_raw, factor=float(artifact_factor), method="mad", pad=int(artifact_pad)
        )
        art_exc_raw = detect_artifacts_by_derivative(
            res.t_exc, res.exc_raw, factor=float(artifact_factor), method="mad", pad=int(artifact_pad)
        )

        if require_shared:
            art_iso, art_exc = shared_artifacts_by_time(res.t_iso, art_iso_raw, res.t_exc, art_exc_raw)
        else:
            art_iso, art_exc = art_iso_raw, art_exc_raw
    else:
        art_iso = np.zeros_like(res.iso_raw, dtype=bool)
        art_exc = np.zeros_like(res.exc_raw, dtype=bool)

    res.art_iso = np.asarray(art_iso, dtype=bool)
    res.art_exc = np.asarray(art_exc, dtype=bool)

    # 2) Clean holes
    res.iso_clean_holes = remove_with_holes(res.iso_raw, res.art_iso)
    res.exc_clean_holes = remove_with_holes(res.exc_raw, res.art_exc)

    # 3) Interpolated versions
    res.iso_clean_interp = linear_interpolate_by_time(res.t_iso, res.iso_clean_holes)
    res.exc_clean_interp = linear_interpolate_by_time(res.t_exc, res.exc_clean_holes)

    # 4) Align iso->exc (no time interpolation; just mapping)
    res.iso_on_exc_holes = align_iso_to_exc_no_interp(res.t_iso, res.iso_clean_holes, res.t_exc, mode=align_mode)
    res.iso_on_exc_interp = align_iso_to_exc_no_interp(res.t_iso, res.iso_clean_interp, res.t_exc, mode=align_mode)

    res.iso_on_exc_holes = np.asarray(res.iso_on_exc_holes, dtype=float)
    res.iso_on_exc_interp = np.asarray(res.iso_on_exc_interp, dtype=float)

    # Holes pipeline: keep holes where exc had artifacts
    res.iso_on_exc_holes[res.art_exc] = np.nan

    # 5) Activate pipeline choice (recomputes fit, dFF, zF, etc.)
    set_interpolation_mode(res, use_interpolation=bool(use_linear_interp))


# ---------------------- CSV analysis ----------------------
# ========================================================================================================
# MAIN ANALYSIS PIPELINE - analyze_csv()
# ========================================================================================================
# ENTRY POINT: Called from run_analysis() in MainAppTk (line ~4890)
# OUTPUT: Dictionary of ChannelResult objects (one per G-column found)
# WHAT IT DOES: Complete data processing pipeline for a single CSV file
#
# CALCULATION STEPS IN ORDER:
# 1. Load CSV and separate ISO (LED=6) and EXC (LED=7) samples
# 2. For each G-column (channel):
#    a. Detect artifacts using detect_artifacts_by_derivative()
#    b. Remove artifacts (create iso_clean_holes and exc_clean_holes)
#    c. Optionally interpolate holes (iso_clean_interp, exc_clean_interp)
#    d. Align ISO to EXC timestamps
#    e. Compute fit and ΔF/F using _compute_fit_and_downstream()
#    f. Compute z-score normalizations using recompute_normalizations()
#
# RESULT: Each ChannelResult object contains all pre-computed data for:
#    - Raw plot (exc_raw, iso_raw)
#    - Artifact plot (art_iso, art_exc)
#    - Slope normality plot (slopes computed on-demand in _draw_slope_normality)
#    - Fit plot (fitted_iso_on_exc, dF, dFF)
#    - Normalization plot (dFF, zF_global, zF_interval)
#    - Frequency plot (dFF_nointerp via FFT)

def analyze_csv(
    csv_path: str,
    artifact_enabled: bool,
    artifact_factor: float,
    artifact_method: str,  # kept for compatibility; MAD-only
    artifact_pad: int,
    require_shared: bool,
    align_mode: str,
    fit_windows: List[Tuple[float, float]],
    acq_fps_hz: Optional[float],
    smooth_window: int,
    zf_interval_start_s: float,
    zf_interval_end_s: float,
    use_linear_interp: bool,
) -> Dict[str, ChannelResult]:
    """
    COMPLETE ANALYSIS WORKFLOW FOR ONE CSV FILE.
    
    CSV STRUCTURE EXPECTED:
    - SystemTimestamp: Time column (float, seconds or milliseconds)
    - LedState: 6 (ISO), 7 (EXC), 255 (off)
    - G0, G1, G2, ... : PMT/detector channels
    
    OUTPUT:
    - One ChannelResult object per G-column (G0, G1, G2, etc.)
    - Each contains all computed/intermediate values needed for 7 plot types
    """
    # Fast load:
    #   1) Read ONLY the header to discover which G* columns exist
    #   2) Then read only the columns we actually need (much faster for wide CSVs)
    header = pd.read_csv(csv_path, nrows=0)

    for col in ["SystemTimestamp", "LedState"]:
        if col not in header.columns:
            raise ValueError(f"Missing required column '{col}'.")

    g_cols = find_g_columns(header.columns)
    if not g_cols:
        raise ValueError("No G* columns found (e.g., G0, G1...).")

    usecols = ["SystemTimestamp", "LedState"] + list(g_cols)
    df = pd.read_csv(csv_path, usecols=usecols)

    # Use LedState==7 as t0 if present; else first timestamp
    if (df["LedState"] == 7).any():
        start_idx = df.index[df["LedState"] == 7][0]
        t0 = float(df.loc[start_idx, "SystemTimestamp"])
    else:
        t0 = float(df["SystemTimestamp"].iloc[0])

    t_full = df["SystemTimestamp"].astype(float).to_numpy() - t0
    led = df["LedState"].to_numpy()

    mask_iso = led == ISO_STATE
    mask_exc = led == EXC_STATE

    t_iso = t_full[mask_iso]
    t_exc = t_full[mask_exc]

    if t_iso.size == 0 or t_exc.size == 0:
        raise ValueError("No iso/exc samples found. Check LedState coding.")

    # Effective sampling frequency for excitatory series (for frequency analysis tab)
    acq_fps_val = float(acq_fps_hz) if (acq_fps_hz is not None and np.isfinite(acq_fps_hz)) else 0.0
    if acq_fps_val > 0:
        eff_fs = acq_fps_val / 2.0
    else:
        eff_fs = estimate_fs_from_t(t_exc)
        acq_fps_val = eff_fs * 2.0 if np.isfinite(eff_fs) else 0.0

    results: Dict[str, ChannelResult] = {}

    for gcol in g_cols:
        iso_raw = df.loc[mask_iso, gcol].to_numpy(dtype=float)
        exc_raw = df.loc[mask_exc, gcol].to_numpy(dtype=float)

        if iso_raw.size == 0 or exc_raw.size == 0:
            continue

        if artifact_enabled:
            # MAD-only
            art_iso_raw = detect_artifacts_by_derivative(
                t_iso, iso_raw, factor=artifact_factor, method="mad", pad=artifact_pad
            )
            art_exc_raw = detect_artifacts_by_derivative(
                t_exc, exc_raw, factor=artifact_factor, method="mad", pad=artifact_pad
            )

            if require_shared:
                art_iso, art_exc = shared_artifacts_by_time(t_iso, art_iso_raw, t_exc, art_exc_raw)
            else:
                art_iso, art_exc = art_iso_raw, art_exc_raw
        else:
            art_iso = np.zeros_like(iso_raw, dtype=bool)
            art_exc = np.zeros_like(exc_raw, dtype=bool)

        # Cleaned with holes
        iso_clean_holes = remove_with_holes(iso_raw, art_iso)
        exc_clean_holes = remove_with_holes(exc_raw, art_exc)

        # Interpolated versions (fill holes right after artifact remover)
        iso_clean_interp = linear_interpolate_by_time(t_iso, iso_clean_holes)
        exc_clean_interp = linear_interpolate_by_time(t_exc, exc_clean_holes)

        # Align iso->exc WITHOUT time interpolation (just mapping)
        iso_on_exc_holes = align_iso_to_exc_no_interp(t_iso, iso_clean_holes, t_exc, mode=align_mode)
        iso_on_exc_interp = align_iso_to_exc_no_interp(t_iso, iso_clean_interp, t_exc, mode=align_mode)

        iso_on_exc_holes = np.asarray(iso_on_exc_holes, dtype=float)
        iso_on_exc_interp = np.asarray(iso_on_exc_interp, dtype=float)

        # Holes pipeline: keep the artifacts as NaNs (holes)
        iso_on_exc_holes[art_exc] = np.nan

        # Choose ACTIVE pipeline based on toggle
        if use_linear_interp:
            iso_clean = iso_clean_interp
            exc_clean = exc_clean_interp
            iso_on_exc = iso_on_exc_interp
        else:
            iso_clean = iso_clean_holes
            exc_clean = exc_clean_holes
            iso_on_exc = iso_on_exc_holes

        res = ChannelResult(
            gcol=gcol,
            t_iso=t_iso,
            t_exc=t_exc,
            iso_raw=iso_raw,
            exc_raw=exc_raw,
            art_iso=np.asarray(art_iso, dtype=bool),
            art_exc=np.asarray(art_exc, dtype=bool),
            iso_clean_holes=iso_clean_holes,
            exc_clean_holes=exc_clean_holes,
            iso_clean_interp=iso_clean_interp,
            exc_clean_interp=exc_clean_interp,
            iso_on_exc_holes=iso_on_exc_holes,
            iso_on_exc_interp=iso_on_exc_interp,
            use_interpolation=bool(use_linear_interp),
            iso_clean=iso_clean,
            exc_clean=exc_clean,
            iso_on_exc=iso_on_exc,
            windows=list(fit_windows),
            acq_fps_hz=acq_fps_val,
            eff_fs_hz=float(eff_fs),
            slope=np.nan,
            intercept=np.nan,
            r2=np.nan,
            fitted_iso_on_exc=np.full_like(t_exc, np.nan, dtype=float),
            residual=np.full_like(t_exc, np.nan, dtype=float),
            dF=np.full_like(t_exc, np.nan, dtype=float),
            dFF=np.full_like(t_exc, np.nan, dtype=float),
            slope_nointerp=np.nan,
            intercept_nointerp=np.nan,
            r2_nointerp=np.nan,
            fitted_iso_on_exc_nointerp=np.full_like(t_exc, np.nan, dtype=float),
            residual_nointerp=np.full_like(t_exc, np.nan, dtype=float),
            dF_nointerp=np.full_like(t_exc, np.nan, dtype=float),
            dFF_nointerp=np.full_like(t_exc, np.nan, dtype=float),
            smooth_window=max(1, int(smooth_window)),
            zf_interval_start_s=float(zf_interval_start_s),
            zf_interval_end_s=float(zf_interval_end_s),
            zF_global=np.full_like(t_exc, np.nan, dtype=float),
            zF_interval=np.full_like(t_exc, np.nan, dtype=float),
        )

        recompute_fit_and_downstream(res)
        results[gcol] = res

    if not results:
        raise ValueError("No usable G* channels found to plot.")

    return results


# ---------------------- GUI widgets ----------------------
class PlotTabTk(ttk.Frame):
    """
    A single plot display window shown in the GUI.
    
    This widget is a frame that contains:
      - A Matplotlib figure (the actual plot area)
      - A toolbar (zoom, pan, home buttons, etc.)
      - Save and Export buttons
    
    Users can interact with plots by:
      - Double-clicking to edit titles and labels
      - Right-clicking to change colors
      - Ctrl+I to add time markers
      - Using the toolbar buttons to zoom/pan
    
    This class handles the display and interactive editing without re-running
    any analysis - changes made here are purely visual.
    """

    def __init__(
        self,
        master,
        tab_name: str,
        default_filename_prefix: str = "",
        figsize: Tuple[float, float] = (7.2, 4.6),
        dpi: int = 110,
    ):
        super().__init__(master)
        self.tab_name = tab_name
        self.default_filename_prefix = default_filename_prefix

        # Callback for exporting data (set by parent widget)
        # Should return Dict[sheet_name, DataFrame] for Excel export
        self.export_provider: Optional[Callable[[], Dict[str, pd.DataFrame]]] = None

        # Create the Matplotlib figure (the actual plot area)
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        # Embed the figure in the Tkinter window using TkAgg backend
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Add Matplotlib toolbar (zoom, pan, home, save buttons)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        # Create button row at bottom (Save graph, Export data)
        btn_row = ttk.Frame(self)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)

        self.save_btn = ttk.Button(btn_row, text="Save this graph…", command=self.save_plot)
        self.save_btn.pack(side=tk.RIGHT)

        self.export_btn = ttk.Button(btn_row, text="Export data (Excel)…", command=self.export_excel)
        self.export_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Pack the plot canvas on top
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== INTERACTIVE EDITING SYSTEM =====
        # Users can change the appearance of plots without re-running analysis.
        # These settings are stored and applied each time the plot is redrawn.
        
        # Font size overrides (user can set from main toolbar)
        self.axis_label_fontsize: Optional[float] = None   # Size for x/y axis label text
        self.graph_title_fontsize: Optional[float] = None  # Size for axis titles and main title
        self.tick_label_fontsize: Optional[float] = None   # Size for numbers on the axes

        # Text edits - persist across redraws
        self._suptitle_override: Optional[str] = None

        # Overrides keyed by axes index (so they survive fig.clear()+subplots())
        self._title_overrides: Dict[int, str] = {}   # ax index -> title (ax.set_title)
        self._xlabel_overrides: Dict[int, str] = {}  # ax index -> xlabel
        self._ylabel_overrides: Dict[int, str] = {}  # ax index -> ylabel

        self._legend_title_overrides: Dict[int, str] = {}               # ax index -> legend title
        self._legend_label_overrides: Dict[int, Dict[str, str]] = {}    # ax index -> {old->new}
        self._color_overrides: Dict[int, Dict[str, str]] = {}           # ax index -> {label->hex}

        # Fallback per-artist color override (least persistent)
        self._artist_color_overrides: Dict[int, str] = {}               # id(artist) -> hex
        # ----------------- Time markers (Ctrl+I) -----------------
        # Up to 4 vertical dotted lines per figure; stored and reapplied on redraws.
        # Each item: {"x": float, "label": str, "style_idx": int}
        self._time_markers: List[Dict[str, Any]] = []
        self._time_marker_artists: List[Any] = []


        self._cid_button_press = self.canvas.mpl_connect("button_press_event", self._on_mpl_button_press)

    def _ax_key(self, ax) -> int:
        try:
            return int(self.fig.axes.index(ax))
        except Exception:
            return id(ax)

    def save_plot(self):
        suggested = f"{self.default_filename_prefix}_{self.tab_name}".strip("_")
        path = filedialog.asksaveasfilename(
            title="Save graph",
            defaultextension=".png",
            initialfile=f"{suggested}.png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.fig.savefig(path, bbox_inches="tight")

    @staticmethod
    def _safe_sheet_name(name: str, used: set) -> str:
        # Excel constraints: max 31 chars; invalid : \ / ? * [ ]
        name = re.sub(r"[:\\/?*\[\]]+", "_", str(name)).strip()
        if not name:
            name = "Sheet"
        name = name[:31]

        base = name
        k = 2
        while name in used:
            suffix = f"_{k}"
            name = (base[: max(1, 31 - len(suffix))] + suffix)[:31]
            k += 1
        used.add(name)
        return name

    def _payload_from_artists_fallback(self) -> Dict[str, pd.DataFrame]:
        """
        Fallback exporter: extracts visible line/scatter data from the figure.
        Creates one sheet per artist.
        """
        payload: Dict[str, pd.DataFrame] = {}
        used: set = set()

        for ai, ax in enumerate(self.fig.axes, start=1):
            # Lines
            for li, line in enumerate(ax.get_lines(), start=1):
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                label = line.get_label()
                if not label or label.startswith("_"):
                    label = f"line{li}"
                sheet = self._safe_sheet_name(f"ax{ai}_{label}", used)
                payload[sheet] = pd.DataFrame({"x": x, "y": y})

            # Scatter (PathCollection)
            for ci, coll in enumerate(getattr(ax, "collections", []), start=1):
                try:
                    offs = coll.get_offsets()
                except Exception:
                    continue
                if offs is None or len(offs) == 0:
                    continue
                offs = np.asarray(offs, dtype=float)
                if offs.ndim != 2 or offs.shape[1] < 2:
                    continue
                sheet = self._safe_sheet_name(f"ax{ai}_scatter{ci}", used)
                payload[sheet] = pd.DataFrame({"x": offs[:, 0], "y": offs[:, 1]})

        if not payload:
            payload[self._safe_sheet_name("empty", used)] = pd.DataFrame({"note": ["No plottable artists found."]})

        return payload

    # ---------------------- Minimal .xlsx writer (standard library only) ----------------------
    @staticmethod
    def _excel_col_name(n_1based: int) -> str:
        """1->A, 2->B, ..., 26->Z, 27->AA ..."""
        n = int(n_1based)
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    @staticmethod
    def _is_nan(v: Any) -> bool:
        try:
            return v is None or (isinstance(v, float) and not np.isfinite(v)) or (
                isinstance(v, np.floating) and not np.isfinite(v)
            )
        except Exception:
            return v is None

    def _write_worksheet_xml(self, zf: zipfile.ZipFile, sheet_path: str, df: pd.DataFrame) -> None:
        """
        Stream a worksheet XML into the zip (avoids building huge strings in memory).
        Uses inline strings; numbers are written as numeric <v>.
        """
        cols = list(df.columns)
        ncols = len(cols)

        def write_cell(fh, row_idx_1based: int, col_idx_1based: int, value: Any, force_str: bool = False):
            if self._is_nan(value):
                return  # omit blank cell

            col_letter = self._excel_col_name(col_idx_1based)
            cell_ref = f"{col_letter}{row_idx_1based}"

            # bool first (since bool is subclass of int)
            if isinstance(value, (bool, np.bool_)) and not force_str:
                v = "1" if bool(value) else "0"
                fh.write(f'<c r="{cell_ref}" t="b"><v>{v}</v></c>'.encode("utf-8"))
                return

            # numeric
            if not force_str and isinstance(value, (int, np.integer)):
                fh.write(f'<c r="{cell_ref}"><v>{int(value)}</v></c>'.encode("utf-8"))
                return
            if not force_str and isinstance(value, (float, np.floating)):
                if np.isfinite(value):
                    fh.write(f'<c r="{cell_ref}"><v>{float(value)}</v></c>'.encode("utf-8"))
                return

            # everything else -> string
            s = str(value)
            if len(s) > 32767:
                s = s[:32767]  # Excel cell text limit
            s = _xml_escape(s)
            fh.write(
                f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{s}</t></is></c>'.encode("utf-8")
            )

        with zf.open(sheet_path, "w") as fh:
            fh.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
            fh.write(b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
            fh.write(b"<sheetData>")

            # Header row (row 1)
            fh.write(b'<row r="1">')
            for j, c in enumerate(cols, start=1):
                write_cell(fh, 1, j, c, force_str=True)
            fh.write(b"</row>")

            # Data rows (row 2..)
            for i, row in enumerate(df.itertuples(index=False, name=None), start=2):
                fh.write(f'<row r="{i}">'.encode("utf-8"))
                for j in range(ncols):
                    write_cell(fh, i, j + 1, row[j], force_str=False)
                fh.write(b"</row>")

            fh.write(b"</sheetData></worksheet>")

    def _write_xlsx_minimal(self, path: str, payload: Dict[str, pd.DataFrame]) -> None:
        """
        Create a valid .xlsx file with multiple sheets using only the standard library.
        """
        used: set = set()
        sheets: List[Tuple[str, pd.DataFrame]] = []
        for sheet_name, df in payload.items():
            safe = self._safe_sheet_name(sheet_name, used)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            sheets.append((safe, df))

        # [Content_Types].xml
        ct_lines = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
            '<Override PartName="/docProps/core.xml" '
            'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        for i in range(1, len(sheets) + 1):
            ct_lines.append(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        ct_lines.append("</Types>")
        content_types_xml = "\n".join(ct_lines)

        # _rels/.rels
        rels_xml = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="xl/workbook.xml"/>',
                '<Relationship Id="rId2" '
                'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
                'Target="docProps/core.xml"/>',
                '<Relationship Id="rId3" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
                'Target="docProps/app.xml"/>',
                "</Relationships>",
            ]
        )

        # docProps/core.xml (minimal)
        core_xml = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                "<cp:coreProperties "
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" '
                'xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
                "<dc:title>Fiberlyse Export</dc:title>",
                "<dc:creator>Fiberlyse</dc:creator>",
                "<cp:lastModifiedBy>Fiberlyse</cp:lastModifiedBy>",
                "</cp:coreProperties>",
            ]
        )

        # docProps/app.xml (minimal)
        app_xml = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">',
                "<Application>Fiberlyse</Application>",
                "</Properties>",
            ]
        )

        # xl/workbook.xml
        wb_lines = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
            "<sheets>",
        ]
        for i, (sheet_name, _df) in enumerate(sheets, start=1):
            name_esc = _xml_escape(sheet_name)
            wb_lines.append(f'<sheet name="{name_esc}" sheetId="{i}" r:id="rId{i}"/>')
        wb_lines += ["</sheets>", "</workbook>"]
        workbook_xml = "\n".join(wb_lines)

        # xl/_rels/workbook.xml.rels
        wb_rels_lines = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for i in range(1, len(sheets) + 1):
            wb_rels_lines.append(
                f'<Relationship Id="rId{i}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i}.xml"/>'
            )
        wb_rels_lines.append(
            f'<Relationship Id="rId{len(sheets) + 1}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            f'Target="styles.xml"/>'
        )
        wb_rels_lines.append("</Relationships>")
        workbook_rels_xml = "\n".join(wb_rels_lines)

        # xl/styles.xml (minimal)
        styles_xml = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                '<fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>',
                '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>',
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>',
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>',
                '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>',
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>',
                "</styleSheet>",
            ]
        )

        # Write zip
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml.encode("utf-8"))
            zf.writestr("_rels/.rels", rels_xml.encode("utf-8"))
            zf.writestr("docProps/core.xml", core_xml.encode("utf-8"))
            zf.writestr("docProps/app.xml", app_xml.encode("utf-8"))
            zf.writestr("xl/workbook.xml", workbook_xml.encode("utf-8"))
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml.encode("utf-8"))
            zf.writestr("xl/styles.xml", styles_xml.encode("utf-8"))

            for i, (_sheet_name, df) in enumerate(sheets, start=1):
                self._write_worksheet_xml(zf, f"xl/worksheets/sheet{i}.xml", df)

    def export_excel(self):
        """
        Export the current plot's data to an Excel file (.xlsx).
        
        WHAT IT DOES:
        - Opens a "Save File" dialog
        - Suggests a filename based on the plot type and current tab
        - Calls an export provider function to gather the data
        - Writes a multi-sheet Excel workbook with organized data
        - Shows confirmation message with the saved file path
        
        WHY YOU'D USE THIS:
        - Share data with collaborators who use Excel
        - Further analysis in R, Python, or other tools
        - Create publication-ready figures from the data
        - Backup analysis results in human-readable format
        
        EXPORTED DATA STRUCTURE:
        - Multiple sheets, one per measurement or category
        - Each sheet has ISO/EXC fluorescence values (raw and normalized)
        - Artifacts flagged, slopes calculated, frequencies computed
        - Easily importable to statistics software
        
        EXAMPLE FILENAMES:
        - "batch_compare_[timestamp].xlsx" (for comparison plots)
        - "batch_average_[timestamp].xlsx" (for average plots)
        - "raw_data_[timestamp].xlsx" (for raw channel data)
        """
        suggested = f"{self.default_filename_prefix}_{self.tab_name}".strip("_")
        path = filedialog.asksaveasfilename(
            title="Export data (Excel)",
            defaultextension=".xlsx",
            initialfile=f"{suggested}.xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            if callable(self.export_provider):
                payload = self.export_provider()
            else:
                payload = self._payload_from_artists_fallback()

            if not isinstance(payload, dict) or not payload:
                raise ValueError("Export provider returned no data.")

            # Write real .xlsx without openpyxl/xlsxwriter
            self._write_xlsx_minimal(path, payload)

            messagebox.showinfo("Export complete", f"Saved Excel file:\n{path}")

        except Exception as e:
            messagebox.showerror("Export failed", f"Could not export Excel data:\n\n{e}")

    # ----------------- Interactive editing helpers -----------------
    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        """
        Set universal x/y label fontsize override for this figure.
        Use None to disable override.
        """
        if fontsize is None:
            self.axis_label_fontsize = None
            self.redraw()
            return
        try:
            fs = float(fontsize)
        except Exception:
            return
        if not np.isfinite(fs) or fs <= 0:
            return
        self.axis_label_fontsize = fs
        self.redraw()

    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        """
        Set universal graph title fontsize override for this figure.
        This affects ax.set_title() and fig.suptitle().
        Use None to disable override.
        """
        if fontsize is None:
            self.graph_title_fontsize = None
            self.redraw()
            return
        try:
            fs = float(fontsize)
        except Exception:
            return
        if not np.isfinite(fs) or fs <= 0:
            return
        self.graph_title_fontsize = fs
        self.redraw()

    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        """
        Set universal tick label fontsize override for this figure.
        Use None to disable override.
        """
        if fontsize is None:
            self.tick_label_fontsize = None
            self.redraw()
            return
        try:
            fs = float(fontsize)
        except Exception:
            return
        if not np.isfinite(fs) or fs <= 0:
            return
        self.tick_label_fontsize = fs
        self.redraw()

    def _toolbar_is_active(self) -> bool:
        # Avoid popping dialogs while pan/zoom tools are active
        try:
            return bool(getattr(self.toolbar, "mode", ""))
        except Exception:
            return False

    @staticmethod
    def _normalize_hex_color(s: str) -> str:
        s = (s or "").strip()
        if not s:
            raise ValueError("Empty color")

        if not s.startswith("#"):
            s = "#" + s

        # #RGB / #RGBA -> expand
        if re.fullmatch(r"#[0-9a-fA-F]{3}", s):
            r, g, b = s[1], s[2], s[3]
            return f"#{r}{r}{g}{g}{b}{b}"
        if re.fullmatch(r"#[0-9a-fA-F]{4}", s):
            r, g, b, a = s[1], s[2], s[3], s[4]
            return f"#{r}{r}{g}{g}{b}{b}{a}{a}"

        # #RRGGBB or #RRGGBBAA
        if re.fullmatch(r"#[0-9a-fA-F]{6}", s) or re.fullmatch(r"#[0-9a-fA-F]{8}", s):
            return s

        raise ValueError("Expected hex like #RRGGBB (or #RRGGBBAA)")

    @staticmethod
    def _artist_contains(artist, event) -> bool:
        try:
            hit, _ = artist.contains(event)
            return bool(hit)
        except Exception:
            return False

    @staticmethod
    def _resolve_label(label: str, mapping: Dict[str, str], max_hops: int = 12) -> str:
        # Follow mapping chains safely: A->B, B->C => C
        cur = str(label)
        for _ in range(max_hops):
            nxt = mapping.get(cur)
            if not nxt or nxt == cur:
                return cur
            cur = nxt
        return cur

    @staticmethod
    def _iter_axes_color_targets(ax):
        # Only "plot elements" (not axis spines/ticks)
        for art in list(getattr(ax, "lines", [])):
            yield art
        for art in list(getattr(ax, "collections", [])):
            yield art
        for art in list(getattr(ax, "patches", [])):
            # Skip the axes background patch
            if art is getattr(ax, "patch", None):
                continue
            yield art

    @staticmethod
    def _set_artist_color(artist, hex_color: str) -> None:
        # Many artists accept set_color; Collections/Patches may need face/edge
        if hasattr(artist, "set_color"):
            try:
                artist.set_color(hex_color)
                return
            except Exception:
                pass
        if hasattr(artist, "set_facecolor"):
            try:
                artist.set_facecolor(hex_color)
            except Exception:
                pass
        if hasattr(artist, "set_edgecolor"):
            try:
                artist.set_edgecolor(hex_color)
            except Exception:
                pass

    @staticmethod
    def _artist_current_hex(artist) -> Optional[str]:
        try:
            # Text/Line2D
            if hasattr(artist, "get_color"):
                return mcolors.to_hex(artist.get_color(), keep_alpha=False)
        except Exception:
            pass
        try:
            # Collections/Patches
            if hasattr(artist, "get_facecolor"):
                fc = artist.get_facecolor()
                if fc is not None and len(fc):
                    return mcolors.to_hex(fc[0], keep_alpha=False)
        except Exception:
            pass
        return None

    def _legend_handles(self, leg):
        handles = getattr(leg, "legend_handles", None)
        if handles is None:
            handles = getattr(leg, "legendHandles", [])
        return list(handles) if handles is not None else []

    def _hit_test_editable_text(self, event):
        """
        Return a tuple describing what was hit for editing:
          ("legend_text", ax, txt)
          ("legend_title", ax, txt)
          ("graph_title", ax, txt)    # ax.title
          ("xlabel", ax, txt)         # ax.xaxis.label
          ("ylabel", ax, txt)         # ax.yaxis.label
          ("suptitle", None, txt)
        """
        fig = self.fig

        # Legend text + title
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg:
                for txt in leg.get_texts():
                    if self._artist_contains(txt, event):
                        return ("legend_text", ax, txt)
                lt = leg.get_title()
                if lt and lt.get_text() is not None and self._artist_contains(lt, event):
                    return ("legend_title", ax, lt)

        # Axes titles + axis labels
        for ax in fig.axes:
            t = ax.title
            if t and t.get_text() is not None and self._artist_contains(t, event):
                return ("graph_title", ax, t)

            xl = ax.xaxis.label
            if xl and xl.get_text() is not None and self._artist_contains(xl, event):
                return ("xlabel", ax, xl)

            yl = ax.yaxis.label
            if yl and yl.get_text() is not None and self._artist_contains(yl, event):
                return ("ylabel", ax, yl)

        # Figure suptitle
        st = getattr(fig, "_suptitle", None)
        if st is not None and self._artist_contains(st, event):
            return ("suptitle", None, st)

        return None

    def _find_legend_entry_at_event(self, ax, event):
        leg = ax.get_legend()
        if not leg:
            return None

        texts = list(leg.get_texts())
        handles = self._legend_handles(leg)

        # text hit
        for i, txt in enumerate(texts):
            if self._artist_contains(txt, event):
                return ("legend", txt.get_text(), i)

        # handle hit
        for i, h in enumerate(handles):
            if self._artist_contains(h, event):
                label = texts[i].get_text() if i < len(texts) else getattr(h, "get_label", lambda: "")()
                return ("legend", str(label), i)

        return None

    def _find_axes_artist_at_event(self, ax, event):
        # Prefer artists with higher zorder
        cands = list(self._iter_axes_color_targets(ax))
        cands.sort(key=lambda a: float(getattr(a, "get_zorder", lambda: 0.0)()), reverse=True)
        for art in cands:
            if not getattr(art, "get_visible", lambda: True)():
                continue
            if self._artist_contains(art, event):
                return art
        return None

    def _update_label_override(self, ax_key: int, old: str, new: str) -> None:
        m = self._legend_label_overrides.setdefault(ax_key, {})

        # If "old" is already a mapped-to value, update keys that point to it
        for k in list(m.keys()):
            if m[k] == old:
                m[k] = new

        m[old] = new

        # Move color overrides to the new label if needed
        cmap = self._color_overrides.get(ax_key)
        if cmap and old in cmap and new != old:
            cmap[new] = cmap.pop(old)

    def _rename_axes_artists_label(self, ax, old: str, new: str) -> None:
        for art in self._iter_axes_color_targets(ax):
            try:
                if hasattr(art, "get_label") and hasattr(art, "set_label"):
                    if str(art.get_label()) == str(old):
                        art.set_label(str(new))
            except Exception:
                pass

    def _apply_user_overrides(self) -> None:
        fig = self.fig

        # suptitle text + fontsize override
        st = getattr(fig, "_suptitle", None)
        if self._suptitle_override is not None:
            if st is None:
                st = fig.suptitle(self._suptitle_override)
            else:
                st.set_text(self._suptitle_override)

        if self.graph_title_fontsize is not None and st is not None:
            try:
                st.set_fontsize(self.graph_title_fontsize)
            except Exception:
                pass

        for i, ax in enumerate(fig.axes):
            ax_key = int(i)

            # Universal axis label fontsize
            if self.axis_label_fontsize is not None:
                try:
                    ax.xaxis.label.set_fontsize(self.axis_label_fontsize)
                except Exception:
                    pass
                try:
                    ax.yaxis.label.set_fontsize(self.axis_label_fontsize)
                except Exception:
                    pass

            # Universal tick label fontsize
            if self.tick_label_fontsize is not None:
                try:
                    ax.tick_params(axis="both", labelsize=self.tick_label_fontsize)
                except Exception:
                    pass

            # Universal graph title fontsize
            if self.graph_title_fontsize is not None:
                try:
                    ax.title.set_fontsize(self.graph_title_fontsize)
                except Exception:
                    pass

            # Graph title text override
            if ax_key in self._title_overrides:
                try:
                    ax.title.set_text(self._title_overrides[ax_key])
                except Exception:
                    pass

            # Axis label text overrides
            if ax_key in self._xlabel_overrides:
                try:
                    ax.set_xlabel(self._xlabel_overrides[ax_key])
                except Exception:
                    pass
            if ax_key in self._ylabel_overrides:
                try:
                    ax.set_ylabel(self._ylabel_overrides[ax_key])
                except Exception:
                    pass

            # Legend title override
            leg = ax.get_legend()
            if leg and ax_key in self._legend_title_overrides:
                try:
                    leg.set_title(self._legend_title_overrides[ax_key])
                except Exception:
                    try:
                        leg.get_title().set_text(self._legend_title_overrides[ax_key])
                    except Exception:
                        pass

            # Legend label overrides
            label_map = self._legend_label_overrides.get(ax_key, {})
            if label_map:
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, "get_label") and hasattr(art, "set_label"):
                            lab = str(art.get_label())
                            if lab in label_map:
                                art.set_label(self._resolve_label(lab, label_map))
                    except Exception:
                        pass

                if leg:
                    for txt in leg.get_texts():
                        try:
                            txt.set_text(self._resolve_label(txt.get_text(), label_map))
                        except Exception:
                            pass

            # Color overrides by label
            cmap = self._color_overrides.get(ax_key, {})
            if cmap:
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, "get_label"):
                            lab = str(art.get_label())
                            if lab in cmap:
                                self._set_artist_color(art, cmap[lab])
                    except Exception:
                        pass

                # Update legend handles to match label colors
                if leg:
                    texts = list(leg.get_texts())
                    handles = self._legend_handles(leg)
                    for j, txt in enumerate(texts):
                        lab = str(txt.get_text())
                        if lab in cmap and j < len(handles):
                            self._set_artist_color(handles[j], cmap[lab])

        # Fallback per-artist colors (only persists until that artist is destroyed)
        for ax in fig.axes:
            for art in ax.get_children():
                c = self._artist_color_overrides.get(id(art))
                if c:
                    self._set_artist_color(art, c)

    # ----------------- Mouse event handler -----------------
    def _on_mpl_button_press(self, event):
        if event is None:
            return
        if self._toolbar_is_active():
            return

        # Double left click => edit title / axis labels / legend text
        if getattr(event, "dblclick", False) and getattr(event, "button", None) == 1:
            hit = self._hit_test_editable_text(event)
            if not hit:
                return

            kind, ax, txt = hit
            old_text = str(txt.get_text())

            new_text = simpledialog.askstring(
                "Edit text",
                "Enter new text:",
                initialvalue=old_text,
                parent=self.winfo_toplevel(),
            )
            if new_text is None:
                return

            new_text = str(new_text)

            if kind == "suptitle":
                self._suptitle_override = new_text
                txt.set_text(new_text)
                self.redraw()
                return

            if ax is None:
                return

            ax_key = self._ax_key(ax)

            if kind == "graph_title":
                self._title_overrides[ax_key] = new_text
                txt.set_text(new_text)
                self.redraw()
                return

            if kind == "xlabel":
                self._xlabel_overrides[ax_key] = new_text
                try:
                    ax.set_xlabel(new_text)
                except Exception:
                    txt.set_text(new_text)
                self.redraw()
                return

            if kind == "ylabel":
                self._ylabel_overrides[ax_key] = new_text
                try:
                    ax.set_ylabel(new_text)
                except Exception:
                    txt.set_text(new_text)
                self.redraw()
                return

            if kind == "legend_title":
                self._legend_title_overrides[ax_key] = new_text
                txt.set_text(new_text)
                self.redraw()
                return

            if kind == "legend_text":
                self._update_label_override(ax_key, old_text, new_text)
                self._rename_axes_artists_label(ax, old_text, new_text)
                txt.set_text(new_text)
                self.redraw()
                return

        # Right click => recolor a plot element (or legend entry)
        if getattr(event, "button", None) == 3:
            ax = getattr(event, "inaxes", None)

            # If clicked on legend entry, recolor by legend label
            if ax is not None:
                leg_hit = self._find_legend_entry_at_event(ax, event)
            else:
                leg_hit = None

            if leg_hit and ax is not None:
                _kind, label, _i = leg_hit
                ax_key = self._ax_key(ax)

                current = None
                # Try get current color from first matching artist
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, "get_label") and str(art.get_label()) == str(label):
                            current = self._artist_current_hex(art)
                            break
                    except Exception:
                        pass

                initial = current or self._color_overrides.get(ax_key, {}).get(label, "#000000")

                s = simpledialog.askstring(
                    "Set color",
                    f"Hex color for '{label}' (#RRGGBB or #RRGGBBAA):",
                    initialvalue=initial,
                    parent=self.winfo_toplevel(),
                )
                if s is None:
                    return
                try:
                    hex_color = self._normalize_hex_color(s)
                except Exception as e:
                    messagebox.showerror("Invalid color", f"{e}")
                    return

                self._color_overrides.setdefault(ax_key, {})[str(label)] = hex_color
                self.redraw()
                return

            # Otherwise: try hit-test an artist in the axes and recolor that
            if ax is None:
                return

            art = self._find_axes_artist_at_event(ax, event)
            if art is None:
                return

            # Prefer to store by label (persists across redraws), else fall back to artist-id
            label = None
            try:
                if hasattr(art, "get_label"):
                    label = str(art.get_label())
                    if label.startswith("_"):
                        label = None
            except Exception:
                label = None

            initial = self._artist_current_hex(art) or "#000000"
            s = simpledialog.askstring(
                "Set color",
                "Hex color (#RRGGBB or #RRGGBBAA):",
                initialvalue=initial,
                parent=self.winfo_toplevel(),
            )
            if s is None:
                return
            try:
                hex_color = self._normalize_hex_color(s)
            except Exception as e:
                messagebox.showerror("Invalid color", f"{e}")
                return

            if label is not None:
                self._color_overrides.setdefault(self._ax_key(ax), {})[label] = hex_color
            else:
                self._artist_color_overrides[id(art)] = hex_color

            self.redraw()

    # ----------------- Time marker helpers (Ctrl+I) -----------------
    @staticmethod
    def _axis_looks_like_time(ax) -> bool:
        """
        Heuristic to decide whether an axis is a 'time axis' where a vertical timestamp marker
        should be drawn.
        - Prefer explicit x-labels that mention time/seconds.
        - Fallback: inspect line x-data for a mostly increasing, non-trivial range.
        """
        try:
            xl = str(ax.get_xlabel() or "").lower()
        except Exception:
            xl = ""
        if ("time" in xl) or ("sec" in xl) or ("(s" in xl):
            return True

        try:
            for line in list(getattr(ax, "lines", [])):
                try:
                    x = np.asarray(line.get_xdata(), dtype=float)
                except Exception:
                    continue
                if x.size < 10:
                    continue
                xf = x[np.isfinite(x)]
                if xf.size < 10:
                    continue
                # Must have a reasonable span
                if float(np.nanmax(xf) - np.nanmin(xf)) < 1.0:
                    continue
                # Mostly non-decreasing
                dx = np.diff(xf)
                if dx.size and float(np.nanmean(dx >= 0)) > 0.8:
                    return True
        except Exception:
            pass

        return False

    def _clear_time_marker_artists(self) -> None:
        for art in list(getattr(self, "_time_marker_artists", [])):
            try:
                art.remove()
            except Exception:
                pass
        self._time_marker_artists = []

    def _apply_time_markers(self) -> None:
        """(Re)draw stored time markers on all time-like axes in this figure."""
        self._clear_time_marker_artists()

        marks = list(getattr(self, "_time_markers", []) or [])
        if not marks:
            return

        # Four distinct dotted-ish styles
        styles = [
            dict(linestyle=":", linewidth=1.3),
            dict(linestyle=(0, (1, 1)), linewidth=1.3),
            dict(linestyle=(0, (1, 3)), linewidth=1.3),
            dict(linestyle=(0, (3, 1, 1, 1)), linewidth=1.3),
        ]

        for ax in list(getattr(self.fig, "axes", [])):
            if not self._axis_looks_like_time(ax):
                continue

            added_any = False
            for m in marks:
                try:
                    x = float(m.get("x"))
                except Exception:
                    continue
                if not np.isfinite(x):
                    continue

                label = str(m.get("label") or f"t={x:g}s")
                try:
                    si = int(m.get("style_idx", 0))
                except Exception:
                    si = 0
                si = max(0, min(3, si))

                try:
                    art = ax.axvline(
                        x,
                        label=label,
                        color="0.2",
                        alpha=0.85,
                        zorder=9,
                        **styles[si],
                    )
                    self._time_marker_artists.append(art)
                    added_any = True
                except Exception:
                    pass

            # Ensure the markers appear in the legend (if the plot uses a legend)
            if added_any:
                try:
                    # Keep existing legend loc if possible
                    leg = ax.get_legend()
                    if leg is not None:
                        try:
                            loc = getattr(leg, "_loc", "best")
                        except Exception:
                            loc = "best"
                    else:
                        loc = "best"
                    ax.legend(loc=loc)
                except Exception:
                    pass

    def add_time_marker(self, x_s: float, label: Optional[str] = None) -> None:
        """
        Add a vertical dotted timestamp marker (seconds) to this plot (max 4).
        The marker is persisted and reapplied on redraws.
        """
        try:
            x = float(x_s)
        except Exception:
            return
        if not np.isfinite(x):
            return

        cur = list(getattr(self, "_time_markers", []) or [])
        if len(cur) >= 4:
            try:
                messagebox.showwarning(
                    "Time markers",
                    "You can add up to 4 time markers per plot.\n\n"
                    "Remove one with Ctrl+I, then Ctrl+Backspace (in the dialog).",
                )
            except Exception:
                pass
            return

        si = len(cur)  # 0..3 distinct styles by insertion order
        lab = str(label).strip() if (label is not None) else ""
        if not lab:
            lab = f"t={x:g}s"

        cur.append({"x": x, "label": lab, "style_idx": int(si)})
        self._time_markers = cur
        self.redraw()

    def remove_last_time_marker(self) -> bool:
        """Remove the most recently added time marker. Returns True if removed."""
        cur = list(getattr(self, "_time_markers", []) or [])
        if not cur:
            return False
        cur.pop(-1)
        self._time_markers = cur
        self.redraw()
        return True

    def clear_time_markers(self) -> None:
        """Remove all time markers from this plot."""
        self._time_markers = []
        self.redraw()

    def redraw(self):
        # Apply user overrides (titles/labels/legend labels/colors + font sizes)
        self._apply_user_overrides()

        # Apply any user-added time markers (Ctrl+I) and update legends
        self._apply_time_markers()

        # Re-apply overrides so marker artists also participate in label/color overrides
        self._apply_user_overrides()

        self.canvas.draw_idle()



class ChannelTabsTk(ttk.Frame):
    """
    Display all analysis results for one mouse/channel (one G column).
    
    When you click on a mouse tab in the main window, it shows 7 different "inner tabs":
      1. Raw: The original signal from this mouse (ISO and EXC channels)
      2. Slope Normality: Statistical test of the fit (are ISO and EXC correlated?)
      3. Artifact Remover: Which samples were flagged as artifacts?
      4. Fit: The linear regression line (ISO vs EXC) and residuals
      5. Normalization: The final normalized signal (your chosen method)
      6. Normalization (smoothed): Same, but with smoothing applied
      7. Frequency analysis: Which frequencies are present in the data?
    
    This class manages all 7 plots and handles the logic of what to show/hide and
    when to redraw (to keep the GUI responsive with large datasets).
    """

    def __init__(self, master, res: ChannelResult, norm_mode: str, parent_app=None):
        super().__init__(master)
        self.res = res  # Cache of analysis results for this mouse
        self.norm_mode = norm_mode  # Which normalization method is selected
        self.parent_app = parent_app  # Reference to MainAppTk for accessing all results

        # Create a notebook (tabbed interface) for the 7 plots
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        prefix = res.gcol  # Mouse label (e.g., "G0", "G1")

        # ===== ANALYSIS RESULTS TABS =====
        # Create 7 plot windows (one for each analysis type)
        self.tab_raw = PlotTabTk(self.tabs, "Raw", prefix)
        self.tab_slope = PlotTabTk(self.tabs, "SlopeNormality", prefix, figsize=(8.2, 5.8))
        self.tab_art = PlotTabTk(self.tabs, "ArtifactRemover", prefix)
        self.tab_fit = PlotTabTk(self.tabs, "Fit", prefix)

        # Normalization tabs (result of the processing pipeline)
        self.tab_norm = PlotTabTk(self.tabs, "Normalization", prefix)
        self.tab_norm_smooth = PlotTabTk(self.tabs, "Normalization_smoothed", prefix)

        # Frequency analysis (only relevant for ΔF/F, hides other normalization modes)
        self.tab_freq = PlotTabTk(self.tabs, "Freq analysis", prefix, figsize=(8.6, 6.4))

        self._configure_tabs()

        # ===== PERFORMANCE OPTIMIZATION =====
        # Reason: When loading 8 CSV files with multiple mice each, creating all plots
        # upfront is very slow (takes seconds to load). Instead:
        #   - Only draw the visible tab immediately
        #   - Draw other tabs the first time user clicks them
        # This makes the GUI feel fast and responsive.
        
        self._slope_drawn_once = False  # Skip expensive Shapiro-Wilk test until needed
        self._freq_drawn_version = None  # Skip frequency analysis until needed
        self._fit_initialized = False  # Skip residual plots until needed

        # When user clicks a different inner tab, draw it
        self.tabs.bind("<<NotebookTabChanged>>", self._on_inner_tab_changed)

        # Setup export functions for each tab
        self._attach_exporters()

        # Draw the first tab (Raw data) right away
        self._draw_current_tab()

    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [
            self.tab_raw,
            self.tab_slope,
            self.tab_art,
            self.tab_fit,
            self.tab_norm,
            self.tab_norm_smooth,
            self.tab_freq,
        ]:
            try:
                tab.set_axis_label_fontsize(fontsize)
            except Exception:
                pass

    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [
            self.tab_raw,
            self.tab_slope,
            self.tab_art,
            self.tab_fit,
            self.tab_norm,
            self.tab_norm_smooth,
            self.tab_freq,
        ]:
            try:
                tab.set_tick_label_fontsize(fontsize)
            except Exception:
                pass

    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [
            self.tab_raw,
            self.tab_slope,
            self.tab_art,
            self.tab_fit,
            self.tab_norm,
            self.tab_norm_smooth,
            self.tab_freq,
        ]:
            try:
                tab.set_graph_title_fontsize(fontsize)
            except Exception:
                pass

    
    # ===== INNER TAB DRAWING (Lazy rendering of 7 analysis views) =====
    # PERFORMANCE OPTIMIZATION:
    # Even within each mouse, we have 7 tabs (Raw, Slope, Artifacts, etc.)
    # Some tabs require expensive calculations (frequency analysis, slope normality)
    # So we only draw tabs when user clicks on them, and cache the result
    #
    # HOW IT WORKS:
    #   1. Raw tab: Always fast (just plot the data)
    #   2. Slope normality: Compute once, cache result
    #   3. Artifacts, Fit, Normalization: Fast (just plot)
    #   4. Frequency: Compute once, cache result (FFT is expensive)


    def _on_inner_tab_changed(self, _event=None) -> None:
        """
        Triggered when user clicks on different INNER tab within a mouse.
        
        INNER TABS vs OUTER TABS Explained:
        ===================================
        OUTER tabs: Pick which mouse to view (F1:G0, F1:G1, ...or Compare/Average)
        INNER tabs: Pick which analysis within that mouse (Raw, Slope, Artifacts, etc.)
        
        THIS METHOD:
        - Detects which inner tab user clicked on
        - Calls _draw_current_tab() to render that tab
        
        PERFORMANCE:
        - Each inner tab only draws when the user clicks it (lazy rendering)
        - Some tabs are cached: click once (slow), click again (instant)
        - Example: Frequency analysis uses FFT (expensive), cached after first draw
        """
        self._draw_current_tab()

    def _draw_current_tab(self) -> None:
        """
        Draw the currently selected inner tab (whichever one user is viewing).
        
        WHAT IT DOES:
        - Identifies which of the 7 tabs is selected
        - Calls the appropriate _draw_* method for that tab
        - Uses caching for expensive operations
        
        THE 7 INNER TABS:
        1. Raw: Original ISO/EXC fluorescence (always fast, no caching)
        2. Slope Normality: Check if slopes are normally distributed (compute once)
        3. Artifact Map: Shows which samples were flagged as artifacts
        4. Exponential Fit: Overlay of exponential decay fit (fast)
        5. Normalization: ΔF/F or z-score view (changes with normalization setting)
        6. Normalized + Smoothed: ΔF/F with noise reduction applied
        7. Frequency Analysis: Power spectrum / FFT (compute once, slow)
        
        CACHING STRATEGY:
        - _slope_drawn_once: Slope normality is expensive, compute only once
        - _freq_drawn_once: Frequency analysis is very expensive (FFT), compute only once
        - Others: Redrawn every time (they're fast or depend on changing settings)
        """
        sel = self.tabs.select()
        if not sel:
            return

        # Raw (fast)
        if sel == str(self.tab_raw):
            self._draw_raw()
            return

        # Slope normality (compute once, raw data never changes)
        if sel == str(self.tab_slope):
            if not self._slope_drawn_once:
                self._draw_slope_normality()
                self._slope_drawn_once = True
            else:
                # Re-apply any user style overrides (fonts, titles) without recomputing stats
                self.tab_slope.redraw()
            return

        # Artifact remover view (depends on pipeline)
        if sel == str(self.tab_art):
            self._draw_artifact()
            return

        # Fit view (needs SpanSelector exactly once)
        if sel == str(self.tab_fit):
            if not self._fit_initialized:
                self._draw_fit_and_attach_selector()
                self._fit_initialized = True
            else:
                # Make sure the fit plot reflects the latest arrays/settings
                self.refresh_after_pipeline_change()
            return

        # Normalization (ΔF/F or zF)
        if sel == str(self.tab_norm):
            self._draw_norm()
            return

        # Smoothed normalization
        if sel == str(self.tab_norm_smooth):
            self._draw_norm_smooth()
            return

        # Frequency analysis (heavy). Only meaningful in ΔF/F mode.
        if self.tab_freq is not None and sel == str(self.tab_freq):
            current_version = int(getattr(self.res, "_data_version", 0))
            if self._freq_drawn_version != current_version:
                self._draw_frequency()
                self._freq_drawn_version = current_version
            else:
                # Nothing changed; just redraw to apply style overrides
                self.tab_freq.redraw()
            return

    def get_active_plot_tab(self) -> Optional[PlotTabTk]:
        """
        Return the PlotTabTk instance for the currently selected INNER tab.
        Used by global hotkeys (Ctrl+I markers).
        """
        sel = self.tabs.select()
        if not sel:
            return None

        # Compare by widget path names
        if sel == str(self.tab_raw):
            return self.tab_raw
        if sel == str(self.tab_slope):
            return self.tab_slope
        if sel == str(self.tab_art):
            return self.tab_art
        if sel == str(self.tab_fit):
            return self.tab_fit
        if sel == str(self.tab_norm):
            return self.tab_norm
        if sel == str(self.tab_norm_smooth):
            return self.tab_norm_smooth
        if self.tab_freq is not None and sel == str(self.tab_freq):
            return self.tab_freq

        return None

    def _norm_tab_texts(self) -> Tuple[str, str]:
        if self.norm_mode == NORM_DFF:
            return "ΔF/F", "ΔF/F smoothed"
        if self.norm_mode == NORM_ZF_GLOBAL:
            return "zF (global)", "zF smoothed"
        if self.norm_mode == NORM_ZF_INTERVAL:
            return "zF - interval based", "zF smoothed"
        return "Normalization", "Smoothed"

    
    def _configure_tabs(self):
        """
        Configure the inner Notebook tabs.

        IMPORTANT (speed + UX):
        -----------------------
        Earlier versions removed and re-added every tab whenever normalization mode changed.
        That is slow and can reset the user's selected tab.

        Instead we now:
          - Add the base tabs once (Raw / Slope / Artifact / Fit / Norm / Smoothed)
          - Update the labels for the normalization tabs when the mode changes
          - Add/remove the Frequency analysis tab only when needed (ΔF/F only)
        """
        norm_txt, smooth_txt = self._norm_tab_texts()

        # First-time setup: notebook is empty -> add tabs in the desired order
        if len(self.tabs.tabs()) == 0:
            self.tabs.add(self.tab_raw, text="Raw")
            self.tabs.add(self.tab_slope, text="Slope normality")
            self.tabs.add(self.tab_art, text="Artifact remover")
            self.tabs.add(self.tab_fit, text="Fit")
            self.tabs.add(self.tab_norm, text=norm_txt)
            self.tabs.add(self.tab_norm_smooth, text=smooth_txt)
            if self.norm_mode == NORM_DFF:
                self.tabs.add(self.tab_freq, text="Freq analysis")
            return

        # Update normalization tab labels in-place
        self.tabs.tab(self.tab_norm, text=norm_txt)
        self.tabs.tab(self.tab_norm_smooth, text=smooth_txt)

        # Frequency tab is only meaningful for ΔF/F
        freq_present = str(self.tab_freq) in self.tabs.tabs()

        if self.norm_mode == NORM_DFF:
            if not freq_present:
                self.tabs.add(self.tab_freq, text="Freq analysis")
            else:
                self.tabs.tab(self.tab_freq, text="Freq analysis")
        else:
            if freq_present:
                # If user is currently on the freq tab, move them somewhere sensible first
                if self.tabs.select() == str(self.tab_freq):
                    self.tabs.select(self.tab_norm)
                self.tabs.forget(self.tab_freq)

    def set_norm_mode(self, norm_mode: str):
        """
        Change normalization display mode (ΔF/F / global zF / interval zF).

        We re-label the normalization tabs and redraw the currently visible tab.
        """
        self.norm_mode = norm_mode
        self._configure_tabs()

        # Redraw the currently visible tab so the user sees the new mode immediately.
        self._draw_current_tab()

    def _draw_static_tabs(self):
        self._draw_raw()
        self._draw_slope_normality()
        self._draw_artifact()
        self._draw_fit_and_attach_selector()

    
    def refresh_after_pipeline_change(self):
        """
        Called when interpolation mode or artifact pipeline changes.

        PERFORMANCE NOTE:
        -----------------
        Earlier versions redrew many tabs immediately (artifact + fit + all normalization tabs).
        With many mice this becomes slow.

        We now redraw *only the currently visible inner tab*.
        Other tabs will be refreshed automatically when the user clicks them.
        """
        sel = self.tabs.select()
        if not sel:
            return

        # Artifact tab (depends on pipeline)
        if sel == str(self.tab_art):
            self._draw_artifact()
            return

        # Fit tab (depends on pipeline, must not recreate SpanSelector unnecessarily)
        if sel == str(self.tab_fit):
            if not self._fit_initialized:
                self._draw_fit_and_attach_selector()
                self._fit_initialized = True
            else:
                try:
                    self._line_exc.set_ydata(self.res.exc_clean)
                    self._line_fit.set_ydata(self.res.fitted_iso_on_exc)
                    self._fit_info.set_text(self._fit_info_text())
                    self.tab_fit.redraw()
                except Exception:
                    # Fallback: if something went out of sync, rebuild fit artists once
                    self._draw_fit_and_attach_selector()
                    self._fit_initialized = True
            return

        # Normalization tab (depends on pipeline and norm mode)
        if sel == str(self.tab_norm):
            self._draw_norm()
            return

        # Smoothed normalization tab
        if sel == str(self.tab_norm_smooth):
            self._draw_norm_smooth()
            return

        # Frequency analysis (heavy): only redraw if visible and in ΔF/F mode
        if self.norm_mode == NORM_DFF and self.tab_freq is not None and sel == str(self.tab_freq):
            self._draw_frequency()
            self._freq_drawn_version = int(getattr(self.res, "_data_version", 0))
            return

        # Raw / slope tabs are unaffected by these pipeline changes.

    def _draw_norm_dependent_tabs(self):
        self._draw_norm()
        self._draw_norm_smooth()
        if self.norm_mode == NORM_DFF:
            self._draw_frequency()

    # ===== RAW DATA PLOT =====
    # CALCULATION USED: Pre-computed in analyze_csv() - no additional calcs here
    # DATA PLOTTED: exc_raw, iso_raw (original signal from CSV)
    # PURPOSE: Show the raw signal before any processing
    def _draw_raw(self):
        ax = self.tab_raw.ax
        ax.clear()
        ax.plot(self.res.t_exc, self.res.exc_raw, label="Excitatory raw")
        ax.plot(self.res.t_iso, self.res.iso_raw, label="Isosbestic raw")
        ax.set_title(f"{self.res.gcol} - Raw")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal")
        ax.legend(loc="best")
        self.tab_raw.redraw()

    # ===== SLOPE NORMALITY PLOT =====
    # CALCULATIONS DONE HERE:
    # 1. compute_slopes(t_iso, iso_raw) → array of dy/dt for isosbestic
    # 2. compute_slopes(t_exc, exc_raw) → array of dy/dt for excitatory
    # 3. SCIPY STATS (if available): Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling tests
    # 4. Q-Q plot points: theoretical normal quantiles vs ordered slopes
    # PURPOSE: Shows if signal changes are normally distributed (diagnostic of signal quality)
    def _draw_slope_normality(self):
        fig = self.tab_slope.fig
        fig.clear()
        axs = fig.subplots(2, 2)

        ax_hist_iso, ax_hist_exc = axs[0, 0], axs[0, 1]
        ax_qq_iso, ax_qq_exc = axs[1, 0], axs[1, 1]

        # CALCULATION STEP 1: Compute dy/dt for both channels
        slopes_iso = compute_slopes(self.res.t_iso, self.res.iso_raw)
        slopes_exc = compute_slopes(self.res.t_exc, self.res.exc_raw)

        MAX_HIST_N = 200_000
        slopes_iso_hist = subsample_even(slopes_iso, MAX_HIST_N)
        slopes_exc_hist = subsample_even(slopes_exc, MAX_HIST_N)

        def _fmt(x: float) -> str:
            return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.3g}"

        def draw_hist(ax, slopes: np.ndarray, title: str):
            ax.clear()
            if slopes.size < 3:
                ax.text(0.5, 0.5, "Not enough finite slopes", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                ax.set_xlabel("dy/dt")
                ax.set_ylabel("Density")
                return
            ax.hist(slopes, bins=80, density=True, alpha=0.75)
            ax.set_title(title)
            ax.set_xlabel("dy/dt")
            ax.set_ylabel("Density")

        def draw_qq(ax, slopes_all: np.ndarray, title: str):
            ax.clear()
            if slopes_all.size < 3:
                ax.text(0.5, 0.5, "Not enough finite slopes", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                ax.set_xlabel("Normal theoretical quantiles")
                ax.set_ylabel("Ordered slopes")
                return

            theor, ordered, m, b, r = qq_plot_points(slopes_all, max_n=DEFAULT_SLOPE_TEST_MAX_N)
            if theor.size:
                ax.scatter(theor, ordered, s=8, alpha=0.75)
                if np.isfinite(m) and np.isfinite(b):
                    xline = np.array([float(np.min(theor)), float(np.max(theor))])
                    ax.plot(xline, m * xline + b, linewidth=1.2)

            ax.set_title(title + (f" (r={r:.3f})" if np.isfinite(r) else ""))
            ax.set_xlabel("Normal theoretical quantiles")
            ax.set_ylabel("Ordered slopes")

            summ = normality_summary(slopes_all, alpha=0.05, max_n=DEFAULT_SLOPE_TEST_MAX_N)
            lines = [
                f"n={summ['n']} (test n={summ['n_test']})",
                f"mean={_fmt(summ['mean'])}  std={_fmt(summ['std'])}",
            ]

            if _HAVE_SCIPY_STATS:
                lines += [
                    f"Shapiro p={_fmt_p(summ['shapiro_p'])}",
                    f"D’Agostino p={_fmt_p(summ['dagostino_p'])}",
                    f"AD stat={_fmt(summ['anderson_stat'])}  cv5={_fmt(summ['anderson_cv5'])}",
                    f"α=0.05 verdict: {summ['verdict']}",
                ]
            else:
                lines += ["SciPy not installed: no p-values", f"α=0.05 verdict: {summ['verdict']}"]

            if summ.get("note"):
                lines.append(str(summ["note"]).strip())

            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.12),
            )

        draw_hist(ax_hist_iso, slopes_iso_hist, "Isosbestic slopes (raw) – histogram")
        draw_hist(ax_hist_exc, slopes_exc_hist, "Excitatory slopes (raw) – histogram")
        draw_qq(ax_qq_iso, slopes_iso, "Isosbestic slopes (raw) – Q–Q")
        draw_qq(ax_qq_exc, slopes_exc, "Excitatory slopes (raw) – Q–Q")

        fig.suptitle(f"{self.res.gcol} – Slope normality (dy/dt on raw)", y=0.99, fontsize=11)
        fig.tight_layout(rect=(0, 0.0, 1, 0.96))
        self.tab_slope.redraw()

    # ===== ARTIFACT MAP PLOT =====
    # CALCULATION USED: detect_artifacts_by_derivative() (called in analyze_csv)
    # DATA PLOTTED: art_iso, art_exc (boolean arrays showing flagged samples)
    # PROCESSING: Overlays red regions showing where artifacts were detected
    # PURPOSE: Visualize which samples were flagged for removal and why
    def _draw_artifact(self):
        ax = self.tab_art.ax
        ax.clear()

        # Raw (gray)
        ax.plot(self.res.t_exc, self.res.exc_raw, color="0.75", linewidth=1.0, label="Exc raw")
        ax.plot(self.res.t_iso, self.res.iso_raw, color="0.75", linewidth=1.0, linestyle="--", label="Iso raw")

        # Cleaned (holes)
        ax.plot(self.res.t_exc, self.res.exc_clean_holes, linewidth=1.2, label="Exc cleaned (holes)")
        ax.plot(self.res.t_iso, self.res.iso_clean_holes, linewidth=1.2, label="Iso cleaned (holes)")

        # If interpolation ON: highlight ONLY the added (filled) segments in a different color
        if self.res.use_interpolation:

            def _interp_overlay(y_holes: np.ndarray, y_interp: np.ndarray) -> np.ndarray:
                y_holes = np.asarray(y_holes, dtype=float)
                y_interp = np.asarray(y_interp, dtype=float)
                overlay = np.full_like(y_interp, np.nan, dtype=float)

                nan_mask = ~np.isfinite(y_holes)
                for i0, i1 in _contiguous_true_runs(nan_mask):
                    # include 1 neighbor point on both sides so the line connects visually
                    j0 = max(0, i0 - 1)
                    j1 = min(len(overlay) - 1, i1 + 1)
                    overlay[j0 : j1 + 1] = y_interp[j0 : j1 + 1]
                return overlay

            exc_fill = _interp_overlay(self.res.exc_clean_holes, self.res.exc_clean_interp)
            iso_fill = _interp_overlay(self.res.iso_clean_holes, self.res.iso_clean_interp)

            ax.plot(
                self.res.t_exc,
                exc_fill,
                linewidth=2.0,
                color="tab:orange",
                label="Exc interpolated (filled parts)",
            )
            ax.plot(
                self.res.t_iso,
                iso_fill,
                linewidth=2.0,
                color="tab:orange",
                linestyle="--",
                label="Iso interpolated (filled parts)",
            )

        # Artifact markers
        if np.any(self.res.art_exc):
            ax.scatter(
                self.res.t_exc[self.res.art_exc],
                self.res.exc_raw[self.res.art_exc],
                s=12,
                color="red",
                label="Shared artifacts",
                zorder=5,
            )
        if np.any(self.res.art_iso):
            ax.scatter(
                self.res.t_iso[self.res.art_iso],
                self.res.iso_raw[self.res.art_iso],
                s=12,
                color="red",
                label="_nolegend_",
                zorder=5,
            )

        ax.set_title(f"{self.res.gcol} - Artifact remover (red=shared, orange=filled)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal")
        ax.legend(loc="best")
        self.tab_art.redraw()

    # ===== EXPONENTIAL FIT PLOT =====
    # CALCULATION USED: _compute_fit_and_downstream() (called in analyze_csv)
    # DATA PLOTTED: 
    #   - exc_clean: excitatory signal (after artifact removal, active pipeline)
    #   - fitted_iso_on_exc: a*iso_on_exc + b (linear fit)
    # INTERACTIVE: User can drag to select new fit window → triggers update_fit_window()
    # PURPOSE: Visualize how well ISO explains EXC, assess fit quality with R²
    def _draw_fit_and_attach_selector(self):
        ax = self.tab_fit.ax
        ax.clear()

        self._fit_window_artists = []
        for (a, b) in self.res.windows:
            span = ax.axvspan(a, b, alpha=0.15)
            v1 = ax.axvline(a, linestyle=":", linewidth=1.0, alpha=0.7)
            v2 = ax.axvline(b, linestyle=":", linewidth=1.0, alpha=0.7)
            self._fit_window_artists.extend([span, v1, v2])

        (self._line_exc,) = ax.plot(self.res.t_exc, self.res.exc_clean, label="Exc (active)")
        (self._line_fit,) = ax.plot(self.res.t_exc, self.res.fitted_iso_on_exc, label="Fitted iso → exc (active)")

        self._fit_info = ax.text(
            0.99,
            0.02,
            self._fit_info_text(),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.08, lw=0.6),
        )

        ax.set_title(f"{self.res.gcol} - Fit (drag to select new window)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal")
        ax.legend(loc="best")

        def onselect(xmin, xmax):
            if xmin == xmax:
                return
            left, right = sorted([float(xmin), float(xmax)])
            self.update_fit_window(left, right)

        self._span_selector = SpanSelector(ax, onselect, "horizontal", useblit=True)
        self.tab_fit.redraw()

    def _fit_info_text(self) -> str:
        a = self.res.slope
        b = self.res.intercept
        r2 = self.res.r2
        tag = "interp" if self.res.use_interpolation else "holes"
        return f"{tag}\nslope={a:.4f}\ninterc={b:.4f}\nR²={r2:.4f}"

    def _get_norm_series(self) -> Tuple[np.ndarray, str, str, str]:
        if self.norm_mode == NORM_DFF:
            return self.res.dFF, "ΔF/F", "ΔF/F", f"{self.res.gcol} - ΔF/F"

        if self.norm_mode == NORM_ZF_GLOBAL:
            return self.res.zF_global, "zF", "zF (global)", f"{self.res.gcol} - zF (global, GUI)"

        if self.norm_mode == NORM_ZF_INTERVAL:
            a = self.res.zf_interval_start_s
            b = self.res.zf_interval_end_s
            return (
                self.res.zF_interval,
                "zF",
                f"zF (interval stats: {min(a, b):g}–{max(a, b):g} s)",
                f"{self.res.gcol} - zF - interval based",
            )

        return self.res.dFF, "Signal", "Signal", f"{self.res.gcol} - Normalization"

    def _get_norm_series_for_result(self, res: ChannelResult) -> Tuple[np.ndarray, str, str, str]:
        """Get normalization series for an arbitrary ChannelResult (used for computing global ylim)."""
        if self.norm_mode == NORM_DFF:
            return res.dFF, "ΔF/F", "ΔF/F", f"{res.gcol} - ΔF/F"

        if self.norm_mode == NORM_ZF_GLOBAL:
            return res.zF_global, "zF", "zF (global)", f"{res.gcol} - zF (global, GUI)"

        if self.norm_mode == NORM_ZF_INTERVAL:
            a = res.zf_interval_start_s
            b = res.zf_interval_end_s
            return (
                res.zF_interval,
                "zF",
                f"zF (interval stats: {min(a, b):g}–{max(a, b):g} s)",
                f"{res.gcol} - zF - interval based",
            )

        return res.dFF, "Signal", "Signal", f"{res.gcol} - Normalization"

    # ===== NORMALIZATION PLOT =====
    # CALCULATION USED: Computed in recompute_fit_and_downstream() via recompute_normalizations()
    # NORMALIZATIONS AVAILABLE:
    #   - ΔF/F: (EXC - fitted_ISO) / fitted_ISO (most common)
    #   - zF global: (ΔF/F - mean) / std (across entire recording)
    #   - zF interval: (ΔF/F - mean) / std (only within user-selected window)
    # PURPOSE: Show normalized neuronal signal with motion artifact removed
    def _draw_norm(self):
        ax = self.tab_norm.ax
        ax.clear()

        y, ylabel, label, title = self._get_norm_series()
        ax.plot(self.res.t_exc, y, label=label)

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        self.tab_norm.redraw()

    # ===== NORMALIZED + SMOOTHED PLOT =====
    # CALCULATIONS:
    #   1. Start with normalized signal (dFF, zF_global, or zF_interval)
    #   2. Apply moving average smoothing: window = self.res.smooth_window samples
    #   3. Smoothing is cached in res._smooth_cache for performance
    # PURPOSE: Same as normalization but with noise reduction applied
    # NOTE: This is display-only smoothing; frequency analysis uses NON-SMOOTHED dFF_nointerp
    def _draw_norm_smooth(self):
        ax = self.tab_norm_smooth.ax
        ax.clear()

        y, ylabel, label, title = self._get_norm_series()
        w = int(getattr(self.res, "smooth_window", DEFAULT_SMOOTH_WINDOW))

        y_s = get_smoothed_norm_array(self.res, self.norm_mode, window_size=w)

        (line_raw,) = ax.plot(
            self.res.t_exc,
            y,
            alpha=0.25,
            linewidth=1.0,
            zorder=1,
            label=f"{label} (raw)",
        )

        ax.plot(
            self.res.t_exc,
            y_s,
            linewidth=1.4,
            zorder=3,
            color=line_raw.get_color(),
            label=f"{ylabel} smoothed (win={w})",
        )

        ax.set_title(f"{title} - smoothed")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        
        # FORCE UNIFORM Y-AXIS ACROSS ALL FILES
        # Compute global min/max from all loaded files so that y-axis limits are consistent
        if self.parent_app is not None and hasattr(self.parent_app, '_results'):
            try:
                all_results = self.parent_app._results
                global_y_min = np.inf
                global_y_max = -np.inf
                
                # Iterate through all results and find the overall min/max
                for mid, res_other in all_results.items():
                    y_other, _, _, _ = self._get_norm_series_for_result(res_other)
                    y_s_other = get_smoothed_norm_array(res_other, self.norm_mode, window_size=w)
                    
                    # Use smoothed data for limits
                    finite_vals = y_s_other[np.isfinite(y_s_other)]
                    if len(finite_vals) > 0:
                        global_y_min = min(global_y_min, np.nanmin(finite_vals))
                        global_y_max = max(global_y_max, np.nanmax(finite_vals))
                
                # Apply the global limits with 5% padding
                if np.isfinite(global_y_min) and np.isfinite(global_y_max):
                    padding = (global_y_max - global_y_min) * 0.05
                    ax.set_ylim(global_y_min - padding, global_y_max + padding)
            except Exception:
                # If anything fails, just let matplotlib auto-scale
                pass
        
        self.tab_norm_smooth.redraw()

    # ===== FREQUENCY ANALYSIS PLOT =====
    # CRITICAL CALCULATION DETAILS:
    #   1. ALWAYS uses dFF_nointerp (non-interpolated ΔF/F with NaN holes preserved)
    #   2. 6 frequency bands computed via FFT or Butterworth filtering
    #   3. Sampling rate: eff_fs_hz = acq_fps / 2 (from analyze_csv line ~1270)
    #   4. FFT magnitude is normalized by sampling rate
    #   5. SciPy signal.butter() used if available (segment-wise filtering around NaNs)
    #   6. Falls back to plain FFT if SciPy unavailable (fails if NaN holes present)
    # PURPOSE: Extract power in different frequency bands (e.g., 0.1-1 Hz neuronal, 1-4 Hz, etc.)
    # NOTE: Frequency bands defined by FREQ_BANDS constant at top of file
    def _draw_frequency(self):
        """
        FREQUENCY ANALYSIS - The most computationally expensive tab.
        
        IMPORTANT: Frequency analysis ALWAYS uses the NON-INTERPOLATED ΔF/F (holes preserved),
        and it DOES NOT interpolate across holes. This ensures we're analyzing only real data.
        
        SIGNAL PROCESSING METHOD:
        - If SciPy available: Butterworth filter (segment-wise, handles NaNs gracefully)
        - If no SciPy: FFT only (fails if NaN holes present in data)
        - Each frequency band gets its own plot showing power over time
        """
        fig = self.tab_freq.fig
        fig.clear()
        axs = fig.subplots(3, 2)
        axes = np.asarray(axs).flatten()

        t = np.asarray(self.res.t_exc, dtype=float)
        # CALCULATION: Use NON-INTERPOLATED ΔF/F to preserve data integrity for frequency analysis
        dff = np.asarray(self.res.dFF_nointerp, dtype=float)

        fs = float(getattr(self.res, "eff_fs_hz", np.nan))
        acq = float(getattr(self.res, "acq_fps_hz", 0.0))

        if not np.isfinite(fs) or fs <= 0:
            fs = estimate_fs_from_t(t)

        have_holes = np.any(~np.isfinite(dff))

        if _HAVE_SCIPY_SIGNAL:
            method = "Butterworth (segment-wise, no interpolation)"
        else:
            method = "FFT (only if no NaNs)" if not have_holes else "Unavailable (SciPy signal required for NaN holes)"

        supt = (
            f"{self.res.gcol} – Band-limited ΔF/F (NO interpolation)\n"
            f"Uses non-interpolated ΔF/F; holes preserved. "
            f"Acq FPS={acq:.3g} Hz → eff fs={fs:.3g} Hz | {method}"
        )
        fig.suptitle(supt, y=0.995, fontsize=11)

        for i, (low_hz, high_hz) in enumerate(FREQ_BANDS):
            ax = axes[i]
            ax.clear()

            label = f"{high_hz:g}–{low_hz:g} Hz"

            if not np.any(np.isfinite(dff)):
                ax.text(0.5, 0.5, "ΔF/F is all NaN", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ΔF/F")
                continue

            if _HAVE_SCIPY_SIGNAL:
                y_band = bandpass_butterworth_segmentwise_no_interp(dff, low_hz, high_hz, fs, order=DEFAULT_BUTTER_ORDER)
            else:
                y_band = bandpass_fft_no_interp(dff, low_hz, high_hz, fs)

            if not np.any(np.isfinite(y_band)):
                ax.text(
                    0.5,
                    0.5,
                    "Band unavailable\n(check SciPy / NaNs / Nyquist)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                ax.plot(t, y_band, label=label, linewidth=1.0)
                ax.legend(loc="best")

            ax.set_title(label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F")

        for j in range(len(FREQ_BANDS), len(axes)):
            axes[j].axis("off")

        fig.tight_layout(rect=(0, 0.0, 1, 0.93))
        self.tab_freq.redraw()

    
    def update_fit_window(self, start_s: float, end_s: float) -> None:
        """
        Called when the user drags the SpanSelector in the Fit tab.

        PERFORMANCE NOTE:
        -----------------
        Selecting a new fit window can happen many times in quick succession.
        Redrawing *every* downstream tab (especially Frequency analysis) each time is slow.

        We therefore:
          - Recompute the fit and all downstream arrays (ΔF/F, zF, etc.)
          - Update the Fit tab immediately (because the user is looking at it)
          - Do NOT redraw other tabs right now (they will redraw lazily on next click)
        """
        
        if not isinstance(self.res.windows, list) or not self.res.windows:
         self.res.windows = [(float(start_s), float(end_s))]
        else:
          self.res.windows[0] = (float(start_s), float(end_s))



        recompute_fit_and_downstream(self.res)

        # Update the Fit tab view (lines + info label) without touching other tabs
        if not self._fit_initialized:
            self._draw_fit_and_attach_selector()
            self._fit_initialized = True
            return

        try:
            self._line_exc.set_ydata(self.res.exc_clean)
            self._line_fit.set_ydata(self.res.fitted_iso_on_exc)
            self._fit_info.set_text(self._fit_info_text())
            self.tab_fit.redraw()
        except Exception:
            # Fallback: rebuild fit artists if something went out of sync
            self._draw_fit_and_attach_selector()
            self._fit_initialized = True

    def _attach_exporters(self):
        self.tab_raw.export_provider = self._export_raw
        self.tab_slope.export_provider = self._export_slope_normality
        self.tab_art.export_provider = self._export_artifact
        self.tab_fit.export_provider = self._export_fit
        self.tab_norm.export_provider = self._export_norm
        self.tab_norm_smooth.export_provider = self._export_norm_smoothed
        self.tab_freq.export_provider = self._export_frequency

    def _meta_df(self) -> pd.DataFrame:
        wtxt = "; ".join([f"[{min(a, b):g},{max(a, b):g}]" for (a, b) in self.res.windows]) if self.res.windows else ""
        rows = [
            ("gcol", self.res.gcol),
            ("use_interpolation", bool(self.res.use_interpolation)),
            ("smooth_window", int(getattr(self.res, "smooth_window", DEFAULT_SMOOTH_WINDOW))),
            ("acq_fps_hz", float(getattr(self.res, "acq_fps_hz", np.nan))),
            ("eff_fs_hz", float(getattr(self.res, "eff_fs_hz", np.nan))),
            ("fit_windows_s", wtxt),
            ("slope_active", float(self.res.slope)),
            ("intercept_active", float(self.res.intercept)),
            ("r2_active", float(self.res.r2)),
            ("zf_interval_start_s", float(getattr(self.res, "zf_interval_start_s", DEFAULT_ZF_INTERVAL_START_S))),
            ("zf_interval_end_s", float(getattr(self.res, "zf_interval_end_s", DEFAULT_ZF_INTERVAL_END_S))),
            ("norm_mode", str(self.norm_mode)),
        ]
        return pd.DataFrame(rows, columns=["key", "value"])

    def _export_raw(self) -> Dict[str, pd.DataFrame]:
        exc = pd.DataFrame({"t_s": self.res.t_exc, "exc_raw": self.res.exc_raw})
        iso = pd.DataFrame({"t_s": self.res.t_iso, "iso_raw": self.res.iso_raw})
        return {"exc_raw": exc, "iso_raw": iso, "meta": self._meta_df()}

    def _export_slope_normality(self) -> Dict[str, pd.DataFrame]:
        slopes_iso = compute_slopes(self.res.t_iso, self.res.iso_raw)
        slopes_exc = compute_slopes(self.res.t_exc, self.res.exc_raw)

        theor_i, ord_i, m_i, b_i, r_i = qq_plot_points(slopes_iso, max_n=DEFAULT_SLOPE_TEST_MAX_N)
        theor_e, ord_e, m_e, b_e, r_e = qq_plot_points(slopes_exc, max_n=DEFAULT_SLOPE_TEST_MAX_N)

        return {
            "slopes_iso": pd.DataFrame({"slope_dy_dt": slopes_iso}),
            "slopes_exc": pd.DataFrame({"slope_dy_dt": slopes_exc}),
            "qq_iso": pd.DataFrame({"theor": theor_i, "ordered": ord_i}),
            "qq_exc": pd.DataFrame({"theor": theor_e, "ordered": ord_e}),
            "qq_fit_meta": pd.DataFrame(
                [
                    ("iso_slope", m_i),
                    ("iso_intercept", b_i),
                    ("iso_r", r_i),
                    ("exc_slope", m_e),
                    ("exc_intercept", b_e),
                    ("exc_r", r_e),
                ],
                columns=["key", "value"],
            ),
            "meta": self._meta_df(),
        }

    def _export_artifact(self) -> Dict[str, pd.DataFrame]:
        exc = pd.DataFrame(
            {
                "t_s": self.res.t_exc,
                "exc_raw": self.res.exc_raw,
                "exc_clean_holes": self.res.exc_clean_holes,
                "exc_clean_interp": self.res.exc_clean_interp,
                "artifact_exc": self.res.art_exc.astype(bool),
            }
        )
        iso = pd.DataFrame(
            {
                "t_s": self.res.t_iso,
                "iso_raw": self.res.iso_raw,
                "iso_clean_holes": self.res.iso_clean_holes,
                "iso_clean_interp": self.res.iso_clean_interp,
                "artifact_iso": self.res.art_iso.astype(bool),
            }
        )
        return {"exc_artifact": exc, "iso_artifact": iso, "meta": self._meta_df()}

    def _export_fit(self) -> Dict[str, pd.DataFrame]:
        df = pd.DataFrame(
            {
                "t_s": self.res.t_exc,
                "exc_active": self.res.exc_clean,
                "iso_on_exc_active": self.res.iso_on_exc,
                "fitted_iso_on_exc_active": self.res.fitted_iso_on_exc,
                "residual_active": self.res.residual,
                "dF_active": self.res.dF,
                "dFF_active": self.res.dFF,
                "zF_global": self.res.zF_global,
                "zF_interval": self.res.zF_interval,
            }
        )
        return {"fit_active": df, "meta": self._meta_df()}

    def _export_norm(self) -> Dict[str, pd.DataFrame]:
        y, ylabel, label, title = self._get_norm_series()
        df = pd.DataFrame({"t_s": self.res.t_exc, "value": y})
        return {"normalization": df, "meta": self._meta_df()}

    def _export_norm_smoothed(self) -> Dict[str, pd.DataFrame]:
        y, ylabel, label, title = self._get_norm_series()
        w = int(getattr(self.res, "smooth_window", DEFAULT_SMOOTH_WINDOW))
        y_s = get_smoothed_norm_array(self.res, self.norm_mode, window_size=w)
        df = pd.DataFrame({"t_s": self.res.t_exc, "value_raw": y, f"value_smoothed_win{w}": y_s})
        return {"normalization_smoothed": df, "meta": self._meta_df()}

    def _export_frequency(self) -> Dict[str, pd.DataFrame]:
        """
        Mirrors the plotting logic: ALWAYS uses non-interpolated ΔF/F (holes),
        and does NOT interpolate across holes.
        """
        t = np.asarray(self.res.t_exc, dtype=float)
        dff = np.asarray(self.res.dFF_nointerp, dtype=float)

        fs = float(getattr(self.res, "eff_fs_hz", np.nan))
        acq = float(getattr(self.res, "acq_fps_hz", 0.0))
        if not np.isfinite(fs) or fs <= 0:
            fs = estimate_fs_from_t(t)

        out = {"t_s": t, "dFF_nointerp": dff}

        for (low_hz, high_hz) in FREQ_BANDS:
            col = f"band_{low_hz:g}_{high_hz:g}_Hz"
            if _HAVE_SCIPY_SIGNAL:
                y_band = bandpass_butterworth_segmentwise_no_interp(dff, low_hz, high_hz, fs, order=DEFAULT_BUTTER_ORDER)
            else:
                y_band = bandpass_fft_no_interp(dff, low_hz, high_hz, fs)
            out[col] = y_band

        df = pd.DataFrame(out)

        meta = self._meta_df()
        extra = pd.DataFrame(
            [
                ("freq_acq_fps_hz", acq),
                ("freq_eff_fs_hz", fs),
                ("method", "butter_sos_segmentwise" if _HAVE_SCIPY_SIGNAL else "fft"),
            ],
            columns=["key", "value"],
        )
        meta2 = pd.concat([meta, extra], ignore_index=True)

        return {"freq_bands": df, "meta": meta2}


class BatchCompareTk(ttk.Frame):
    """Batch compare: pick multiple mice and plot them together."""

    def __init__(self, master, app: "MainAppTk"):
        super().__init__(master)
        self.app = app

        self.selected_ids: List[str] = []
        self._available_ids: List[str] = []
        self._display: Dict[str, str] = {}

        # Layout: controls (left) + plot (right)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=8)
        controls.grid(row=0, column=0, sticky="nsw")

        plot_holder = ttk.Frame(self)
        plot_holder.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(8, 8))
        plot_holder.rowconfigure(0, weight=1)
        plot_holder.columnconfigure(0, weight=1)

        # Controls
        ttk.Label(controls, text="Available mice:").grid(row=0, column=0, sticky="w")

        self.lst_available = tk.Listbox(controls, selectmode=tk.EXTENDED, height=10, width=28)
        sb_av = ttk.Scrollbar(controls, orient="vertical", command=self.lst_available.yview)
        self.lst_available.config(yscrollcommand=sb_av.set)
        self.lst_available.grid(row=1, column=0, sticky="nsew")
        sb_av.grid(row=1, column=1, sticky="ns")

        btns = ttk.Frame(controls)
        btns.grid(row=2, column=0, columnspan=2, pady=(6, 10), sticky="ew")
        self.btn_add = ttk.Button(btns, text="Add →", command=self.add_selected)
        self.btn_add.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_remove = ttk.Button(btns, text="← Remove", command=self.remove_selected)
        self.btn_remove.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        ttk.Label(controls, text="Selected for compare:").grid(row=3, column=0, sticky="w")

        self.lst_selected = tk.Listbox(controls, selectmode=tk.EXTENDED, height=10, width=28)
        sb_sel = ttk.Scrollbar(controls, orient="vertical", command=self.lst_selected.yview)
        self.lst_selected.config(yscrollcommand=sb_sel.set)
        self.lst_selected.grid(row=4, column=0, sticky="nsew")
        sb_sel.grid(row=4, column=1, sticky="ns")

        opts = ttk.Frame(controls)
        opts.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.var_smoothed = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Plot smoothed trace", variable=self.var_smoothed, command=self.refresh_plot).pack(
            side=tk.TOP, anchor="w"
        )

        self.btn_plot = ttk.Button(opts, text="Plot selected", command=self.refresh_plot)
        self.btn_plot.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        self.btn_clear = ttk.Button(opts, text="Clear selection", command=self.clear_selection)
        self.btn_clear.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        hint = ttk.Label(
            controls,
            text="Tip: Add time markers on this plot with Ctrl+I\n(remove last with Ctrl+I then Backspace).",
            foreground="gray35",
            justify="left",
        )
        hint.grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 0))

        # Plot tab
        self.plot = PlotTabTk(plot_holder, "Compare", default_filename_prefix="batch_compare", figsize=(8.2, 5.2))
        self.plot.grid(row=0, column=0, sticky="nsew")
        self.plot.export_provider = self._export_compare

        # Bind double-click add/remove for convenience
        self.lst_available.bind("<Double-Button-1>", lambda _e: self.add_selected())
        self.lst_selected.bind("<Double-Button-1>", lambda _e: self.remove_selected())

        self.refresh_available()
        self.refresh_plot()

    def refresh_available(self) -> None:
        items = self.app.get_mouse_items()
        self._display = {mid: label for mid, label in items}

        # Drop any selected that no longer exists
        self.selected_ids = [mid for mid in self.selected_ids if mid in self._display]

        self._available_ids = [mid for mid, _lab in items if mid not in set(self.selected_ids)]

        self.lst_available.delete(0, tk.END)
        for mid in self._available_ids:
            self.lst_available.insert(tk.END, self._display.get(mid, mid))

        self.lst_selected.delete(0, tk.END)
        for mid in self.selected_ids:
            self.lst_selected.insert(tk.END, self._display.get(mid, mid))

    def add_selected(self) -> None:
        sel = list(self.lst_available.curselection())
        if not sel:
            return
        for i in sel:
            if 0 <= int(i) < len(self._available_ids):
                mid = self._available_ids[int(i)]
                if mid not in self.selected_ids:
                    self.selected_ids.append(mid)
        self.refresh_available()
        self.refresh_plot()

    def remove_selected(self) -> None:
        sel = sorted(list(self.lst_selected.curselection()), reverse=True)
        if not sel:
            return
        for i in sel:
            if 0 <= int(i) < len(self.selected_ids):
                self.selected_ids.pop(int(i))
        self.refresh_available()
        self.refresh_plot()

    def clear_selection(self) -> None:
        self.selected_ids = []
        self.refresh_available()
        self.refresh_plot()

    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_axis_label_fontsize(fontsize)
        except Exception:
            pass

    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_graph_title_fontsize(fontsize)
        except Exception:
            pass

    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_tick_label_fontsize(fontsize)
        except Exception:
            pass

    def refresh_plot(self) -> None:
        ax = self.plot.ax
        ax.clear()

        mode = self.app.get_norm_mode()
        smooth_win = self.app.get_smooth_window()
        do_smooth = bool(self.var_smoothed.get())

        ylabel = "Signal"
        plotted = 0

        for mid in self.selected_ids:
            res = self.app.get_mouse_result(mid)
            if res is None:
                continue
            t, y, ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:
                y = get_smoothed_norm_array(res, mode, window_size=smooth_win)

            ax.plot(t, y, label=self._display.get(mid, mid))
            plotted += 1

        ax.set_title(f"Batch compare ({plotted} mouse{'es' if plotted != 1 else ''})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        if plotted:
            ax.legend(loc="best")

        self.plot.redraw()

    def _export_compare(self) -> Dict[str, pd.DataFrame]:
        mode = self.app.get_norm_mode()
        smooth_win = self.app.get_smooth_window()
        do_smooth = bool(self.var_smoothed.get())

        payload: Dict[str, pd.DataFrame] = {}
        for mid in self.selected_ids:
            res = self.app.get_mouse_result(mid)
            if res is None:
                continue
            t, y, _ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:
                y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
            payload[self._display.get(mid, mid)] = pd.DataFrame({"t_s": t, "value": y})

        payload["meta"] = pd.DataFrame(
            [
                ("mode", mode),
                ("smoothed", do_smooth),
                ("smooth_window", smooth_win),
                ("n_selected", len(self.selected_ids)),
            ],
            columns=["key", "value"],
        )
        return payload


class BatchAverageTk(ttk.Frame):
    """Batch average: choose two sets (Group A and Group B) and plot their means."""

    def __init__(self, master, app: "MainAppTk"):
        super().__init__(master)
        self.app = app

        self.group_a: List[str] = []
        self.group_b: List[str] = []
        self._available_ids: List[str] = []
        self._display: Dict[str, str] = {}

        # Layout: controls (left) + plot (right)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=8)
        controls.grid(row=0, column=0, sticky="nsw")

        plot_holder = ttk.Frame(self)
        plot_holder.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(8, 8))
        plot_holder.rowconfigure(0, weight=1)
        plot_holder.columnconfigure(0, weight=1)

        # Controls
        ttk.Label(controls, text="Available mice:").grid(row=0, column=0, sticky="w")

        self.lst_available = tk.Listbox(controls, selectmode=tk.EXTENDED, height=9, width=28)
        sb_av = ttk.Scrollbar(controls, orient="vertical", command=self.lst_available.yview)
        self.lst_available.config(yscrollcommand=sb_av.set)
        self.lst_available.grid(row=1, column=0, sticky="nsew")
        sb_av.grid(row=1, column=1, sticky="ns")

        btns = ttk.Frame(controls)
        btns.grid(row=2, column=0, columnspan=2, pady=(6, 10), sticky="ew")
        ttk.Button(btns, text="Add to A →", command=self.add_to_a).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Add to B →", command=self.add_to_b).pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        # Group A
        ttk.Label(controls, text="Group A:").grid(row=3, column=0, sticky="w")
        self.lst_a = tk.Listbox(controls, selectmode=tk.EXTENDED, height=6, width=28)
        sb_a = ttk.Scrollbar(controls, orient="vertical", command=self.lst_a.yview)
        self.lst_a.config(yscrollcommand=sb_a.set)
        self.lst_a.grid(row=4, column=0, sticky="nsew")
        sb_a.grid(row=4, column=1, sticky="ns")
        ttk.Button(controls, text="Remove from A", command=self.remove_from_a).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(4, 10)
        )

        # Group B
        ttk.Label(controls, text="Group B:").grid(row=6, column=0, sticky="w")
        self.lst_b = tk.Listbox(controls, selectmode=tk.EXTENDED, height=6, width=28)
        sb_b = ttk.Scrollbar(controls, orient="vertical", command=self.lst_b.yview)
        self.lst_b.config(yscrollcommand=sb_b.set)
        self.lst_b.grid(row=7, column=0, sticky="nsew")
        sb_b.grid(row=7, column=1, sticky="ns")
        ttk.Button(controls, text="Remove from B", command=self.remove_from_b).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(4, 10)
        )

        # Options
        opts = ttk.Frame(controls)
        opts.grid(row=9, column=0, columnspan=2, sticky="ew")

        self.var_show_individual = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Show individual traces", variable=self.var_show_individual, command=self.refresh_plot).pack(
            side=tk.TOP, anchor="w"
        )

        self.var_show_sem = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Show SEM shading", variable=self.var_show_sem, command=self.refresh_plot).pack(
            side=tk.TOP, anchor="w"
        )

        self.var_smoothed = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Average smoothed traces", variable=self.var_smoothed, command=self.refresh_plot).pack(
            side=tk.TOP, anchor="w"
        )

        self.btn_plot = ttk.Button(opts, text="Plot averages", command=self.refresh_plot)
        self.btn_plot.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        self.btn_clear = ttk.Button(opts, text="Clear groups", command=self.clear_groups)
        self.btn_clear.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        hint = ttk.Label(
            controls,
            text="Tip: Ctrl+I adds a vertical time marker on this plot.",
            foreground="gray35",
            justify="left",
        )
        hint.grid(row=10, column=0, columnspan=2, sticky="w", pady=(12, 0))

        # Plot tab
        self.plot = PlotTabTk(plot_holder, "Average", default_filename_prefix="batch_average", figsize=(8.2, 5.2))
        self.plot.grid(row=0, column=0, sticky="nsew")
        self.plot.export_provider = self._export_average

        self.refresh_available()
        self.refresh_plot()

    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_axis_label_fontsize(fontsize)
        except Exception:
            pass

    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_graph_title_fontsize(fontsize)
        except Exception:
            pass

    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:
            self.plot.set_tick_label_fontsize(fontsize)
        except Exception:
            pass

    def refresh_available(self) -> None:
        items = self.app.get_mouse_items()
        self._display = {mid: label for mid, label in items}

        # Drop any missing ids from groups
        all_ids = set(self._display.keys())
        self.group_a = [mid for mid in self.group_a if mid in all_ids]
        self.group_b = [mid for mid in self.group_b if mid in all_ids]

        used = set(self.group_a) | set(self.group_b)
        self._available_ids = [mid for mid, _lab in items if mid not in used]

        self.lst_available.delete(0, tk.END)
        for mid in self._available_ids:
            self.lst_available.insert(tk.END, self._display.get(mid, mid))

        self.lst_a.delete(0, tk.END)
        for mid in self.group_a:
            self.lst_a.insert(tk.END, self._display.get(mid, mid))

        self.lst_b.delete(0, tk.END)
        for mid in self.group_b:
            self.lst_b.insert(tk.END, self._display.get(mid, mid))

    def _add_from_available(self, target: List[str]) -> None:
        sel = list(self.lst_available.curselection())
        if not sel:
            return
        for i in sel:
            if 0 <= int(i) < len(self._available_ids):
                mid = self._available_ids[int(i)]
                if mid not in target:
                    target.append(mid)
        self.refresh_available()
        self.refresh_plot()

    def add_to_a(self) -> None:
        self._add_from_available(self.group_a)

    def add_to_b(self) -> None:
        self._add_from_available(self.group_b)

    def remove_from_a(self) -> None:
        sel = sorted(list(self.lst_a.curselection()), reverse=True)
        if not sel:
            return
        for i in sel:
            if 0 <= int(i) < len(self.group_a):
                self.group_a.pop(int(i))
        self.refresh_available()
        self.refresh_plot()

    def remove_from_b(self) -> None:
        sel = sorted(list(self.lst_b.curselection()), reverse=True)
        if not sel:
            return
        for i in sel:
            if 0 <= int(i) < len(self.group_b):
                self.group_b.pop(int(i))
        self.refresh_available()
        self.refresh_plot()

    def clear_groups(self) -> None:
        self.group_a = []
        self.group_b = []
        self.refresh_available()
        self.refresh_plot()

    @staticmethod
    def _interp_to_common_grid(t_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate each series onto the intersection grid (based on shortest time vector)."""
        if not t_list or not y_list:
            return np.array([], dtype=float), np.empty((0, 0), dtype=float)

        # Use the shortest time vector as the common grid (keeps it simple & deterministic)
        lens = [len(t) for t in t_list]
        j = int(np.argmin(lens))
        t0 = np.asarray(t_list[j], dtype=float)
        if t0.size == 0:
            return np.array([], dtype=float), np.empty((0, 0), dtype=float)

        Y = []
        for t, y in zip(t_list, y_list):
            t = np.asarray(t, dtype=float)
            y = np.asarray(y, dtype=float)
            if t.size < 2 or y.size != t.size:
                continue

            # Fill NaNs before interpolation so that the mean isn't spuriously NaN everywhere
            y_filled = _fill_nans_linear_1d(y)  # fast NumPy interpolation (same practical behaviour as pandas)

            try:
                yi = np.interp(t0, t, y_filled).astype(float)
            except Exception:
                continue
            Y.append(yi)

        if not Y:
            return t0, np.empty((0, t0.size), dtype=float)

        return t0, np.vstack(Y)

    def _group_stats(self, ids: List[str], mode: str, smooth_win: int, do_smooth: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        t_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for mid in ids:
            res = self.app.get_mouse_result(mid)
            if res is None:
                continue
            t, y, _ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:
                y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
            t_list.append(t)
            y_list.append(y)

        t0, Y = self._interp_to_common_grid(t_list, y_list)
        if Y.size == 0:
            return t0, np.full_like(t0, np.nan, dtype=float), np.full_like(t0, np.nan, dtype=float), 0

        mean = np.nanmean(Y, axis=0)
        # SEM: std/sqrt(n) using finite counts per timepoint
        n_eff = np.sum(np.isfinite(Y), axis=0)
        std = np.nanstd(Y, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sem = std / np.sqrt(np.where(n_eff > 0, n_eff, np.nan))

        return t0, mean, sem, int(Y.shape[0])

    def refresh_plot(self) -> None:
        ax = self.plot.ax
        ax.clear()

        mode = self.app.get_norm_mode()
        smooth_win = self.app.get_smooth_window()
        do_smooth = bool(self.var_smoothed.get())

        ylabel = "Signal"
        show_individual = bool(self.var_show_individual.get())
        show_sem = bool(self.var_show_sem.get())

        # Individual traces (light)
        if show_individual:
            for mid in self.group_a:
                res = self.app.get_mouse_result(mid)
                if res is None:
                    continue
                t, y, ylabel = get_series_for_norm_mode(res, mode)
                if do_smooth:
                    y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
                ax.plot(t, y, alpha=0.25, linewidth=1.0, label="_nolegend_")

            for mid in self.group_b:
                res = self.app.get_mouse_result(mid)
                if res is None:
                    continue
                t, y, ylabel = get_series_for_norm_mode(res, mode)
                if do_smooth:
                    y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
                ax.plot(t, y, alpha=0.25, linewidth=1.0, linestyle="--", label="_nolegend_")

        # Group A mean + sem
        tA, mA, sA, nA = self._group_stats(self.group_a, mode, smooth_win, do_smooth)
        if nA > 0 and tA.size:
            (lineA,) = ax.plot(tA, mA, linewidth=2.0, label=f"Group A (n={nA})")
            if show_sem and np.any(np.isfinite(sA)):
                c = lineA.get_color()
                ax.fill_between(tA, mA - sA, mA + sA, alpha=0.2, color=c, linewidth=0)

        # Group B mean + sem
        tB, mB, sB, nB = self._group_stats(self.group_b, mode, smooth_win, do_smooth)
        if nB > 0 and tB.size:
            (lineB,) = ax.plot(tB, mB, linewidth=2.0, linestyle="--", label=f"Group B (n={nB})")
            if show_sem and np.any(np.isfinite(sB)):
                c = lineB.get_color()
                ax.fill_between(tB, mB - sB, mB + sB, alpha=0.2, color=c, linewidth=0)

        ax.set_title("Batch average (Group A vs Group B)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)

        if (nA > 0) or (nB > 0):
            ax.legend(loc="best")

        self.plot.redraw()

    def _export_average(self) -> Dict[str, pd.DataFrame]:
        mode = self.app.get_norm_mode()
        smooth_win = self.app.get_smooth_window()
        do_smooth = bool(self.var_smoothed.get())

        tA, mA, sA, nA = self._group_stats(self.group_a, mode, smooth_win, do_smooth)
        tB, mB, sB, nB = self._group_stats(self.group_b, mode, smooth_win, do_smooth)

        payload: Dict[str, pd.DataFrame] = {}

        payload["groupA_mean_sem"] = pd.DataFrame({"t_s": tA, "mean": mA, "sem": sA})
        payload["groupB_mean_sem"] = pd.DataFrame({"t_s": tB, "mean": mB, "sem": sB})

        payload["groupA_members"] = pd.DataFrame({"mouse": [self._display.get(mid, mid) for mid in self.group_a]})
        payload["groupB_members"] = pd.DataFrame({"mouse": [self._display.get(mid, mid) for mid in self.group_b]})

        payload["meta"] = pd.DataFrame(
            [
                ("mode", mode),
                ("avg_smoothed", do_smooth),
                ("smooth_window", smooth_win),
                ("n_groupA", nA),
                ("n_groupB", nB),
            ],
            columns=["key", "value"],
        )
        return payload


def _basename_no_ext(path: str) -> str:
    base = os.path.basename(str(path))
    name, _ext = os.path.splitext(base)
    name = name.strip()
    return name if name else (base if base else "file")


def _unique_aliases(paths: List[str]) -> List[str]:
    used: set = set()
    aliases: List[str] = []
    for p in paths:
        base = _basename_no_ext(p)
        alias = base
        k = 2
        while alias in used:
            alias = f"{base}_{k}"
            k += 1
        used.add(alias)
        aliases.append(alias)
    return aliases


def get_series_for_norm_mode(res: ChannelResult, mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (t, y, ylabel) for the requested normalization mode."""
    if mode == NORM_DFF:
        return np.asarray(res.t_exc, dtype=float), np.asarray(res.dFF, dtype=float), "ΔF/F"
    if mode == NORM_ZF_GLOBAL:
        return np.asarray(res.t_exc, dtype=float), np.asarray(res.zF_global, dtype=float), "zF"
    if mode == NORM_ZF_INTERVAL:
        return np.asarray(res.t_exc, dtype=float), np.asarray(res.zF_interval, dtype=float), "zF"
    return np.asarray(res.t_exc, dtype=float), np.asarray(res.dFF, dtype=float), "Signal"


class MainAppTk:
    """
    THE MAIN APPLICATION WINDOW.
    
    This is the "master controller" of the entire program. It:
    
    1. LOADS DATA
       - Lets you select up to 8 CSV files from your computer
       - Reads mouse signal data (G0, G1, etc.), LED states, and timestamps
       
    2. RUNS ANALYSIS
       - Processes the raw data (artifact removal, normalization, frequency analysis)
       - Runs in a background thread so the GUI stays responsive
       - Can be configured with controls: MAD factor, smoothing, interpolation, etc.
       
    3. DISPLAYS RESULTS
       - One "mouse tab" for each signal column found in the CSV (G0, G1, etc.)
       - Each mouse tab contains 7 "inner tabs" showing different analyses
       - You can see raw data, artifacts, fits, normalized signal, frequency content
       
    4. MULTI-FILE VIEWING
       - You can analyze multiple CSV files in one run
       - Use the "File" drop-down to switch which file's channels are shown
       
    5. INTERACTIVE EDITING
       - Double-click chart titles to rename them
       - Right-click elements to recolor them
       - Ctrl+I to add time markers
       - Adjust font sizes from the toolbar
       
    6. EXPORT
       - Save each graph as PNG/SVG/PDF
       - Export underlying data to Excel
    
    The window is organized into sections (top to bottom):
      - FILE SELECTION: Add/clear CSVs and Run button
      - CONTROLS: Artifact remover sensitivity, smoothing, interpolation, FPS
      - NORMALIZATION: Choose ΔF/F or z-score method, set font sizes
      - TABS: A notebook showing the selected file's mouse/channel tabs
      - STATUS BAR: Shows current operation or error messages
    """
    
    def __init__(self, initial_csvs: Optional[List[str]] = None, autorun: bool = False):
        self.root = tk.Tk()
        self.root.title("Fiberlyse")

        # Up to 8 CSVs for one analysis run
        self.csv_paths: List[str] = []
        if initial_csvs:
            for p in list(initial_csvs):
                if p and p not in self.csv_paths:
                    self.csv_paths.append(p)
        self.csv_paths = self.csv_paths[:8]

        # Aggregated results: mouse_id -> ChannelResult
        self._results: Optional[Dict[str, ChannelResult]] = None
        self._mouse_display: Dict[str, str] = {}
        self._mouse_order: List[str] = []

        self._analysis_thread: Optional[threading.Thread] = None
        self._channel_widgets: Dict[str, ChannelTabsTk] = {}

        self.compare_widget: Optional[BatchCompareTk] = None
        self.average_widget: Optional[BatchAverageTk] = None

        # Per-file viewing dropdown (populated after analysis)
        self._result_file_order: List[str] = []
        self._file_alias_by_key: Dict[str, str] = {}
        self._file_path_by_key: Dict[str, str] = {}

        # Font overrides (None until user presses Enter)
        self._axis_label_fs_override: Optional[float] = None
        self._graph_title_fs_override: Optional[float] = None
        # override for tick label fontsize (numbers on axes)
        self._tick_label_fs_override: Optional[float] = None

        # Ctrl+I marker hotkey state

        # ------------------ Top row (file list + run) ------------------
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.lbl_file = ttk.Label(top, text="")
        self.lbl_file.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.btn_add = ttk.Button(top, text="Add CSV(s)…", command=self.add_csvs)
        self.btn_add.grid(row=0, column=1, sticky="e", padx=(0, 6))

        self.btn_clear = ttk.Button(top, text="Clear files", command=self.clear_csvs)
        self.btn_clear.grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.btn_run = ttk.Button(top, text="Run analysis", command=self.run_analysis)
        self.btn_run.grid(row=0, column=3, sticky="e")

        self._update_files_label()

        # ===== CONTROLS ROW (Artifact Detection & Signal Processing Parameters) =====
        # WHAT: This row configures artifact detection, sampling, and interpolation
        # WHY: These settings control how the raw ISO/EXC signals are cleaned and prepared
        # before being analyzed
        row2 = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        row2.pack(side=tk.TOP, fill=tk.X)

        # ---- ARTIFACT REMOVER SECTION ----
        # Automatically flag suspicious points (movement, glitches) as "artifacts"
        # Marked points are excluded from analysis (or interpolated to fill the gap)
        self.var_artifact_enabled = tk.BooleanVar(value=True)
        self.chk_artifact_enabled = ttk.Checkbutton(
            row2,
            text="Enable artifact remover (MAD)",
            variable=self.var_artifact_enabled,
            command=self.on_artifact_enabled_toggled,
        )
        self.chk_artifact_enabled.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 12))

        # Factor: How sensitive the artifact detection is
        # HIGHER value = less sensitive (fewer things flagged as artifacts)
        # LOWER value = more sensitive (more things flagged)
        # ~11.9 is a good middle ground for most data
        ttk.Label(row2, text="Factor:").grid(row=0, column=2, sticky="w")
        self.var_factor = tk.StringVar(value="11.9")
        self.spin_factor = ttk.Spinbox(row2, from_=0.1, to=1000.0, increment=0.1, textvariable=self.var_factor, width=8)
        self.spin_factor.grid(row=0, column=3, padx=(6, 12), sticky="w")

        # Pad: How many neighboring samples to also mark as artifacts
        # Prevents isolated artifacts from being ignored (fills in brief gaps)
        # Higher values create wider artifact regions around detected problems
        ttk.Label(row2, text="Pad (samples):").grid(row=0, column=4, sticky="w")
        self.var_pad = tk.StringVar(value="1")
        self.spin_pad = ttk.Spinbox(row2, from_=0, to=50, increment=1, textvariable=self.var_pad, width=6)
        self.spin_pad.grid(row=0, column=5, padx=(6, 12), sticky="w")

        # Shared artifacts: Only flag a point if BOTH channels (ISO and EXC) show the same problem
        # Reduces false positives from single-channel noise (e.g., one PMT having a glitch)
        self.var_shared = tk.BooleanVar(value=True)
        self.chk_shared = ttk.Checkbutton(row2, text="Require shared artifacts", variable=self.var_shared)
        self.chk_shared.grid(row=0, column=6, padx=(0, 12), sticky="w")

        # ---- ACQUISITION SETTINGS ----
        # The frame rate of your camera (frames per second)
        # Note: The system alternates ISO/EXC for each frame, so effective rate = FPS / 2
        ttk.Label(row2, text="Acq FPS (Hz):").grid(row=0, column=7, sticky="w")
        self.var_acq_fps = tk.StringVar(value=str(DEFAULT_ACQ_FPS_HZ))
        self.spin_acq_fps = ttk.Spinbox(row2, from_=0.0, to=10000.0, increment=0.1, textvariable=self.var_acq_fps, width=8)
        self.spin_acq_fps.grid(row=0, column=8, padx=(6, 6), sticky="w")
        ttk.Label(row2, text="(eff = FPS/2)").grid(row=0, column=9, sticky="w")

        # ---- NOISE SMOOTHING ----
        # When enabled, nearby points are averaged together to reduce noise in the signal
        # Larger window = more smoothing = cleaner display but less fine detail
        # Smaller window = preserves more of the original data but noisier
        ttk.Label(row2, text="Smooth win (samples):").grid(row=0, column=10, sticky="w")
        self.var_smooth_win = tk.StringVar(value=str(DEFAULT_SMOOTH_WINDOW))
        self.spin_smooth_win = ttk.Spinbox(row2, from_=1, to=100000, increment=1, textvariable=self.var_smooth_win, width=8)
        self.spin_smooth_win.grid(row=0, column=11, padx=(6, 12), sticky="w")

        # ---- FILLING ARTIFACT GAPS ----
        # After artifacts are removed, should we fill the "holes" with synthetic data?
        # YES = smoother display but with interpolated (fake) data in the gaps
        # NO = pure data only, but with visible gaps where artifacts were
        # Note: Frequency analysis ALWAYS uses the non-interpolated (pure) data regardless
        self.var_interp = tk.BooleanVar(value=DEFAULT_USE_LINEAR_INTERP)
        self.chk_interp = ttk.Checkbutton(
            row2,
            text="Linear interpolate holes (after artifact removal)",
            variable=self.var_interp,
            command=self.on_interp_toggled,
        )
        self.chk_interp.grid(row=0, column=12, padx=(0, 0), sticky="w")

        # ===== NORMALIZATION & DISPLAY CONTROLS ROW =====
        # WHAT: This row controls how the signal is normalized and how plots are displayed
        # WHY: Different normalization methods reveal different patterns in the data
        # Font controls let you make text readable on different displays
        row3 = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        row3.pack(side=tk.TOP, fill=tk.X)

        # ---- NORMALIZATION METHOD ----
        # Choose which equation to use for normalizing the signal:
        #   ΔF/F: Simple percent change from baseline (most common for fiber photometry)
        #   zF (global): Z-score normalized across the entire recording
        #   zF (interval): Z-score normalized to a specific time window you select
        ttk.Label(row3, text="Normalization view:").grid(row=0, column=0, sticky="w")
        self.var_norm = tk.StringVar(value=NORM_DFF)
        self.cmb_norm = ttk.Combobox(row3, values=NORM_CHOICES, textvariable=self.var_norm, state="readonly", width=24)
        self.cmb_norm.grid(row=0, column=1, padx=(6, 12), sticky="w")
        self.cmb_norm.bind("<<ComboboxSelected>>", lambda _e: self.on_norm_mode_changed())

        # ---- INTERVAL-BASED Z-SCORE WINDOW ----
        # Only used when selected normalization is "zF - interval based"
        # This window defines which part of the recording is used to compute the baseline stats
        self.lbl_interval = ttk.Label(row3, text="Interval (s):")
        self.var_interval_start = tk.StringVar(value=str(DEFAULT_ZF_INTERVAL_START_S))
        self.var_interval_end = tk.StringVar(value=str(DEFAULT_ZF_INTERVAL_END_S))
        self.spin_interval_start = ttk.Spinbox(row3, from_=-1e9, to=1e9, increment=1.0, textvariable=self.var_interval_start, width=10)
        self.lbl_interval_to = ttk.Label(row3, text="to")
        self.spin_interval_end = ttk.Spinbox(row3, from_=-1e9, to=1e9, increment=1.0, textvariable=self.var_interval_end, width=10)

        self.btn_apply_norm = ttk.Button(row3, text="Apply interval", command=self.apply_normalization)

        for w in [self.spin_interval_start, self.spin_interval_end]:
            w.bind("<Return>", lambda _e: self.apply_normalization())

        self.lbl_interval.grid(row=0, column=2, sticky="w")
        self.spin_interval_start.grid(row=0, column=3, padx=(6, 4), sticky="w")
        self.lbl_interval_to.grid(row=0, column=4, sticky="w")
        self.spin_interval_end.grid(row=0, column=5, padx=(4, 12), sticky="w")
        self.btn_apply_norm.grid(row=0, column=6, sticky="w")

        # ---- FONT SIZE CONTROLS (for readability) ----
        # Adjust text size in plots for different screen sizes and vision needs
        # Press ENTER in any field to apply and see changes immediately in the plot
        
        # Axis label font size (labels like "Time (s)" and "Signal (mV)")
        # Generally 8-12pt for readable labels; larger screens can use 12-14pt
        ttk.Label(row3, text="Axis label font:").grid(row=0, column=7, sticky="w", padx=(18, 0))
        self.var_axis_label_fs = tk.StringVar(value=f"{DEFAULT_AXIS_LABEL_FONTSIZE:g}")
        self.spin_axis_label_fs = ttk.Spinbox(
            row3,
            from_=6,
            to=60,
            increment=1,
            textvariable=self.var_axis_label_fs,
            width=6,
        )
        self.spin_axis_label_fs.grid(row=0, column=8, padx=(6, 12), sticky="w")
        self.spin_axis_label_fs.bind("<Return>", lambda _e: self.apply_axis_label_fontsize())

        # Graph title font size (axes titles + experiment name at top)
        # Generally 10-16pt; larger for emphasis on what you're looking at
        ttk.Label(row3, text="Graph title font:").grid(row=0, column=9, sticky="w", padx=(10, 0))
        self.var_graph_title_fs = tk.StringVar(value=f"{DEFAULT_GRAPH_TITLE_FONTSIZE:g}")
        self.spin_graph_title_fs = ttk.Spinbox(
            row3,
            from_=6,
            to=80,
            increment=1,
            textvariable=self.var_graph_title_fs,
            width=6,
        )
        self.spin_graph_title_fs.grid(row=0, column=10, padx=(6, 0), sticky="w")
        self.spin_graph_title_fs.bind("<Return>", lambda _e: self.apply_graph_title_fontsize())

        # Tick label font size (numbers on x/y axes like "0", "10", "100")
        # Generally 6-10pt; smaller numbers are fine for tick labels
        ttk.Label(row3, text="Tick label font:").grid(row=0, column=11, sticky="w", padx=(10, 0))
        self.var_tick_label_fs = tk.StringVar(value=f"{DEFAULT_TICK_LABEL_FONTSIZE:g}")
        self.spin_tick_label_fs = ttk.Spinbox(
            row3,
            from_=4,
            to=40,
            increment=1,
            textvariable=self.var_tick_label_fs,
            width=6,
        )
        self.spin_tick_label_fs.grid(row=0, column=12, padx=(6, 0), sticky="w")
        self.spin_tick_label_fs.bind("<Return>", lambda _e: self.apply_tick_label_fontsize())

        self.update_norm_controls_visibility()

        # ------------------ File chooser (placed directly above G-channel tabs) ------------------
        view_row = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        view_row.pack(side=tk.TOP, fill=tk.X)
        view_row.columnconfigure(2, weight=1)

        ttk.Label(view_row, text="File:").grid(row=0, column=0, sticky="w")
        self.var_view_file = tk.StringVar(value="")
        self.cmb_view_file = ttk.Combobox(
            view_row,
            textvariable=self.var_view_file,
            values=[],
            width=16,
            state="disabled",
        )
        self.cmb_view_file.grid(row=0, column=1, padx=(6, 12), sticky="w")
        self.cmb_view_file.bind("<<ComboboxSelected>>", self.on_view_file_changed)

        self.lbl_view_hint = ttk.Label(view_row, text="Run analysis to load the file selector.")
        self.lbl_view_hint.grid(row=0, column=2, sticky="w")

        # ------------------ Outer tabs ------------------
        self.outer_tabs = ttk.Notebook(self.root)
        self.outer_tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        # Lazy mouse-tab loading (performance):
        # In batch mode you can have dozens of mice. Creating a full set of matplotlib canvases
        # for every mouse up-front is very slow.
        #
        # Instead we create lightweight placeholder frames first and instantiate the heavy
        # ChannelTabsTk only when the user selects that mouse tab.
        self._mouse_frames: Dict[str, ttk.Frame] = {}
        self._frame_to_mid: Dict[str, str] = {}

        # When the user changes the OUTER tab, lazy-load the selected mouse tab if needed.
        self.outer_tabs.bind("<<NotebookTabChanged>>", self._on_outer_tab_changed)

        # ------------------ Status bar ------------------
        self.status = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize artifact control states
        self._sync_artifact_controls_state()

        # Ctrl+I time marker hotkeys
        self._install_time_marker_hotkeys()

        # Ctrl+J file-number mapping hotkey
        self._install_file_map_hotkeys()

        if self.csv_paths and autorun:
            self.root.after(0, self.run_analysis)

    def _install_file_map_hotkeys(self) -> None:
        """Ctrl+J: show which CSV corresponds to File 1/File 2/..."""
        self.root.bind_all("<Control-j>", self._on_ctrl_j, add="+")
        self.root.bind_all("<Control-J>", self._on_ctrl_j, add="+")

    def _on_ctrl_j(self, _event=None) -> None:
        # Helper for Ctrl+J; wrap call so errors don’t silently abort
        try:
            self._show_file_number_map_dialog()
        except Exception as e:
            # Print to console and update status bar for visibility
            try:
                print(f"_on_ctrl_j error: {e}", file=sys.stderr)
            except Exception:
                pass
            try:
                self.status.set(f"Error showing file map: {e}")
            except Exception:
                pass

    def _show_file_number_map_dialog(self) -> None:
        paths = list(self.csv_paths)[:8]
        if not paths:
            messagebox.showinfo(
                "File number mapping (Ctrl+J)",
                "No CSV files selected.\n\nUse 'Add CSV(s)…' first.",
            )
            return

        top = tk.Toplevel(self.root)
        top.title("File number mapping (Ctrl+J)")
        top.transient(self.root)
        top.grab_set()

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text="These are the names used in the file drop-down:",
            justify="left",
        ).pack(anchor="w")

        txt_frame = ttk.Frame(frm)
        txt_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        txt = tk.Text(txt_frame, width=90, height=min(18, 3 + len(paths) * 2), wrap="none")
        sb = ttk.Scrollbar(txt_frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)

        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        lines = []
        for i, p in enumerate(paths, start=1):
            display_name = f"File {i}"
            basename = os.path.basename(p)
            lines.append(f"{display_name}: {basename}\n    {p}\n")

        txt.insert("1.0", "\n".join(lines).rstrip() + "\n")
        txt.configure(state="disabled")

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X)

        def copy_to_clipboard():
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append("\n".join(lines).rstrip() + "\n")
            except Exception:
                pass

        ttk.Button(btn_row, text="Copy", command=copy_to_clipboard).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Close", command=top.destroy).pack(side=tk.RIGHT)

        self.root.wait_window(top)

    # ------------------ Batch file controls ------------------
    @staticmethod
    def _mouse_file_key(mouse_id: str) -> str:
        return str(mouse_id).split(":", 1)[0]

    @staticmethod
    def _mouse_channel_label(mouse_id: str) -> str:
        parts = str(mouse_id).split(":", 1)
        return parts[1] if len(parts) == 2 and parts[1] else str(mouse_id)

    def _set_result_file_choices(self, file_meta: List[Tuple[str, str, str]]) -> None:
        self._result_file_order = [file_key for file_key, _alias, _path in file_meta]
        self._file_alias_by_key = {file_key: alias for file_key, alias, _path in file_meta}
        self._file_path_by_key = {file_key: path for file_key, _alias, path in file_meta}

        values = [self._file_alias_by_key[file_key] for file_key in self._result_file_order]
        if not values:
            self.var_view_file.set("")
            self.cmb_view_file.config(values=(), state="disabled")
            self.lbl_view_hint.config(text="Run analysis to load the file selector.")
            return

        current = (self.var_view_file.get() or "").strip()
        if current not in values:
            current = values[0]

        self.cmb_view_file.config(values=tuple(values), state="readonly")
        self.var_view_file.set(current)
        if len(values) == 1:
            self.lbl_view_hint.config(text="Showing the selected file's G channels below.")
        else:
            self.lbl_view_hint.config(text="Pick a file to show its G channels below.")

    def _get_selected_view_file_key(self) -> Optional[str]:
        if not self._result_file_order:
            return None

        selected_alias = (self.var_view_file.get() or "").strip()
        for file_key in self._result_file_order:
            if self._file_alias_by_key.get(file_key) == selected_alias:
                return file_key

        fallback = self._result_file_order[0]
        self.var_view_file.set(self._file_alias_by_key.get(fallback, ""))
        return fallback

    def on_view_file_changed(self, _event=None) -> None:
        if not self._results:
            return

        self._refresh_visible_tabs()

        file_key = self._get_selected_view_file_key()
        alias = self._file_alias_by_key.get(file_key, "") if file_key else ""
        if alias:
            self.status.set(f"Viewing file: {alias}.")

    def _update_files_label(self):
        if not self.csv_paths:
            self.lbl_file.config(text="No CSV files selected.")
            return
        if len(self.csv_paths) == 1:
            self.lbl_file.config(text=self.csv_paths[0])
            return
        first = os.path.basename(self.csv_paths[0])
        self.lbl_file.config(text=f"{len(self.csv_paths)} CSV files selected (first: {first})")

    def add_csvs(self):
        """
        Open file browser to select one or more CSV files for analysis.
        
        WHAT IT DOES:
        - Opens a file picker dialog where you can choose CSV files
        - Adds selected files to the list shown in the "Files:" line
        - Prevents duplicate files (won't add same file twice)
        - Limits you to 8 files maximum (system limitation)
        
        WHY THIS MATTERS:
        - Your CSV file contains the raw ISO/EXC signal recordings
        - Each file represents one imaging session
        - You can analyze multiple sessions in one batch run
        
        USER WORKFLOW:
        1. Click "Add CSV(s)…" button
        2. Navigate to your data folder
        3. Select one or more .csv files (use Ctrl+Click for multiple)
        4. Files appear in the list at the top
        5. Click "Run analysis" to process them
        """
        paths = filedialog.askopenfilenames(title="Select CSV file(s)", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not paths:
            return
        for p in paths:
            if p and p not in self.csv_paths:
                self.csv_paths.append(p)
        if len(self.csv_paths) > 8:
            messagebox.showwarning(
                "Too many files",
                "You can analyze up to 8 CSV files at a time.\n\nOnly the first 8 will be kept.",
            )
            self.csv_paths = self.csv_paths[:8]
        self._update_files_label()
        self.status.set(f"Selected {len(self.csv_paths)} file(s).")

    def clear_csvs(self):
        self.csv_paths = []
        self._update_files_label()
        self.status.set("Cleared file list.")

    # ------------------ Helpers for batch tabs ------------------
    def get_mouse_items(self) -> List[Tuple[str, str]]:
        """Return ordered list of (mouse_id, display_label) for batch selection UIs."""
        items: List[Tuple[str, str]] = []
        for mid in list(self._mouse_order):
            items.append((mid, self._mouse_display.get(mid, mid)))
        return items

    def get_mouse_result(self, mouse_id: str) -> Optional[ChannelResult]:
        if not self._results:
            return None
        return self._results.get(mouse_id)

    def get_mouse_display(self, mouse_id: str) -> str:
        return self._mouse_display.get(mouse_id, mouse_id)

    def get_norm_mode(self) -> str:
        return self._read_norm_mode()

    def get_smooth_window(self) -> int:
        try:
            return max(1, int(float(self.var_smooth_win.get())))
        except Exception:
            return DEFAULT_SMOOTH_WINDOW

    # ------------------ Time marker hotkeys ------------------
    def _install_time_marker_hotkeys(self) -> None:
        """
        Install the keyboard shortcut for *time markers*.

        What the user sees / does:
        --------------------------
        - Press **Ctrl + I** anywhere in the app.
        - A small dialog opens where they can:
            * ADD a marker: type a time in seconds and press Enter / OK.
            * REMOVE the last marker: press Ctrl + Backspace in the dialog.

        Why this exists:
        ----------------
        When you compare/average mice, it is often useful to mark important events
        (e.g., "stim on", "cue", "injection") with vertical lines.

        Important details:
        ------------------
        - Markers are stored inside the active PlotTabTk (so they survive redraws).
        - The marker line is added to the legend so it can be renamed later
          (double-click the legend text).
        - Up to 4 markers are supported; each one gets a different line style.
        """
        # Bind both lowercase and uppercase “i” because OS/keyboards can report either.
        self.root.bind_all("<Control-i>", self._on_ctrl_i, add="+")
        self.root.bind_all("<Control-I>", self._on_ctrl_i, add="+")

    def _on_ctrl_i(self, _event=None) -> None:
        """
        Ctrl+I handler.

        We first figure out which plot is currently visible (the “active” plot),
        and then show the dialog that lets the user add or remove a marker.
        """
        tab = self.find_active_plot_tab()
        if tab is None:
            # If no plot is active (rare), do nothing.
            return

        self._show_time_marker_dialog(tab)

    def _show_time_marker_dialog(self, tab: "PlotTabTk") -> None:
        """
        Open a small *modal* dialog that asks for a time marker.

        "Modal" means the user must close this dialog before interacting with the
        rest of the GUI again (this avoids confusion).
        """
        # Create a new small window on top of the main GUI
        top = tk.Toplevel(self.root)
        top.title("Time marker (Ctrl+I)")
        top.transient(self.root)   # keep it on top of the main window
        top.grab_set()             # make it modal

        # A string variable that holds what the user types
        time_var = tk.StringVar(value=getattr(self, "_last_marker_time_str", ""))

        # Instruction text (written for non-programmers)
        info = (
            "Add a vertical time marker to the *currently active* plot.\n\n"
            "• To ADD: type a time in seconds and press Enter / OK.\n"
            "• To REMOVE the last marker: press Ctrl + Backspace.\n\n"
            "Tip: marker lines appear in the legend, so you can rename them by double-clicking."
        )

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text=info, justify="left").pack(anchor="w")

        ttk.Label(frm, text="Time (seconds):").pack(anchor="w", pady=(10, 0))
        entry = ttk.Entry(frm, textvariable=time_var, width=28)
        entry.pack(anchor="w", pady=(2, 8))
        entry.focus_set()

        # Buttons row (OK / Cancel). We do NOT add a "Remove" button because
        # the requested removal shortcut is Ctrl+Backspace.
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, pady=(4, 0))

        def close_dialog() -> None:
            """Close the dialog safely."""
            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()

        def do_add(_e=None) -> str:
            """
            Try to add a marker.

            Returns "break" so Tkinter does not also beep or do default key actions.
            """
            s = (time_var.get() or "").strip()
            if not s:
                messagebox.showerror("Missing time", "Please type a time in seconds (for example: 12.5).")
                return "break"

            try:
                t = float(s)
            except Exception:
                messagebox.showerror("Invalid time", f"Could not read '{s}' as a number of seconds.")
                return "break"

            if not np.isfinite(t):
                messagebox.showerror("Invalid time", "Time must be a finite number.")
                return "break"

            # Remember what the user typed so next time the dialog can pre-fill it
            self._last_marker_time_str = s

            # Ask the active plot to add the marker
            tab.add_time_marker(t)

            close_dialog()
            return "break"

        def do_remove(_e=None) -> str:
            """
            Remove the most recently added marker line.

            This is triggered by Ctrl+Backspace.
            """
            tab.remove_last_time_marker()
            close_dialog()
            return "break"

        def do_cancel(_e=None) -> str:
            """User cancelled."""
            close_dialog()
            return "break"

        ttk.Button(btn_row, text="OK (add)", command=do_add).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Cancel", command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8))

        # Keyboard shortcuts inside the dialog:
        #   Enter      -> add marker
        #   Esc        -> cancel
        #   Ctrl+Backspace -> remove last marker
        top.bind("<Return>", do_add)
        top.bind("<Escape>", do_cancel)

        # Bind to both the window and the entry widget to be robust
        top.bind("<Control-BackSpace>", do_remove)
        entry.bind("<Control-BackSpace>", do_remove)

        # Wait here until the dialog is closed (because it is modal)
        self.root.wait_window(top)

# ------------------ Existing GUI helpers ------------------
    def _sync_artifact_controls_state(self):
        enabled = bool(self.var_artifact_enabled.get())
        self.spin_factor.config(state=("normal" if enabled else "disabled"))
        self.spin_pad.config(state=("normal" if enabled else "disabled"))
        if enabled:
            self.chk_shared.state(["!disabled"])
        else:
            self.chk_shared.state(["disabled"])

    def _read_norm_mode(self) -> str:
        mode = (self.var_norm.get() or NORM_DFF).strip()
        return mode if mode in NORM_CHOICES else NORM_DFF

    def update_norm_controls_visibility(self):
        """
        Show/hide the interval time window controls based on selected normalization view.
        
        WHAT IT DOES:
        - When you select "zF - interval based" normalization, extra fields appear
          to let you specify the time window for z-score calculation
        - When you select other normalizations, those fields disappear (they're not needed)
        
        WHY THIS MATTERS:
        - For ΔF/F and global z-score: the app automatically calculates the baseline
        - For interval-based z-score: YOU need to tell the app which time period to use
          as the "normal" baseline (e.g., first 5 seconds before stimulus)
        - This method ensures only the relevant controls are visible, reducing clutter
        """
        mode = self._read_norm_mode()
        show_interval = mode == NORM_ZF_INTERVAL

        if show_interval:
            self.lbl_interval.grid()
            self.spin_interval_start.grid()
            self.lbl_interval_to.grid()
            self.spin_interval_end.grid()
            self.btn_apply_norm.grid()
        else:
            self.lbl_interval.grid_remove()
            self.spin_interval_start.grid_remove()
            self.lbl_interval_to.grid_remove()
            self.spin_interval_end.grid_remove()
            self.btn_apply_norm.grid_remove()

    def on_norm_mode_changed(self):
        self.update_norm_controls_visibility()
        self.apply_normalization()

    # ------------------ Font size apply (Enter only) ------------------
    def apply_axis_label_fontsize(self):
        """
        Apply universal axis label font size (x/y labels) to all plots.
        Triggered ONLY by pressing Enter in the spinbox.
        """
        try:
            fs = float(self.var_axis_label_fs.get())
        except Exception as e:
            messagebox.showerror("Invalid font size", f"Could not parse axis label font size:\n\n{e}")
            return
        if not np.isfinite(fs) or fs <= 0:
            messagebox.showerror("Invalid font size", "Axis label font size must be a positive number.")
            return

        self._axis_label_fs_override = fs

        for widget in self._channel_widgets.values():
            widget.set_axis_label_fontsize(fs)

        if self.compare_widget is not None:
            self.compare_widget.set_axis_label_fontsize(fs)
        if self.average_widget is not None:
            self.average_widget.set_axis_label_fontsize(fs)

        self.status.set(f"Applied axis label font size = {fs:g} (x/y labels).")

    def apply_graph_title_fontsize(self):
        """
        Apply universal graph title font size (axes titles + figure suptitle) to all plots.
        Triggered ONLY by pressing Enter in the spinbox.
        """
        try:
            fs = float(self.var_graph_title_fs.get())
        except Exception as e:
            messagebox.showerror("Invalid font size", f"Could not parse graph title font size:\n\n{e}")
            return
        if not np.isfinite(fs) or fs <= 0:
            messagebox.showerror("Invalid font size", "Graph title font size must be a positive number.")
            return

        self._graph_title_fs_override = fs

        for widget in self._channel_widgets.values():
            widget.set_graph_title_fontsize(fs)

        if self.compare_widget is not None:
            self.compare_widget.set_graph_title_fontsize(fs)
        if self.average_widget is not None:
            self.average_widget.set_graph_title_fontsize(fs)

        self.status.set(f"Applied graph title font size = {fs:g} (plot titles).")

    def apply_tick_label_fontsize(self):
        """
        Apply universal tick label font size (the numbers on x/y axes).
        Triggered ONLY by pressing Enter in the spinbox.
        """
        try:
            fs = float(self.var_tick_label_fs.get())
        except Exception as e:
            messagebox.showerror("Invalid font size", f"Could not parse tick label font size:\n\n{e}")
            return
        if not np.isfinite(fs) or fs <= 0:
            messagebox.showerror("Invalid font size", "Tick label font size must be a positive number.")
            return

        self._tick_label_fs_override = fs

        for widget in self._channel_widgets.values():
            widget.set_tick_label_fontsize(fs)

        if self.compare_widget is not None:
            self.compare_widget.set_tick_label_fontsize(fs)
        if self.average_widget is not None:
            self.average_widget.set_tick_label_fontsize(fs)

        self.status.set(f"Applied tick label font size = {fs:g} (axis numbers).")

    # ------------------ Artifact enable toggle (NO CSV reload) ------------------
    def on_artifact_enabled_toggled(self):
        enabled = bool(self.var_artifact_enabled.get())
        self._sync_artifact_controls_state()

        # If we already have results loaded, recompute pipeline in-place (no CSV reload)
        if not self._results:
            self.status.set(f"Artifact remover {'ENABLED' if enabled else 'DISABLED'} (will apply on next Run).")
            return

        try:
            artifact_factor = float(self.var_factor.get())
            artifact_pad = int(float(self.var_pad.get()))
            require_shared = bool(self.var_shared.get())
            align_mode = DEFAULT_ALIGN_MODE
            use_interp = bool(self.var_interp.get())
        except Exception as e:
            messagebox.showerror("Invalid settings", f"Could not parse artifact settings:\n\n{e}")
            return

        self.status.set("Updating artifact pipeline…")

        def worker():
            try:
                for res in self._results.values():
                    recompute_artifact_pipeline_inplace(
                        res,
                        artifact_enabled=enabled,
                        artifact_factor=artifact_factor,
                        artifact_pad=artifact_pad,
                        require_shared=require_shared,
                        align_mode=align_mode,
                        use_linear_interp=use_interp,
                    )
                self.root.after(0, self._after_artifact_pipeline_updated)
            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_failed(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _after_artifact_pipeline_updated(self):
        for widget in self._channel_widgets.values():
            widget.refresh_after_pipeline_change()

        if self.compare_widget is not None:
            self.compare_widget.refresh_plot()
        if self.average_widget is not None:
            self.average_widget.refresh_plot()

        self.status.set(
            f"Artifact remover {'ENABLED' if self.var_artifact_enabled.get() else 'DISABLED'} (MAD). "
            f"Interp={'ON' if self.var_interp.get() else 'OFF'}."
        )

    # ------------------ Interpolation toggle ------------------
    def on_interp_toggled(self):
        """
        Toggle linear interpolation without re-running artifact detection.
        Frequency analysis still uses NO interpolation (holes pipeline).
        """
        if not self._results:
            return
        use_interp = bool(self.var_interp.get())
        for res in self._results.values():
            set_interpolation_mode(res, use_interp)

        for widget in self._channel_widgets.values():
            widget.refresh_after_pipeline_change()

        if self.compare_widget is not None:
            self.compare_widget.refresh_plot()
        if self.average_widget is not None:
            self.average_widget.refresh_plot()

        self.status.set(f"Interpolation {'ON' if use_interp else 'OFF'} (frequency analysis still uses NO interpolation).")

    # ------------------ Apply normalization ------------------
    def apply_normalization(self):
        """
        Apply the selected normalization view to all analysis results and refresh plots.
        
        WHAT IT DOES:
        - Reads your normalization choice (ΔF/F, zF global, or zF interval)
        - If you chose interval-based z-score, validates the time window you specified
        - Updates all plot windows with the new normalization view
        - Shows a status message confirming what was applied
        
        WHY THIS MATTERS:
        - Normalization converts raw fluorescence values to a more interpretable form
        - ΔF/F: Shows % change from baseline (most common for fiber photometry)
        - zF (global): Standardizes signal relative to entire recording
        - zF (interval): Standardizes relative to a baseline period you define
        - Different normalizations reveal different patterns in your data
        
        WHEN IT'S CALLED:
        - Automatically when you select a new normalization from the dropdown
        - When you press "Apply interval" after changing the time window
        - After analysis finishes (to use your selected normalization)
        """
        mode = self._read_norm_mode()

        try:
            smooth_win = max(1, int(float(self.var_smooth_win.get())))
        except Exception:
            smooth_win = DEFAULT_SMOOTH_WINDOW

        interval_start: Optional[float] = None
        interval_end: Optional[float] = None
        if mode == NORM_ZF_INTERVAL:
            try:
                interval_start = float(self.var_interval_start.get())
                interval_end = float(self.var_interval_end.get())
            except Exception as e:
                messagebox.showerror("Invalid interval", f"Could not parse interval start/end:\n\n{e}")
                return

        if not self._results:
            if mode == NORM_DFF:
                self.status.set("Normalization view: ΔF/F")
            elif mode == NORM_ZF_GLOBAL:
                self.status.set("Normalization view: zF (global, GUI)")
            elif mode == NORM_ZF_INTERVAL:
                self.status.set("Normalization view: zF - interval based (set interval and apply)")
            return

        for res in self._results.values():
            res.smooth_window = smooth_win
            if mode == NORM_ZF_INTERVAL and interval_start is not None and interval_end is not None:
                res.zf_interval_start_s = interval_start
                res.zf_interval_end_s = interval_end
            recompute_normalizations(res)

        for widget in self._channel_widgets.values():
            widget.set_norm_mode(mode)

        if self.compare_widget is not None:
            self.compare_widget.refresh_plot()
        if self.average_widget is not None:
            self.average_widget.refresh_plot()

        if mode == NORM_DFF:
            self.status.set("Normalization view: ΔF/F.")
        elif mode == NORM_ZF_GLOBAL:
            self.status.set("Normalization view: zF (global, GUI).")
        elif mode == NORM_ZF_INTERVAL:
            a = float(interval_start)
            b = float(interval_end)
            self.status.set(f"Normalization view: zF - interval based. Interval = {min(a, b):g}–{max(a, b):g} s")

    # ===== ANALYSIS EXECUTION =====
    def run_analysis(self):
        """
        Process all selected CSV files with the current settings and display results.
        
        THIS IS THE MAIN ANALYSIS WORKFLOW:
        ====================================
        1. Validate all settings (artifact params, normalization type, smoothing, etc.)
        2. For each CSV file: call analyze_csv() to process the data
        3. Collect results and organize them by filename and G-column
        4. Create tabs for interactive browsing of each result
        5. Display 7 analysis views: Raw, Slope Normality, Artifacts, Fit, Normalization,
           Normalized+Smoothed, and Frequency Analysis
        
        WHAT HAPPENS INTERNALLY:
        - Reads all parameters from the GUI controls (artifact factor, smoothing, etc.)
        - Runs analysis in a background THREAD so the GUI stays responsive
        - Updates status bar to show progress: "Analyzing 1/3: mouse-001..." etc.
        - When complete, builds the interactive tabbed interface for exploring results
        
        WHAT THE USER SEES:
        - "Run analysis" button becomes disabled (gray) while processing
        - Status bar shows "Analyzing..."
        - Once finished, status shows details: number of mice, number of channels, etc.
        - All result tabs appear in the main window for interactive exploration
        
        ERROR HANDLING:
        - If any file fails, shows error dialog with details
        - Button re-enabled so you can fix and try again
        """
        if not self.csv_paths:
            messagebox.showwarning("No files", "Please add one or more CSV files first.")
            return

        try:
            artifact_enabled = bool(self.var_artifact_enabled.get())
            artifact_factor = float(self.var_factor.get())
            artifact_pad = int(float(self.var_pad.get()))
            require_shared = bool(self.var_shared.get())
            align_mode = DEFAULT_ALIGN_MODE

            acq_fps = float(self.var_acq_fps.get())
            smooth_win = max(1, int(float(self.var_smooth_win.get())))

            mode = self._read_norm_mode()

            interval_start = float(self.var_interval_start.get())
            interval_end = float(self.var_interval_end.get())

            use_interp = bool(self.var_interp.get())

        except Exception as e:
            messagebox.showerror("Invalid settings", f"Could not parse settings:\n\n{e}")
            return

        fit_windows = list(DEFAULT_FIT_WINDOWS)

        paths = list(self.csv_paths)[:8]
        file_labels = [f"File {i}" for i in range(1, len(paths) + 1)]
        file_meta = [(f"F{i}", label, path) for i, (path, label) in enumerate(zip(paths, file_labels), start=1)]

        self.btn_run.config(state=tk.DISABLED)
        self.btn_add.config(state=tk.DISABLED)
        self.btn_clear.config(state=tk.DISABLED)
        self.status.set("Analyzing…")

        def worker():
            try:
                aggregated: Dict[str, ChannelResult] = {}
                display: Dict[str, str] = {}
                order: List[str] = []

                for i, (file_key, alias, path) in enumerate(file_meta, start=1):
                    # Update status in UI thread
                    self.root.after(0, lambda i=i, alias=alias: self.status.set(f"Analyzing {i}/{len(paths)}: {alias}…"))

                    per_file = analyze_csv(
                        path,
                        artifact_enabled=artifact_enabled,
                        artifact_factor=artifact_factor,
                        artifact_method="mad",
                        artifact_pad=artifact_pad,
                        require_shared=require_shared,
                        align_mode=align_mode,
                        fit_windows=fit_windows,
                        acq_fps_hz=acq_fps if (np.isfinite(acq_fps) and acq_fps > 0) else None,
                        smooth_window=smooth_win,
                        zf_interval_start_s=interval_start,
                        zf_interval_end_s=interval_end,
                        use_linear_interp=use_interp,
                    )

                    for gcol in sorted(per_file.keys()):
                        mid = f"{file_key}:{gcol}"
                        aggregated[mid] = per_file[gcol]
                        display[mid] = f"{alias}:{gcol}"
                        order.append(mid)

                self.root.after(0, lambda: self.on_analysis_finished(aggregated, display, order, mode, file_meta))
            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_failed(str(e)))

        self._analysis_thread = threading.Thread(target=worker, daemon=True)
        self._analysis_thread.start()

    def on_analysis_finished(
        self,
        results: Dict[str, ChannelResult],
        display: Dict[str, str],
        order: List[str],
        mode: str,
        file_meta: List[Tuple[str, str, str]],
    ):
        """
        Called when background analysis thread completes successfully.
        
        WHAT IT DOES:
        - Saves analysis results to internal storage
        - Rebuilds all tabs to display the results
        - Updates status bar with summary statistics
        - Re-enables buttons so you can run another analysis or modify settings
        
        SHOWS IN STATUS BAR:
        - Number of files analyzed
        - Number of mice (unique G-columns) found
        - Normalization mode that's active
        - Smoothing window size used
        - Whether artifact removal was on/off
        - Whether interpolation was used
        - Sampling rate calculated from your FPS setting
        """
        self._results = results
        self._mouse_display = dict(display)
        self._mouse_order = list(order)
        self._set_result_file_choices(file_meta)

        self.build_tabs(norm_mode=mode)

        try:
            any_key = next(iter(results.keys()))
            r0 = results[any_key]
            self.status.set(
                f"Done. Files={len(file_meta)} | Mice={len(results)} | Mode={mode}. Smooth win={r0.smooth_window}. "
                f"Artifacts={'ON' if self.var_artifact_enabled.get() else 'OFF'}. "
                f"Interp={'ON' if r0.use_interpolation else 'OFF'}. "
                f"Freq eff fs={r0.eff_fs_hz:.3g} Hz (Acq FPS={r0.acq_fps_hz:.3g})."
            )
        except Exception:
            self.status.set("Done.")

        self.btn_run.config(state=tk.NORMAL)
        self.btn_add.config(state=tk.NORMAL)
        self.btn_clear.config(state=tk.NORMAL)

        # Ensure UI reflects current norm mode and interval visibility
        self.on_norm_mode_changed()

    def on_analysis_failed(self, msg: str):
        """
        Called when background analysis thread encounters an error.
        
        WHAT IT DOES:
        - Displays error dialog with details about what went wrong
        - Re-enables buttons so you can try again
        
        COMMON ERRORS:
        - File not found (path no longer valid)
        - CSV format incorrect (missing ISO/EXC columns, wrong headers)
        - Math error (can't divide by zero, NaN values, etc.)
        - Invalid settings (negative numbers where not allowed, etc.)
        """
        messagebox.showerror("Analysis failed", msg)
        self.status.set("Analysis failed.")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_add.config(state=tk.NORMAL)
        self.btn_clear.config(state=tk.NORMAL)

    
    def build_tabs(self, norm_mode: str):
        """
        Build the interactive tabbed interface for browsing analysis results.

        The notebook still uses lazy loading for speed, but now the user first
        chooses a single file from the "File" drop-down. The notebook then
        shows only that file's channel tabs.
        """
        outer_tabs = self.outer_tabs

        # Remove existing tabs from the notebook
        for tab_id in outer_tabs.tabs():
            outer_tabs.forget(tab_id)

        # Destroy widgets from any previous analysis run so stale results do not linger.
        for frame in list(getattr(self, "_mouse_frames", {}).values()):
            try:
                frame.destroy()
            except Exception:
                pass
        if self.compare_widget is not None:
            try:
                self.compare_widget.destroy()
            except Exception:
                pass
        if self.average_widget is not None:
            try:
                self.average_widget.destroy()
            except Exception:
                pass

        # Clear widget references
        self._channel_widgets.clear()

        # Lazy-loading maps:
        #   mid -> lightweight ttk.Frame used as the notebook tab content
        #   frame_widget_name -> mid (reverse lookup when user clicks a tab)
        self._mouse_frames = {}
        self._frame_to_mid = {}
        self.compare_widget = None
        self.average_widget = None

        if not self._results:
            return

        # Create a placeholder frame for each mouse id, but only show the
        # currently selected file in the notebook.
        mids = [mid for mid in self._mouse_order if mid in self._results]
        for mid in mids:
            frame = ttk.Frame(outer_tabs)
            display_label = self._mouse_display.get(mid, mid)

            # A tiny placeholder label so the tab isn't blank before first click
            ttk.Label(
                frame,
                text=f"Click this tab to load plots for {display_label} (lazy-loaded for speed).",
                foreground="#666666",
            ).pack(anchor="w", padx=10, pady=10)

            self._mouse_frames[mid] = frame

        self._refresh_visible_tabs()

    def _refresh_visible_tabs(self) -> None:
        outer_tabs = self.outer_tabs

        for tab_id in outer_tabs.tabs():
            outer_tabs.forget(tab_id)

        self._frame_to_mid = {}

        file_key = self._get_selected_view_file_key()
        mids: List[str] = []
        for mid in self._mouse_order:
            if mid not in self._results:
                continue
            if file_key is not None and self._mouse_file_key(mid) != file_key:
                continue
            mids.append(mid)

        for mid in mids:
            frame = self._mouse_frames.get(mid)
            if frame is None:
                continue

            tab_label = self._mouse_channel_label(mid) if file_key is not None else self._mouse_display.get(mid, mid)
            outer_tabs.add(frame, text=tab_label)
            self._frame_to_mid[str(frame)] = mid

        if mids:
            outer_tabs.select(self._mouse_frames[mids[0]])
            self._ensure_mouse_widget(mids[0])

    def _on_outer_tab_changed(self, _event=None) -> None:
        """
        Triggered when user clicks on a different tab in the main tab bar.
        
        KEY FEATURE - Lazy Loading Implementation:
        ==========================================
        This is where the performance optimization happens:
        1. User clicks on "F1:G0" tab
        2. This method is called by the operating system
        3. If that tab hasn't been drawn yet, we call _ensure_mouse_widget()
        4. All 7 plots appear (takes ~1 second on first click)
        5. Performance stays fast: no analysis, just drawing existing results
        
        WHY LAZY LOADING?
        - Batch mode: 8 files × 4 columns = 32 tabs × 7 plots = 224 graphs
        - Drawing all 224 plots immediately would freeze GUI for 30+ seconds
        - Lazy loading: Only draw plots when user views them (~1 second per tab)
        - Subsequent clicks on same tab are instant (plots cached in memory)
        
        INTERNAL LOGIC:
        - Per-mouse tabs are still placeholder frames until first click
        """
        sel = self.outer_tabs.select()
        if not sel:
            return

        mid = self._frame_to_mid.get(sel)
        if mid:
            self._ensure_mouse_widget(mid)

    def _ensure_mouse_widget(self, mid: str) -> None:
        """
        Create the ChannelTabsTk plotting widget if it hasn't been created yet.
        
        WHAT IT DOES:
        - Checks if we've already drawn plots for this mouse: if yes, do nothing
        - If no, instantiate ChannelTabsTk (creates all 7 matplotlib figures)
        - Remove placeholder label
        - Pack widget into its frame so it's visible
        - Apply any font-size overrides the user set earlier
        
        CALL CHAIN:
        _on_outer_tab_changed() → _ensure_mouse_widget() → ChannelTabsTk()
                                                                    ↓
                                                    Creates all 7 matplotlib plots
        
        PERFORMANCE:
        - This is the "expensive" step where matplotlib draws graphs to memory
        - Takes ~1 second for a typical results object with 1500+ data points
        - After this, switching tabs is instant (plots already in memory)
        """
        if mid in self._channel_widgets:
            return

        frame = self._mouse_frames.get(mid)
        if frame is None:
            return

        # Remove placeholder children (labels, etc.)
        for child in frame.winfo_children():
            child.destroy()

        # Build the per-mouse plotting notebook
        res = self._results[mid]
        widget = ChannelTabsTk(frame, res, norm_mode=self._read_norm_mode(), parent_app=self)
        widget.pack(fill=tk.BOTH, expand=True)

        # Apply any "global" font-size overrides that the user set earlier.
        if self._axis_label_fs_override is not None:
            widget.set_axis_label_fontsize(self._axis_label_fs_override)
        if self._graph_title_fs_override is not None:
            widget.set_graph_title_fontsize(self._graph_title_fs_override)
        if self._tick_label_fs_override is not None:
            widget.set_tick_label_fontsize(self._tick_label_fs_override)

        self._channel_widgets[mid] = widget

    def find_active_plot_tab(self) -> Optional[PlotTabTk]:
        """
        Return the PlotTabTk that is currently visible, so global hotkeys (Ctrl+I)
        know which plot to modify.

        Returns:
            PlotTabTk or None
        """
        sel = self.outer_tabs.select()
        if not sel:
            return None

        # Mouse tab
        mid = self._frame_to_mid.get(sel)
        if not mid:
            return None

        # Ensure the mouse widget exists (lazy-load)
        self._ensure_mouse_widget(mid)

        cw = self._channel_widgets.get(mid)
        if cw is None:
            return None

        return cw.get_active_plot_tab()



# ---------------------- CLI + Entry point ----------------------
# The GUI is intended to be run as a normal Python script:
#   python fiberlyse_gui_batch_markers_v3_optimized.py
#
# Optional: you can pass one or more CSV files on the command line:
#   python fiberlyse_gui_batch_markers_v3_optimized.py --csv file1.csv file2.csv


def parse_cli():
    parser = argparse.ArgumentParser(description="Fiberlyse GUI (batch + markers)")

    # Accept 0..N CSV paths. If none are provided, the user can browse in the GUI.
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[],
        help="Optional CSV file(s) to load automatically (space-separated).",
    )
    return parser.parse_args()


def main():
    args = parse_cli()

    app = MainAppTk()

    # If CSV files were provided on the command line, pre-load them and auto-run analysis.
    if getattr(args, "csv", None):
        app.csv_paths = list(args.csv)
        try:
            app._update_files_label()
        except Exception:
            pass

        # Start analysis (runs in a background thread; GUI stays responsive)
        try:
            app.run_analysis()
        except Exception:
            # If something goes wrong, still start the GUI so the user can try manually.
            pass

    app.root.mainloop()


if __name__ == "__main__":
    main()


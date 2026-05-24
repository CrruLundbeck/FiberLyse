from __future__ import annotations;import sys;import os;import re;import argparse;import threading;import zipfile;from xml.sax.saxutils import escape as _xml_escape;from dataclasses import dataclass;from typing import Dict, List, Tuple, Optional, Callable, Any;import numpy as np;import pandas as pd;from statistics import NormalDist
try:import tkinter as tk;from tkinter import ttk, filedialog, messagebox, simpledialog
except Exception as e:raise SystemExit(f'Tkinter is not available in this Python environment.\n\nOn Windows, Tkinter usually comes with the official python.org installer.\nIf your company Python build excludes Tkinter, you’ll need IT to install a Python distribution that includes Tk/Tcl.\n\nOriginal error: {e}')
import matplotlib;matplotlib.use('TkAgg');from matplotlib.figure import Figure;from matplotlib.widgets import SpanSelector;from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk;from matplotlib import colors as mcolors;from matplotlib.font_manager import FontProperties
try:from scipy import stats;_HAVE_SCIPY_STATS = True
except Exception:stats = None;_HAVE_SCIPY_STATS = False
try:from scipy.signal import butter, sosfiltfilt, filtfilt;_HAVE_SCIPY_SIGNAL = True
except Exception:butter = None;sosfiltfilt = None;filtfilt = None;_HAVE_SCIPY_SIGNAL = False
ISO_STATE = 1;EXC_STATE = 2;DEFAULT_FIT_WINDOWS = [(0.0, 6500.0)];DEFAULT_ALIGN_MODE = 'nearest';DEFAULT_SLOPE_TEST_MAX_N = 5000;DEFAULT_ACQ_FPS_HZ = 40.0;DEFAULT_BUTTER_ORDER = 2;DEFAULT_SMOOTH_WINDOW = 20;DEFAULT_ZF_INTERVAL_START_S = 0.0;DEFAULT_ZF_INTERVAL_END_S = 6500.0;DEFAULT_USE_LINEAR_INTERP = True
def _fontsize_points(v: Any, fallback: float) -> float:
    try:return float(FontProperties(size=v).get_size_in_points())
    except Exception:
        try:return float(v)
        except Exception:return float(fallback)
DEFAULT_AXIS_LABEL_FONTSIZE = _fontsize_points(matplotlib.rcParams.get('axes.labelsize', 10), fallback=10.0);DEFAULT_GRAPH_TITLE_FONTSIZE = _fontsize_points(matplotlib.rcParams.get('axes.titlesize', 12), fallback=12.0);DEFAULT_TICK_LABEL_FONTSIZE = _fontsize_points(matplotlib.rcParams.get('xtick.labelsize', matplotlib.rcParams.get('ytick.labelsize', 8)), fallback=8.0);NORMALIZATION_OPTION_DF_OVER_F = 'ΔF/F';NORMALIZATION_OPTION_ZF_GLOBAL = 'zF (global, GUI)';NORMALIZATION_OPTION_ZF_INTERVAL = 'zF - interval based';NORMALIZATION_ALL_OPTIONS = [NORMALIZATION_OPTION_DF_OVER_F, NORMALIZATION_OPTION_ZF_GLOBAL, NORMALIZATION_OPTION_ZF_INTERVAL];NORM_DFF = NORMALIZATION_OPTION_DF_OVER_F;NORM_ZF_GLOBAL = NORMALIZATION_OPTION_ZF_GLOBAL;NORM_ZF_INTERVAL = NORMALIZATION_OPTION_ZF_INTERVAL;NORM_CHOICES = NORMALIZATION_ALL_OPTIONS;FREQ_BANDS: List[Tuple[float, float]] = [(5.0, 10.0), (2.5, 5.0), (1.25, 2.5), (0.6, 1.25), (0.3, 0.6), (0.15, 0.3)]
def find_g_columns(columns) -> List[str]:
    g_cols = [c for c in columns if re.match('^G\\d*$', str(c)) or str(c).startswith('G')];ordered = [c for c in columns if c in g_cols]
    for drop in ['FrameCounter', 'LedState', 'SystemTimestamp', 'ComputerTimestamp']:
        if drop in ordered:ordered.remove(drop)
    return ordered
def detect_artifacts_by_derivative(t: np.ndarray, y: np.ndarray, factor: float, method: str='mad', pad: int=1) -> np.ndarray:
    t = np.asarray(t, dtype=float);y = np.asarray(y, dtype=float);valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 3:return np.zeros_like(y, dtype=bool)
    tv = t[valid];yv = y[valid];dt = np.diff(tv);dy = np.diff(yv);dt_safe = np.where(dt == 0, np.nan, dt);slopes = dy / dt_safe;slopes_valid = slopes[np.isfinite(slopes)]
    if slopes_valid.size < 3:return np.zeros_like(y, dtype=bool)
    method = (method or 'mad').strip().lower()
    if method != 'mad':raise ValueError("Only 'mad' artifact method is supported (sd removed).")
    center = float(np.median(slopes_valid));mad = float(np.median(np.abs(slopes_valid - center)));scale = float(1.4826 * mad)
    if not np.isfinite(scale) or scale <= 0:return np.zeros_like(y, dtype=bool)
    bad_seg = np.abs(slopes - center) > factor * scale;art_valid = np.zeros_like(yv, dtype=bool);bad_idx = np.where(bad_seg)[0]
    for k in bad_idx:i0 = max(0, k - pad);i1 = min(len(art_valid) - 1, k + 1 + pad);art_valid[i0:i1 + 1] = True
    art = np.zeros_like(y, dtype=bool);art[np.where(valid)[0]] = art_valid;return art
def compute_slopes(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float);y = np.asarray(y, dtype=float);valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 3:return np.array([], dtype=float)
    tv = t[valid];yv = y[valid];dt = np.diff(tv);dy = np.diff(yv);dt = np.where(dt == 0, np.nan, dt)
    with np.errstate(divide='ignore', invalid='ignore'):slopes = dy / dt
    slopes = slopes[np.isfinite(slopes)];return slopes
def subsample_even(x: np.ndarray, max_n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if max_n is None or max_n <= 0 or x.size <= max_n:return x
    idx = np.linspace(0, x.size - 1, max_n, dtype=int);return x[idx]
def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):return 'n/a'
    if p < 0.0001:return f'{p:.1e}'
    return f'{p:.4f}'
def normality_summary(x: np.ndarray, alpha: float=0.05, max_n: int=DEFAULT_SLOPE_TEST_MAX_N) -> dict:
    x = np.asarray(x, dtype=float);x = x[np.isfinite(x)];out = {'n': int(x.size), 'n_test': int(min(x.size, max_n if max_n is not None and max_n > 0 else x.size)), 'mean': np.nan, 'std': np.nan, 'median': np.nan, 'mad': np.nan, 'shapiro_p': np.nan, 'dagostino_p': np.nan, 'anderson_stat': np.nan, 'anderson_cv5': np.nan, 'anderson_ok': None, 'verdict': 'n/a', 'note': ''}
    if x.size < 3:out['note'] = 'Need ≥3 finite slopes';return out
    x_test = subsample_even(x, max_n=max_n);out['mean'] = float(np.mean(x_test));out['std'] = float(np.std(x_test, ddof=1)) if x_test.size >= 2 else np.nan;out['median'] = float(np.median(x_test));out['mad'] = float(np.median(np.abs(x_test - out['median'])))
    if not _HAVE_SCIPY_STATS:out['note'] = 'SciPy not installed: skipping formal tests';out['verdict'] = 'n/a';return out
    try:
        if x_test.size >= 3:_, p = stats.shapiro(x_test);out['shapiro_p'] = float(p)
    except Exception as e:out['note'] += f'Shapiro failed: {e}. '
    try:
        if x_test.size >= 8:_, p = stats.normaltest(x_test);out['dagostino_p'] = float(p)
    except Exception as e:out['note'] += f'K² failed: {e}. '
    try:
        ad = stats.anderson(x_test, dist='norm');out['anderson_stat'] = float(ad.statistic);cv5 = np.nan
        for sl, cv in zip(ad.significance_level, ad.critical_values):
            if float(sl) == 5.0:cv5 = float(cv);break
        if not np.isfinite(cv5) and len(ad.critical_values) > 0:i = int(np.argmin(np.abs(np.asarray(ad.significance_level, dtype=float) - 5.0)));cv5 = float(ad.critical_values[i])
        out['anderson_cv5'] = cv5;out['anderson_ok'] = bool(np.isfinite(cv5) and ad.statistic < cv5)
    except Exception as e:out['note'] += f'AD failed: {e}. '
    rejects = []
    if np.isfinite(out['shapiro_p']):rejects.append(out['shapiro_p'] <= alpha)
    if np.isfinite(out['dagostino_p']):rejects.append(out['dagostino_p'] <= alpha)
    if out['anderson_ok'] is not None:rejects.append(not out['anderson_ok'])
    out['verdict'] = 'n/a' if not rejects else 'REJECT' if any(rejects) else 'OK';return out
def qq_plot_points(x: np.ndarray, max_n: int=DEFAULT_SLOPE_TEST_MAX_N) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    x = np.asarray(x, dtype=float);x = x[np.isfinite(x)]
    if x.size < 3:return (np.array([], dtype=float), np.array([], dtype=float), np.nan, np.nan, np.nan)
    xq = subsample_even(x, max_n=max_n)
    if _HAVE_SCIPY_STATS:(theor, ordered), (slope, intercept, r) = stats.probplot(xq, dist='norm');return (np.asarray(theor), np.asarray(ordered), float(slope), float(intercept), float(r))
    ordered = np.sort(xq);n = ordered.size;p = (np.arange(1, n + 1) - 0.5) / n;theor = np.array([NormalDist().inv_cdf(float(pi)) for pi in p], dtype=float);slope, intercept = np.polyfit(theor, ordered, 1);r = float(np.corrcoef(theor, ordered)[0, 1]) if n >= 2 else np.nan;return (theor, ordered, float(slope), float(intercept), r)
def shared_artifacts_by_time(t_iso: np.ndarray, art_iso: np.ndarray, t_exc: np.ndarray, art_exc: np.ndarray, tol: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    t_iso = np.asarray(t_iso, dtype=float);t_exc = np.asarray(t_exc, dtype=float);art_iso = np.asarray(art_iso, dtype=bool);art_exc = np.asarray(art_exc, dtype=bool)
    if not art_iso.any() or not art_exc.any():return (np.zeros_like(art_iso, dtype=bool), np.zeros_like(art_exc, dtype=bool))
    if tol is None:dt_iso = np.median(np.diff(t_iso)) if t_iso.size > 1 else np.inf;dt_exc = np.median(np.diff(t_exc)) if t_exc.size > 1 else np.inf;base_dt = min(dt_iso, dt_exc);tol = 0.5 * base_dt if np.isfinite(base_dt) else 0.0
    iso_idx = np.where(art_iso)[0];exc_idx = np.where(art_exc)[0];iso_times = t_iso[iso_idx];exc_times = t_exc[exc_idx];iso_shared = np.zeros_like(art_iso, dtype=bool);exc_shared = np.zeros_like(art_exc, dtype=bool)
    for ei, te in zip(exc_idx, exc_times):
        if np.any(np.abs(iso_times - te) <= tol):exc_shared[ei] = True
    for ii, ti in zip(iso_idx, iso_times):
        if np.any(np.abs(exc_times - ti) <= tol):iso_shared[ii] = True
    return (iso_shared, exc_shared)
def remove_with_holes(y: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:y = np.asarray(y, dtype=float).copy();y[np.asarray(artifact_mask, dtype=bool)] = np.nan;return y
def linear_interpolate_by_time(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float);y = np.asarray(y, dtype=float);out = y.copy();valid = np.isfinite(t) & np.isfinite(out)
    if valid.sum() < 2:return out
    out = np.interp(t, t[valid], out[valid]).astype(float);return out
def align_iso_to_exc_no_interp(t_iso: np.ndarray, y_iso: np.ndarray, t_exc: np.ndarray, mode: str='nearest') -> np.ndarray:
    t_iso = np.asarray(t_iso, dtype=float);y_iso = np.asarray(y_iso, dtype=float);t_exc = np.asarray(t_exc, dtype=float)
    if t_iso.size == 0:return np.full_like(t_exc, np.nan, dtype=float)
    idx = np.searchsorted(t_iso, t_exc, side='left');mode = (mode or 'nearest').strip().lower()
    if mode == 'prev':j = idx - 1
    elif mode == 'next':j = idx
    elif mode == 'nearest':j_prev = np.clip(idx - 1, 0, len(t_iso) - 1);j_next = np.clip(idx, 0, len(t_iso) - 1);d_prev = np.abs(t_exc - t_iso[j_prev]);d_next = np.abs(t_exc - t_iso[j_next]);j = np.where(d_next < d_prev, j_next, j_prev)
    else:raise ValueError("align mode must be 'prev', 'next', or 'nearest'")
    j = np.clip(j, 0, len(t_iso) - 1);return y_iso[j]
def fit_linear(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:X = np.vstack([x, np.ones_like(x)]).T;(a, b), *_ = np.linalg.lstsq(X, y, rcond=None);return (float(a), float(b))
def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:ss_res = np.nansum((y - yhat) ** 2);ss_tot = np.nansum((y - np.nanmean(y)) ** 2);return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
def _fill_nans_linear_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:return x.copy()
    out = x.copy();finite = np.isfinite(out)
    if not np.any(finite):return out
    idx = np.where(finite)[0]
    if idx.size == 1:out[~finite] = out[idx[0]];return out
    xp = idx.astype(float);fp = out[idx].astype(float);x_all = np.arange(out.size, dtype=float);out = np.interp(x_all, xp, fp).astype(float);return out
def _moving_average_centered_fast(x: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=float);n = x.size
    if n == 0:return x.copy()
    w = int(max(1, window_size))
    if w == 1:return x.copy()
    left = (w - 1) // 2;right = w // 2;idx = np.arange(n, dtype=int);start = idx - left;end = idx + right;start = np.clip(start, 0, n - 1);end = np.clip(end, 0, n - 1);prefix = np.concatenate([[0.0], np.cumsum(x, dtype=float)]);sums = prefix[end + 1] - prefix[start];counts = (end - start + 1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):y = sums / counts
    return np.asarray(y, dtype=float)
def smooth_like_batch(x: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=float);n = x.size
    if n == 0:return x.copy()
    try:w = int(window_size)
    except Exception:w = DEFAULT_SMOOTH_WINDOW
    w = max(1, w)
    if w == 1:return x.copy()
    nan_mask = ~np.isfinite(x);x_filled = _fill_nans_linear_1d(x);padlen = 3 * (w - 1)
    if _HAVE_SCIPY_SIGNAL and filtfilt is not None and (n > padlen):
        b = np.ones(w, dtype=float) / float(w);a = 1.0
        try:y = filtfilt(b, a, x_filled)
        except Exception:y = _moving_average_centered_fast(x_filled, w)
    else:y = _moving_average_centered_fast(x_filled, w)
    y = np.asarray(y, dtype=float);y[nan_mask] = np.nan;return y
def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:return []
    idx = np.where(mask)[0]
    if idx.size == 0:return []
    runs: List[Tuple[int, int]] = [];s = idx[0];p = idx[0]
    for k in idx[1:]:
        if k == p + 1:p = k
        else:runs.append((s, p));s = p = k
    runs.append((s, p));return runs
def bandpass_butterworth_segmentwise_no_interp(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int=DEFAULT_BUTTER_ORDER) -> np.ndarray:
    x = np.asarray(x, dtype=float);out = np.full_like(x, np.nan, dtype=float)
    if not _HAVE_SCIPY_SIGNAL or butter is None or sosfiltfilt is None:return out
    if not np.isfinite(fs) or fs <= 0:return out
    nyq = fs / 2.0;high_hz = min(float(high_hz), nyq - 1e-06)
    if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:return out
    low = float(low_hz) / nyq;high = float(high_hz) / nyq
    try:sos = butter(int(order), [low, high], btype='band', output='sos')
    except Exception:return out
    finite = np.isfinite(x)
    for i0, i1 in _contiguous_true_runs(finite):
        seg = x[i0:i1 + 1]
        if seg.size < 8:continue
        try:yseg = sosfiltfilt(sos, seg)
        except Exception:
            try:yseg = sosfiltfilt(sos, seg, padlen=0)
            except Exception:continue
        out[i0:i1 + 1] = yseg
    return out
def bandpass_fft_no_interp(x: np.ndarray, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(fs) or fs <= 0:return np.full_like(x, np.nan, dtype=float)
    if not np.all(np.isfinite(x)):return np.full_like(x, np.nan, dtype=float)
    nyq = fs / 2.0;high_hz = min(float(high_hz), nyq - 1e-06)
    if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:return np.full_like(x, np.nan, dtype=float)
    n = x.size
    if n < 4:return np.full_like(x, np.nan, dtype=float)
    X = np.fft.rfft(x);freqs = np.fft.rfftfreq(n, d=1.0 / fs);mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz));Xf = np.where(mask, X, 0.0);y = np.fft.irfft(Xf, n=n);return np.asarray(y, dtype=float)
def estimate_fs_from_t(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float);dt = np.diff(t);dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:return np.nan
    med = float(np.median(dt))
    if med <= 0 or not np.isfinite(med):return np.nan
    return float(1.0 / med)
def _nanstd_safe(x: np.ndarray, ddof: int) -> float:
    x = np.asarray(x, dtype=float);x = x[np.isfinite(x)]
    if x.size < 2:return np.nan
    try:return float(np.std(x, ddof=int(ddof)))
    except Exception:return np.nan
def zscore_global_gui(dff: np.ndarray, ddof: int=0) -> np.ndarray:
    dff = np.asarray(dff, dtype=float);mu = float(np.nanmean(dff));sigma = _nanstd_safe(dff, ddof=ddof)
    if np.isfinite(sigma) and sigma > 1e-12:return (dff - mu) / sigma
    return np.full_like(dff, np.nan, dtype=float)
def zscore_interval_based(dff: np.ndarray, t: np.ndarray, start_s: float, end_s: float, ddof: int=1) -> np.ndarray:
    dff = np.asarray(dff, dtype=float);t = np.asarray(t, dtype=float);a = float(min(start_s, end_s));b = float(max(start_s, end_s));mask = (t >= a) & (t <= b) & np.isfinite(dff)
    if mask.sum() < 2:mask = np.isfinite(dff)
    mu = float(np.nanmean(dff[mask])) if mask.any() else np.nan;sigma = _nanstd_safe(dff[mask], ddof=ddof) if mask.any() else np.nan
    if np.isfinite(sigma) and sigma > 1e-12:return (dff - mu) / sigma
    return np.full_like(dff, np.nan, dtype=float)
def get_norm_array(res: 'ChannelResult', norm_mode: str) -> np.ndarray:
    if norm_mode == NORM_DFF:return np.asarray(res.dFF, dtype=float)
    if norm_mode == NORM_ZF_GLOBAL:return np.asarray(res.zF_global, dtype=float)
    if norm_mode == NORM_ZF_INTERVAL:return np.asarray(res.zF_interval, dtype=float)
    return np.asarray(res.dFF, dtype=float)
def get_smoothed_norm_array(res: 'ChannelResult', norm_mode: str, window_size: int) -> np.ndarray:
    try:w = int(window_size)
    except Exception:w = DEFAULT_SMOOTH_WINDOW
    w = max(1, w);cache = getattr(res, '_smooth_cache', None)
    if cache is None or not isinstance(cache, dict):cache = {};setattr(res, '_smooth_cache', cache)
    key = (str(norm_mode), int(w))
    if key in cache:return cache[key]
    y = get_norm_array(res, norm_mode);y_s = smooth_like_batch(y, window_size=w)
    if len(cache) > 12:cache.clear()
    cache[key] = y_s;return y_s
@dataclass
class ChannelResult:gcol: str;t_iso: np.ndarray;t_exc: np.ndarray;iso_raw: np.ndarray;exc_raw: np.ndarray;art_iso: np.ndarray;art_exc: np.ndarray;iso_clean_holes: np.ndarray;exc_clean_holes: np.ndarray;iso_clean_interp: np.ndarray;exc_clean_interp: np.ndarray;iso_on_exc_holes: np.ndarray;iso_on_exc_interp: np.ndarray;use_interpolation: bool;iso_clean: np.ndarray;exc_clean: np.ndarray;iso_on_exc: np.ndarray;windows: List[Tuple[float, float]];acq_fps_hz: float;eff_fs_hz: float;slope: float;intercept: float;r2: float;fitted_iso_on_exc: np.ndarray;residual: np.ndarray;dF: np.ndarray;dFF: np.ndarray;slope_nointerp: float;intercept_nointerp: float;r2_nointerp: float;fitted_iso_on_exc_nointerp: np.ndarray;residual_nointerp: np.ndarray;dF_nointerp: np.ndarray;dFF_nointerp: np.ndarray;smooth_window: int;zf_interval_start_s: float;zf_interval_end_s: float;zF_global: np.ndarray;zF_interval: np.ndarray
def recompute_normalizations(res: ChannelResult) -> None:
    res.zF_global = zscore_global_gui(res.dFF, ddof=0);res.zF_interval = zscore_interval_based(res.dFF, res.t_exc, start_s=res.zf_interval_start_s, end_s=res.zf_interval_end_s, ddof=1);cache = getattr(res, '_smooth_cache', None)
    if isinstance(cache, dict):cache.clear()
    else:setattr(res, '_smooth_cache', {})
def _compute_fit_and_downstream(t_exc: np.ndarray, exc: np.ndarray, iso_on_exc: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_exc = np.asarray(t_exc, dtype=float);exc = np.asarray(exc, dtype=float);iso_on_exc = np.asarray(iso_on_exc, dtype=float);fit_mask = np.zeros_like(t_exc, dtype=bool)
    for a, b in windows:fit_mask |= (t_exc >= float(min(a, b))) & (t_exc <= float(max(a, b)))
    finite = np.isfinite(exc) & np.isfinite(iso_on_exc);fit_idx = fit_mask & finite
    if fit_idx.sum() < 2:fit_idx = finite
    if fit_idx.sum() >= 2:a_fit, b_fit = fit_linear(exc[fit_idx], iso_on_exc[fit_idx]);fitted = a_fit * iso_on_exc + b_fit;r2 = r2_score(exc[fit_idx], fitted[fit_idx])
    else:a_fit, b_fit, r2 = (np.nan, np.nan, np.nan);fitted = np.full_like(iso_on_exc, np.nan, dtype=float)
    residual = exc - fitted;dF = residual
    with np.errstate(divide='ignore', invalid='ignore'):dFF = np.divide(dF, fitted, out=np.full_like(dF, np.nan), where=np.isfinite(fitted) & (np.abs(fitted) > 1e-12))
    return (float(a_fit), float(b_fit), float(r2), fitted, residual, dF, dFF)
def recompute_fit_and_downstream(res: ChannelResult) -> None:
    a, b, r2, fitted, resid, dF, dFF = _compute_fit_and_downstream(res.t_exc, res.exc_clean, res.iso_on_exc, res.windows);res.slope = a;res.intercept = b;res.r2 = r2;res.fitted_iso_on_exc = fitted;res.residual = resid;res.dF = dF;res.dFF = dFF;a2, b2, r22, fitted2, resid2, dF2, dFF2 = _compute_fit_and_downstream(res.t_exc, res.exc_clean_holes, res.iso_on_exc_holes, res.windows);res.slope_nointerp = a2;res.intercept_nointerp = b2;res.r2_nointerp = r22;res.fitted_iso_on_exc_nointerp = fitted2;res.residual_nointerp = resid2;res.dF_nointerp = dF2;res.dFF_nointerp = dFF2;recompute_normalizations(res);res._data_version = int(getattr(res, '_data_version', 0)) + 1;fcache = getattr(res, '_freq_cache', None)
    if isinstance(fcache, dict):fcache.clear()
    else:setattr(res, '_freq_cache', {})
def set_interpolation_mode(res: ChannelResult, use_interpolation: bool) -> None:
    res.use_interpolation = bool(use_interpolation)
    if res.use_interpolation:res.iso_clean = res.iso_clean_interp;res.exc_clean = res.exc_clean_interp;res.iso_on_exc = res.iso_on_exc_interp
    else:res.iso_clean = res.iso_clean_holes;res.exc_clean = res.exc_clean_holes;res.iso_on_exc = res.iso_on_exc_holes
    recompute_fit_and_downstream(res)
def recompute_artifact_pipeline_inplace(res: ChannelResult, artifact_enabled: bool, artifact_factor: float, artifact_pad: int, require_shared: bool, align_mode: str, use_linear_interp: bool) -> None:
    if artifact_enabled:
        art_iso_raw = detect_artifacts_by_derivative(res.t_iso, res.iso_raw, factor=float(artifact_factor), method='mad', pad=int(artifact_pad));art_exc_raw = detect_artifacts_by_derivative(res.t_exc, res.exc_raw, factor=float(artifact_factor), method='mad', pad=int(artifact_pad))
        if require_shared:art_iso, art_exc = shared_artifacts_by_time(res.t_iso, art_iso_raw, res.t_exc, art_exc_raw)
        else:art_iso, art_exc = (art_iso_raw, art_exc_raw)
    else:art_iso = np.zeros_like(res.iso_raw, dtype=bool);art_exc = np.zeros_like(res.exc_raw, dtype=bool)
    res.art_iso = np.asarray(art_iso, dtype=bool);res.art_exc = np.asarray(art_exc, dtype=bool);res.iso_clean_holes = remove_with_holes(res.iso_raw, res.art_iso);res.exc_clean_holes = remove_with_holes(res.exc_raw, res.art_exc);res.iso_clean_interp = linear_interpolate_by_time(res.t_iso, res.iso_clean_holes);res.exc_clean_interp = linear_interpolate_by_time(res.t_exc, res.exc_clean_holes);res.iso_on_exc_holes = align_iso_to_exc_no_interp(res.t_iso, res.iso_clean_holes, res.t_exc, mode=align_mode);res.iso_on_exc_interp = align_iso_to_exc_no_interp(res.t_iso, res.iso_clean_interp, res.t_exc, mode=align_mode);res.iso_on_exc_holes = np.asarray(res.iso_on_exc_holes, dtype=float);res.iso_on_exc_interp = np.asarray(res.iso_on_exc_interp, dtype=float);res.iso_on_exc_holes[res.art_exc] = np.nan;set_interpolation_mode(res, use_interpolation=bool(use_linear_interp))
def analyze_csv(csv_path: str, artifact_enabled: bool, artifact_factor: float, artifact_method: str, artifact_pad: int, require_shared: bool, align_mode: str, fit_windows: List[Tuple[float, float]], acq_fps_hz: Optional[float], smooth_window: int, zf_interval_start_s: float, zf_interval_end_s: float, use_linear_interp: bool) -> Dict[str, ChannelResult]:
    header = pd.read_csv(csv_path, nrows=0)
    for col in ['SystemTimestamp', 'LedState']:
        if col not in header.columns:raise ValueError(f"Missing required column '{col}'.")
    g_cols = find_g_columns(header.columns)
    if not g_cols:raise ValueError('No G* columns found (e.g., G0, G1...).')
    usecols = ['SystemTimestamp', 'LedState'] + list(g_cols);df = pd.read_csv(csv_path, usecols=usecols)
    if (df['LedState'] == 7).any():start_idx = df.index[df['LedState'] == 7][0];t0 = float(df.loc[start_idx, 'SystemTimestamp'])
    else:t0 = float(df['SystemTimestamp'].iloc[0])
    t_full = df['SystemTimestamp'].astype(float).to_numpy() - t0;led = df['LedState'].to_numpy();mask_iso = led == ISO_STATE;mask_exc = led == EXC_STATE;t_iso = t_full[mask_iso];t_exc = t_full[mask_exc]
    if t_iso.size == 0 or t_exc.size == 0:raise ValueError('No iso/exc samples found. Check LedState coding.')
    acq_fps_val = float(acq_fps_hz) if acq_fps_hz is not None and np.isfinite(acq_fps_hz) else 0.0
    if acq_fps_val > 0:eff_fs = acq_fps_val / 2.0
    else:eff_fs = estimate_fs_from_t(t_exc);acq_fps_val = eff_fs * 2.0 if np.isfinite(eff_fs) else 0.0
    results: Dict[str, ChannelResult] = {}
    for gcol in g_cols:
        iso_raw = df.loc[mask_iso, gcol].to_numpy(dtype=float);exc_raw = df.loc[mask_exc, gcol].to_numpy(dtype=float)
        if iso_raw.size == 0 or exc_raw.size == 0:continue
        if artifact_enabled:
            art_iso_raw = detect_artifacts_by_derivative(t_iso, iso_raw, factor=artifact_factor, method='mad', pad=artifact_pad);art_exc_raw = detect_artifacts_by_derivative(t_exc, exc_raw, factor=artifact_factor, method='mad', pad=artifact_pad)
            if require_shared:art_iso, art_exc = shared_artifacts_by_time(t_iso, art_iso_raw, t_exc, art_exc_raw)
            else:art_iso, art_exc = (art_iso_raw, art_exc_raw)
        else:art_iso = np.zeros_like(iso_raw, dtype=bool);art_exc = np.zeros_like(exc_raw, dtype=bool)
        iso_clean_holes = remove_with_holes(iso_raw, art_iso);exc_clean_holes = remove_with_holes(exc_raw, art_exc);iso_clean_interp = linear_interpolate_by_time(t_iso, iso_clean_holes);exc_clean_interp = linear_interpolate_by_time(t_exc, exc_clean_holes);iso_on_exc_holes = align_iso_to_exc_no_interp(t_iso, iso_clean_holes, t_exc, mode=align_mode);iso_on_exc_interp = align_iso_to_exc_no_interp(t_iso, iso_clean_interp, t_exc, mode=align_mode);iso_on_exc_holes = np.asarray(iso_on_exc_holes, dtype=float);iso_on_exc_interp = np.asarray(iso_on_exc_interp, dtype=float);iso_on_exc_holes[art_exc] = np.nan
        if use_linear_interp:iso_clean = iso_clean_interp;exc_clean = exc_clean_interp;iso_on_exc = iso_on_exc_interp
        else:iso_clean = iso_clean_holes;exc_clean = exc_clean_holes;iso_on_exc = iso_on_exc_holes
        res = ChannelResult(gcol=gcol, t_iso=t_iso, t_exc=t_exc, iso_raw=iso_raw, exc_raw=exc_raw, art_iso=np.asarray(art_iso, dtype=bool), art_exc=np.asarray(art_exc, dtype=bool), iso_clean_holes=iso_clean_holes, exc_clean_holes=exc_clean_holes, iso_clean_interp=iso_clean_interp, exc_clean_interp=exc_clean_interp, iso_on_exc_holes=iso_on_exc_holes, iso_on_exc_interp=iso_on_exc_interp, use_interpolation=bool(use_linear_interp), iso_clean=iso_clean, exc_clean=exc_clean, iso_on_exc=iso_on_exc, windows=list(fit_windows), acq_fps_hz=acq_fps_val, eff_fs_hz=float(eff_fs), slope=np.nan, intercept=np.nan, r2=np.nan, fitted_iso_on_exc=np.full_like(t_exc, np.nan, dtype=float), residual=np.full_like(t_exc, np.nan, dtype=float), dF=np.full_like(t_exc, np.nan, dtype=float), dFF=np.full_like(t_exc, np.nan, dtype=float), slope_nointerp=np.nan, intercept_nointerp=np.nan, r2_nointerp=np.nan, fitted_iso_on_exc_nointerp=np.full_like(t_exc, np.nan, dtype=float), residual_nointerp=np.full_like(t_exc, np.nan, dtype=float), dF_nointerp=np.full_like(t_exc, np.nan, dtype=float), dFF_nointerp=np.full_like(t_exc, np.nan, dtype=float), smooth_window=max(1, int(smooth_window)), zf_interval_start_s=float(zf_interval_start_s), zf_interval_end_s=float(zf_interval_end_s), zF_global=np.full_like(t_exc, np.nan, dtype=float), zF_interval=np.full_like(t_exc, np.nan, dtype=float));recompute_fit_and_downstream(res);results[gcol] = res
    if not results:raise ValueError('No usable G* channels found to plot.')
    return results
class PlotTabTk(ttk.Frame):
    def __init__(self, master, tab_name: str, default_filename_prefix: str='', figsize: Tuple[float, float]=(7.2, 4.6), dpi: int=110):super().__init__(master);self.tab_name = tab_name;self.default_filename_prefix = default_filename_prefix;self.export_provider: Optional[Callable[[], Dict[str, pd.DataFrame]]] = None;self.fig = Figure(figsize=figsize, dpi=dpi);self.ax = self.fig.add_subplot(111);self.canvas = FigureCanvasTkAgg(self.fig, master=self);self.canvas_widget = self.canvas.get_tk_widget();self.toolbar = NavigationToolbar2Tk(self.canvas, self);self.toolbar.update();btn_row = ttk.Frame(self);btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6);self.save_btn = ttk.Button(btn_row, text='Save this graph…', command=self.save_plot);self.save_btn.pack(side=tk.RIGHT);self.export_btn = ttk.Button(btn_row, text='Export data (Excel)…', command=self.export_excel);self.export_btn.pack(side=tk.RIGHT, padx=(0, 8));self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True);self.axis_label_fontsize: Optional[float] = None;self.graph_title_fontsize: Optional[float] = None;self.tick_label_fontsize: Optional[float] = None;self._suptitle_override: Optional[str] = None;self._title_overrides: Dict[int, str] = {};self._xlabel_overrides: Dict[int, str] = {};self._ylabel_overrides: Dict[int, str] = {};self._legend_title_overrides: Dict[int, str] = {};self._legend_label_overrides: Dict[int, Dict[str, str]] = {};self._color_overrides: Dict[int, Dict[str, str]] = {};self._artist_color_overrides: Dict[int, str] = {};self._time_markers: List[Dict[str, Any]] = [];self._time_marker_artists: List[Any] = [];self._cid_button_press = self.canvas.mpl_connect('button_press_event', self._on_mpl_button_press)
    def _ax_key(self, ax) -> int:
        try:return int(self.fig.axes.index(ax))
        except Exception:return id(ax)
    def save_plot(self):
        suggested = f'{self.default_filename_prefix}_{self.tab_name}'.strip('_');path = filedialog.asksaveasfilename(title='Save graph', defaultextension='.png', initialfile=f'{suggested}.png', filetypes=[('PNG', '*.png'), ('SVG', '*.svg'), ('PDF', '*.pdf'), ('All files', '*.*')])
        if not path:return
        self.fig.savefig(path, bbox_inches='tight')
    @staticmethod
    def _safe_sheet_name(name: str, used: set) -> str:
        name = re.sub('[:\\\\/?*\\[\\]]+', '_', str(name)).strip()
        if not name:name = 'Sheet'
        name = name[:31];base = name;k = 2
        while name in used:suffix = f'_{k}';name = (base[:max(1, 31 - len(suffix))] + suffix)[:31];k += 1
        used.add(name);return name
    def _payload_from_artists_fallback(self) -> Dict[str, pd.DataFrame]:
        payload: Dict[str, pd.DataFrame] = {};used: set = set()
        for ai, ax in enumerate(self.fig.axes, start=1):
            for li, line in enumerate(ax.get_lines(), start=1):
                x = np.asarray(line.get_xdata(), dtype=float);y = np.asarray(line.get_ydata(), dtype=float);label = line.get_label()
                if not label or label.startswith('_'):label = f'line{li}'
                sheet = self._safe_sheet_name(f'ax{ai}_{label}', used);payload[sheet] = pd.DataFrame({'x': x, 'y': y})
            for ci, coll in enumerate(getattr(ax, 'collections', []), start=1):
                try:offs = coll.get_offsets()
                except Exception:continue
                if offs is None or len(offs) == 0:continue
                offs = np.asarray(offs, dtype=float)
                if offs.ndim != 2 or offs.shape[1] < 2:continue
                sheet = self._safe_sheet_name(f'ax{ai}_scatter{ci}', used);payload[sheet] = pd.DataFrame({'x': offs[:, 0], 'y': offs[:, 1]})
        if not payload:payload[self._safe_sheet_name('empty', used)] = pd.DataFrame({'note': ['No plottable artists found.']})
        return payload
    @staticmethod
    def _excel_col_name(n_1based: int) -> str:
        n = int(n_1based);s = ''
        while n > 0:n, r = divmod(n - 1, 26);s = chr(65 + r) + s
        return s
    @staticmethod
    def _is_nan(v: Any) -> bool:
        try:return v is None or (isinstance(v, float) and (not np.isfinite(v))) or (isinstance(v, np.floating) and (not np.isfinite(v)))
        except Exception:return v is None
    def _write_worksheet_xml(self, zf: zipfile.ZipFile, sheet_path: str, df: pd.DataFrame) -> None:
        cols = list(df.columns);ncols = len(cols)
        def write_cell(fh, row_idx_1based: int, col_idx_1based: int, value: Any, force_str: bool=False):
            if self._is_nan(value):return
            col_letter = self._excel_col_name(col_idx_1based);cell_ref = f'{col_letter}{row_idx_1based}'
            if isinstance(value, (bool, np.bool_)) and (not force_str):v = '1' if bool(value) else '0';fh.write(f'<c r="{cell_ref}" t="b"><v>{v}</v></c>'.encode('utf-8'));return
            if not force_str and isinstance(value, (int, np.integer)):fh.write(f'<c r="{cell_ref}"><v>{int(value)}</v></c>'.encode('utf-8'));return
            if not force_str and isinstance(value, (float, np.floating)):
                if np.isfinite(value):fh.write(f'<c r="{cell_ref}"><v>{float(value)}</v></c>'.encode('utf-8'))
                return
            s = str(value)
            if len(s) > 32767:s = s[:32767]
            s = _xml_escape(s);fh.write(f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{s}</t></is></c>'.encode('utf-8'))
        with zf.open(sheet_path, 'w') as fh:
            fh.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>');fh.write(b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">');fh.write(b'<sheetData>');fh.write(b'<row r="1">')
            for j, c in enumerate(cols, start=1):write_cell(fh, 1, j, c, force_str=True)
            fh.write(b'</row>')
            for i, row in enumerate(df.itertuples(index=False, name=None), start=2):
                fh.write(f'<row r="{i}">'.encode('utf-8'))
                for j in range(ncols):write_cell(fh, i, j + 1, row[j], force_str=False)
                fh.write(b'</row>')
            fh.write(b'</sheetData></worksheet>')
    def _write_xlsx_minimal(self, path: str, payload: Dict[str, pd.DataFrame]) -> None:
        used: set = set();sheets: List[Tuple[str, pd.DataFrame]] = []
        for sheet_name, df in payload.items():
            safe = self._safe_sheet_name(sheet_name, used)
            if not isinstance(df, pd.DataFrame):df = pd.DataFrame(df)
            sheets.append((safe, df))
        ct_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">', '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', '<Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>', '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>', '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>', '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>']
        for i in range(1, len(sheets) + 1):ct_lines.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        ct_lines.append('</Types>');content_types_xml = '\n'.join(ct_lines);rels_xml = '\n'.join(['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">', '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>', '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>', '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>', '</Relationships>']);core_xml = '\n'.join(['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">', '<dc:title>Fiberlyse Export</dc:title>', '<dc:creator>Fiberlyse</dc:creator>', '<cp:lastModifiedBy>Fiberlyse</cp:lastModifiedBy>', '</cp:coreProperties>']);app_xml = '\n'.join(['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">', '<Application>Fiberlyse</Application>', '</Properties>']);wb_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">', '<sheets>']
        for i, (sheet_name, _df) in enumerate(sheets, start=1):name_esc = _xml_escape(sheet_name);wb_lines.append(f'<sheet name="{name_esc}" sheetId="{i}" r:id="rId{i}"/>')
        wb_lines += ['</sheets>', '</workbook>'];workbook_xml = '\n'.join(wb_lines);wb_rels_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        for i in range(1, len(sheets) + 1):wb_rels_lines.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        wb_rels_lines.append(f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>');wb_rels_lines.append('</Relationships>');workbook_rels_xml = '\n'.join(wb_rels_lines);styles_xml = '\n'.join(['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">', '<fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>', '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>', '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>', '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>', '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>', '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>', '</styleSheet>'])
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('[Content_Types].xml', content_types_xml.encode('utf-8'));zf.writestr('_rels/.rels', rels_xml.encode('utf-8'));zf.writestr('docProps/core.xml', core_xml.encode('utf-8'));zf.writestr('docProps/app.xml', app_xml.encode('utf-8'));zf.writestr('xl/workbook.xml', workbook_xml.encode('utf-8'));zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels_xml.encode('utf-8'));zf.writestr('xl/styles.xml', styles_xml.encode('utf-8'))
            for i, (_sheet_name, df) in enumerate(sheets, start=1):self._write_worksheet_xml(zf, f'xl/worksheets/sheet{i}.xml', df)
    def export_excel(self):
        suggested = f'{self.default_filename_prefix}_{self.tab_name}'.strip('_');path = filedialog.asksaveasfilename(title='Export data (Excel)', defaultextension='.xlsx', initialfile=f'{suggested}.xlsx', filetypes=[('Excel workbook', '*.xlsx'), ('All files', '*.*')])
        if not path:return
        try:
            if callable(self.export_provider):payload = self.export_provider()
            else:payload = self._payload_from_artists_fallback()
            if not isinstance(payload, dict) or not payload:raise ValueError('Export provider returned no data.')
            self._write_xlsx_minimal(path, payload);messagebox.showinfo('Export complete', f'Saved Excel file:\n{path}')
        except Exception as e:messagebox.showerror('Export failed', f'Could not export Excel data:\n\n{e}')
    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        if fontsize is None:self.axis_label_fontsize = None;self.redraw();return
        try:fs = float(fontsize)
        except Exception:return
        if not np.isfinite(fs) or fs <= 0:return
        self.axis_label_fontsize = fs;self.redraw()
    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        if fontsize is None:self.graph_title_fontsize = None;self.redraw();return
        try:fs = float(fontsize)
        except Exception:return
        if not np.isfinite(fs) or fs <= 0:return
        self.graph_title_fontsize = fs;self.redraw()
    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        if fontsize is None:self.tick_label_fontsize = None;self.redraw();return
        try:fs = float(fontsize)
        except Exception:return
        if not np.isfinite(fs) or fs <= 0:return
        self.tick_label_fontsize = fs;self.redraw()
    def _toolbar_is_active(self) -> bool:
        try:return bool(getattr(self.toolbar, 'mode', ''))
        except Exception:return False
    @staticmethod
    def _normalize_hex_color(s: str) -> str:
        s = (s or '').strip()
        if not s:raise ValueError('Empty color')
        if not s.startswith('#'):s = '#' + s
        if re.fullmatch('#[0-9a-fA-F]{3}', s):r, g, b = (s[1], s[2], s[3]);return f'#{r}{r}{g}{g}{b}{b}'
        if re.fullmatch('#[0-9a-fA-F]{4}', s):r, g, b, a = (s[1], s[2], s[3], s[4]);return f'#{r}{r}{g}{g}{b}{b}{a}{a}'
        if re.fullmatch('#[0-9a-fA-F]{6}', s) or re.fullmatch('#[0-9a-fA-F]{8}', s):return s
        raise ValueError('Expected hex like #RRGGBB (or #RRGGBBAA)')
    @staticmethod
    def _artist_contains(artist, event) -> bool:
        try:hit, _ = artist.contains(event);return bool(hit)
        except Exception:return False
    @staticmethod
    def _resolve_label(label: str, mapping: Dict[str, str], max_hops: int=12) -> str:
        cur = str(label)
        for _ in range(max_hops):
            nxt = mapping.get(cur)
            if not nxt or nxt == cur:return cur
            cur = nxt
        return cur
    @staticmethod
    def _iter_axes_color_targets(ax):
        for art in list(getattr(ax, 'lines', [])):yield art
        for art in list(getattr(ax, 'collections', [])):yield art
        for art in list(getattr(ax, 'patches', [])):
            if art is getattr(ax, 'patch', None):continue
            yield art
    @staticmethod
    def _set_artist_color(artist, hex_color: str) -> None:
        if hasattr(artist, 'set_color'):
            try:artist.set_color(hex_color);return
            except Exception:pass
        if hasattr(artist, 'set_facecolor'):
            try:artist.set_facecolor(hex_color)
            except Exception:pass
        if hasattr(artist, 'set_edgecolor'):
            try:artist.set_edgecolor(hex_color)
            except Exception:pass
    @staticmethod
    def _artist_current_hex(artist) -> Optional[str]:
        try:
            if hasattr(artist, 'get_color'):return mcolors.to_hex(artist.get_color(), keep_alpha=False)
        except Exception:pass
        try:
            if hasattr(artist, 'get_facecolor'):
                fc = artist.get_facecolor()
                if fc is not None and len(fc):return mcolors.to_hex(fc[0], keep_alpha=False)
        except Exception:pass
        return None
    def _legend_handles(self, leg):
        handles = getattr(leg, 'legend_handles', None)
        if handles is None:handles = getattr(leg, 'legendHandles', [])
        return list(handles) if handles is not None else []
    def _hit_test_editable_text(self, event):
        fig = self.fig
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg:
                for txt in leg.get_texts():
                    if self._artist_contains(txt, event):return ('legend_text', ax, txt)
                lt = leg.get_title()
                if lt and lt.get_text() is not None and self._artist_contains(lt, event):return ('legend_title', ax, lt)
        for ax in fig.axes:
            t = ax.title
            if t and t.get_text() is not None and self._artist_contains(t, event):return ('graph_title', ax, t)
            xl = ax.xaxis.label
            if xl and xl.get_text() is not None and self._artist_contains(xl, event):return ('xlabel', ax, xl)
            yl = ax.yaxis.label
            if yl and yl.get_text() is not None and self._artist_contains(yl, event):return ('ylabel', ax, yl)
        st = getattr(fig, '_suptitle', None)
        if st is not None and self._artist_contains(st, event):return ('suptitle', None, st)
        return None
    def _find_legend_entry_at_event(self, ax, event):
        leg = ax.get_legend()
        if not leg:return None
        texts = list(leg.get_texts());handles = self._legend_handles(leg)
        for i, txt in enumerate(texts):
            if self._artist_contains(txt, event):return ('legend', txt.get_text(), i)
        for i, h in enumerate(handles):
            if self._artist_contains(h, event):label = texts[i].get_text() if i < len(texts) else getattr(h, 'get_label', lambda: '')();return ('legend', str(label), i)
        return None
    def _find_axes_artist_at_event(self, ax, event):
        cands = list(self._iter_axes_color_targets(ax));cands.sort(key=lambda a: float(getattr(a, 'get_zorder', lambda: 0.0)()), reverse=True)
        for art in cands:
            if not getattr(art, 'get_visible', lambda: True)():continue
            if self._artist_contains(art, event):return art
        return None
    def _update_label_override(self, ax_key: int, old: str, new: str) -> None:
        m = self._legend_label_overrides.setdefault(ax_key, {})
        for k in list(m.keys()):
            if m[k] == old:m[k] = new
        m[old] = new;cmap = self._color_overrides.get(ax_key)
        if cmap and old in cmap and (new != old):cmap[new] = cmap.pop(old)
    def _rename_axes_artists_label(self, ax, old: str, new: str) -> None:
        for art in self._iter_axes_color_targets(ax):
            try:
                if hasattr(art, 'get_label') and hasattr(art, 'set_label'):
                    if str(art.get_label()) == str(old):art.set_label(str(new))
            except Exception:pass
    def _apply_user_overrides(self) -> None:
        fig = self.fig;st = getattr(fig, '_suptitle', None)
        if self._suptitle_override is not None:
            if st is None:st = fig.suptitle(self._suptitle_override)
            else:st.set_text(self._suptitle_override)
        if self.graph_title_fontsize is not None and st is not None:
            try:st.set_fontsize(self.graph_title_fontsize)
            except Exception:pass
        for i, ax in enumerate(fig.axes):
            ax_key = int(i)
            if self.axis_label_fontsize is not None:
                try:ax.xaxis.label.set_fontsize(self.axis_label_fontsize)
                except Exception:pass
                try:ax.yaxis.label.set_fontsize(self.axis_label_fontsize)
                except Exception:pass
            if self.tick_label_fontsize is not None:
                try:ax.tick_params(axis='both', labelsize=self.tick_label_fontsize)
                except Exception:pass
            if self.graph_title_fontsize is not None:
                try:ax.title.set_fontsize(self.graph_title_fontsize)
                except Exception:pass
            if ax_key in self._title_overrides:
                try:ax.title.set_text(self._title_overrides[ax_key])
                except Exception:pass
            if ax_key in self._xlabel_overrides:
                try:ax.set_xlabel(self._xlabel_overrides[ax_key])
                except Exception:pass
            if ax_key in self._ylabel_overrides:
                try:ax.set_ylabel(self._ylabel_overrides[ax_key])
                except Exception:pass
            leg = ax.get_legend()
            if leg and ax_key in self._legend_title_overrides:
                try:leg.set_title(self._legend_title_overrides[ax_key])
                except Exception:
                    try:leg.get_title().set_text(self._legend_title_overrides[ax_key])
                    except Exception:pass
            label_map = self._legend_label_overrides.get(ax_key, {})
            if label_map:
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, 'get_label') and hasattr(art, 'set_label'):
                            lab = str(art.get_label())
                            if lab in label_map:art.set_label(self._resolve_label(lab, label_map))
                    except Exception:pass
                if leg:
                    for txt in leg.get_texts():
                        try:txt.set_text(self._resolve_label(txt.get_text(), label_map))
                        except Exception:pass
            cmap = self._color_overrides.get(ax_key, {})
            if cmap:
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, 'get_label'):
                            lab = str(art.get_label())
                            if lab in cmap:self._set_artist_color(art, cmap[lab])
                    except Exception:pass
                if leg:
                    texts = list(leg.get_texts());handles = self._legend_handles(leg)
                    for j, txt in enumerate(texts):
                        lab = str(txt.get_text())
                        if lab in cmap and j < len(handles):self._set_artist_color(handles[j], cmap[lab])
        for ax in fig.axes:
            for art in ax.get_children():
                c = self._artist_color_overrides.get(id(art))
                if c:self._set_artist_color(art, c)
    def _on_mpl_button_press(self, event):
        if event is None:return
        if self._toolbar_is_active():return
        if getattr(event, 'dblclick', False) and getattr(event, 'button', None) == 1:
            hit = self._hit_test_editable_text(event)
            if not hit:return
            kind, ax, txt = hit;old_text = str(txt.get_text());new_text = simpledialog.askstring('Edit text', 'Enter new text:', initialvalue=old_text, parent=self.winfo_toplevel())
            if new_text is None:return
            new_text = str(new_text)
            if kind == 'suptitle':self._suptitle_override = new_text;txt.set_text(new_text);self.redraw();return
            if ax is None:return
            ax_key = self._ax_key(ax)
            if kind == 'graph_title':self._title_overrides[ax_key] = new_text;txt.set_text(new_text);self.redraw();return
            if kind == 'xlabel':
                self._xlabel_overrides[ax_key] = new_text
                try:ax.set_xlabel(new_text)
                except Exception:txt.set_text(new_text)
                self.redraw();return
            if kind == 'ylabel':
                self._ylabel_overrides[ax_key] = new_text
                try:ax.set_ylabel(new_text)
                except Exception:txt.set_text(new_text)
                self.redraw();return
            if kind == 'legend_title':self._legend_title_overrides[ax_key] = new_text;txt.set_text(new_text);self.redraw();return
            if kind == 'legend_text':self._update_label_override(ax_key, old_text, new_text);self._rename_axes_artists_label(ax, old_text, new_text);txt.set_text(new_text);self.redraw();return
        if getattr(event, 'button', None) == 3:
            ax = getattr(event, 'inaxes', None)
            if ax is not None:leg_hit = self._find_legend_entry_at_event(ax, event)
            else:leg_hit = None
            if leg_hit and ax is not None:
                _kind, label, _i = leg_hit;ax_key = self._ax_key(ax);current = None
                for art in self._iter_axes_color_targets(ax):
                    try:
                        if hasattr(art, 'get_label') and str(art.get_label()) == str(label):current = self._artist_current_hex(art);break
                    except Exception:pass
                initial = current or self._color_overrides.get(ax_key, {}).get(label, '#000000');s = simpledialog.askstring('Set color', f"Hex color for '{label}' (#RRGGBB or #RRGGBBAA):", initialvalue=initial, parent=self.winfo_toplevel())
                if s is None:return
                try:hex_color = self._normalize_hex_color(s)
                except Exception as e:messagebox.showerror('Invalid color', f'{e}');return
                self._color_overrides.setdefault(ax_key, {})[str(label)] = hex_color;self.redraw();return
            if ax is None:return
            art = self._find_axes_artist_at_event(ax, event)
            if art is None:return
            label = None
            try:
                if hasattr(art, 'get_label'):
                    label = str(art.get_label())
                    if label.startswith('_'):label = None
            except Exception:label = None
            initial = self._artist_current_hex(art) or '#000000';s = simpledialog.askstring('Set color', 'Hex color (#RRGGBB or #RRGGBBAA):', initialvalue=initial, parent=self.winfo_toplevel())
            if s is None:return
            try:hex_color = self._normalize_hex_color(s)
            except Exception as e:messagebox.showerror('Invalid color', f'{e}');return
            if label is not None:self._color_overrides.setdefault(self._ax_key(ax), {})[label] = hex_color
            else:self._artist_color_overrides[id(art)] = hex_color
            self.redraw()
    @staticmethod
    def _axis_looks_like_time(ax) -> bool:
        try:xl = str(ax.get_xlabel() or '').lower()
        except Exception:xl = ''
        if 'time' in xl or 'sec' in xl or '(s' in xl:return True
        try:
            for line in list(getattr(ax, 'lines', [])):
                try:x = np.asarray(line.get_xdata(), dtype=float)
                except Exception:continue
                if x.size < 10:continue
                xf = x[np.isfinite(x)]
                if xf.size < 10:continue
                if float(np.nanmax(xf) - np.nanmin(xf)) < 1.0:continue
                dx = np.diff(xf)
                if dx.size and float(np.nanmean(dx >= 0)) > 0.8:return True
        except Exception:pass
        return False
    def _clear_time_marker_artists(self) -> None:
        for art in list(getattr(self, '_time_marker_artists', [])):
            try:art.remove()
            except Exception:pass
        self._time_marker_artists = []
    def _apply_time_markers(self) -> None:
        self._clear_time_marker_artists();marks = list(getattr(self, '_time_markers', []) or [])
        if not marks:return
        styles = [dict(linestyle=':', linewidth=1.3), dict(linestyle=(0, (1, 1)), linewidth=1.3), dict(linestyle=(0, (1, 3)), linewidth=1.3), dict(linestyle=(0, (3, 1, 1, 1)), linewidth=1.3)]
        for ax in list(getattr(self.fig, 'axes', [])):
            if not self._axis_looks_like_time(ax):continue
            added_any = False
            for m in marks:
                try:x = float(m.get('x'))
                except Exception:continue
                if not np.isfinite(x):continue
                label = str(m.get('label') or f't={x:g}s')
                try:si = int(m.get('style_idx', 0))
                except Exception:si = 0
                si = max(0, min(3, si))
                try:art = ax.axvline(x, label=label, color='0.2', alpha=0.85, zorder=9, **styles[si]);self._time_marker_artists.append(art);added_any = True
                except Exception:pass
            if added_any or ax.get_legend() is not None:
                try:
                    if ax.get_legend() is not None:ax.get_legend().remove()
                    ax.legend(loc='best')
                except Exception:pass
    def add_time_marker(self, x_s: float, label: Optional[str]=None) -> None:
        try:x = float(x_s)
        except Exception:return
        if not np.isfinite(x):return
        cur = list(getattr(self, '_time_markers', []) or [])
        if len(cur) >= 4:
            try:messagebox.showwarning('Time markers', 'You can add up to 4 time markers per plot.\n\nRemove one with Ctrl+I, then Ctrl+Backspace (in the dialog).')
            except Exception:pass
            return
        si = len(cur);lab = str(label).strip() if label is not None else ''
        if not lab:lab = f't={x:g}s'
        cur.append({'x': x, 'label': lab, 'style_idx': int(si)});self._time_markers = cur;self.redraw()
    def remove_last_time_marker(self) -> bool:
        cur = list(getattr(self, '_time_markers', []) or [])
        if not cur:return False
        cur.pop(-1);self._time_markers = cur;self.redraw();return True
    def clear_time_markers(self) -> None:self._time_markers = [];self.redraw()
    def redraw(self):self._apply_user_overrides();self._apply_time_markers();self._apply_user_overrides();self.canvas.draw_idle()
class ChannelTabsTk(ttk.Frame):
    def __init__(self, master, res: ChannelResult, norm_mode: str, parent_app=None):super().__init__(master);self.res = res;self.norm_mode = norm_mode;self.parent_app = parent_app;self.tabs = ttk.Notebook(self);self.tabs.pack(fill=tk.BOTH, expand=True);prefix = res.gcol;self.tab_raw = PlotTabTk(self.tabs, 'Raw', prefix);self.tab_slope = PlotTabTk(self.tabs, 'SlopeNormality', prefix, figsize=(8.2, 5.8));self.tab_art = PlotTabTk(self.tabs, 'ArtifactRemover', prefix);self.tab_fit = PlotTabTk(self.tabs, 'Fit', prefix);self.tab_norm = PlotTabTk(self.tabs, 'Normalization', prefix);self.tab_norm_smooth = PlotTabTk(self.tabs, 'Normalization_smoothed', prefix);self.tab_freq = PlotTabTk(self.tabs, 'Freq analysis', prefix, figsize=(8.6, 6.4));self._configure_tabs();self._slope_drawn_once = False;self._freq_drawn_version = None;self._fit_initialized = False;self.tabs.bind('<<NotebookTabChanged>>', self._on_inner_tab_changed);self._attach_exporters();self._draw_current_tab()
    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [self.tab_raw, self.tab_slope, self.tab_art, self.tab_fit, self.tab_norm, self.tab_norm_smooth, self.tab_freq]:
            try:tab.set_axis_label_fontsize(fontsize)
            except Exception:pass
    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [self.tab_raw, self.tab_slope, self.tab_art, self.tab_fit, self.tab_norm, self.tab_norm_smooth, self.tab_freq]:
            try:tab.set_tick_label_fontsize(fontsize)
            except Exception:pass
    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        for tab in [self.tab_raw, self.tab_slope, self.tab_art, self.tab_fit, self.tab_norm, self.tab_norm_smooth, self.tab_freq]:
            try:tab.set_graph_title_fontsize(fontsize)
            except Exception:pass
    def _on_inner_tab_changed(self, _event=None) -> None:self._draw_current_tab()
    def _draw_current_tab(self) -> None:
        sel = self.tabs.select()
        if not sel:return
        if sel == str(self.tab_raw):self._draw_raw();return
        if sel == str(self.tab_slope):
            if not self._slope_drawn_once:self._draw_slope_normality();self._slope_drawn_once = True
            else:self.tab_slope.redraw()
            return
        if sel == str(self.tab_art):self._draw_artifact();return
        if sel == str(self.tab_fit):
            if not self._fit_initialized:self._draw_fit_and_attach_selector();self._fit_initialized = True
            else:self.refresh_after_pipeline_change()
            return
        if sel == str(self.tab_norm):self._draw_norm();return
        if sel == str(self.tab_norm_smooth):self._draw_norm_smooth();return
        if self.tab_freq is not None and sel == str(self.tab_freq):
            current_version = int(getattr(self.res, '_data_version', 0))
            if self._freq_drawn_version != current_version:self._draw_frequency();self._freq_drawn_version = current_version
            else:self.tab_freq.redraw()
            return
    def get_active_plot_tab(self) -> Optional[PlotTabTk]:
        sel = self.tabs.select()
        if not sel:return None
        if sel == str(self.tab_raw):return self.tab_raw
        if sel == str(self.tab_slope):return self.tab_slope
        if sel == str(self.tab_art):return self.tab_art
        if sel == str(self.tab_fit):return self.tab_fit
        if sel == str(self.tab_norm):return self.tab_norm
        if sel == str(self.tab_norm_smooth):return self.tab_norm_smooth
        if self.tab_freq is not None and sel == str(self.tab_freq):return self.tab_freq
        return None
    def _norm_tab_texts(self) -> Tuple[str, str]:
        if self.norm_mode == NORM_DFF:return ('ΔF/F', 'ΔF/F smoothed')
        if self.norm_mode == NORM_ZF_GLOBAL:return ('zF (global)', 'zF smoothed')
        if self.norm_mode == NORM_ZF_INTERVAL:return ('zF - interval based', 'zF smoothed')
        return ('Normalization', 'Smoothed')
    def _configure_tabs(self):
        norm_txt, smooth_txt = self._norm_tab_texts()
        if len(self.tabs.tabs()) == 0:
            self.tabs.add(self.tab_raw, text='Raw');self.tabs.add(self.tab_slope, text='Slope normality');self.tabs.add(self.tab_art, text='Artifact remover');self.tabs.add(self.tab_fit, text='Fit');self.tabs.add(self.tab_norm, text=norm_txt);self.tabs.add(self.tab_norm_smooth, text=smooth_txt)
            if self.norm_mode == NORM_DFF:self.tabs.add(self.tab_freq, text='Freq analysis')
            return
        self.tabs.tab(self.tab_norm, text=norm_txt);self.tabs.tab(self.tab_norm_smooth, text=smooth_txt);freq_present = str(self.tab_freq) in self.tabs.tabs()
        if self.norm_mode == NORM_DFF:
            if not freq_present:self.tabs.add(self.tab_freq, text='Freq analysis')
            else:self.tabs.tab(self.tab_freq, text='Freq analysis')
        elif freq_present:
            if self.tabs.select() == str(self.tab_freq):self.tabs.select(self.tab_norm)
            self.tabs.forget(self.tab_freq)
    def set_norm_mode(self, norm_mode: str):self.norm_mode = norm_mode;self._configure_tabs();self._draw_current_tab()
    def _draw_static_tabs(self):self._draw_raw();self._draw_slope_normality();self._draw_artifact();self._draw_fit_and_attach_selector()
    def refresh_after_pipeline_change(self):
        sel = self.tabs.select()
        if not sel:return
        if sel == str(self.tab_art):self._draw_artifact();return
        if sel == str(self.tab_fit):
            if not self._fit_initialized:self._draw_fit_and_attach_selector();self._fit_initialized = True
            else:
                try:self._line_exc.set_ydata(self.res.exc_clean);self._line_fit.set_ydata(self.res.fitted_iso_on_exc);self._fit_info.set_text(self._fit_info_text());self.tab_fit.redraw()
                except Exception:self._draw_fit_and_attach_selector();self._fit_initialized = True
            return
        if sel == str(self.tab_norm):self._draw_norm();return
        if sel == str(self.tab_norm_smooth):self._draw_norm_smooth();return
        if self.norm_mode == NORM_DFF and self.tab_freq is not None and (sel == str(self.tab_freq)):self._draw_frequency();self._freq_drawn_version = int(getattr(self.res, '_data_version', 0));return
    def _draw_norm_dependent_tabs(self):
        self._draw_norm();self._draw_norm_smooth()
        if self.norm_mode == NORM_DFF:self._draw_frequency()
    def _draw_raw(self):ax = self.tab_raw.ax;ax.clear();ax.plot(self.res.t_exc, self.res.exc_raw, label='Excitatory raw');ax.plot(self.res.t_iso, self.res.iso_raw, label='Isosbestic raw');ax.set_title(f'{self.res.gcol} - Raw');ax.set_xlabel('Time (s)');ax.set_ylabel('Signal');ax.legend(loc='best');self.tab_raw.redraw()
    def _draw_slope_normality(self):
        fig = self.tab_slope.fig;fig.clear();axs = fig.subplots(2, 2);ax_hist_iso, ax_hist_exc = (axs[0, 0], axs[0, 1]);ax_qq_iso, ax_qq_exc = (axs[1, 0], axs[1, 1]);slopes_iso = compute_slopes(self.res.t_iso, self.res.iso_raw);slopes_exc = compute_slopes(self.res.t_exc, self.res.exc_raw);MAX_HIST_N = 200000;slopes_iso_hist = subsample_even(slopes_iso, MAX_HIST_N);slopes_exc_hist = subsample_even(slopes_exc, MAX_HIST_N)
        def _fmt(x: float) -> str:return 'n/a' if x is None or not np.isfinite(x) else f'{x:.3g}'
        def draw_hist(ax, slopes: np.ndarray, title: str):
            ax.clear()
            if slopes.size < 3:ax.text(0.5, 0.5, 'Not enough finite slopes', ha='center', va='center', transform=ax.transAxes);ax.set_title(title);ax.set_xlabel('dy/dt');ax.set_ylabel('Density');return
            ax.hist(slopes, bins=80, density=True, alpha=0.75);ax.set_title(title);ax.set_xlabel('dy/dt');ax.set_ylabel('Density')
        def draw_qq(ax, slopes_all: np.ndarray, title: str):
            ax.clear()
            if slopes_all.size < 3:ax.text(0.5, 0.5, 'Not enough finite slopes', ha='center', va='center', transform=ax.transAxes);ax.set_title(title);ax.set_xlabel('Normal theoretical quantiles');ax.set_ylabel('Ordered slopes');return
            theor, ordered, m, b, r = qq_plot_points(slopes_all, max_n=DEFAULT_SLOPE_TEST_MAX_N)
            if theor.size:
                ax.scatter(theor, ordered, s=8, alpha=0.75)
                if np.isfinite(m) and np.isfinite(b):xline = np.array([float(np.min(theor)), float(np.max(theor))]);ax.plot(xline, m * xline + b, linewidth=1.2)
            ax.set_title(title + (f' (r={r:.3f})' if np.isfinite(r) else ''));ax.set_xlabel('Normal theoretical quantiles');ax.set_ylabel('Ordered slopes');summ = normality_summary(slopes_all, alpha=0.05, max_n=DEFAULT_SLOPE_TEST_MAX_N);lines = [f"n={summ['n']} (test n={summ['n_test']})", f"mean={_fmt(summ['mean'])}  std={_fmt(summ['std'])}"]
            if _HAVE_SCIPY_STATS:lines += [f"Shapiro p={_fmt_p(summ['shapiro_p'])}", f"D’Agostino p={_fmt_p(summ['dagostino_p'])}", f"AD stat={_fmt(summ['anderson_stat'])}  cv5={_fmt(summ['anderson_cv5'])}", f"α=0.05 verdict: {summ['verdict']}"]
            else:lines += ['SciPy not installed: no p-values', f"α=0.05 verdict: {summ['verdict']}"]
            if summ.get('note'):lines.append(str(summ['note']).strip())
            ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, ha='left', va='top', fontsize=8, family='monospace', bbox=dict(boxstyle='round,pad=0.3', alpha=0.12))
        draw_hist(ax_hist_iso, slopes_iso_hist, 'Isosbestic slopes (raw) – histogram');draw_hist(ax_hist_exc, slopes_exc_hist, 'Excitatory slopes (raw) – histogram');draw_qq(ax_qq_iso, slopes_iso, 'Isosbestic slopes (raw) – Q–Q');draw_qq(ax_qq_exc, slopes_exc, 'Excitatory slopes (raw) – Q–Q');fig.suptitle(f'{self.res.gcol} – Slope normality (dy/dt on raw)', y=0.99, fontsize=11);fig.tight_layout(rect=(0, 0.0, 1, 0.96));self.tab_slope.redraw()
    def _draw_artifact(self):
        ax = self.tab_art.ax;ax.clear();ax.plot(self.res.t_exc, self.res.exc_raw, color='0.75', linewidth=1.0, label='Exc raw');ax.plot(self.res.t_iso, self.res.iso_raw, color='0.75', linewidth=1.0, linestyle='--', label='Iso raw');ax.plot(self.res.t_exc, self.res.exc_clean_holes, linewidth=1.2, label='Exc cleaned (holes)');ax.plot(self.res.t_iso, self.res.iso_clean_holes, linewidth=1.2, label='Iso cleaned (holes)')
        if self.res.use_interpolation:
            def _interp_overlay(y_holes: np.ndarray, y_interp: np.ndarray) -> np.ndarray:
                y_holes = np.asarray(y_holes, dtype=float);y_interp = np.asarray(y_interp, dtype=float);overlay = np.full_like(y_interp, np.nan, dtype=float);nan_mask = ~np.isfinite(y_holes)
                for i0, i1 in _contiguous_true_runs(nan_mask):j0 = max(0, i0 - 1);j1 = min(len(overlay) - 1, i1 + 1);overlay[j0:j1 + 1] = y_interp[j0:j1 + 1]
                return overlay
            exc_fill = _interp_overlay(self.res.exc_clean_holes, self.res.exc_clean_interp);iso_fill = _interp_overlay(self.res.iso_clean_holes, self.res.iso_clean_interp);ax.plot(self.res.t_exc, exc_fill, linewidth=2.0, color='tab:orange', label='Exc interpolated (filled parts)');ax.plot(self.res.t_iso, iso_fill, linewidth=2.0, color='tab:orange', linestyle='--', label='Iso interpolated (filled parts)')
        if np.any(self.res.art_exc):ax.scatter(self.res.t_exc[self.res.art_exc], self.res.exc_raw[self.res.art_exc], s=12, color='red', label='Shared artifacts', zorder=5)
        if np.any(self.res.art_iso):ax.scatter(self.res.t_iso[self.res.art_iso], self.res.iso_raw[self.res.art_iso], s=12, color='red', label='_nolegend_', zorder=5)
        ax.set_title(f'{self.res.gcol} - Artifact remover (red=shared, orange=filled)');ax.set_xlabel('Time (s)');ax.set_ylabel('Signal');ax.legend(loc='best');self.tab_art.redraw()
    def _draw_fit_and_attach_selector(self):
        ax = self.tab_fit.ax;ax.clear();self._fit_window_artists = []
        for a, b in self.res.windows:span = ax.axvspan(a, b, alpha=0.15);v1 = ax.axvline(a, linestyle=':', linewidth=1.0, alpha=0.7);v2 = ax.axvline(b, linestyle=':', linewidth=1.0, alpha=0.7);self._fit_window_artists.extend([span, v1, v2])
        self._line_exc, = ax.plot(self.res.t_exc, self.res.exc_clean, label='Exc (active)');self._line_fit, = ax.plot(self.res.t_exc, self.res.fitted_iso_on_exc, label='Fitted iso → exc (active)');self._fit_info = ax.text(0.99, 0.02, self._fit_info_text(), transform=ax.transAxes, ha='right', va='bottom', fontsize=11, bbox=dict(boxstyle='round,pad=0.25', alpha=0.08, lw=0.6));ax.set_title(f'{self.res.gcol} - Fit (drag to select new window)');ax.set_xlabel('Time (s)');ax.set_ylabel('Signal');ax.legend(loc='best')
        def onselect(xmin, xmax):
            if xmin == xmax:return
            left, right = sorted([float(xmin), float(xmax)]);self.update_fit_window(left, right)
        self._span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True);self.tab_fit.redraw()
    def _fit_info_text(self) -> str:a = self.res.slope;b = self.res.intercept;r2 = self.res.r2;tag = 'interp' if self.res.use_interpolation else 'holes';return f'{tag}\nslope={a:.4f}\ninterc={b:.4f}\nR²={r2:.4f}'
    def _get_norm_series(self) -> Tuple[np.ndarray, str, str, str]:
        if self.norm_mode == NORM_DFF:return (self.res.dFF, 'ΔF/F', 'ΔF/F', f'{self.res.gcol} - ΔF/F')
        if self.norm_mode == NORM_ZF_GLOBAL:return (self.res.zF_global, 'zF', 'zF (global)', f'{self.res.gcol} - zF (global, GUI)')
        if self.norm_mode == NORM_ZF_INTERVAL:a = self.res.zf_interval_start_s;b = self.res.zf_interval_end_s;return (self.res.zF_interval, 'zF', f'zF (interval stats: {min(a, b):g}–{max(a, b):g} s)', f'{self.res.gcol} - zF - interval based')
        return (self.res.dFF, 'Signal', 'Signal', f'{self.res.gcol} - Normalization')
    def _get_norm_series_for_result(self, res: ChannelResult) -> Tuple[np.ndarray, str, str, str]:
        if self.norm_mode == NORM_DFF:return (res.dFF, 'ΔF/F', 'ΔF/F', f'{res.gcol} - ΔF/F')
        if self.norm_mode == NORM_ZF_GLOBAL:return (res.zF_global, 'zF', 'zF (global)', f'{res.gcol} - zF (global, GUI)')
        if self.norm_mode == NORM_ZF_INTERVAL:a = res.zf_interval_start_s;b = res.zf_interval_end_s;return (res.zF_interval, 'zF', f'zF (interval stats: {min(a, b):g}–{max(a, b):g} s)', f'{res.gcol} - zF - interval based')
        return (res.dFF, 'Signal', 'Signal', f'{res.gcol} - Normalization')
    def _draw_norm(self):ax = self.tab_norm.ax;ax.clear();y, ylabel, label, title = self._get_norm_series();ax.plot(self.res.t_exc, y, label=label);ax.set_title(title);ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel);ax.legend(loc='best');self.tab_norm.redraw()
    def _draw_norm_smooth(self):
        ax = self.tab_norm_smooth.ax;ax.clear();y, ylabel, label, title = self._get_norm_series();w = int(getattr(self.res, 'smooth_window', DEFAULT_SMOOTH_WINDOW));y_s = get_smoothed_norm_array(self.res, self.norm_mode, window_size=w);line_raw, = ax.plot(self.res.t_exc, y, alpha=0.25, linewidth=1.0, zorder=1, label=f'{label} (raw)');ax.plot(self.res.t_exc, y_s, linewidth=1.4, zorder=3, color=line_raw.get_color(), label=f'{ylabel} smoothed (win={w})');ax.set_title(f'{title} - smoothed');ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel);ax.legend(loc='best')
        if self.parent_app is not None and hasattr(self.parent_app, '_results'):
            try:
                all_results = self.parent_app._results;global_y_min = np.inf;global_y_max = -np.inf
                for mid, res_other in all_results.items():
                    y_other, _, _, _ = self._get_norm_series_for_result(res_other);y_s_other = get_smoothed_norm_array(res_other, self.norm_mode, window_size=w);finite_vals = y_s_other[np.isfinite(y_s_other)]
                    if len(finite_vals) > 0:global_y_min = min(global_y_min, np.nanmin(finite_vals));global_y_max = max(global_y_max, np.nanmax(finite_vals))
                if np.isfinite(global_y_min) and np.isfinite(global_y_max):padding = (global_y_max - global_y_min) * 0.05;ax.set_ylim(global_y_min - padding, global_y_max + padding)
            except Exception:pass
        self.tab_norm_smooth.redraw()
    def _draw_frequency(self):
        fig = self.tab_freq.fig;fig.clear();axs = fig.subplots(3, 2);axes = np.asarray(axs).flatten();t = np.asarray(self.res.t_exc, dtype=float);dff = np.asarray(self.res.dFF_nointerp, dtype=float);fs = float(getattr(self.res, 'eff_fs_hz', np.nan));acq = float(getattr(self.res, 'acq_fps_hz', 0.0))
        if not np.isfinite(fs) or fs <= 0:fs = estimate_fs_from_t(t)
        have_holes = np.any(~np.isfinite(dff))
        if _HAVE_SCIPY_SIGNAL:method = 'Butterworth (segment-wise, no interpolation)'
        else:method = 'FFT (only if no NaNs)' if not have_holes else 'Unavailable (SciPy signal required for NaN holes)'
        supt = f'{self.res.gcol} – Band-limited ΔF/F (NO interpolation)\nUses non-interpolated ΔF/F; holes preserved. Acq FPS={acq:.3g} Hz → eff fs={fs:.3g} Hz | {method}';fig.suptitle(supt, y=0.995, fontsize=11)
        for i, (low_hz, high_hz) in enumerate(FREQ_BANDS):
            ax = axes[i];ax.clear();label = f'{high_hz:g}–{low_hz:g} Hz'
            if not np.any(np.isfinite(dff)):ax.text(0.5, 0.5, 'ΔF/F is all NaN', ha='center', va='center', transform=ax.transAxes);ax.set_title(label);ax.set_xlabel('Time (s)');ax.set_ylabel('ΔF/F');continue
            if _HAVE_SCIPY_SIGNAL:y_band = bandpass_butterworth_segmentwise_no_interp(dff, low_hz, high_hz, fs, order=DEFAULT_BUTTER_ORDER)
            else:y_band = bandpass_fft_no_interp(dff, low_hz, high_hz, fs)
            if not np.any(np.isfinite(y_band)):ax.text(0.5, 0.5, 'Band unavailable\n(check SciPy / NaNs / Nyquist)', ha='center', va='center', transform=ax.transAxes)
            else:ax.plot(t, y_band, label=label, linewidth=1.0);ax.legend(loc='best')
            ax.set_title(label);ax.set_xlabel('Time (s)');ax.set_ylabel('ΔF/F')
        for j in range(len(FREQ_BANDS), len(axes)):axes[j].axis('off')
        fig.tight_layout(rect=(0, 0.0, 1, 0.93));self.tab_freq.redraw()
    def update_fit_window(self, start_s: float, end_s: float) -> None:
        if not isinstance(self.res.windows, list) or not self.res.windows:self.res.windows = [(float(start_s), float(end_s))]
        else:self.res.windows[0] = (float(start_s), float(end_s))
        recompute_fit_and_downstream(self.res)
        if not self._fit_initialized:self._draw_fit_and_attach_selector();self._fit_initialized = True;return
        try:self._line_exc.set_ydata(self.res.exc_clean);self._line_fit.set_ydata(self.res.fitted_iso_on_exc);self._fit_info.set_text(self._fit_info_text());self.tab_fit.redraw()
        except Exception:self._draw_fit_and_attach_selector();self._fit_initialized = True
    def _attach_exporters(self):self.tab_raw.export_provider = self._export_raw;self.tab_slope.export_provider = self._export_slope_normality;self.tab_art.export_provider = self._export_artifact;self.tab_fit.export_provider = self._export_fit;self.tab_norm.export_provider = self._export_norm;self.tab_norm_smooth.export_provider = self._export_norm_smoothed;self.tab_freq.export_provider = self._export_frequency
    def _meta_df(self) -> pd.DataFrame:wtxt = '; '.join([f'[{min(a, b):g},{max(a, b):g}]' for a, b in self.res.windows]) if self.res.windows else '';rows = [('gcol', self.res.gcol), ('use_interpolation', bool(self.res.use_interpolation)), ('smooth_window', int(getattr(self.res, 'smooth_window', DEFAULT_SMOOTH_WINDOW))), ('acq_fps_hz', float(getattr(self.res, 'acq_fps_hz', np.nan))), ('eff_fs_hz', float(getattr(self.res, 'eff_fs_hz', np.nan))), ('fit_windows_s', wtxt), ('slope_active', float(self.res.slope)), ('intercept_active', float(self.res.intercept)), ('r2_active', float(self.res.r2)), ('zf_interval_start_s', float(getattr(self.res, 'zf_interval_start_s', DEFAULT_ZF_INTERVAL_START_S))), ('zf_interval_end_s', float(getattr(self.res, 'zf_interval_end_s', DEFAULT_ZF_INTERVAL_END_S))), ('norm_mode', str(self.norm_mode))];return pd.DataFrame(rows, columns=['key', 'value'])
    def _export_raw(self) -> Dict[str, pd.DataFrame]:exc = pd.DataFrame({'t_s': self.res.t_exc, 'exc_raw': self.res.exc_raw});iso = pd.DataFrame({'t_s': self.res.t_iso, 'iso_raw': self.res.iso_raw});return {'exc_raw': exc, 'iso_raw': iso, 'meta': self._meta_df()}
    def _export_slope_normality(self) -> Dict[str, pd.DataFrame]:slopes_iso = compute_slopes(self.res.t_iso, self.res.iso_raw);slopes_exc = compute_slopes(self.res.t_exc, self.res.exc_raw);theor_i, ord_i, m_i, b_i, r_i = qq_plot_points(slopes_iso, max_n=DEFAULT_SLOPE_TEST_MAX_N);theor_e, ord_e, m_e, b_e, r_e = qq_plot_points(slopes_exc, max_n=DEFAULT_SLOPE_TEST_MAX_N);return {'slopes_iso': pd.DataFrame({'slope_dy_dt': slopes_iso}), 'slopes_exc': pd.DataFrame({'slope_dy_dt': slopes_exc}), 'qq_iso': pd.DataFrame({'theor': theor_i, 'ordered': ord_i}), 'qq_exc': pd.DataFrame({'theor': theor_e, 'ordered': ord_e}), 'qq_fit_meta': pd.DataFrame([('iso_slope', m_i), ('iso_intercept', b_i), ('iso_r', r_i), ('exc_slope', m_e), ('exc_intercept', b_e), ('exc_r', r_e)], columns=['key', 'value']), 'meta': self._meta_df()}
    def _export_artifact(self) -> Dict[str, pd.DataFrame]:exc = pd.DataFrame({'t_s': self.res.t_exc, 'exc_raw': self.res.exc_raw, 'exc_clean_holes': self.res.exc_clean_holes, 'exc_clean_interp': self.res.exc_clean_interp, 'artifact_exc': self.res.art_exc.astype(bool)});iso = pd.DataFrame({'t_s': self.res.t_iso, 'iso_raw': self.res.iso_raw, 'iso_clean_holes': self.res.iso_clean_holes, 'iso_clean_interp': self.res.iso_clean_interp, 'artifact_iso': self.res.art_iso.astype(bool)});return {'exc_artifact': exc, 'iso_artifact': iso, 'meta': self._meta_df()}
    def _export_fit(self) -> Dict[str, pd.DataFrame]:df = pd.DataFrame({'t_s': self.res.t_exc, 'exc_active': self.res.exc_clean, 'iso_on_exc_active': self.res.iso_on_exc, 'fitted_iso_on_exc_active': self.res.fitted_iso_on_exc, 'residual_active': self.res.residual, 'dF_active': self.res.dF, 'dFF_active': self.res.dFF, 'zF_global': self.res.zF_global, 'zF_interval': self.res.zF_interval});return {'fit_active': df, 'meta': self._meta_df()}
    def _export_norm(self) -> Dict[str, pd.DataFrame]:y, ylabel, label, title = self._get_norm_series();df = pd.DataFrame({'t_s': self.res.t_exc, 'value': y});return {'normalization': df, 'meta': self._meta_df()}
    def _export_norm_smoothed(self) -> Dict[str, pd.DataFrame]:y, ylabel, label, title = self._get_norm_series();w = int(getattr(self.res, 'smooth_window', DEFAULT_SMOOTH_WINDOW));y_s = get_smoothed_norm_array(self.res, self.norm_mode, window_size=w);df = pd.DataFrame({'t_s': self.res.t_exc, 'value_raw': y, f'value_smoothed_win{w}': y_s});return {'normalization_smoothed': df, 'meta': self._meta_df()}
    def _export_frequency(self) -> Dict[str, pd.DataFrame]:
        t = np.asarray(self.res.t_exc, dtype=float);dff = np.asarray(self.res.dFF_nointerp, dtype=float);fs = float(getattr(self.res, 'eff_fs_hz', np.nan));acq = float(getattr(self.res, 'acq_fps_hz', 0.0))
        if not np.isfinite(fs) or fs <= 0:fs = estimate_fs_from_t(t)
        out = {'t_s': t, 'dFF_nointerp': dff}
        for low_hz, high_hz in FREQ_BANDS:
            col = f'band_{low_hz:g}_{high_hz:g}_Hz'
            if _HAVE_SCIPY_SIGNAL:y_band = bandpass_butterworth_segmentwise_no_interp(dff, low_hz, high_hz, fs, order=DEFAULT_BUTTER_ORDER)
            else:y_band = bandpass_fft_no_interp(dff, low_hz, high_hz, fs)
            out[col] = y_band
        df = pd.DataFrame(out);meta = self._meta_df();extra = pd.DataFrame([('freq_acq_fps_hz', acq), ('freq_eff_fs_hz', fs), ('method', 'butter_sos_segmentwise' if _HAVE_SCIPY_SIGNAL else 'fft')], columns=['key', 'value']);meta2 = pd.concat([meta, extra], ignore_index=True);return {'freq_bands': df, 'meta': meta2}
class BatchCompareTk(ttk.Frame):
    def __init__(self, master, app: 'MainAppTk'):super().__init__(master);self.app = app;self.selected_ids: List[str] = [];self._available_ids: List[str] = [];self._display: Dict[str, str] = {};self.columnconfigure(1, weight=1);self.rowconfigure(0, weight=1);controls = ttk.Frame(self, padding=8);controls.grid(row=0, column=0, sticky='nsw');plot_holder = ttk.Frame(self);plot_holder.grid(row=0, column=1, sticky='nsew', padx=(0, 8), pady=(8, 8));plot_holder.rowconfigure(0, weight=1);plot_holder.columnconfigure(0, weight=1);ttk.Label(controls, text='Available mice:').grid(row=0, column=0, sticky='w');self.lst_available = tk.Listbox(controls, selectmode=tk.EXTENDED, height=10, width=28);sb_av = ttk.Scrollbar(controls, orient='vertical', command=self.lst_available.yview);self.lst_available.config(yscrollcommand=sb_av.set);self.lst_available.grid(row=1, column=0, sticky='nsew');sb_av.grid(row=1, column=1, sticky='ns');btns = ttk.Frame(controls);btns.grid(row=2, column=0, columnspan=2, pady=(6, 10), sticky='ew');self.btn_add = ttk.Button(btns, text='Add →', command=self.add_selected);self.btn_add.pack(side=tk.LEFT, fill=tk.X, expand=True);self.btn_remove = ttk.Button(btns, text='← Remove', command=self.remove_selected);self.btn_remove.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True);ttk.Label(controls, text='Selected for compare:').grid(row=3, column=0, sticky='w');self.lst_selected = tk.Listbox(controls, selectmode=tk.EXTENDED, height=10, width=28);sb_sel = ttk.Scrollbar(controls, orient='vertical', command=self.lst_selected.yview);self.lst_selected.config(yscrollcommand=sb_sel.set);self.lst_selected.grid(row=4, column=0, sticky='nsew');sb_sel.grid(row=4, column=1, sticky='ns');opts = ttk.Frame(controls);opts.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(10, 0));self.var_smoothed = tk.BooleanVar(value=False);ttk.Checkbutton(opts, text='Plot smoothed trace', variable=self.var_smoothed, command=self.refresh_plot).pack(side=tk.TOP, anchor='w');self.btn_plot = ttk.Button(opts, text='Plot selected', command=self.refresh_plot);self.btn_plot.pack(side=tk.TOP, fill=tk.X, pady=(8, 0));self.btn_clear = ttk.Button(opts, text='Clear selection', command=self.clear_selection);self.btn_clear.pack(side=tk.TOP, fill=tk.X, pady=(6, 0));hint = ttk.Label(controls, text='Tip: Add time markers on this plot with Ctrl+I\n(remove last with Ctrl+I then Backspace).', foreground='gray35', justify='left');hint.grid(row=6, column=0, columnspan=2, sticky='w', pady=(12, 0));self.plot = PlotTabTk(plot_holder, 'Compare', default_filename_prefix='batch_compare', figsize=(8.2, 5.2));self.plot.grid(row=0, column=0, sticky='nsew');self.plot.export_provider = self._export_compare;self.lst_available.bind('<Double-Button-1>', lambda _e: self.add_selected());self.lst_selected.bind('<Double-Button-1>', lambda _e: self.remove_selected());self.refresh_available();self.refresh_plot()
    def refresh_available(self) -> None:
        items = self.app.get_mouse_items();self._display = {mid: label for mid, label in items};self.selected_ids = [mid for mid in self.selected_ids if mid in self._display];self._available_ids = [mid for mid, _lab in items if mid not in set(self.selected_ids)];self.lst_available.delete(0, tk.END)
        for mid in self._available_ids:self.lst_available.insert(tk.END, self._display.get(mid, mid))
        self.lst_selected.delete(0, tk.END)
        for mid in self.selected_ids:self.lst_selected.insert(tk.END, self._display.get(mid, mid))
    def add_selected(self) -> None:
        sel = list(self.lst_available.curselection())
        if not sel:return
        for i in sel:
            if 0 <= int(i) < len(self._available_ids):
                mid = self._available_ids[int(i)]
                if mid not in self.selected_ids:self.selected_ids.append(mid)
        self.refresh_available();self.refresh_plot()
    def remove_selected(self) -> None:
        sel = sorted(list(self.lst_selected.curselection()), reverse=True)
        if not sel:return
        for i in sel:
            if 0 <= int(i) < len(self.selected_ids):self.selected_ids.pop(int(i))
        self.refresh_available();self.refresh_plot()
    def clear_selection(self) -> None:self.selected_ids = [];self.refresh_available();self.refresh_plot()
    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_axis_label_fontsize(fontsize)
        except Exception:pass
    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_graph_title_fontsize(fontsize)
        except Exception:pass
    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_tick_label_fontsize(fontsize)
        except Exception:pass
    def refresh_plot(self) -> None:
        ax = self.plot.ax;ax.clear();mode = self.app.get_norm_mode();smooth_win = self.app.get_smooth_window();do_smooth = bool(self.var_smoothed.get());ylabel = 'Signal';plotted = 0
        for mid in self.selected_ids:
            res = self.app.get_mouse_result(mid)
            if res is None:continue
            t, y, ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
            ax.plot(t, y, label=self._display.get(mid, mid));plotted += 1
        ax.set_title(f"Batch compare ({plotted} mouse{('es' if plotted != 1 else '')})");ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel)
        if plotted:ax.legend(loc='best')
        self.plot.redraw()
    def _export_compare(self) -> Dict[str, pd.DataFrame]:
        mode = self.app.get_norm_mode();smooth_win = self.app.get_smooth_window();do_smooth = bool(self.var_smoothed.get());payload: Dict[str, pd.DataFrame] = {}
        for mid in self.selected_ids:
            res = self.app.get_mouse_result(mid)
            if res is None:continue
            t, y, _ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
            payload[self._display.get(mid, mid)] = pd.DataFrame({'t_s': t, 'value': y})
        payload['meta'] = pd.DataFrame([('mode', mode), ('smoothed', do_smooth), ('smooth_window', smooth_win), ('n_selected', len(self.selected_ids))], columns=['key', 'value']);return payload
class BatchAverageTk(ttk.Frame):
    def __init__(self, master, app: 'MainAppTk'):super().__init__(master);self.app = app;self.group_a: List[str] = [];self.group_b: List[str] = [];self._available_ids: List[str] = [];self._display: Dict[str, str] = {};self.columnconfigure(1, weight=1);self.rowconfigure(0, weight=1);controls = ttk.Frame(self, padding=8);controls.grid(row=0, column=0, sticky='nsw');plot_holder = ttk.Frame(self);plot_holder.grid(row=0, column=1, sticky='nsew', padx=(0, 8), pady=(8, 8));plot_holder.rowconfigure(0, weight=1);plot_holder.columnconfigure(0, weight=1);ttk.Label(controls, text='Available mice:').grid(row=0, column=0, sticky='w');self.lst_available = tk.Listbox(controls, selectmode=tk.EXTENDED, height=9, width=28);sb_av = ttk.Scrollbar(controls, orient='vertical', command=self.lst_available.yview);self.lst_available.config(yscrollcommand=sb_av.set);self.lst_available.grid(row=1, column=0, sticky='nsew');sb_av.grid(row=1, column=1, sticky='ns');btns = ttk.Frame(controls);btns.grid(row=2, column=0, columnspan=2, pady=(6, 10), sticky='ew');ttk.Button(btns, text='Add to A →', command=self.add_to_a).pack(side=tk.LEFT, fill=tk.X, expand=True);ttk.Button(btns, text='Add to B →', command=self.add_to_b).pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True);ttk.Label(controls, text='Group A:').grid(row=3, column=0, sticky='w');self.lst_a = tk.Listbox(controls, selectmode=tk.EXTENDED, height=6, width=28);sb_a = ttk.Scrollbar(controls, orient='vertical', command=self.lst_a.yview);self.lst_a.config(yscrollcommand=sb_a.set);self.lst_a.grid(row=4, column=0, sticky='nsew');sb_a.grid(row=4, column=1, sticky='ns');ttk.Button(controls, text='Remove from A', command=self.remove_from_a).grid(row=5, column=0, columnspan=2, sticky='ew', pady=(4, 10));ttk.Label(controls, text='Group B:').grid(row=6, column=0, sticky='w');self.lst_b = tk.Listbox(controls, selectmode=tk.EXTENDED, height=6, width=28);sb_b = ttk.Scrollbar(controls, orient='vertical', command=self.lst_b.yview);self.lst_b.config(yscrollcommand=sb_b.set);self.lst_b.grid(row=7, column=0, sticky='nsew');sb_b.grid(row=7, column=1, sticky='ns');ttk.Button(controls, text='Remove from B', command=self.remove_from_b).grid(row=8, column=0, columnspan=2, sticky='ew', pady=(4, 10));opts = ttk.Frame(controls);opts.grid(row=9, column=0, columnspan=2, sticky='ew');self.var_show_individual = tk.BooleanVar(value=False);ttk.Checkbutton(opts, text='Show individual traces', variable=self.var_show_individual, command=self.refresh_plot).pack(side=tk.TOP, anchor='w');self.var_show_sem = tk.BooleanVar(value=True);ttk.Checkbutton(opts, text='Show SEM shading', variable=self.var_show_sem, command=self.refresh_plot).pack(side=tk.TOP, anchor='w');self.var_smoothed = tk.BooleanVar(value=False);ttk.Checkbutton(opts, text='Average smoothed traces', variable=self.var_smoothed, command=self.refresh_plot).pack(side=tk.TOP, anchor='w');self.btn_plot = ttk.Button(opts, text='Plot averages', command=self.refresh_plot);self.btn_plot.pack(side=tk.TOP, fill=tk.X, pady=(8, 0));self.btn_clear = ttk.Button(opts, text='Clear groups', command=self.clear_groups);self.btn_clear.pack(side=tk.TOP, fill=tk.X, pady=(6, 0));hint = ttk.Label(controls, text='Tip: Ctrl+I adds a vertical time marker on this plot.', foreground='gray35', justify='left');hint.grid(row=10, column=0, columnspan=2, sticky='w', pady=(12, 0));self.plot = PlotTabTk(plot_holder, 'Average', default_filename_prefix='batch_average', figsize=(8.2, 5.2));self.plot.grid(row=0, column=0, sticky='nsew');self.plot.export_provider = self._export_average;self.refresh_available();self.refresh_plot()
    def set_axis_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_axis_label_fontsize(fontsize)
        except Exception:pass
    def set_graph_title_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_graph_title_fontsize(fontsize)
        except Exception:pass
    def set_tick_label_fontsize(self, fontsize: Optional[float]) -> None:
        try:self.plot.set_tick_label_fontsize(fontsize)
        except Exception:pass
    def refresh_available(self) -> None:
        items = self.app.get_mouse_items();self._display = {mid: label for mid, label in items};all_ids = set(self._display.keys());self.group_a = [mid for mid in self.group_a if mid in all_ids];self.group_b = [mid for mid in self.group_b if mid in all_ids];used = set(self.group_a) | set(self.group_b);self._available_ids = [mid for mid, _lab in items if mid not in used];self.lst_available.delete(0, tk.END)
        for mid in self._available_ids:self.lst_available.insert(tk.END, self._display.get(mid, mid))
        self.lst_a.delete(0, tk.END)
        for mid in self.group_a:self.lst_a.insert(tk.END, self._display.get(mid, mid))
        self.lst_b.delete(0, tk.END)
        for mid in self.group_b:self.lst_b.insert(tk.END, self._display.get(mid, mid))
    def _add_from_available(self, target: List[str]) -> None:
        sel = list(self.lst_available.curselection())
        if not sel:return
        for i in sel:
            if 0 <= int(i) < len(self._available_ids):
                mid = self._available_ids[int(i)]
                if mid not in target:target.append(mid)
        self.refresh_available();self.refresh_plot()
    def add_to_a(self) -> None:self._add_from_available(self.group_a)
    def add_to_b(self) -> None:self._add_from_available(self.group_b)
    def remove_from_a(self) -> None:
        sel = sorted(list(self.lst_a.curselection()), reverse=True)
        if not sel:return
        for i in sel:
            if 0 <= int(i) < len(self.group_a):self.group_a.pop(int(i))
        self.refresh_available();self.refresh_plot()
    def remove_from_b(self) -> None:
        sel = sorted(list(self.lst_b.curselection()), reverse=True)
        if not sel:return
        for i in sel:
            if 0 <= int(i) < len(self.group_b):self.group_b.pop(int(i))
        self.refresh_available();self.refresh_plot()
    def clear_groups(self) -> None:self.group_a = [];self.group_b = [];self.refresh_available();self.refresh_plot()
    @staticmethod
    def _interp_to_common_grid(t_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not t_list or not y_list:return (np.array([], dtype=float), np.empty((0, 0), dtype=float))
        lens = [len(t) for t in t_list];j = int(np.argmin(lens));t0 = np.asarray(t_list[j], dtype=float)
        if t0.size == 0:return (np.array([], dtype=float), np.empty((0, 0), dtype=float))
        Y = []
        for t, y in zip(t_list, y_list):
            t = np.asarray(t, dtype=float);y = np.asarray(y, dtype=float)
            if t.size < 2 or y.size != t.size:continue
            y_filled = _fill_nans_linear_1d(y)
            try:yi = np.interp(t0, t, y_filled).astype(float)
            except Exception:continue
            Y.append(yi)
        if not Y:return (t0, np.empty((0, t0.size), dtype=float))
        return (t0, np.vstack(Y))
    def _group_stats(self, ids: List[str], mode: str, smooth_win: int, do_smooth: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        t_list: List[np.ndarray] = [];y_list: List[np.ndarray] = []
        for mid in ids:
            res = self.app.get_mouse_result(mid)
            if res is None:continue
            t, y, _ylabel = get_series_for_norm_mode(res, mode)
            if do_smooth:y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
            t_list.append(t);y_list.append(y)
        t0, Y = self._interp_to_common_grid(t_list, y_list)
        if Y.size == 0:return (t0, np.full_like(t0, np.nan, dtype=float), np.full_like(t0, np.nan, dtype=float), 0)
        mean = np.nanmean(Y, axis=0);n_eff = np.sum(np.isfinite(Y), axis=0);std = np.nanstd(Y, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):sem = std / np.sqrt(np.where(n_eff > 0, n_eff, np.nan))
        return (t0, mean, sem, int(Y.shape[0]))
    def refresh_plot(self) -> None:
        ax = self.plot.ax;ax.clear();mode = self.app.get_norm_mode();smooth_win = self.app.get_smooth_window();do_smooth = bool(self.var_smoothed.get());ylabel = 'Signal';show_individual = bool(self.var_show_individual.get());show_sem = bool(self.var_show_sem.get())
        if show_individual:
            for mid in self.group_a:
                res = self.app.get_mouse_result(mid)
                if res is None:continue
                t, y, ylabel = get_series_for_norm_mode(res, mode)
                if do_smooth:y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
                ax.plot(t, y, alpha=0.25, linewidth=1.0, label='_nolegend_')
            for mid in self.group_b:
                res = self.app.get_mouse_result(mid)
                if res is None:continue
                t, y, ylabel = get_series_for_norm_mode(res, mode)
                if do_smooth:y = get_smoothed_norm_array(res, mode, window_size=smooth_win)
                ax.plot(t, y, alpha=0.25, linewidth=1.0, linestyle='--', label='_nolegend_')
        tA, mA, sA, nA = self._group_stats(self.group_a, mode, smooth_win, do_smooth)
        if nA > 0 and tA.size:
            lineA, = ax.plot(tA, mA, linewidth=2.0, label=f'Group A (n={nA})')
            if show_sem and np.any(np.isfinite(sA)):c = lineA.get_color();ax.fill_between(tA, mA - sA, mA + sA, alpha=0.2, color=c, linewidth=0)
        tB, mB, sB, nB = self._group_stats(self.group_b, mode, smooth_win, do_smooth)
        if nB > 0 and tB.size:
            lineB, = ax.plot(tB, mB, linewidth=2.0, linestyle='--', label=f'Group B (n={nB})')
            if show_sem and np.any(np.isfinite(sB)):c = lineB.get_color();ax.fill_between(tB, mB - sB, mB + sB, alpha=0.2, color=c, linewidth=0)
        ax.set_title('Batch average (Group A vs Group B)');ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel)
        if nA > 0 or nB > 0:ax.legend(loc='best')
        self.plot.redraw()
    def _export_average(self) -> Dict[str, pd.DataFrame]:mode = self.app.get_norm_mode();smooth_win = self.app.get_smooth_window();do_smooth = bool(self.var_smoothed.get());tA, mA, sA, nA = self._group_stats(self.group_a, mode, smooth_win, do_smooth);tB, mB, sB, nB = self._group_stats(self.group_b, mode, smooth_win, do_smooth);payload: Dict[str, pd.DataFrame] = {};payload['groupA_mean_sem'] = pd.DataFrame({'t_s': tA, 'mean': mA, 'sem': sA});payload['groupB_mean_sem'] = pd.DataFrame({'t_s': tB, 'mean': mB, 'sem': sB});payload['groupA_members'] = pd.DataFrame({'mouse': [self._display.get(mid, mid) for mid in self.group_a]});payload['groupB_members'] = pd.DataFrame({'mouse': [self._display.get(mid, mid) for mid in self.group_b]});payload['meta'] = pd.DataFrame([('mode', mode), ('avg_smoothed', do_smooth), ('smooth_window', smooth_win), ('n_groupA', nA), ('n_groupB', nB)], columns=['key', 'value']);return payload
def _basename_no_ext(path: str) -> str:base = os.path.basename(str(path));name, _ext = os.path.splitext(base);name = name.strip();return name if name else base if base else 'file'
def _unique_aliases(paths: List[str]) -> List[str]:
    used: set = set();aliases: List[str] = []
    for p in paths:
        base = _basename_no_ext(p);alias = base;k = 2
        while alias in used:alias = f'{base}_{k}';k += 1
        used.add(alias);aliases.append(alias)
    return aliases
def get_series_for_norm_mode(res: ChannelResult, mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if mode == NORM_DFF:return (np.asarray(res.t_exc, dtype=float), np.asarray(res.dFF, dtype=float), 'ΔF/F')
    if mode == NORM_ZF_GLOBAL:return (np.asarray(res.t_exc, dtype=float), np.asarray(res.zF_global, dtype=float), 'zF')
    if mode == NORM_ZF_INTERVAL:return (np.asarray(res.t_exc, dtype=float), np.asarray(res.zF_interval, dtype=float), 'zF')
    return (np.asarray(res.t_exc, dtype=float), np.asarray(res.dFF, dtype=float), 'Signal')
class MainAppTk:
    def __init__(self, initial_csvs: Optional[List[str]]=None, autorun: bool=False):
        self.root = tk.Tk();self.root.title('Fiberlyse');self.root.resizable(True, True);self.root.geometry('1200x700');self.csv_paths: List[str] = []
        if initial_csvs:
            for p in list(initial_csvs):
                if p and p not in self.csv_paths:self.csv_paths.append(p)
        self.csv_paths = self.csv_paths[:8];self._results: Optional[Dict[str, ChannelResult]] = None;self._mouse_display: Dict[str, str] = {};self._mouse_order: List[str] = [];self._analysis_thread: Optional[threading.Thread] = None;self._channel_widgets: Dict[str, ChannelTabsTk] = {};self.compare_widget: Optional[BatchCompareTk] = None;self.average_widget: Optional[BatchAverageTk] = None;self._result_file_order: List[str] = [];self._file_alias_by_key: Dict[str, str] = {};self._file_path_by_key: Dict[str, str] = {};self._axis_label_fs_override: Optional[float] = None;self._graph_title_fs_override: Optional[float] = None;self._tick_label_fs_override: Optional[float] = None;top = ttk.Frame(self.root, padding=8);top.pack(side=tk.TOP, fill=tk.X);self.lbl_file = ttk.Label(top, text='');self.lbl_file.grid(row=0, column=0, sticky='w', padx=(0, 8));self.btn_add = ttk.Button(top, text='Add CSV(s)…', command=self.add_csvs);self.btn_add.grid(row=0, column=1, sticky='e', padx=(0, 6));self.btn_clear = ttk.Button(top, text='Clear files', command=self.clear_csvs);self.btn_clear.grid(row=0, column=2, sticky='e', padx=(0, 6));self.btn_run = ttk.Button(top, text='Run analysis', command=self.run_analysis);self.btn_run.grid(row=0, column=3, sticky='e');self._update_files_label();row2 = ttk.Frame(self.root, padding=(8, 0, 8, 4));row2.pack(side=tk.TOP, fill=tk.X);self.var_artifact_enabled = tk.BooleanVar(value=True);self.chk_artifact_enabled = ttk.Checkbutton(row2, text='Enable artifact remover (MAD)', variable=self.var_artifact_enabled, command=self.on_artifact_enabled_toggled);self.chk_artifact_enabled.grid(row=0, column=0, columnspan=2, sticky='w', padx=(0, 12));ttk.Label(row2, text='Factor:').grid(row=0, column=2, sticky='w');self.var_factor = tk.StringVar(value='11.9');self.spin_factor = ttk.Spinbox(row2, from_=0.1, to=1000.0, increment=0.1, textvariable=self.var_factor, width=8);self.spin_factor.grid(row=0, column=3, padx=(6, 12), sticky='w');ttk.Label(row2, text='Pad (samples):').grid(row=0, column=4, sticky='w');self.var_pad = tk.StringVar(value='1');self.spin_pad = ttk.Spinbox(row2, from_=0, to=50, increment=1, textvariable=self.var_pad, width=6);self.spin_pad.grid(row=0, column=5, padx=(6, 12), sticky='w');self.var_shared = tk.BooleanVar(value=True);self.chk_shared = ttk.Checkbutton(row2, text='Require shared artifacts', variable=self.var_shared);self.chk_shared.grid(row=0, column=6, padx=(0, 12), sticky='w');ttk.Label(row2, text='Acq FPS (Hz):').grid(row=0, column=7, sticky='w');self.var_acq_fps = tk.StringVar(value=str(DEFAULT_ACQ_FPS_HZ));self.spin_acq_fps = ttk.Spinbox(row2, from_=0.0, to=10000.0, increment=0.1, textvariable=self.var_acq_fps, width=8);self.spin_acq_fps.grid(row=0, column=8, padx=(6, 6), sticky='w');ttk.Label(row2, text='(eff = FPS/2)').grid(row=0, column=9, sticky='w');ttk.Label(row2, text='Smooth win (samples):').grid(row=0, column=10, sticky='w');self.var_smooth_win = tk.StringVar(value=str(DEFAULT_SMOOTH_WINDOW));self.spin_smooth_win = ttk.Spinbox(row2, from_=1, to=100000, increment=1, textvariable=self.var_smooth_win, width=8);self.spin_smooth_win.grid(row=0, column=11, padx=(6, 12), sticky='w');self.var_interp = tk.BooleanVar(value=DEFAULT_USE_LINEAR_INTERP);self.chk_interp = ttk.Checkbutton(row2, text='Linear interpolate holes (after artifact removal)', variable=self.var_interp, command=self.on_interp_toggled);self.chk_interp.grid(row=1, column=0, columnspan=6, sticky='w', pady=(4, 0));row3 = ttk.Frame(self.root, padding=(8, 0, 8, 8));row3.pack(side=tk.TOP, fill=tk.X);ttk.Label(row3, text='Normalization view:').grid(row=0, column=0, sticky='w');self.var_norm = tk.StringVar(value=NORM_DFF);self.cmb_norm = ttk.Combobox(row3, values=NORM_CHOICES, textvariable=self.var_norm, state='readonly', width=24);self.cmb_norm.grid(row=0, column=1, padx=(6, 12), sticky='w');self.cmb_norm.bind('<<ComboboxSelected>>', lambda _e: self.on_norm_mode_changed());self.lbl_interval = ttk.Label(row3, text='Interval (s):');self.var_interval_start = tk.StringVar(value=str(DEFAULT_ZF_INTERVAL_START_S));self.var_interval_end = tk.StringVar(value=str(DEFAULT_ZF_INTERVAL_END_S));self.spin_interval_start = ttk.Spinbox(row3, from_=-1000000000.0, to=1000000000.0, increment=1.0, textvariable=self.var_interval_start, width=10);self.lbl_interval_to = ttk.Label(row3, text='to');self.spin_interval_end = ttk.Spinbox(row3, from_=-1000000000.0, to=1000000000.0, increment=1.0, textvariable=self.var_interval_end, width=10);self.btn_apply_norm = ttk.Button(row3, text='Apply interval', command=self.apply_normalization)
        for w in [self.spin_interval_start, self.spin_interval_end]:w.bind('<Return>', lambda _e: self.apply_normalization())
        self.lbl_interval.grid(row=0, column=2, sticky='w');self.spin_interval_start.grid(row=0, column=3, padx=(6, 4), sticky='w');self.lbl_interval_to.grid(row=0, column=4, sticky='w');self.spin_interval_end.grid(row=0, column=5, padx=(4, 12), sticky='w');self.btn_apply_norm.grid(row=0, column=6, sticky='w');ttk.Label(row3, text='Axis label font:').grid(row=0, column=7, sticky='w', padx=(18, 0));self.var_axis_label_fs = tk.StringVar(value=f'{DEFAULT_AXIS_LABEL_FONTSIZE:g}');self.spin_axis_label_fs = ttk.Spinbox(row3, from_=6, to=60, increment=1, textvariable=self.var_axis_label_fs, width=6);self.spin_axis_label_fs.grid(row=0, column=8, padx=(6, 12), sticky='w');self.spin_axis_label_fs.bind('<Return>', lambda _e: self.apply_axis_label_fontsize());ttk.Label(row3, text='Graph title font:').grid(row=0, column=9, sticky='w', padx=(10, 0));self.var_graph_title_fs = tk.StringVar(value=f'{DEFAULT_GRAPH_TITLE_FONTSIZE:g}');self.spin_graph_title_fs = ttk.Spinbox(row3, from_=6, to=80, increment=1, textvariable=self.var_graph_title_fs, width=6);self.spin_graph_title_fs.grid(row=0, column=10, padx=(6, 0), sticky='w');self.spin_graph_title_fs.bind('<Return>', lambda _e: self.apply_graph_title_fontsize());ttk.Label(row3, text='Tick label font:').grid(row=0, column=11, sticky='w', padx=(10, 0));self.var_tick_label_fs = tk.StringVar(value=f'{DEFAULT_TICK_LABEL_FONTSIZE:g}');self.spin_tick_label_fs = ttk.Spinbox(row3, from_=4, to=40, increment=1, textvariable=self.var_tick_label_fs, width=6);self.spin_tick_label_fs.grid(row=0, column=12, padx=(6, 0), sticky='w');self.spin_tick_label_fs.bind('<Return>', lambda _e: self.apply_tick_label_fontsize());self.update_norm_controls_visibility();view_row = ttk.Frame(self.root, padding=(8, 0, 8, 4));view_row.pack(side=tk.TOP, fill=tk.X);view_row.columnconfigure(2, weight=1);ttk.Label(view_row, text='File:').grid(row=0, column=0, sticky='w');self.var_view_file = tk.StringVar(value='');self.cmb_view_file = ttk.Combobox(view_row, textvariable=self.var_view_file, values=[], width=16, state='disabled');self.cmb_view_file.grid(row=0, column=1, padx=(6, 12), sticky='w');self.cmb_view_file.bind('<<ComboboxSelected>>', self.on_view_file_changed);self.lbl_view_hint = ttk.Label(view_row, text='Run analysis to load the file selector.');self.lbl_view_hint.grid(row=0, column=2, sticky='w');self.outer_tabs = ttk.Notebook(self.root);self.outer_tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 6));self._mouse_frames: Dict[str, ttk.Frame] = {};self._frame_to_mid: Dict[str, str] = {};self.outer_tabs.bind('<<NotebookTabChanged>>', self._on_outer_tab_changed);self.status = tk.StringVar(value='Ready.');status_bar = ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor='w');status_bar.pack(side=tk.BOTTOM, fill=tk.X);self._sync_artifact_controls_state();self._install_time_marker_hotkeys();self._install_file_map_hotkeys()
        if self.csv_paths and autorun:self.root.after(0, self.run_analysis)
    def _install_file_map_hotkeys(self) -> None:self.root.bind_all('<Control-j>', self._on_ctrl_j, add='+');self.root.bind_all('<Control-J>', self._on_ctrl_j, add='+')
    def _on_ctrl_j(self, _event=None) -> None:
        try:self._show_file_number_map_dialog()
        except Exception as e:
            try:print(f'_on_ctrl_j error: {e}', file=sys.stderr)
            except Exception:pass
            try:self.status.set(f'Error showing file map: {e}')
            except Exception:pass
    def _show_file_number_map_dialog(self) -> None:
        paths = list(self.csv_paths)[:8]
        if not paths:messagebox.showinfo('File number mapping (Ctrl+J)', "No CSV files selected.\n\nUse 'Add CSV(s)…' first.");return
        top = tk.Toplevel(self.root);top.title('File number mapping (Ctrl+J)');top.transient(self.root);top.grab_set();frm = ttk.Frame(top, padding=10);frm.pack(fill=tk.BOTH, expand=True);ttk.Label(frm, text='These are the names used in the file drop-down:', justify='left').pack(anchor='w');txt_frame = ttk.Frame(frm);txt_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8));txt = tk.Text(txt_frame, width=90, height=min(18, 3 + len(paths) * 2), wrap='none');sb = ttk.Scrollbar(txt_frame, orient='vertical', command=txt.yview);txt.configure(yscrollcommand=sb.set);txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True);sb.pack(side=tk.RIGHT, fill=tk.Y);lines = []
        for i, p in enumerate(paths, start=1):display_name = f'File {i}';basename = os.path.basename(p);lines.append(f'{display_name}: {basename}\n    {p}\n')
        txt.insert('1.0', '\n'.join(lines).rstrip() + '\n');txt.configure(state='disabled');btn_row = ttk.Frame(frm);btn_row.pack(fill=tk.X)
        def copy_to_clipboard():
            try:self.root.clipboard_clear();self.root.clipboard_append('\n'.join(lines).rstrip() + '\n')
            except Exception:pass
        ttk.Button(btn_row, text='Copy', command=copy_to_clipboard).pack(side=tk.LEFT);ttk.Button(btn_row, text='Close', command=top.destroy).pack(side=tk.RIGHT);self.root.wait_window(top)
    @staticmethod
    def _mouse_file_key(mouse_id: str) -> str:return str(mouse_id).split(':', 1)[0]
    @staticmethod
    def _mouse_channel_label(mouse_id: str) -> str:parts = str(mouse_id).split(':', 1);return parts[1] if len(parts) == 2 and parts[1] else str(mouse_id)
    def _set_result_file_choices(self, file_meta: List[Tuple[str, str, str]]) -> None:
        self._result_file_order = [file_key for file_key, _alias, _path in file_meta];self._file_alias_by_key = {file_key: alias for file_key, alias, _path in file_meta};self._file_path_by_key = {file_key: path for file_key, _alias, path in file_meta};values = [self._file_alias_by_key[file_key] for file_key in self._result_file_order]
        if not values:self.var_view_file.set('');self.cmb_view_file.config(values=(), state='disabled');self.lbl_view_hint.config(text='Run analysis to load the file selector.');return
        current = (self.var_view_file.get() or '').strip()
        if current not in values:current = values[0]
        self.cmb_view_file.config(values=tuple(values), state='readonly');self.var_view_file.set(current)
        if len(values) == 1:self.lbl_view_hint.config(text="Showing the selected file's G channels below.")
        else:self.lbl_view_hint.config(text='Pick a file to show its G channels below.')
    def _get_selected_view_file_key(self) -> Optional[str]:
        if not self._result_file_order:return None
        selected_alias = (self.var_view_file.get() or '').strip()
        for file_key in self._result_file_order:
            if self._file_alias_by_key.get(file_key) == selected_alias:return file_key
        fallback = self._result_file_order[0];self.var_view_file.set(self._file_alias_by_key.get(fallback, ''));return fallback
    def on_view_file_changed(self, _event=None) -> None:
        if not self._results:return
        self._refresh_visible_tabs();file_key = self._get_selected_view_file_key();alias = self._file_alias_by_key.get(file_key, '') if file_key else ''
        if alias:self.status.set(f'Viewing file: {alias}.')
    def _update_files_label(self):
        if not self.csv_paths:self.lbl_file.config(text='No CSV files selected.');return
        if len(self.csv_paths) == 1:self.lbl_file.config(text=self.csv_paths[0]);return
        first = os.path.basename(self.csv_paths[0]);self.lbl_file.config(text=f'{len(self.csv_paths)} CSV files selected (first: {first})')
    def add_csvs(self):
        paths = filedialog.askopenfilenames(title='Select CSV file(s)', filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not paths:return
        for p in paths:
            if p and p not in self.csv_paths:self.csv_paths.append(p)
        if len(self.csv_paths) > 8:messagebox.showwarning('Too many files', 'You can analyze up to 8 CSV files at a time.\n\nOnly the first 8 will be kept.');self.csv_paths = self.csv_paths[:8]
        self._update_files_label();self.status.set(f'Selected {len(self.csv_paths)} file(s).')
    def clear_csvs(self):self.csv_paths = [];self._update_files_label();self.status.set('Cleared file list.')
    def get_mouse_items(self) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for mid in list(self._mouse_order):items.append((mid, self._mouse_display.get(mid, mid)))
        return items
    def get_mouse_result(self, mouse_id: str) -> Optional[ChannelResult]:
        if not self._results:return None
        return self._results.get(mouse_id)
    def get_mouse_display(self, mouse_id: str) -> str:return self._mouse_display.get(mouse_id, mouse_id)
    def get_norm_mode(self) -> str:return self._read_norm_mode()
    def get_smooth_window(self) -> int:
        try:return max(1, int(float(self.var_smooth_win.get())))
        except Exception:return DEFAULT_SMOOTH_WINDOW
    def _install_time_marker_hotkeys(self) -> None:self.root.bind_all('<Control-i>', self._on_ctrl_i, add='+');self.root.bind_all('<Control-I>', self._on_ctrl_i, add='+')
    def _on_ctrl_i(self, _event=None) -> None:
        tab = self.find_active_plot_tab()
        if tab is None:return
        self._show_time_marker_dialog(tab)
    def _show_time_marker_dialog(self, tab: 'PlotTabTk') -> None:
        top = tk.Toplevel(self.root);top.title('Time marker (Ctrl+I)');top.transient(self.root);top.grab_set();time_var = tk.StringVar(value=getattr(self, '_last_marker_time_str', ''));info = 'Add a vertical time marker to the *currently active* plot.\n\n• To ADD: type a time in seconds and press Enter / OK.\n• To REMOVE the last marker: press Ctrl + Backspace.\n\nTip: marker lines appear in the legend, so you can rename them by double-clicking.';frm = ttk.Frame(top, padding=10);frm.pack(fill=tk.BOTH, expand=True);ttk.Label(frm, text=info, justify='left').pack(anchor='w');ttk.Label(frm, text='Time (seconds):').pack(anchor='w', pady=(10, 0));entry = ttk.Entry(frm, textvariable=time_var, width=28);entry.pack(anchor='w', pady=(2, 8));entry.focus_set();btn_row = ttk.Frame(frm);btn_row.pack(fill=tk.X, pady=(4, 0))
        def close_dialog() -> None:
            try:top.grab_release()
            except Exception:pass
            top.destroy()
        def do_add(_e=None) -> str:
            s = (time_var.get() or '').strip()
            if not s:messagebox.showerror('Missing time', 'Please type a time in seconds (for example: 12.5).');return 'break'
            try:t = float(s)
            except Exception:messagebox.showerror('Invalid time', f"Could not read '{s}' as a number of seconds.");return 'break'
            if not np.isfinite(t):messagebox.showerror('Invalid time', 'Time must be a finite number.');return 'break'
            self._last_marker_time_str = s;tab.add_time_marker(t);close_dialog();return 'break'
        def do_remove(_e=None) -> str:tab.remove_last_time_marker();close_dialog();return 'break'
        def do_cancel(_e=None) -> str:close_dialog();return 'break'
        ttk.Button(btn_row, text='OK (add)', command=do_add).pack(side=tk.RIGHT);ttk.Button(btn_row, text='Cancel', command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8));top.bind('<Return>', do_add);top.bind('<Escape>', do_cancel);top.bind('<Control-BackSpace>', do_remove);entry.bind('<Control-BackSpace>', do_remove);self.root.wait_window(top)
    def _sync_artifact_controls_state(self):
        enabled = bool(self.var_artifact_enabled.get());self.spin_factor.config(state='normal' if enabled else 'disabled');self.spin_pad.config(state='normal' if enabled else 'disabled')
        if enabled:self.chk_shared.state(['!disabled'])
        else:self.chk_shared.state(['disabled'])
    def _read_norm_mode(self) -> str:mode = (self.var_norm.get() or NORM_DFF).strip();return mode if mode in NORM_CHOICES else NORM_DFF
    def update_norm_controls_visibility(self):
        mode = self._read_norm_mode();show_interval = mode == NORM_ZF_INTERVAL
        if show_interval:self.lbl_interval.grid();self.spin_interval_start.grid();self.lbl_interval_to.grid();self.spin_interval_end.grid();self.btn_apply_norm.grid()
        else:self.lbl_interval.grid_remove();self.spin_interval_start.grid_remove();self.lbl_interval_to.grid_remove();self.spin_interval_end.grid_remove();self.btn_apply_norm.grid_remove()
    def on_norm_mode_changed(self):self.update_norm_controls_visibility();self.apply_normalization()
    def apply_axis_label_fontsize(self):
        try:fs = float(self.var_axis_label_fs.get())
        except Exception as e:messagebox.showerror('Invalid font size', f'Could not parse axis label font size:\n\n{e}');return
        if not np.isfinite(fs) or fs <= 0:messagebox.showerror('Invalid font size', 'Axis label font size must be a positive number.');return
        self._axis_label_fs_override = fs
        for widget in self._channel_widgets.values():widget.set_axis_label_fontsize(fs)
        if self.compare_widget is not None:self.compare_widget.set_axis_label_fontsize(fs)
        if self.average_widget is not None:self.average_widget.set_axis_label_fontsize(fs)
        self.status.set(f'Applied axis label font size = {fs:g} (x/y labels).')
    def apply_graph_title_fontsize(self):
        try:fs = float(self.var_graph_title_fs.get())
        except Exception as e:messagebox.showerror('Invalid font size', f'Could not parse graph title font size:\n\n{e}');return
        if not np.isfinite(fs) or fs <= 0:messagebox.showerror('Invalid font size', 'Graph title font size must be a positive number.');return
        self._graph_title_fs_override = fs
        for widget in self._channel_widgets.values():widget.set_graph_title_fontsize(fs)
        if self.compare_widget is not None:self.compare_widget.set_graph_title_fontsize(fs)
        if self.average_widget is not None:self.average_widget.set_graph_title_fontsize(fs)
        self.status.set(f'Applied graph title font size = {fs:g} (plot titles).')
    def apply_tick_label_fontsize(self):
        try:fs = float(self.var_tick_label_fs.get())
        except Exception as e:messagebox.showerror('Invalid font size', f'Could not parse tick label font size:\n\n{e}');return
        if not np.isfinite(fs) or fs <= 0:messagebox.showerror('Invalid font size', 'Tick label font size must be a positive number.');return
        self._tick_label_fs_override = fs
        for widget in self._channel_widgets.values():widget.set_tick_label_fontsize(fs)
        if self.compare_widget is not None:self.compare_widget.set_tick_label_fontsize(fs)
        if self.average_widget is not None:self.average_widget.set_tick_label_fontsize(fs)
        self.status.set(f'Applied tick label font size = {fs:g} (axis numbers).')
    def on_artifact_enabled_toggled(self):
        enabled = bool(self.var_artifact_enabled.get());self._sync_artifact_controls_state()
        if not self._results:self.status.set(f"Artifact remover {('ENABLED' if enabled else 'DISABLED')} (will apply on next Run).");return
        try:artifact_factor = float(self.var_factor.get());artifact_pad = int(float(self.var_pad.get()));require_shared = bool(self.var_shared.get());align_mode = DEFAULT_ALIGN_MODE;use_interp = bool(self.var_interp.get())
        except Exception as e:messagebox.showerror('Invalid settings', f'Could not parse artifact settings:\n\n{e}');return
        self.status.set('Updating artifact pipeline…')
        def worker():
            try:
                for res in self._results.values():recompute_artifact_pipeline_inplace(res, artifact_enabled=enabled, artifact_factor=artifact_factor, artifact_pad=artifact_pad, require_shared=require_shared, align_mode=align_mode, use_linear_interp=use_interp)
                self.root.after(0, self._after_artifact_pipeline_updated)
            except Exception as e:self.root.after(0, lambda: self.on_analysis_failed(str(e)))
        threading.Thread(target=worker, daemon=True).start()
    def _after_artifact_pipeline_updated(self):
        for widget in self._channel_widgets.values():widget.refresh_after_pipeline_change()
        if self.compare_widget is not None:self.compare_widget.refresh_plot()
        if self.average_widget is not None:self.average_widget.refresh_plot()
        self.status.set(f"Artifact remover {('ENABLED' if self.var_artifact_enabled.get() else 'DISABLED')} (MAD). Interp={('ON' if self.var_interp.get() else 'OFF')}.")
    def on_interp_toggled(self):
        if not self._results:return
        use_interp = bool(self.var_interp.get())
        for res in self._results.values():set_interpolation_mode(res, use_interp)
        for widget in self._channel_widgets.values():widget.refresh_after_pipeline_change()
        if self.compare_widget is not None:self.compare_widget.refresh_plot()
        if self.average_widget is not None:self.average_widget.refresh_plot()
        self.status.set(f"Interpolation {('ON' if use_interp else 'OFF')} (frequency analysis still uses NO interpolation).")
    def apply_normalization(self):
        mode = self._read_norm_mode()
        try:smooth_win = max(1, int(float(self.var_smooth_win.get())))
        except Exception:smooth_win = DEFAULT_SMOOTH_WINDOW
        interval_start: Optional[float] = None;interval_end: Optional[float] = None
        if mode == NORM_ZF_INTERVAL:
            try:interval_start = float(self.var_interval_start.get());interval_end = float(self.var_interval_end.get())
            except Exception as e:messagebox.showerror('Invalid interval', f'Could not parse interval start/end:\n\n{e}');return
        if not self._results:
            if mode == NORM_DFF:self.status.set('Normalization view: ΔF/F')
            elif mode == NORM_ZF_GLOBAL:self.status.set('Normalization view: zF (global, GUI)')
            elif mode == NORM_ZF_INTERVAL:self.status.set('Normalization view: zF - interval based (set interval and apply)')
            return
        for res in self._results.values():
            res.smooth_window = smooth_win
            if mode == NORM_ZF_INTERVAL and interval_start is not None and (interval_end is not None):res.zf_interval_start_s = interval_start;res.zf_interval_end_s = interval_end
            recompute_normalizations(res)
        for widget in self._channel_widgets.values():widget.set_norm_mode(mode)
        if self.compare_widget is not None:self.compare_widget.refresh_plot()
        if self.average_widget is not None:self.average_widget.refresh_plot()
        if mode == NORM_DFF:self.status.set('Normalization view: ΔF/F.')
        elif mode == NORM_ZF_GLOBAL:self.status.set('Normalization view: zF (global, GUI).')
        elif mode == NORM_ZF_INTERVAL:a = float(interval_start);b = float(interval_end);self.status.set(f'Normalization view: zF - interval based. Interval = {min(a, b):g}–{max(a, b):g} s')
    def run_analysis(self):
        if not self.csv_paths:messagebox.showwarning('No files', 'Please add one or more CSV files first.');return
        try:artifact_enabled = bool(self.var_artifact_enabled.get());artifact_factor = float(self.var_factor.get());artifact_pad = int(float(self.var_pad.get()));require_shared = bool(self.var_shared.get());align_mode = DEFAULT_ALIGN_MODE;acq_fps = float(self.var_acq_fps.get());smooth_win = max(1, int(float(self.var_smooth_win.get())));mode = self._read_norm_mode();interval_start = float(self.var_interval_start.get());interval_end = float(self.var_interval_end.get());use_interp = bool(self.var_interp.get())
        except Exception as e:messagebox.showerror('Invalid settings', f'Could not parse settings:\n\n{e}');return
        fit_windows = list(DEFAULT_FIT_WINDOWS);paths = list(self.csv_paths)[:8];file_labels = [f'File {i}' for i in range(1, len(paths) + 1)];file_meta = [(f'F{i}', label, path) for i, (path, label) in enumerate(zip(paths, file_labels), start=1)];self.btn_run.config(state=tk.DISABLED);self.btn_add.config(state=tk.DISABLED);self.btn_clear.config(state=tk.DISABLED);self.status.set('Analyzing…')
        def worker():
            try:
                aggregated: Dict[str, ChannelResult] = {};display: Dict[str, str] = {};order: List[str] = []
                for i, (file_key, alias, path) in enumerate(file_meta, start=1):
                    self.root.after(0, lambda i=i, alias=alias: self.status.set(f'Analyzing {i}/{len(paths)}: {alias}…'));per_file = analyze_csv(path, artifact_enabled=artifact_enabled, artifact_factor=artifact_factor, artifact_method='mad', artifact_pad=artifact_pad, require_shared=require_shared, align_mode=align_mode, fit_windows=fit_windows, acq_fps_hz=acq_fps if np.isfinite(acq_fps) and acq_fps > 0 else None, smooth_window=smooth_win, zf_interval_start_s=interval_start, zf_interval_end_s=interval_end, use_linear_interp=use_interp)
                    for gcol in sorted(per_file.keys()):mid = f'{file_key}:{gcol}';aggregated[mid] = per_file[gcol];display[mid] = f'{alias}:{gcol}';order.append(mid)
                self.root.after(0, lambda: self.on_analysis_finished(aggregated, display, order, mode, file_meta))
            except Exception as e:self.root.after(0, lambda: self.on_analysis_failed(str(e)))
        self._analysis_thread = threading.Thread(target=worker, daemon=True);self._analysis_thread.start()
    def on_analysis_finished(self, results: Dict[str, ChannelResult], display: Dict[str, str], order: List[str], mode: str, file_meta: List[Tuple[str, str, str]]):
        self._results = results;self._mouse_display = dict(display);self._mouse_order = list(order);self._set_result_file_choices(file_meta);self.build_tabs(norm_mode=mode)
        try:any_key = next(iter(results.keys()));r0 = results[any_key];self.status.set(f"Done. Files={len(file_meta)} | Mice={len(results)} | Mode={mode}. Smooth win={r0.smooth_window}. Artifacts={('ON' if self.var_artifact_enabled.get() else 'OFF')}. Interp={('ON' if r0.use_interpolation else 'OFF')}. Freq eff fs={r0.eff_fs_hz:.3g} Hz (Acq FPS={r0.acq_fps_hz:.3g}).")
        except Exception:self.status.set('Done.')
        self.btn_run.config(state=tk.NORMAL);self.btn_add.config(state=tk.NORMAL);self.btn_clear.config(state=tk.NORMAL);self.on_norm_mode_changed()
    def on_analysis_failed(self, msg: str):messagebox.showerror('Analysis failed', msg);self.status.set('Analysis failed.');self.btn_run.config(state=tk.NORMAL);self.btn_add.config(state=tk.NORMAL);self.btn_clear.config(state=tk.NORMAL)
    def build_tabs(self, norm_mode: str):
        outer_tabs = self.outer_tabs
        for tab_id in outer_tabs.tabs():outer_tabs.forget(tab_id)
        for frame in list(getattr(self, '_mouse_frames', {}).values()):
            try:frame.destroy()
            except Exception:pass
        if self.compare_widget is not None:
            try:self.compare_widget.destroy()
            except Exception:pass
        if self.average_widget is not None:
            try:self.average_widget.destroy()
            except Exception:pass
        self._channel_widgets.clear();self._mouse_frames = {};self._frame_to_mid = {};self.compare_widget = None;self.average_widget = None
        if not self._results:return
        mids = [mid for mid in self._mouse_order if mid in self._results]
        for mid in mids:frame = ttk.Frame(outer_tabs);display_label = self._mouse_display.get(mid, mid);ttk.Label(frame, text=f'Click this tab to load plots for {display_label} (lazy-loaded for speed).', foreground='#666666').pack(anchor='w', padx=10, pady=10);self._mouse_frames[mid] = frame
        self._refresh_visible_tabs()
    def _refresh_visible_tabs(self) -> None:
        outer_tabs = self.outer_tabs
        for tab_id in outer_tabs.tabs():outer_tabs.forget(tab_id)
        self._frame_to_mid = {};file_key = self._get_selected_view_file_key();mids: List[str] = []
        for mid in self._mouse_order:
            if mid not in self._results:continue
            if file_key is not None and self._mouse_file_key(mid) != file_key:continue
            mids.append(mid)
        for mid in mids:
            frame = self._mouse_frames.get(mid)
            if frame is None:continue
            tab_label = self._mouse_channel_label(mid) if file_key is not None else self._mouse_display.get(mid, mid);outer_tabs.add(frame, text=tab_label);self._frame_to_mid[str(frame)] = mid
        if mids:outer_tabs.select(self._mouse_frames[mids[0]]);self._ensure_mouse_widget(mids[0])
    def _on_outer_tab_changed(self, _event=None) -> None:
        sel = self.outer_tabs.select()
        if not sel:return
        mid = self._frame_to_mid.get(sel)
        if mid:self._ensure_mouse_widget(mid)
    def _ensure_mouse_widget(self, mid: str) -> None:
        if mid in self._channel_widgets:return
        frame = self._mouse_frames.get(mid)
        if frame is None:return
        for child in frame.winfo_children():child.destroy()
        res = self._results[mid];widget = ChannelTabsTk(frame, res, norm_mode=self._read_norm_mode(), parent_app=self);widget.pack(fill=tk.BOTH, expand=True)
        if self._axis_label_fs_override is not None:widget.set_axis_label_fontsize(self._axis_label_fs_override)
        if self._graph_title_fs_override is not None:widget.set_graph_title_fontsize(self._graph_title_fs_override)
        if self._tick_label_fs_override is not None:widget.set_tick_label_fontsize(self._tick_label_fs_override)
        self._channel_widgets[mid] = widget
    def find_active_plot_tab(self) -> Optional[PlotTabTk]:
        sel = self.outer_tabs.select()
        if not sel:return None
        mid = self._frame_to_mid.get(sel)
        if not mid:return None
        self._ensure_mouse_widget(mid);cw = self._channel_widgets.get(mid)
        if cw is None:return None
        return cw.get_active_plot_tab()
def parse_cli():parser = argparse.ArgumentParser(description='Fiberlyse GUI (batch + markers)');parser.add_argument('--csv', nargs='*', default=[], help='Optional CSV file(s) to load automatically (space-separated).');return parser.parse_args()
def main():
    args = parse_cli();app = MainAppTk()
    if getattr(args, 'csv', None):
        app.csv_paths = list(args.csv)
        try:app._update_files_label()
        except Exception:pass
        try:app.run_analysis()
        except Exception:pass
    app.root.mainloop()
def _fiberlyse_auc_insert_zero_crossings(x, y):
    x = np.asarray(x, dtype=float);y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:return (x, y)
    xs = [float(x[0])];ys = [float(y[0])]
    for k in range(1, x.size):
        x0 = float(x[k - 1]);x1 = float(x[k]);y0 = float(y[k - 1]);y1 = float(y[k])
        if np.isfinite(y0) and np.isfinite(y1) and (y0 != 0.0) and (y1 != 0.0):
            if y0 < 0.0 < y1 or y1 < 0.0 < y0:
                denom = y1 - y0
                if abs(denom) > 1e-15:
                    frac = -y0 / denom;xc = x0 + frac * (x1 - x0)
                    if min(x0, x1) < xc < max(x0, x1):xs.append(float(xc));ys.append(0.0)
        xs.append(x1);ys.append(y1)
    return (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
def _fiberlyse_auc_stats_for_xy(x, y, start_x, end_x, baseline_y=0.0):
    try:lo = float(min(start_x, end_x));hi = float(max(start_x, end_x));baseline = float(baseline_y)
    except Exception:return None
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:return None
    if not np.isfinite(baseline):baseline = 0.0
    x = np.asarray(x, dtype=float).reshape(-1);y = np.asarray(y, dtype=float).reshape(-1);n = min(x.size, y.size)
    if n < 2:return None
    x = x[:n];y = y[:n];finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:return None
    trapz = getattr(np, 'trapezoid', None)
    if trapz is None:trapz = np.trapz
    signed_auc = 0.0;abs_auc = 0.0;positive_auc = 0.0;negative_auc = 0.0;coverage_x = 0.0;n_points_used = 0
    for i0, i1 in _contiguous_true_runs(finite):
        xs = x[i0:i1 + 1].astype(float);ys = y[i0:i1 + 1].astype(float)
        if xs.size < 2:continue
        if np.any(np.diff(xs) < 0):order = np.argsort(xs);xs = xs[order];ys = ys[order]
        unique_x, unique_idx = np.unique(xs, return_index=True);xs = unique_x.astype(float);ys = ys[unique_idx].astype(float)
        if xs.size < 2:continue
        left = max(lo, float(xs[0]));right = min(hi, float(xs[-1]))
        if right <= left:continue
        inside = (xs >= left) & (xs <= right);xi = xs[inside].astype(float);yi = ys[inside].astype(float)
        if xi.size == 0:xi = np.array([left, right], dtype=float);yi = np.interp(xi, xs, ys).astype(float)
        else:
            if not np.isclose(float(xi[0]), left, rtol=0.0, atol=1e-12):xi = np.insert(xi, 0, left);yi = np.insert(yi, 0, float(np.interp(left, xs, ys)))
            if not np.isclose(float(xi[-1]), right, rtol=0.0, atol=1e-12):xi = np.append(xi, right);yi = np.append(yi, float(np.interp(right, xs, ys)))
        if xi.size < 2:continue
        yb = yi - baseline;xi_z, yb_z = _fiberlyse_auc_insert_zero_crossings(xi, yb);signed_auc += float(trapz(yb_z, xi_z));abs_auc += float(trapz(np.abs(yb_z), xi_z));positive_auc += float(trapz(np.maximum(yb_z, 0.0), xi_z));negative_auc += float(trapz(np.minimum(yb_z, 0.0), xi_z));coverage_x += float(right - left);n_points_used += int(xi_z.size)
    if coverage_x <= 0.0 or n_points_used < 2:return None
    requested_x = hi - lo;coverage_percent = 100.0 * coverage_x / requested_x if requested_x > 0 else np.nan;mean_minus_baseline = signed_auc / coverage_x if coverage_x > 0 else np.nan;return {'signed_auc': float(signed_auc), 'abs_auc': float(abs_auc), 'positive_auc': float(positive_auc), 'negative_auc': float(negative_auc), 'coverage_x': float(coverage_x), 'coverage_percent': float(coverage_percent), 'mean_minus_baseline': float(mean_minus_baseline), 'n_points_used': float(n_points_used)}
def _fiberlyse_plot_calculate_auc_for_interval(self, start_x, end_x, baseline_y=0.0):
    rows = []
    try:self._apply_user_overrides()
    except Exception:pass
    for ai, ax in enumerate(list(getattr(self.fig, 'axes', [])), start=1):
        axis_title = str(ax.get_title() or '').strip()
        if not axis_title:axis_title = f'Axis {ai}'
        for line in list(getattr(ax, 'lines', [])):
            try:
                if not line.get_visible():continue
            except Exception:continue
            label = str(line.get_label() or '').strip()
            if not label or label.startswith('_'):continue
            try:x = np.asarray(line.get_xdata(), dtype=float);y = np.asarray(line.get_ydata(), dtype=float)
            except Exception:continue
            stats = _fiberlyse_auc_stats_for_xy(x=x, y=y, start_x=start_x, end_x=end_x, baseline_y=baseline_y)
            if stats is None:continue
            row = {'axis': axis_title, 'trace': label};row.update(stats);rows.append(row)
    return rows
def _fiberlyse_auc_fmt(v):
    try:vf = float(v)
    except Exception:return 'n/a'
    if not np.isfinite(vf):return 'n/a'
    return f'{vf:.8g}'
def _fiberlyse_plot_format_auc_results_text(self, rows, start_x, end_x, baseline_y):
    lo = float(min(start_x, end_x));hi = float(max(start_x, end_x));lines = [f'AUC results for plot: {self.tab_name}', f'Interval on x-axis: {lo:g} to {hi:g}', f'Baseline: B = {baseline_y:g}', '', 'PLAIN-LANGUAGE EXPLANATION', '', 'Signed AUC:', 'Net area after baseline subtraction. Area above baseline is positive; area below baseline is negative. Positive and negative parts can cancel each other out.', '', 'Absolute AUC:', 'Total area away from baseline, regardless of direction. Both increases and decreases count as positive area.', '', 'Positive AUC:', 'Only the area above baseline. Any part of the signal below baseline is treated as zero.', '', 'Negative AUC:', 'Only the area below baseline. This value is usually negative.', '', 'Coverage:', 'The amount of your selected x/time interval that had usable finite data. NaN holes or missing regions are not integrated across.', '', 'Coverage %:', 'The percentage of the requested interval that actually had usable data.', '', 'Mean - baseline:', 'The average baseline-corrected signal over the usable part of the selected interval.', '', 'Points used:', 'The number of x/y points used for integration, including inserted interval-boundary points and zero-crossing points.', '', 'MATHEMATICAL FORMULAS', '', 'Notation:', 'B = baseline', 'yᵦ(x) = y(x) - B', 'U = usable parts of [x_start, x_end] after removing NaN/missing segments', 'Δxᵢ = xᵢ₊₁ - xᵢ', '', 'Trapezoidal rule used by the calculator:', '∫ f(x) dx ≈ Σᵢ ½ · [f(xᵢ) + f(xᵢ₊₁)] · Δxᵢ', '', 'Signed AUC:', 'A_signed ≈ Σᵢ ½ · [yᵦᵢ + yᵦᵢ₊₁] · Δxᵢ', '', 'Absolute AUC:', 'A_abs ≈ Σᵢ ½ · [|yᵦᵢ| + |yᵦᵢ₊₁|] · Δxᵢ', '', 'Positive AUC:', 'A_pos ≈ Σᵢ ½ · [max(yᵦᵢ, 0) + max(yᵦᵢ₊₁, 0)] · Δxᵢ', '', 'Negative AUC:', 'A_neg ≈ Σᵢ ½ · [min(yᵦᵢ, 0) + min(yᵦᵢ₊₁, 0)] · Δxᵢ', '', 'Coverage:', 'C = Σₖ (bₖ - aₖ)', 'where [aₖ, bₖ] are the usable finite data segments inside the selected interval.', '', 'Coverage %:', 'C_% = 100 · C / (x_end - x_start)', '', 'Mean - baseline:', 'μᵦ = A_signed / C', '', 'On time plots, x is time in seconds, so AUC units are y-units · seconds.', '', '\t'.join(['Axis', 'Trace', 'Signed AUC', 'Absolute AUC', 'Positive AUC', 'Negative AUC', 'Coverage', 'Coverage %', 'Mean - baseline', 'Points used'])]
    for r in rows:lines.append('\t'.join([str(r.get('axis', '')), str(r.get('trace', '')), _fiberlyse_auc_fmt(r.get('signed_auc')), _fiberlyse_auc_fmt(r.get('abs_auc')), _fiberlyse_auc_fmt(r.get('positive_auc')), _fiberlyse_auc_fmt(r.get('negative_auc')), _fiberlyse_auc_fmt(r.get('coverage_x')), _fiberlyse_auc_fmt(r.get('coverage_percent')), _fiberlyse_auc_fmt(r.get('mean_minus_baseline')), _fiberlyse_auc_fmt(r.get('n_points_used'))]))
    return '\n'.join(lines)
def _fiberlyse_plot_default_auc_interval_strings(self):
    ax_default = None
    try:
        for ax in list(getattr(self.fig, 'axes', [])):
            if self._axis_looks_like_time(ax):ax_default = ax;break
    except Exception:ax_default = None
    if ax_default is None:
        try:ax_default = self.fig.axes[0] if self.fig.axes else None
        except Exception:ax_default = None
    if ax_default is None:return ('0', '1')
    try:
        x0, x1 = ax_default.get_xlim();lo = min(float(x0), float(x1));hi = max(float(x0), float(x1))
        if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):return (f'{lo:g}', f'{hi:g}')
    except Exception:pass
    return ('0', '1')
def _fiberlyse_tree_sort(tree, col, reverse=False):
    data = []
    for item in tree.get_children(''):
        val = tree.set(item, col)
        try:key = float(val)
        except Exception:key = str(val).lower()
        data.append((key, item))
    data.sort(reverse=bool(reverse))
    for index, (_key, item) in enumerate(data):tree.move(item, '', index)
    tree.heading(col, command=lambda: _fiberlyse_tree_sort(tree, col, not reverse))
def _fiberlyse_plot_show_auc_results_window(self, rows, start_x, end_x, baseline_y):
    parent = self.winfo_toplevel();text_out = self._format_auc_results_text(rows, start_x, end_x, baseline_y);lo = float(min(start_x, end_x));hi = float(max(start_x, end_x));requested = hi - lo;top = tk.Toplevel(parent);top.title('AUC results');top.transient(parent);top.geometry('1160x840');main = ttk.Frame(top, padding=10);main.pack(fill=tk.BOTH, expand=True);main.rowconfigure(1, weight=1);main.rowconfigure(2, weight=0);main.columnconfigure(0, weight=1);summary = f'Plot: {self.tab_name} | Interval: {lo:g} to {hi:g} (requested width {requested:g}) | Baseline: B = {baseline_y:g}';ttk.Label(main, text=summary, justify='left').grid(row=0, column=0, sticky='w');table_frame = ttk.LabelFrame(main, text='AUC table');table_frame.grid(row=1, column=0, sticky='nsew', pady=(8, 8));table_frame.rowconfigure(0, weight=1);table_frame.columnconfigure(0, weight=1);columns = ['axis', 'trace', 'signed_auc', 'abs_auc', 'positive_auc', 'negative_auc', 'coverage_x', 'coverage_percent', 'mean_minus_baseline', 'n_points_used'];headings = {'axis': 'Axis', 'trace': 'Trace', 'signed_auc': 'Signed AUC', 'abs_auc': 'Absolute AUC', 'positive_auc': 'Positive AUC', 'negative_auc': 'Negative AUC', 'coverage_x': 'Coverage', 'coverage_percent': 'Coverage %', 'mean_minus_baseline': 'Mean - baseline', 'n_points_used': 'Points used'};widths = {'axis': 220, 'trace': 190, 'signed_auc': 105, 'abs_auc': 105, 'positive_auc': 105, 'negative_auc': 105, 'coverage_x': 90, 'coverage_percent': 90, 'mean_minus_baseline': 120, 'n_points_used': 90};tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12);ysb = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview);xsb = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview);tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set);tree.grid(row=0, column=0, sticky='nsew');ysb.grid(row=0, column=1, sticky='ns');xsb.grid(row=1, column=0, sticky='ew')
    for col in columns:tree.heading(col, text=headings[col], command=lambda c=col: _fiberlyse_tree_sort(tree, c, False));tree.column(col, width=widths.get(col, 100), minwidth=60, stretch=col in ['axis', 'trace'])
    for r in rows:
        vals = []
        for col in columns:
            if col in ['axis', 'trace']:vals.append(str(r.get(col, '')))
            else:vals.append(_fiberlyse_auc_fmt(r.get(col)))
        tree.insert('', tk.END, values=vals)
    explanation_frame = ttk.LabelFrame(main, text='Explanation and formulas');explanation_frame.grid(row=2, column=0, sticky='ew', pady=(0, 8));explanation_frame.rowconfigure(0, weight=1);explanation_frame.columnconfigure(0, weight=1);explanation = 'PLAIN-LANGUAGE EXPLANATION\n\nSigned AUC:\nNet area after baseline subtraction. Area above baseline is positive; area below baseline is negative. Positive and negative parts can cancel each other out.\n\nAbsolute AUC:\nTotal area away from baseline, regardless of direction. Both increases and decreases count as positive area.\n\nPositive AUC:\nOnly the area above baseline. Any part of the signal below baseline is treated as zero.\n\nNegative AUC:\nOnly the area below baseline. This value is usually negative.\n\nCoverage:\nThe amount of your selected x/time interval that had usable finite data. NaN holes or missing regions are not integrated across.\n\nCoverage %:\nThe percentage of the requested interval that actually had usable data.\n\nMean - baseline:\nThe average baseline-corrected signal over the usable part of the selected interval.\n\nPoints used:\nThe number of x/y points used for integration, including inserted interval-boundary points and zero-crossing points.\n\nMATHEMATICAL FORMULAS\n\nNotation:\nB = baseline\nyᵦ(x) = y(x) - B\nU = usable parts of [x_start, x_end] after removing NaN/missing segments\nΔxᵢ = xᵢ₊₁ - xᵢ\n\nTrapezoidal rule:\n∫ f(x) dx ≈ Σᵢ ½ · [f(xᵢ) + f(xᵢ₊₁)] · Δxᵢ\n\nSigned AUC:\nA_signed ≈ Σᵢ ½ · [yᵦᵢ + yᵦᵢ₊₁] · Δxᵢ\n\nAbsolute AUC:\nA_abs ≈ Σᵢ ½ · [|yᵦᵢ| + |yᵦᵢ₊₁|] · Δxᵢ\n\nPositive AUC:\nA_pos ≈ Σᵢ ½ · [max(yᵦᵢ, 0) + max(yᵦᵢ₊₁, 0)] · Δxᵢ\n\nNegative AUC:\nA_neg ≈ Σᵢ ½ · [min(yᵦᵢ, 0) + min(yᵦᵢ₊₁, 0)] · Δxᵢ\n\nCoverage:\nC = Σₖ (bₖ - aₖ)\nwhere [aₖ, bₖ] are the usable finite data segments inside the selected interval.\n\nCoverage %:\nC_% = 100 · C / (x_end - x_start)\n\nMean - baseline:\nμᵦ = A_signed / C\n\nOn time plots, x is time in seconds, so AUC units are y-units · seconds.';txt = tk.Text(explanation_frame, height=18, wrap='word');exp_scroll = ttk.Scrollbar(explanation_frame, orient='vertical', command=txt.yview);txt.configure(yscrollcommand=exp_scroll.set);txt.grid(row=0, column=0, sticky='nsew', padx=(8, 0), pady=6);exp_scroll.grid(row=0, column=1, sticky='ns', padx=(4, 8), pady=6);txt.insert('1.0', explanation);txt.configure(state='disabled');btn_row = ttk.Frame(main);btn_row.grid(row=3, column=0, sticky='ew')
    def copy_table_to_clipboard():
        try:parent.clipboard_clear();parent.clipboard_append(text_out)
        except Exception:pass
    ttk.Button(btn_row, text='Copy table + explanations + formulas', command=copy_table_to_clipboard).pack(side=tk.LEFT);ttk.Button(btn_row, text='Close', command=top.destroy).pack(side=tk.RIGHT)
def _fiberlyse_plot_show_auc_dialog(self):
    parent = self.winfo_toplevel()
    if not getattr(self.fig, 'axes', []):messagebox.showinfo('AUC', 'No axes/plot found in this graph.');return
    default_start, default_end = self._default_auc_interval_strings();start_var = tk.StringVar(value=str(getattr(self, '_last_auc_start_str', default_start) or default_start));end_var = tk.StringVar(value=str(getattr(self, '_last_auc_end_str', default_end) or default_end));baseline_var = tk.StringVar(value=str(getattr(self, '_last_auc_baseline_str', '0') or '0'));top = tk.Toplevel(parent);top.title('AUC interval');top.transient(parent);top.grab_set();frm = ttk.Frame(top, padding=10);frm.pack(fill=tk.BOTH, expand=True);ttk.Label(frm, text='Calculate AUC for every visible line trace in the current graph.\n\nFor time plots, enter times in seconds.\nAUC is computed relative to the baseline B you enter below.\nUse B = 0 if you want ordinary area relative to zero.', justify='left').grid(row=0, column=0, columnspan=2, sticky='w');ttk.Label(frm, text='Start x/time:').grid(row=1, column=0, sticky='w', pady=(10, 2));entry_start = ttk.Entry(frm, textvariable=start_var, width=18);entry_start.grid(row=1, column=1, sticky='w', pady=(10, 2));ttk.Label(frm, text='End x/time:').grid(row=2, column=0, sticky='w', pady=2);entry_end = ttk.Entry(frm, textvariable=end_var, width=18);entry_end.grid(row=2, column=1, sticky='w', pady=2);ttk.Label(frm, text='Baseline B:').grid(row=3, column=0, sticky='w', pady=2);ttk.Entry(frm, textvariable=baseline_var, width=18).grid(row=3, column=1, sticky='w', pady=2);small_help = 'Formula setup: yᵦ(x) = y(x) - B. Use B = 0 if you do not want to subtract another baseline.';ttk.Label(frm, text=small_help, foreground='gray35', justify='left', wraplength=420).grid(row=4, column=0, columnspan=2, sticky='w', pady=(6, 0));btn_row = ttk.Frame(frm);btn_row.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(10, 0))
    def close_dialog():
        try:top.grab_release()
        except Exception:pass
        try:top.destroy()
        except Exception:pass
    def do_calculate(_e=None):
        try:start_x = float((start_var.get() or '').strip());end_x = float((end_var.get() or '').strip());baseline_y = float((baseline_var.get() or '0').strip())
        except Exception as e:messagebox.showerror('Invalid AUC interval', f'Could not parse the AUC inputs:\n\n{e}');return 'break'
        if not np.isfinite(start_x) or not np.isfinite(end_x):messagebox.showerror('Invalid AUC interval', 'Start and end must be finite numbers.');return 'break'
        if start_x == end_x:messagebox.showerror('Invalid AUC interval', 'Start and end cannot be the same value.');return 'break'
        if not np.isfinite(baseline_y):baseline_y = 0.0
        rows = self.calculate_auc_for_interval(start_x=start_x, end_x=end_x, baseline_y=baseline_y);self._last_auc_start_str = (start_var.get() or '').strip();self._last_auc_end_str = (end_var.get() or '').strip();self._last_auc_baseline_str = (baseline_var.get() or '0').strip();close_dialog()
        if not rows:messagebox.showinfo('AUC', 'No visible labeled line traces overlapped that interval.\n\nScatter-only plots, histogram patches, hidden lines, and vertical marker lines are ignored.');return 'break'
        self._show_auc_results_window(rows, start_x, end_x, baseline_y);return 'break'
    def do_cancel(_e=None):close_dialog();return 'break'
    ttk.Button(btn_row, text='Calculate AUC', command=do_calculate).pack(side=tk.RIGHT);ttk.Button(btn_row, text='Cancel', command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8));top.bind('<Return>', do_calculate);top.bind('<Escape>', do_cancel);entry_start.focus_set();parent.wait_window(top)
PlotTabTk.calculate_auc_for_interval = _fiberlyse_plot_calculate_auc_for_interval;PlotTabTk._format_auc_results_text = _fiberlyse_plot_format_auc_results_text;PlotTabTk._default_auc_interval_strings = _fiberlyse_plot_default_auc_interval_strings;PlotTabTk._show_auc_results_window = _fiberlyse_plot_show_auc_results_window;PlotTabTk.show_auc_dialog = _fiberlyse_plot_show_auc_dialog;_FIBERLYSE_ORIGINAL_PLOTTABTK_INIT = PlotTabTk.__init__
def _fiberlyse_plottabtk_init_with_auc(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_INIT(self, *args, **kwargs)
    try:
        existing = getattr(self, 'auc_btn', None)
        if existing is not None:
            try:
                if bool(existing.winfo_exists()):return
            except Exception:pass
        btn_parent = self.save_btn.master;self.auc_btn = ttk.Button(btn_parent, text='AUC interval...', command=self.show_auc_dialog);self.auc_btn.pack(side=tk.RIGHT, padx=(0, 8))
    except Exception:pass
PlotTabTk.__init__ = _fiberlyse_plottabtk_init_with_auc
def _fiberlyse_install_auc_hotkeys(self):self.root.bind_all('<Control-u>', self._on_ctrl_u, add='+');self.root.bind_all('<Control-U>', self._on_ctrl_u, add='+')
def _fiberlyse_on_ctrl_u(self, _event=None):
    tab = self.find_active_plot_tab()
    if tab is None:return
    tab.show_auc_dialog()
MainAppTk._install_auc_hotkeys = _fiberlyse_install_auc_hotkeys;MainAppTk._on_ctrl_u = _fiberlyse_on_ctrl_u;_FIBERLYSE_ORIGINAL_MAINAPPTK_INIT = MainAppTk.__init__
def _fiberlyse_mainapptk_init_with_auc(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_MAINAPPTK_INIT(self, *args, **kwargs)
    try:self._install_auc_hotkeys()
    except Exception:pass
MainAppTk.__init__ = _fiberlyse_mainapptk_init_with_auc;_FIBERLYSE_ORIGINAL_FIND_ACTIVE_PLOT_TAB = MainAppTk.find_active_plot_tab
def _fiberlyse_find_active_plot_tab_with_auc(self):
    sel = self.outer_tabs.select()
    if not sel:return None
    try:selected_widget = self.root.nametowidget(sel)
    except Exception:selected_widget = None
    try:
        if self.compare_widget is not None and selected_widget is self.compare_widget:return self.compare_widget.plot
    except Exception:pass
    try:
        if self.average_widget is not None and selected_widget is self.average_widget:return self.average_widget.plot
    except Exception:pass
    return _FIBERLYSE_ORIGINAL_FIND_ACTIVE_PLOT_TAB(self)
MainAppTk.find_active_plot_tab = _fiberlyse_find_active_plot_tab_with_auc

# ---- Fiberlyse axis tick interval slider extension ----
# Adds a per-plot dialog for adjusting x/y major tick spacing with sliders.
try:
    from matplotlib.ticker import MultipleLocator as _FiberlyseMultipleLocator, AutoLocator as _FiberlyseAutoLocator
except Exception:
    _FiberlyseMultipleLocator = None
    _FiberlyseAutoLocator = None

def _fiberlyse_axis_float_or_none(value):
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return float(v)

def _fiberlyse_nice_interval_from_axis(ax, axis_name: str) -> float:
    try:
        lim = ax.get_xlim() if axis_name == 'x' else ax.get_ylim()
        span = abs(float(lim[1]) - float(lim[0]))
    except Exception:
        span = np.nan
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    try:
        axis = ax.xaxis if axis_name == 'x' else ax.yaxis
        ticks = np.asarray(axis.get_majorticklocs(), dtype=float)
        ticks = ticks[np.isfinite(ticks)]
        diffs = np.diff(np.sort(np.unique(ticks)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            val = float(np.median(diffs))
            if np.isfinite(val) and val > 0:
                return val
    except Exception:
        pass
    return max(span / 8.0, 1e-12)

def _fiberlyse_plot_axis_interval_bounds(self, axis_name: str):
    spans = []
    for ax in list(getattr(self.fig, 'axes', [])):
        try:
            lim = ax.get_xlim() if axis_name == 'x' else ax.get_ylim()
            span = abs(float(lim[1]) - float(lim[0]))
            if np.isfinite(span) and span > 0:
                spans.append(span)
        except Exception:
            pass
    span = max(spans) if spans else 1.0
    cur = getattr(self, f'_axis_tick_interval_{axis_name}', None)
    cur = _fiberlyse_axis_float_or_none(cur)
    if cur is None:
        ax0 = None
        try:
            ax0 = self.fig.axes[0] if self.fig.axes else None
        except Exception:
            ax0 = None
        cur = _fiberlyse_nice_interval_from_axis(ax0, axis_name) if ax0 is not None else max(span / 8.0, 1e-12)
    lo = max(span / 1000.0, cur / 100.0, 1e-12)
    hi = max(span, cur * 100.0, lo * 10.0)
    cur = min(max(cur, lo), hi)
    return lo, hi, cur

def _fiberlyse_slider_value_to_pos(value: float, lo: float, hi: float) -> float:
    value = max(float(value), float(lo))
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 10.0)
    try:
        llo = np.log10(lo); lhi = np.log10(hi); lv = np.log10(value)
        if not np.isfinite(llo) or not np.isfinite(lhi) or lhi <= llo:
            return 50.0
        return float(100.0 * (lv - llo) / (lhi - llo))
    except Exception:
        return 50.0

def _fiberlyse_slider_pos_to_value(pos: float, lo: float, hi: float) -> float:
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 10.0)
    try:
        p = min(max(float(pos), 0.0), 100.0) / 100.0
        return float(10.0 ** (np.log10(lo) + p * (np.log10(hi) - np.log10(lo))))
    except Exception:
        return float(lo)

def _fiberlyse_plot_apply_axis_tick_intervals(self, reset_auto: bool=False) -> None:
    if _FiberlyseMultipleLocator is None:
        return
    x_interval = _fiberlyse_axis_float_or_none(getattr(self, '_axis_tick_interval_x', None))
    y_interval = _fiberlyse_axis_float_or_none(getattr(self, '_axis_tick_interval_y', None))
    for ax in list(getattr(self.fig, 'axes', [])):
        try:
            if x_interval is not None:
                ax.xaxis.set_major_locator(_FiberlyseMultipleLocator(x_interval))
            elif reset_auto and _FiberlyseAutoLocator is not None:
                ax.xaxis.set_major_locator(_FiberlyseAutoLocator())
        except Exception:
            pass
        try:
            if y_interval is not None:
                ax.yaxis.set_major_locator(_FiberlyseMultipleLocator(y_interval))
            elif reset_auto and _FiberlyseAutoLocator is not None:
                ax.yaxis.set_major_locator(_FiberlyseAutoLocator())
        except Exception:
            pass

def _fiberlyse_plot_set_axis_tick_intervals(self, x_interval=None, y_interval=None) -> None:
    self._axis_tick_interval_x = _fiberlyse_axis_float_or_none(x_interval)
    self._axis_tick_interval_y = _fiberlyse_axis_float_or_none(y_interval)
    try:
        self._apply_axis_tick_intervals(reset_auto=True)
    except Exception:
        pass
    try:
        self.canvas.draw_idle()
    except Exception:
        pass

def _fiberlyse_plot_show_axis_interval_dialog(self):
    parent = self.winfo_toplevel()
    if not getattr(self.fig, 'axes', []):
        try:messagebox.showinfo('Axis intervals', 'No axes/plot found in this graph.')
        except Exception:pass
        return
    x_lo, x_hi, x_cur = self._axis_interval_bounds('x')
    y_lo, y_hi, y_cur = self._axis_interval_bounds('y')
    orig_x = getattr(self, '_axis_tick_interval_x', None)
    orig_y = getattr(self, '_axis_tick_interval_y', None)
    x_auto = tk.BooleanVar(value=_fiberlyse_axis_float_or_none(orig_x) is None)
    y_auto = tk.BooleanVar(value=_fiberlyse_axis_float_or_none(orig_y) is None)
    x_entry = tk.StringVar(value=f'{(_fiberlyse_axis_float_or_none(orig_x) or x_cur):.8g}')
    y_entry = tk.StringVar(value=f'{(_fiberlyse_axis_float_or_none(orig_y) or y_cur):.8g}')
    x_scale = tk.DoubleVar(value=_fiberlyse_slider_value_to_pos(_fiberlyse_axis_float_or_none(orig_x) or x_cur, x_lo, x_hi))
    y_scale = tk.DoubleVar(value=_fiberlyse_slider_value_to_pos(_fiberlyse_axis_float_or_none(orig_y) or y_cur, y_lo, y_hi))
    top = tk.Toplevel(parent);top.title('Axis intervals');top.transient(parent);top.grab_set();top.resizable(True, False)
    frm = ttk.Frame(top, padding=10);frm.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frm, text='Adjust the major tick interval for the current plot. Use Auto to return an axis to Matplotlib defaults.', justify='left').grid(row=0, column=0, columnspan=5, sticky='w', pady=(0, 10))
    busy = {'value': False}
    def parse_entry(s, label):
        try:v = float((s or '').strip())
        except Exception:raise ValueError(f'{label} interval must be a number.')
        if not np.isfinite(v) or v <= 0:raise ValueError(f'{label} interval must be a positive finite number.')
        return float(v)
    def apply_from_controls(show_errors=False):
        try:
            xi = None if bool(x_auto.get()) else parse_entry(x_entry.get(), 'X')
            yi = None if bool(y_auto.get()) else parse_entry(y_entry.get(), 'Y')
        except Exception as e:
            if show_errors:messagebox.showerror('Invalid axis interval', str(e), parent=top)
            return False
        self.set_axis_tick_intervals(xi, yi)
        return True
    def on_scale(axis_name):
        if busy['value']:
            return
        busy['value'] = True
        try:
            if axis_name == 'x':
                val = _fiberlyse_slider_pos_to_value(x_scale.get(), x_lo, x_hi);x_entry.set(f'{val:.8g}');x_auto.set(False)
            else:
                val = _fiberlyse_slider_pos_to_value(y_scale.get(), y_lo, y_hi);y_entry.set(f'{val:.8g}');y_auto.set(False)
        finally:
            busy['value'] = False
        apply_from_controls(show_errors=False)
    def on_entry_return(_event=None):
        apply_from_controls(show_errors=True)
        try:
            xv = parse_entry(x_entry.get(), 'X')
            x_scale.set(_fiberlyse_slider_value_to_pos(xv, x_lo, x_hi))
        except Exception:
            pass
        try:
            yv = parse_entry(y_entry.get(), 'Y')
            y_scale.set(_fiberlyse_slider_value_to_pos(yv, y_lo, y_hi))
        except Exception:
            pass
        return 'break'
    def on_auto_changed():
        apply_from_controls(show_errors=False)
    def row(r, label, lo, hi, var_scale, var_entry, var_auto, axis_name):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky='w', padx=(0, 8), pady=4)
        scale = ttk.Scale(frm, from_=0.0, to=100.0, orient=tk.HORIZONTAL, variable=var_scale, command=lambda _v, n=axis_name: on_scale(n))
        scale.grid(row=r, column=1, sticky='ew', padx=(0, 8), pady=4)
        ent = ttk.Entry(frm, textvariable=var_entry, width=14)
        ent.grid(row=r, column=2, sticky='w', padx=(0, 8), pady=4)
        ent.bind('<Return>', on_entry_return)
        chk = ttk.Checkbutton(frm, text='Auto', variable=var_auto, command=on_auto_changed)
        chk.grid(row=r, column=3, sticky='w', padx=(0, 8), pady=4)
        ttk.Label(frm, text=f'range: {lo:.3g} to {hi:.3g}', foreground='gray35').grid(row=r, column=4, sticky='w', pady=4)
    frm.columnconfigure(1, weight=1)
    row(1, 'X tick interval:', x_lo, x_hi, x_scale, x_entry, x_auto, 'x')
    row(2, 'Y tick interval:', y_lo, y_hi, y_scale, y_entry, y_auto, 'y')
    hint = ttk.Label(frm, text='Tip: press Enter in a number box to apply an exact interval. The sliders use a logarithmic scale for fine control.', foreground='gray35', justify='left')
    hint.grid(row=3, column=0, columnspan=5, sticky='w', pady=(8, 0))
    btn_row = ttk.Frame(frm);btn_row.grid(row=4, column=0, columnspan=5, sticky='ew', pady=(12, 0))
    def close_dialog():
        try:top.grab_release()
        except Exception:pass
        try:top.destroy()
        except Exception:pass
    def do_reset():
        x_auto.set(True);y_auto.set(True);self.set_axis_tick_intervals(None, None)
    def do_ok(_event=None):
        if apply_from_controls(show_errors=True):close_dialog()
        return 'break'
    def do_cancel(_event=None):
        self.set_axis_tick_intervals(orig_x, orig_y);close_dialog();return 'break'
    ttk.Button(btn_row, text='Reset both to Auto', command=do_reset).pack(side=tk.LEFT)
    ttk.Button(btn_row, text='OK', command=do_ok).pack(side=tk.RIGHT)
    ttk.Button(btn_row, text='Cancel', command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8))
    top.bind('<Return>', do_ok);top.bind('<Escape>', do_cancel)
    parent.wait_window(top)

def _fiberlyse_install_axis_interval_hotkeys(self):
    self.root.bind_all('<Control-k>', self._on_ctrl_k_axis_intervals, add='+')
    self.root.bind_all('<Control-K>', self._on_ctrl_k_axis_intervals, add='+')

def _fiberlyse_on_ctrl_k_axis_intervals(self, _event=None):
    tab = self.find_active_plot_tab()
    if tab is None:
        return
    tab.show_axis_interval_dialog()

PlotTabTk._axis_interval_bounds = _fiberlyse_plot_axis_interval_bounds
PlotTabTk._apply_axis_tick_intervals = _fiberlyse_plot_apply_axis_tick_intervals
PlotTabTk.set_axis_tick_intervals = _fiberlyse_plot_set_axis_tick_intervals
PlotTabTk.show_axis_interval_dialog = _fiberlyse_plot_show_axis_interval_dialog
_FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_AXIS_INTERVALS = PlotTabTk.redraw
def _fiberlyse_plottabtk_redraw_with_axis_intervals(self):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_AXIS_INTERVALS(self)
    try:self._apply_axis_tick_intervals(reset_auto=False)
    except Exception:pass
    try:self.canvas.draw_idle()
    except Exception:pass
PlotTabTk.redraw = _fiberlyse_plottabtk_redraw_with_axis_intervals
_FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_AXIS_INTERVALS = PlotTabTk.__init__
def _fiberlyse_plottabtk_init_with_axis_intervals(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_AXIS_INTERVALS(self, *args, **kwargs)
    try:
        existing = getattr(self, 'axis_interval_btn', None)
        if existing is not None:
            try:
                if bool(existing.winfo_exists()):return
            except Exception:pass
        btn_parent = self.save_btn.master
        self.axis_interval_btn = ttk.Button(btn_parent, text='Axis intervals...', command=self.show_axis_interval_dialog)
        self.axis_interval_btn.pack(side=tk.RIGHT, padx=(0, 8))
    except Exception:pass
PlotTabTk.__init__ = _fiberlyse_plottabtk_init_with_axis_intervals
MainAppTk._install_axis_interval_hotkeys = _fiberlyse_install_axis_interval_hotkeys
MainAppTk._on_ctrl_k_axis_intervals = _fiberlyse_on_ctrl_k_axis_intervals
_FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_AXIS_INTERVALS = MainAppTk.__init__
def _fiberlyse_mainapptk_init_with_axis_intervals(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_AXIS_INTERVALS(self, *args, **kwargs)
    try:self._install_axis_interval_hotkeys()
    except Exception:pass
MainAppTk.__init__ = _fiberlyse_mainapptk_init_with_axis_intervals
# ---- End Fiberlyse axis tick interval slider extension ----

# ---- Fiberlyse axis visible range extension ----
# Extends the axis interval dialog so each plot can also set visible x/y ranges.

def _fiberlyse_axis_range_tuple_or_none(value):
    if value is None:
        return None
    try:
        a, b = value
        a = float(a); b = float(b)
    except Exception:
        return None
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return None
    if b < a:
        a, b = b, a
    return (float(a), float(b))

def _fiberlyse_plot_current_axis_range(self, axis_name: str):
    try:
        ax = self.fig.axes[0] if self.fig.axes else None
    except Exception:
        ax = None
    if ax is None:
        return (0.0, 1.0)
    try:
        lim = ax.get_xlim() if axis_name == 'x' else ax.get_ylim()
        a = float(lim[0]); b = float(lim[1])
        if np.isfinite(a) and np.isfinite(b) and a != b:
            if b < a:
                a, b = b, a
            return (float(a), float(b))
    except Exception:
        pass
    return (0.0, 1.0)

def _fiberlyse_plot_axis_range_slider_bounds(self, axis_name: str):
    vals = []
    for ax in list(getattr(self.fig, 'axes', [])):
        try:
            cur = ax.get_xlim() if axis_name == 'x' else ax.get_ylim()
            for v in cur:
                vf = float(v)
                if np.isfinite(vf):
                    vals.append(vf)
        except Exception:
            pass
        try:
            dl = getattr(ax, 'dataLim', None)
            if dl is not None:
                interval = dl.intervalx if axis_name == 'x' else dl.intervaly
                for v in interval:
                    vf = float(v)
                    if np.isfinite(vf):
                        vals.append(vf)
        except Exception:
            pass
        try:
            for line in list(getattr(ax, 'lines', [])):
                arr = np.asarray(line.get_xdata() if axis_name == 'x' else line.get_ydata(), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    vals.append(float(np.nanmin(arr))); vals.append(float(np.nanmax(arr)))
        except Exception:
            pass
        try:
            for coll in list(getattr(ax, 'collections', [])):
                offs = coll.get_offsets()
                if offs is None or len(offs) == 0:
                    continue
                offs = np.asarray(offs, dtype=float)
                if offs.ndim == 2 and offs.shape[1] >= 2:
                    arr = offs[:, 0] if axis_name == 'x' else offs[:, 1]
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        vals.append(float(np.nanmin(arr))); vals.append(float(np.nanmax(arr)))
        except Exception:
            pass
    stored = _fiberlyse_axis_range_tuple_or_none(getattr(self, f'_axis_visible_range_{axis_name}', None))
    if stored is not None:
        vals.extend([stored[0], stored[1]])
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        vals = [0.0, 1.0]
    lo = float(min(vals)); hi = float(max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1.0)
        lo -= pad; hi += pad
    span = hi - lo
    pad = max(span * 0.05, 1e-12)
    lo -= pad; hi += pad
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))

def _fiberlyse_plot_apply_axis_ranges(self, reset_auto: bool=False) -> None:
    x_range = _fiberlyse_axis_range_tuple_or_none(getattr(self, '_axis_visible_range_x', None))
    y_range = _fiberlyse_axis_range_tuple_or_none(getattr(self, '_axis_visible_range_y', None))
    for ax in list(getattr(self.fig, 'axes', [])):
        try:
            if x_range is not None:
                ax.set_xlim(x_range[0], x_range[1], auto=False)
            elif reset_auto:
                ax.autoscale(enable=True, axis='x')
                try:ax.autoscale_view(scalex=True, scaley=False)
                except Exception:pass
        except Exception:
            pass
        try:
            if y_range is not None:
                ax.set_ylim(y_range[0], y_range[1], auto=False)
            elif reset_auto:
                ax.autoscale(enable=True, axis='y')
                try:ax.autoscale_view(scalex=False, scaley=True)
                except Exception:pass
        except Exception:
            pass

def _fiberlyse_plot_set_axis_ranges(self, x_range=None, y_range=None) -> None:
    self._axis_visible_range_x = _fiberlyse_axis_range_tuple_or_none(x_range)
    self._axis_visible_range_y = _fiberlyse_axis_range_tuple_or_none(y_range)
    try:
        self._apply_axis_ranges(reset_auto=True)
    except Exception:
        pass
    try:
        self._apply_axis_tick_intervals(reset_auto=False)
    except Exception:
        pass
    try:
        self.canvas.draw_idle()
    except Exception:
        pass

def _fiberlyse_clamp(v, lo, hi):
    try:
        v = float(v); lo = float(lo); hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        return float(min(max(v, lo), hi))
    except Exception:
        return v

def _fiberlyse_plot_show_axis_interval_range_dialog(self):
    parent = self.winfo_toplevel()
    if not getattr(self.fig, 'axes', []):
        try:messagebox.showinfo('Axis controls', 'No axes/plot found in this graph.')
        except Exception:pass
        return

    x_lo, x_hi, x_cur = self._axis_interval_bounds('x')
    y_lo, y_hi, y_cur = self._axis_interval_bounds('y')
    xb_lo, xb_hi = self._axis_range_slider_bounds('x')
    yb_lo, yb_hi = self._axis_range_slider_bounds('y')

    orig_tick_x = getattr(self, '_axis_tick_interval_x', None)
    orig_tick_y = getattr(self, '_axis_tick_interval_y', None)
    orig_range_x = getattr(self, '_axis_visible_range_x', None)
    orig_range_y = getattr(self, '_axis_visible_range_y', None)

    cur_x_range = _fiberlyse_axis_range_tuple_or_none(orig_range_x) or self._current_axis_range('x')
    cur_y_range = _fiberlyse_axis_range_tuple_or_none(orig_range_y) or self._current_axis_range('y')

    x_auto = tk.BooleanVar(value=_fiberlyse_axis_float_or_none(orig_tick_x) is None)
    y_auto = tk.BooleanVar(value=_fiberlyse_axis_float_or_none(orig_tick_y) is None)
    xr_auto = tk.BooleanVar(value=_fiberlyse_axis_range_tuple_or_none(orig_range_x) is None)
    yr_auto = tk.BooleanVar(value=_fiberlyse_axis_range_tuple_or_none(orig_range_y) is None)

    x_entry = tk.StringVar(value=f'{(_fiberlyse_axis_float_or_none(orig_tick_x) or x_cur):.8g}')
    y_entry = tk.StringVar(value=f'{(_fiberlyse_axis_float_or_none(orig_tick_y) or y_cur):.8g}')
    x_scale = tk.DoubleVar(value=_fiberlyse_slider_value_to_pos(_fiberlyse_axis_float_or_none(orig_tick_x) or x_cur, x_lo, x_hi))
    y_scale = tk.DoubleVar(value=_fiberlyse_slider_value_to_pos(_fiberlyse_axis_float_or_none(orig_tick_y) or y_cur, y_lo, y_hi))

    xr_min_entry = tk.StringVar(value=f'{cur_x_range[0]:.8g}')
    xr_max_entry = tk.StringVar(value=f'{cur_x_range[1]:.8g}')
    yr_min_entry = tk.StringVar(value=f'{cur_y_range[0]:.8g}')
    yr_max_entry = tk.StringVar(value=f'{cur_y_range[1]:.8g}')
    xr_min_scale = tk.DoubleVar(value=_fiberlyse_clamp(cur_x_range[0], xb_lo, xb_hi))
    xr_max_scale = tk.DoubleVar(value=_fiberlyse_clamp(cur_x_range[1], xb_lo, xb_hi))
    yr_min_scale = tk.DoubleVar(value=_fiberlyse_clamp(cur_y_range[0], yb_lo, yb_hi))
    yr_max_scale = tk.DoubleVar(value=_fiberlyse_clamp(cur_y_range[1], yb_lo, yb_hi))

    top = tk.Toplevel(parent)
    top.title('Axis intervals and visible range')
    top.transient(parent)
    top.grab_set()
    top.resizable(True, False)
    frm = ttk.Frame(top, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frm, text='Adjust tick spacing and the visible x/y range for the current plot. Use Auto to return an item to the plot default.', justify='left').pack(anchor='w', pady=(0, 10))

    tick_box = ttk.LabelFrame(frm, text='Tick intervals')
    tick_box.pack(fill=tk.X, expand=True)
    range_box = ttk.LabelFrame(frm, text='Visible range')
    range_box.pack(fill=tk.X, expand=True, pady=(10, 0))
    tick_box.columnconfigure(1, weight=1)
    range_box.columnconfigure(2, weight=1)

    busy = {'value': False}

    def parse_interval_entry(s, label):
        try:v = float((s or '').strip())
        except Exception:raise ValueError(f'{label} tick interval must be a number.')
        if not np.isfinite(v) or v <= 0:raise ValueError(f'{label} tick interval must be a positive finite number.')
        return float(v)

    def parse_range_entries(min_s, max_s, label):
        try:a = float((min_s or '').strip()); b = float((max_s or '').strip())
        except Exception:raise ValueError(f'{label} visible range values must be numbers.')
        if not np.isfinite(a) or not np.isfinite(b):raise ValueError(f'{label} visible range values must be finite numbers.')
        if b <= a:raise ValueError(f'{label} visible range maximum must be greater than the minimum.')
        return (float(a), float(b))

    def apply_from_controls(show_errors=False):
        try:
            xi = None if bool(x_auto.get()) else parse_interval_entry(x_entry.get(), 'X')
            yi = None if bool(y_auto.get()) else parse_interval_entry(y_entry.get(), 'Y')
            xr = None if bool(xr_auto.get()) else parse_range_entries(xr_min_entry.get(), xr_max_entry.get(), 'X')
            yr = None if bool(yr_auto.get()) else parse_range_entries(yr_min_entry.get(), yr_max_entry.get(), 'Y')
        except Exception as e:
            if show_errors:messagebox.showerror('Invalid axis controls', str(e), parent=top)
            return False
        self.set_axis_tick_intervals(xi, yi)
        self.set_axis_ranges(xr, yr)
        return True

    def update_interval_slider_from_entry(axis_name):
        try:
            if axis_name == 'x':
                xv = parse_interval_entry(x_entry.get(), 'X')
                x_scale.set(_fiberlyse_slider_value_to_pos(xv, x_lo, x_hi))
            else:
                yv = parse_interval_entry(y_entry.get(), 'Y')
                y_scale.set(_fiberlyse_slider_value_to_pos(yv, y_lo, y_hi))
        except Exception:
            pass

    def update_range_sliders_from_entries(axis_name):
        try:
            if axis_name == 'x':
                a, b = parse_range_entries(xr_min_entry.get(), xr_max_entry.get(), 'X')
                xr_min_scale.set(_fiberlyse_clamp(a, xb_lo, xb_hi)); xr_max_scale.set(_fiberlyse_clamp(b, xb_lo, xb_hi))
            else:
                a, b = parse_range_entries(yr_min_entry.get(), yr_max_entry.get(), 'Y')
                yr_min_scale.set(_fiberlyse_clamp(a, yb_lo, yb_hi)); yr_max_scale.set(_fiberlyse_clamp(b, yb_lo, yb_hi))
        except Exception:
            pass

    def on_interval_scale(axis_name):
        if busy['value']:
            return
        busy['value'] = True
        try:
            if axis_name == 'x':
                val = _fiberlyse_slider_pos_to_value(x_scale.get(), x_lo, x_hi); x_entry.set(f'{val:.8g}'); x_auto.set(False)
            else:
                val = _fiberlyse_slider_pos_to_value(y_scale.get(), y_lo, y_hi); y_entry.set(f'{val:.8g}'); y_auto.set(False)
        finally:
            busy['value'] = False
        apply_from_controls(show_errors=False)

    def on_range_scale(axis_name, which):
        if busy['value']:
            return
        busy['value'] = True
        try:
            if axis_name == 'x':
                b_lo, b_hi = xb_lo, xb_hi; eps = max((b_hi - b_lo) * 0.001, 1e-12)
                a = float(xr_min_scale.get()); b = float(xr_max_scale.get())
                if which == 'min' and a >= b:
                    b = min(b_hi, a + eps); xr_max_scale.set(b)
                    if b <= a:
                        a = max(b_lo, b - eps); xr_min_scale.set(a)
                elif which == 'max' and b <= a:
                    a = max(b_lo, b - eps); xr_min_scale.set(a)
                    if b <= a:
                        b = min(b_hi, a + eps); xr_max_scale.set(b)
                xr_min_entry.set(f'{a:.8g}'); xr_max_entry.set(f'{b:.8g}'); xr_auto.set(False)
            else:
                b_lo, b_hi = yb_lo, yb_hi; eps = max((b_hi - b_lo) * 0.001, 1e-12)
                a = float(yr_min_scale.get()); b = float(yr_max_scale.get())
                if which == 'min' and a >= b:
                    b = min(b_hi, a + eps); yr_max_scale.set(b)
                    if b <= a:
                        a = max(b_lo, b - eps); yr_min_scale.set(a)
                elif which == 'max' and b <= a:
                    a = max(b_lo, b - eps); yr_min_scale.set(a)
                    if b <= a:
                        b = min(b_hi, a + eps); yr_max_scale.set(b)
                yr_min_entry.set(f'{a:.8g}'); yr_max_entry.set(f'{b:.8g}'); yr_auto.set(False)
        finally:
            busy['value'] = False
        apply_from_controls(show_errors=False)

    def on_entry_return(_event=None):
        ok = apply_from_controls(show_errors=True)
        if ok:
            update_interval_slider_from_entry('x'); update_interval_slider_from_entry('y')
            update_range_sliders_from_entries('x'); update_range_sliders_from_entries('y')
        return 'break'

    def on_auto_changed():
        apply_from_controls(show_errors=False)

    def tick_row(r, label, lo, hi, var_scale, var_entry, var_auto, axis_name):
        ttk.Label(tick_box, text=label).grid(row=r, column=0, sticky='w', padx=(8, 8), pady=4)
        scale = ttk.Scale(tick_box, from_=0.0, to=100.0, orient=tk.HORIZONTAL, variable=var_scale, command=lambda _v, n=axis_name: on_interval_scale(n))
        scale.grid(row=r, column=1, sticky='ew', padx=(0, 8), pady=4)
        ent = ttk.Entry(tick_box, textvariable=var_entry, width=14)
        ent.grid(row=r, column=2, sticky='w', padx=(0, 8), pady=4)
        ent.bind('<Return>', on_entry_return)
        ttk.Checkbutton(tick_box, text='Auto', variable=var_auto, command=on_auto_changed).grid(row=r, column=3, sticky='w', padx=(0, 8), pady=4)
        ttk.Label(tick_box, text=f'slider: {lo:.3g} to {hi:.3g}', foreground='gray35').grid(row=r, column=4, sticky='w', padx=(0, 8), pady=4)

    def range_row(r, label, b_lo, b_hi, min_scale, max_scale, min_entry, max_entry, auto_var, axis_name):
        ttk.Label(range_box, text=label).grid(row=r, column=0, rowspan=2, sticky='w', padx=(8, 8), pady=4)
        ttk.Label(range_box, text='Min:').grid(row=r, column=1, sticky='e', padx=(0, 6), pady=2)
        smin = ttk.Scale(range_box, from_=b_lo, to=b_hi, orient=tk.HORIZONTAL, variable=min_scale, command=lambda _v, n=axis_name: on_range_scale(n, 'min'))
        smin.grid(row=r, column=2, sticky='ew', padx=(0, 8), pady=2)
        e_min = ttk.Entry(range_box, textvariable=min_entry, width=14)
        e_min.grid(row=r, column=3, sticky='w', padx=(0, 8), pady=2)
        e_min.bind('<Return>', on_entry_return)
        ttk.Checkbutton(range_box, text='Auto', variable=auto_var, command=on_auto_changed).grid(row=r, column=4, rowspan=2, sticky='w', padx=(0, 8), pady=2)
        ttk.Label(range_box, text=f'slider: {b_lo:.3g} to {b_hi:.3g}', foreground='gray35').grid(row=r, column=5, rowspan=2, sticky='w', padx=(0, 8), pady=2)
        ttk.Label(range_box, text='Max:').grid(row=r + 1, column=1, sticky='e', padx=(0, 6), pady=2)
        smax = ttk.Scale(range_box, from_=b_lo, to=b_hi, orient=tk.HORIZONTAL, variable=max_scale, command=lambda _v, n=axis_name: on_range_scale(n, 'max'))
        smax.grid(row=r + 1, column=2, sticky='ew', padx=(0, 8), pady=2)
        e_max = ttk.Entry(range_box, textvariable=max_entry, width=14)
        e_max.grid(row=r + 1, column=3, sticky='w', padx=(0, 8), pady=2)
        e_max.bind('<Return>', on_entry_return)

    tick_row(0, 'X tick interval:', x_lo, x_hi, x_scale, x_entry, x_auto, 'x')
    tick_row(1, 'Y tick interval:', y_lo, y_hi, y_scale, y_entry, y_auto, 'y')
    range_row(0, 'X visible range:', xb_lo, xb_hi, xr_min_scale, xr_max_scale, xr_min_entry, xr_max_entry, xr_auto, 'x')
    range_row(2, 'Y visible range:', yb_lo, yb_hi, yr_min_scale, yr_max_scale, yr_min_entry, yr_max_entry, yr_auto, 'y')

    hint = ttk.Label(frm, text='Tip: for exact limits, type numbers into the Min/Max boxes and press Enter. Ctrl+K opens this dialog for the active plot.', foreground='gray35', justify='left')
    hint.pack(anchor='w', pady=(8, 0))
    btn_row = ttk.Frame(frm)
    btn_row.pack(fill=tk.X, pady=(12, 0))

    def close_dialog():
        try:top.grab_release()
        except Exception:pass
        try:top.destroy()
        except Exception:pass

    def do_reset_all():
        x_auto.set(True); y_auto.set(True); xr_auto.set(True); yr_auto.set(True)
        self.set_axis_tick_intervals(None, None)
        self.set_axis_ranges(None, None)

    def do_ok(_event=None):
        if apply_from_controls(show_errors=True):
            close_dialog()
        return 'break'

    def do_cancel(_event=None):
        self.set_axis_tick_intervals(orig_tick_x, orig_tick_y)
        self.set_axis_ranges(orig_range_x, orig_range_y)
        close_dialog()
        return 'break'

    ttk.Button(btn_row, text='Reset all to Auto', command=do_reset_all).pack(side=tk.LEFT)
    ttk.Button(btn_row, text='OK', command=do_ok).pack(side=tk.RIGHT)
    ttk.Button(btn_row, text='Cancel', command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8))
    top.bind('<Return>', do_ok)
    top.bind('<Escape>', do_cancel)
    parent.wait_window(top)

PlotTabTk._current_axis_range = _fiberlyse_plot_current_axis_range
PlotTabTk._axis_range_slider_bounds = _fiberlyse_plot_axis_range_slider_bounds
PlotTabTk._apply_axis_ranges = _fiberlyse_plot_apply_axis_ranges
PlotTabTk.set_axis_ranges = _fiberlyse_plot_set_axis_ranges
PlotTabTk.show_axis_interval_dialog = _fiberlyse_plot_show_axis_interval_range_dialog

_FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_AXIS_RANGES = PlotTabTk.redraw
def _fiberlyse_plottabtk_redraw_with_axis_ranges(self):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_AXIS_RANGES(self)
    try:self._apply_axis_ranges(reset_auto=False)
    except Exception:pass
    try:self._apply_axis_tick_intervals(reset_auto=False)
    except Exception:pass
    try:self.canvas.draw_idle()
    except Exception:pass
PlotTabTk.redraw = _fiberlyse_plottabtk_redraw_with_axis_ranges

_FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_AXIS_RANGES = PlotTabTk.__init__
def _fiberlyse_plottabtk_init_with_axis_ranges(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_AXIS_RANGES(self, *args, **kwargs)
    try:
        btn = getattr(self, 'axis_interval_btn', None)
        if btn is not None and bool(btn.winfo_exists()):
            btn.config(text='Axis intervals/range...', command=self.show_axis_interval_dialog)
    except Exception:
        pass
PlotTabTk.__init__ = _fiberlyse_plottabtk_init_with_axis_ranges
# ---- End Fiberlyse axis visible range extension ----



# ---- Fiberlyse draggable and resizable legend extension ----
# Adds per-plot legend movement and readability controls.
# Mouse controls:
#   * Left-drag inside a legend to move it.
#   * Drag a legend corner, or Shift/Ctrl-left-drag inside it, to resize the legend text/box.
#   * Use the "Legend box..." button or Ctrl+L for exact settings.

def _fiberlyse_legend_float(value, default=None):
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(v)

def _fiberlyse_legend_clamp(value, lo, hi):
    v = _fiberlyse_legend_float(value, lo)
    try:
        return float(min(max(v, float(lo)), float(hi)))
    except Exception:
        return float(lo)

def _fiberlyse_legend_event_has_resize_modifier(event) -> bool:
    try:
        key = str(getattr(event, 'key', '') or '').lower()
    except Exception:
        key = ''
    return ('shift' in key) or ('control' in key) or ('ctrl' in key)

def _fiberlyse_legend_text_fontsize(leg, fallback: float=10.0) -> float:
    try:
        texts = list(leg.get_texts())
        if texts:
            return float(texts[0].get_fontsize())
    except Exception:
        pass
    try:
        return float(matplotlib.rcParams.get('legend.fontsize', fallback))
    except Exception:
        return float(fallback)

def _fiberlyse_legend_title_fontsize(leg, fallback: Optional[float]=None) -> float:
    if fallback is None:
        fallback = _fiberlyse_legend_text_fontsize(leg, 10.0)
    try:
        title = leg.get_title()
        if title is not None:
            fs = float(title.get_fontsize())
            if np.isfinite(fs) and fs > 0:
                return fs
    except Exception:
        pass
    return float(fallback)

def _fiberlyse_legend_current_anchor_axes(ax, leg, renderer=None):
    try:
        if renderer is None:
            renderer = leg.figure.canvas.get_renderer()
    except Exception:
        renderer = None
    try:
        bbox = leg.get_window_extent(renderer)
        cx = float((bbox.x0 + bbox.x1) / 2.0)
        cy = float((bbox.y0 + bbox.y1) / 2.0)
        xy = ax.transAxes.inverted().transform((cx, cy))
        return (float(xy[0]), float(xy[1]))
    except Exception:
        return (0.5, 0.5)

def _fiberlyse_legend_contains_event(self, event):
    if event is None or getattr(event, 'x', None) is None or getattr(event, 'y', None) is None:
        return None
    renderer = None
    try:
        renderer = self.canvas.get_renderer()
    except Exception:
        try:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()
        except Exception:
            renderer = None
    axes = list(getattr(self.fig, 'axes', []))
    for ax in reversed(axes):
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is None:
            continue
        try:
            if not leg.get_visible():
                continue
        except Exception:
            pass
        try:
            bbox = leg.get_window_extent(renderer)
            if bbox is not None and bbox.contains(float(event.x), float(event.y)):
                return (ax, leg, bbox, renderer)
        except Exception:
            continue
    return None

def _fiberlyse_legend_near_resize_area(bbox, event, margin: float=16.0) -> bool:
    try:
        x = float(event.x); y = float(event.y)
        mx = float(margin)
        near_right = abs(x - float(bbox.x1)) <= mx
        near_left = abs(x - float(bbox.x0)) <= mx
        near_top = abs(y - float(bbox.y1)) <= mx
        near_bottom = abs(y - float(bbox.y0)) <= mx
        return (near_right or near_left) and (near_top or near_bottom)
    except Exception:
        return False

def _fiberlyse_plot_get_legend_state(self, ax=None):
    states = getattr(self, '_legend_box_overrides', None)
    if not isinstance(states, dict):
        states = {}
        self._legend_box_overrides = states
    if ax is None:
        return states
    try:
        key = self._ax_key(ax)
    except Exception:
        key = id(ax)
    state = states.setdefault(key, {})
    if not isinstance(state, dict):
        state = {}
        states[key] = state
    return state

def _fiberlyse_plot_set_legend_cursor(self, cursor: str) -> None:
    try:
        self.canvas_widget.configure(cursor=cursor)
    except Exception:
        try:
            self.canvas_widget.configure(cursor='')
        except Exception:
            pass

def _fiberlyse_plot_apply_legend_box_overrides(self) -> None:
    states = getattr(self, '_legend_box_overrides', None)
    if not isinstance(states, dict):
        states = {}
        self._legend_box_overrides = states
    for ax in list(getattr(self.fig, 'axes', [])):
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is None:
            continue
        try:
            key = self._ax_key(ax)
        except Exception:
            key = id(ax)
        state = states.get(key, {}) if isinstance(states.get(key, {}), dict) else {}
        if bool(state.get('manual_position')):
            try:
                anchor = state.get('anchor_axes', None)
                if anchor is not None:
                    x, y = float(anchor[0]), float(anchor[1])
                    try:
                        if hasattr(leg, 'set_loc'):
                            leg.set_loc('center')
                        else:
                            leg._loc = 10
                    except Exception:
                        try:leg._loc = 10
                        except Exception:pass
                    leg.set_bbox_to_anchor((x, y), transform=ax.transAxes)
            except Exception:
                pass
        elif 'manual_position' in state:
            try:
                if hasattr(leg, 'set_loc'):
                    leg.set_loc('best')
                else:
                    leg._loc = 0
            except Exception:
                try:leg._loc = 0
                except Exception:pass
            try:
                leg.set_bbox_to_anchor(None)
            except Exception:
                pass
        fs = _fiberlyse_legend_float(state.get('fontsize', None), None)
        if fs is not None and fs > 0:
            try:
                for txt in leg.get_texts():
                    txt.set_fontsize(fs)
            except Exception:
                pass
        tfs = _fiberlyse_legend_float(state.get('title_fontsize', None), None)
        if tfs is not None and tfs > 0:
            try:
                title = leg.get_title()
                if title is not None:
                    title.set_fontsize(tfs)
            except Exception:
                pass
        alpha = _fiberlyse_legend_float(state.get('frame_alpha', None), None)
        if alpha is not None:
            try:
                leg.get_frame().set_alpha(_fiberlyse_legend_clamp(alpha, 0.0, 1.0))
            except Exception:
                pass
        face = state.get('facecolor', None)
        if face:
            try:
                leg.get_frame().set_facecolor(face)
            except Exception:
                pass
        try:
            leg.set_draggable(False)
        except Exception:
            pass

def _fiberlyse_plot_on_legend_button_press(self, event):
    if event is None:
        return
    try:
        if self._toolbar_is_active():
            return
    except Exception:
        pass
    try:
        if getattr(event, 'button', None) != 1:
            return
        if bool(getattr(event, 'dblclick', False)):
            return
    except Exception:
        return
    hit = _fiberlyse_legend_contains_event(self, event)
    if hit is None:
        return
    ax, leg, bbox, renderer = hit
    action = 'resize' if (_fiberlyse_legend_event_has_resize_modifier(event) or _fiberlyse_legend_near_resize_area(bbox, event)) else 'move'
    try:
        anchor = _fiberlyse_legend_current_anchor_axes(ax, leg, renderer)
        pointer_axes = ax.transAxes.inverted().transform((float(event.x), float(event.y)))
        offset = (float(anchor[0] - pointer_axes[0]), float(anchor[1] - pointer_axes[1]))
    except Exception:
        anchor = (0.5, 0.5)
        offset = (0.0, 0.0)
    fs0 = _fiberlyse_legend_text_fontsize(leg, 10.0)
    tfs0 = _fiberlyse_legend_title_fontsize(leg, fs0)
    try:
        cx = float((bbox.x0 + bbox.x1) / 2.0); cy = float((bbox.y0 + bbox.y1) / 2.0)
        d0 = float(max(8.0, ((float(event.x) - cx) ** 2.0 + (float(event.y) - cy) ** 2.0) ** 0.5))
    except Exception:
        cx = cy = 0.0; d0 = 80.0
    self._legend_drag_state = {
        'action': action,
        'ax': ax,
        'legend': leg,
        'start_x': float(event.x),
        'start_y': float(event.y),
        'anchor0': anchor,
        'offset': offset,
        'fontsize0': fs0,
        'title_fontsize0': tfs0,
        'center_px': (cx, cy),
        'dist0': d0,
    }
    _fiberlyse_plot_get_legend_state(self, ax)['manual_position'] = True
    _fiberlyse_plot_get_legend_state(self, ax)['anchor_axes'] = tuple(anchor)
    _fiberlyse_plot_set_legend_cursor(self, 'sizing' if action == 'resize' else 'fleur')

def _fiberlyse_plot_on_legend_motion(self, event):
    state = getattr(self, '_legend_drag_state', None)
    if not state:
        hit = _fiberlyse_legend_contains_event(self, event)
        if hit is None:
            _fiberlyse_plot_set_legend_cursor(self, '')
        else:
            _ax, _leg, bbox, _renderer = hit
            cur = 'sizing' if (_fiberlyse_legend_event_has_resize_modifier(event) or _fiberlyse_legend_near_resize_area(bbox, event)) else 'fleur'
            _fiberlyse_plot_set_legend_cursor(self, cur)
        return
    if event is None or getattr(event, 'x', None) is None or getattr(event, 'y', None) is None:
        return
    ax = state.get('ax')
    if ax is None:
        return
    action = str(state.get('action', 'move'))
    st = _fiberlyse_plot_get_legend_state(self, ax)
    if action == 'resize':
        try:
            cx, cy = state.get('center_px', (float(event.x), float(event.y)))
            d0 = float(state.get('dist0', 80.0))
            d = float(max(2.0, ((float(event.x) - float(cx)) ** 2.0 + (float(event.y) - float(cy)) ** 2.0) ** 0.5))
            scale = _fiberlyse_legend_clamp(d / max(d0, 1e-9), 0.35, 3.25)
        except Exception:
            dx = float(event.x) - float(state.get('start_x', event.x))
            scale = _fiberlyse_legend_clamp(1.0 + dx / 220.0, 0.35, 3.25)
        fs = _fiberlyse_legend_clamp(float(state.get('fontsize0', 10.0)) * scale, 5.0, 48.0)
        tfs = _fiberlyse_legend_clamp(float(state.get('title_fontsize0', fs)) * scale, 5.0, 56.0)
        st['fontsize'] = fs
        st['title_fontsize'] = tfs
        st['manual_position'] = True
        st['anchor_axes'] = tuple(state.get('anchor0', (0.5, 0.5)))
    else:
        try:
            pointer_axes = ax.transAxes.inverted().transform((float(event.x), float(event.y)))
            off = state.get('offset', (0.0, 0.0))
            x = float(pointer_axes[0]) + float(off[0])
            y = float(pointer_axes[1]) + float(off[1])
        except Exception:
            x, y = state.get('anchor0', (0.5, 0.5))
        x = _fiberlyse_legend_clamp(x, -0.75, 1.75)
        y = _fiberlyse_legend_clamp(y, -0.75, 1.75)
        st['manual_position'] = True
        st['anchor_axes'] = (x, y)
    try:
        self._apply_legend_box_overrides()
    except Exception:
        pass
    try:
        self.canvas.draw_idle()
    except Exception:
        pass

def _fiberlyse_plot_on_legend_button_release(self, event):
    if getattr(self, '_legend_drag_state', None):
        self._legend_drag_state = None
        _fiberlyse_plot_set_legend_cursor(self, '')
        try:
            self._apply_legend_box_overrides()
            self.canvas.draw_idle()
        except Exception:
            pass

def _fiberlyse_plot_legend_axes_choices(self):
    choices = []
    for i, ax in enumerate(list(getattr(self.fig, 'axes', [])), start=1):
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is None:
            continue
        title = ''
        try:title = str(ax.get_title() or '').strip()
        except Exception:title = ''
        label = f'Axis {i}' + (f': {title}' if title else '')
        choices.append((label, ax, leg))
    return choices

def _fiberlyse_plot_show_legend_box_dialog(self):
    parent = self.winfo_toplevel()
    choices = self._legend_axes_choices()
    if not choices:
        try:messagebox.showinfo('Legend box', 'No legend is visible on the current graph.', parent=parent)
        except Exception:pass
        return
    top = tk.Toplevel(parent);top.title('Legend box');top.transient(parent);top.grab_set();top.resizable(True, False)
    frm = ttk.Frame(top, padding=10);frm.pack(fill=tk.BOTH, expand=True);frm.columnconfigure(1, weight=1)
    ttk.Label(frm, text='Move and resize the legend for the current plot. You can also drag legends directly on the graph.', justify='left').grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 8))
    labels = [c[0] for c in choices]
    axis_var = tk.StringVar(value=labels[0])
    ttk.Label(frm, text='Legend:').grid(row=1, column=0, sticky='w', padx=(0, 8), pady=4)
    cmb = ttk.Combobox(frm, values=labels, textvariable=axis_var, state='readonly', width=42)
    cmb.grid(row=1, column=1, columnspan=3, sticky='ew', pady=4)
    text_size = tk.DoubleVar(value=10.0)
    title_size = tk.DoubleVar(value=10.0)
    alpha_var = tk.DoubleVar(value=0.8)
    auto_pos = tk.BooleanVar(value=False)
    x_var = tk.DoubleVar(value=0.5)
    y_var = tk.DoubleVar(value=0.5)
    busy = {'value': False}

    def current_choice():
        lab = axis_var.get()
        for item in choices:
            if item[0] == lab:
                return item
        return choices[0]

    def load_current():
        busy['value'] = True
        try:
            _label, ax, leg = current_choice()
            state = _fiberlyse_plot_get_legend_state(self, ax)
            fs = _fiberlyse_legend_float(state.get('fontsize', None), None)
            if fs is None:
                fs = _fiberlyse_legend_text_fontsize(leg, 10.0)
            tfs = _fiberlyse_legend_float(state.get('title_fontsize', None), None)
            if tfs is None:
                tfs = _fiberlyse_legend_title_fontsize(leg, fs)
            alpha = _fiberlyse_legend_float(state.get('frame_alpha', None), None)
            if alpha is None:
                try:alpha = float(leg.get_frame().get_alpha())
                except Exception:alpha = 0.8
                if alpha is None or not np.isfinite(alpha):alpha = 0.8
            anchor = state.get('anchor_axes', None)
            if anchor is None:
                anchor = _fiberlyse_legend_current_anchor_axes(ax, leg)
            text_size.set(_fiberlyse_legend_clamp(fs, 5.0, 48.0))
            title_size.set(_fiberlyse_legend_clamp(tfs, 5.0, 56.0))
            alpha_var.set(_fiberlyse_legend_clamp(alpha, 0.0, 1.0))
            auto_pos.set(not bool(state.get('manual_position', False)))
            x_var.set(_fiberlyse_legend_clamp(anchor[0], -0.75, 1.75))
            y_var.set(_fiberlyse_legend_clamp(anchor[1], -0.75, 1.75))
        finally:
            busy['value'] = False

    def apply_live(show_errors=False):
        if busy['value']:
            return True
        try:
            _label, ax, _leg = current_choice()
            state = _fiberlyse_plot_get_legend_state(self, ax)
            state['fontsize'] = _fiberlyse_legend_clamp(text_size.get(), 5.0, 48.0)
            state['title_fontsize'] = _fiberlyse_legend_clamp(title_size.get(), 5.0, 56.0)
            state['frame_alpha'] = _fiberlyse_legend_clamp(alpha_var.get(), 0.0, 1.0)
            if bool(auto_pos.get()):
                state['manual_position'] = False
            else:
                state['manual_position'] = True
                state['anchor_axes'] = (_fiberlyse_legend_clamp(x_var.get(), -0.75, 1.75), _fiberlyse_legend_clamp(y_var.get(), -0.75, 1.75))
            self._apply_legend_box_overrides()
            self.canvas.draw_idle()
            return True
        except Exception as e:
            if show_errors:
                messagebox.showerror('Legend box', f'Could not apply legend settings:\n\n{e}', parent=top)
            return False

    def slider_row(row, label, var, lo, hi, fmt, callback):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=4)
        scale = ttk.Scale(frm, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var, command=lambda _v: callback())
        scale.grid(row=row, column=1, sticky='ew', padx=(0, 8), pady=4)
        entry_var = tk.StringVar(value=fmt(var.get()))
        entry = ttk.Entry(frm, textvariable=entry_var, width=10)
        entry.grid(row=row, column=2, sticky='w', padx=(0, 8), pady=4)
        def sync_entry(*_args):
            try:entry_var.set(fmt(var.get()))
            except Exception:pass
        def entry_return(_e=None):
            try:var.set(float((entry_var.get() or '').strip()))
            except Exception:
                messagebox.showerror('Legend box', f'Could not read {label} as a number.', parent=top)
                sync_entry();return 'break'
            callback();sync_entry();return 'break'
        try:var.trace_add('write', sync_entry)
        except Exception:pass
        entry.bind('<Return>', entry_return)
        return scale, entry

    slider_row(2, 'Text / box size:', text_size, 5.0, 48.0, lambda v: f'{float(v):.1f}', lambda: apply_live(False))
    slider_row(3, 'Title size:', title_size, 5.0, 56.0, lambda v: f'{float(v):.1f}', lambda: apply_live(False))
    slider_row(4, 'Frame opacity:', alpha_var, 0.0, 1.0, lambda v: f'{float(v):.2f}', lambda: apply_live(False))
    ttk.Checkbutton(frm, text='Automatic position', variable=auto_pos, command=lambda: apply_live(False)).grid(row=5, column=0, columnspan=4, sticky='w', pady=(6, 2))
    slider_row(6, 'Position X:', x_var, -0.75, 1.75, lambda v: f'{float(v):.3f}', lambda: (auto_pos.set(False), apply_live(False)))
    slider_row(7, 'Position Y:', y_var, -0.75, 1.75, lambda v: f'{float(v):.3f}', lambda: (auto_pos.set(False), apply_live(False)))
    ttk.Label(frm, text='Mouse: left-drag the legend to move. Drag a legend corner, or Shift/Ctrl-drag inside it, to resize.', foreground='gray35', justify='left').grid(row=8, column=0, columnspan=4, sticky='w', pady=(8, 0))
    btn_row = ttk.Frame(frm);btn_row.grid(row=9, column=0, columnspan=4, sticky='ew', pady=(12, 0))

    def on_axis_changed(_event=None):
        load_current()
    cmb.bind('<<ComboboxSelected>>', on_axis_changed)

    def reset_selected():
        _label, ax, _leg = current_choice()
        try:
            key = self._ax_key(ax)
        except Exception:
            key = id(ax)
        states = getattr(self, '_legend_box_overrides', {})
        if isinstance(states, dict) and key in states:
            states.pop(key, None)
        try:
            leg = ax.get_legend()
            if leg is not None:
                try:
                    if hasattr(leg, 'set_loc'):
                        leg.set_loc('best')
                    else:
                        leg._loc = 0
                except Exception:
                    try:leg._loc = 0
                    except Exception:pass
                try:leg.set_bbox_to_anchor(None)
                except Exception:pass
        except Exception:
            pass
        self.redraw()
        load_current()

    def close_dialog():
        try:top.grab_release()
        except Exception:pass
        try:top.destroy()
        except Exception:pass

    def do_ok(_event=None):
        if apply_live(True):
            close_dialog()
        return 'break'

    def do_cancel(_event=None):
        close_dialog();return 'break'

    ttk.Button(btn_row, text='Reset selected legend', command=reset_selected).pack(side=tk.LEFT)
    ttk.Button(btn_row, text='OK', command=do_ok).pack(side=tk.RIGHT)
    ttk.Button(btn_row, text='Close', command=do_cancel).pack(side=tk.RIGHT, padx=(0, 8))
    top.bind('<Return>', do_ok)
    top.bind('<Escape>', do_cancel)
    load_current()
    parent.wait_window(top)

PlotTabTk._legend_axes_choices = _fiberlyse_plot_legend_axes_choices
PlotTabTk._apply_legend_box_overrides = _fiberlyse_plot_apply_legend_box_overrides
PlotTabTk.show_legend_box_dialog = _fiberlyse_plot_show_legend_box_dialog

_FIBERLYSE_ORIGINAL_APPLY_USER_OVERRIDES_FOR_LEGENDS = PlotTabTk._apply_user_overrides
def _fiberlyse_apply_user_overrides_with_legends(self):
    _FIBERLYSE_ORIGINAL_APPLY_USER_OVERRIDES_FOR_LEGENDS(self)
    try:self._apply_legend_box_overrides()
    except Exception:pass
PlotTabTk._apply_user_overrides = _fiberlyse_apply_user_overrides_with_legends

_FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_LEGENDS = PlotTabTk.redraw
def _fiberlyse_plottabtk_redraw_with_legends(self):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_REDRAW_FOR_LEGENDS(self)
    try:self._apply_legend_box_overrides()
    except Exception:pass
    try:self.canvas.draw_idle()
    except Exception:pass
PlotTabTk.redraw = _fiberlyse_plottabtk_redraw_with_legends

_FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_LEGENDS = PlotTabTk.__init__
def _fiberlyse_plottabtk_init_with_legends(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_LEGENDS(self, *args, **kwargs)
    try:self._legend_box_overrides = {}
    except Exception:pass
    try:self._legend_drag_state = None
    except Exception:pass
    try:
        btn_parent = self.save_btn.master
        self.legend_box_btn = ttk.Button(btn_parent, text='Legend box...', command=self.show_legend_box_dialog)
        self.legend_box_btn.pack(side=tk.RIGHT, padx=(0, 8))
    except Exception:
        pass
    try:self._cid_legend_press = self.canvas.mpl_connect('button_press_event', self._on_legend_button_press)
    except Exception:pass
    try:self._cid_legend_motion = self.canvas.mpl_connect('motion_notify_event', self._on_legend_motion)
    except Exception:pass
    try:self._cid_legend_release = self.canvas.mpl_connect('button_release_event', self._on_legend_button_release)
    except Exception:pass
PlotTabTk.__init__ = _fiberlyse_plottabtk_init_with_legends
PlotTabTk._on_legend_button_press = _fiberlyse_plot_on_legend_button_press
PlotTabTk._on_legend_motion = _fiberlyse_plot_on_legend_motion
PlotTabTk._on_legend_button_release = _fiberlyse_plot_on_legend_button_release


def _fiberlyse_install_legend_hotkeys(self):
    self.root.bind_all('<Control-l>', self._on_ctrl_l_legend_box, add='+')
    self.root.bind_all('<Control-L>', self._on_ctrl_l_legend_box, add='+')

def _fiberlyse_on_ctrl_l_legend_box(self, _event=None):
    tab = self.find_active_plot_tab()
    if tab is None:return
    tab.show_legend_box_dialog()

MainAppTk._install_legend_hotkeys = _fiberlyse_install_legend_hotkeys
MainAppTk._on_ctrl_l_legend_box = _fiberlyse_on_ctrl_l_legend_box
_FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_LEGENDS = MainAppTk.__init__
def _fiberlyse_mainapptk_init_with_legends(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_LEGENDS(self, *args, **kwargs)
    try:self._install_legend_hotkeys()
    except Exception:pass
MainAppTk.__init__ = _fiberlyse_mainapptk_init_with_legends
# ---- End Fiberlyse draggable and resizable legend extension ----


# ---- Begin Fiberlyse synced legend settings across analysis files ----
# When a legend box is moved/resized or edited on one plot, copy the same
# legend settings to the same plot type for every loaded file/channel in the
# current analysis. The stored setting is also applied to lazy-loaded tabs.
try:
    import copy as _fiberlyse_legend_sync_copy
except Exception:
    _fiberlyse_legend_sync_copy = None

def _fiberlyse_legend_sync_deepcopy(obj, fallback=None):
    try:
        if _fiberlyse_legend_sync_copy is not None:
            return _fiberlyse_legend_sync_copy.deepcopy(obj)
    except Exception:
        pass
    try:
        if isinstance(obj, dict):
            return {k: _fiberlyse_legend_sync_deepcopy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fiberlyse_legend_sync_deepcopy(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_fiberlyse_legend_sync_deepcopy(v) for v in obj)
    except Exception:
        pass
    if obj is None:
        return fallback
    return obj

def _fiberlyse_legend_sync_find_app(widget):
    try:
        root = widget.winfo_toplevel()
        app = getattr(root, '_fiberlyse_app', None)
        if app is not None:
            return app
    except Exception:
        pass
    return None

def _fiberlyse_legend_sync_has_legend(plot_tab) -> bool:
    try:
        for ax in list(getattr(plot_tab.fig, 'axes', [])):
            try:
                leg = ax.get_legend()
            except Exception:
                leg = None
            if leg is not None:
                return True
    except Exception:
        pass
    return False

def _fiberlyse_legend_sync_snapshot(plot_tab):
    try:
        plot_tab._apply_legend_box_overrides()
    except Exception:
        pass
    return {
        'legend_box_overrides': _fiberlyse_legend_sync_deepcopy(getattr(plot_tab, '_legend_box_overrides', {}), {}),
        'legend_title_overrides': _fiberlyse_legend_sync_deepcopy(getattr(plot_tab, '_legend_title_overrides', {}), {}),
        'legend_label_overrides': _fiberlyse_legend_sync_deepcopy(getattr(plot_tab, '_legend_label_overrides', {}), {}),
        'color_overrides': _fiberlyse_legend_sync_deepcopy(getattr(plot_tab, '_color_overrides', {}), {}),
    }

def _fiberlyse_legend_sync_apply_to_plot(plot_tab, snapshot, redraw: bool=True) -> None:
    if not isinstance(snapshot, dict):
        return
    try:
        setattr(plot_tab, '_legend_box_overrides', _fiberlyse_legend_sync_deepcopy(snapshot.get('legend_box_overrides', {}), {}))
    except Exception:
        pass
    for attr, key in [
        ('_legend_title_overrides', 'legend_title_overrides'),
        ('_legend_label_overrides', 'legend_label_overrides'),
        ('_color_overrides', 'color_overrides'),
    ]:
        try:
            setattr(plot_tab, attr, _fiberlyse_legend_sync_deepcopy(snapshot.get(key, {}), {}))
        except Exception:
            pass
    if redraw:
        try:
            plot_tab.redraw()
            return
        except Exception:
            pass
        try:
            plot_tab._apply_user_overrides()
        except Exception:
            pass
        try:
            plot_tab._apply_legend_box_overrides()
        except Exception:
            pass
        try:
            plot_tab.canvas.draw_idle()
        except Exception:
            pass

def _fiberlyse_legend_sync_iter_plot_tabs(app):
    seen = set()
    try:
        widgets = list(getattr(app, '_channel_widgets', {}).values())
    except Exception:
        widgets = []
    attr_names = ['tab_raw', 'tab_slope', 'tab_art', 'tab_fit', 'tab_norm', 'tab_norm_smooth', 'tab_freq']
    for cw in widgets:
        for name in attr_names:
            try:
                tab = getattr(cw, name, None)
            except Exception:
                tab = None
            if tab is None:
                continue
            if id(tab) in seen:
                continue
            seen.add(id(tab))
            yield tab
    try:
        cmpw = getattr(app, 'compare_widget', None)
        if cmpw is not None and getattr(cmpw, 'plot', None) is not None and id(cmpw.plot) not in seen:
            seen.add(id(cmpw.plot));yield cmpw.plot
    except Exception:
        pass
    try:
        avgw = getattr(app, 'average_widget', None)
        if avgw is not None and getattr(avgw, 'plot', None) is not None and id(avgw.plot) not in seen:
            seen.add(id(avgw.plot));yield avgw.plot
    except Exception:
        pass

def _fiberlyse_legend_sync_publish(plot_tab, reason: str='legend') -> None:
    try:
        if bool(getattr(plot_tab, '_fiberlyse_legend_sync_suppress', False)):
            return
    except Exception:
        pass
    app = _fiberlyse_legend_sync_find_app(plot_tab)
    if app is None:
        return
    tab_name = str(getattr(plot_tab, 'tab_name', '') or '')
    if not tab_name:
        return
    snapshot = _fiberlyse_legend_sync_snapshot(plot_tab)
    try:
        styles = getattr(app, '_fiberlyse_synced_legend_styles_by_tab', None)
        if not isinstance(styles, dict):
            styles = {}
            setattr(app, '_fiberlyse_synced_legend_styles_by_tab', styles)
        styles[tab_name] = _fiberlyse_legend_sync_deepcopy(snapshot, {})
    except Exception:
        pass
    changed = 0
    for other in _fiberlyse_legend_sync_iter_plot_tabs(app):
        if other is plot_tab:
            continue
        try:
            if str(getattr(other, 'tab_name', '') or '') != tab_name:
                continue
            setattr(other, '_fiberlyse_legend_sync_suppress', True)
            _fiberlyse_legend_sync_apply_to_plot(other, snapshot, redraw=True)
            changed += 1
        except Exception:
            pass
        finally:
            try:setattr(other, '_fiberlyse_legend_sync_suppress', False)
            except Exception:pass
    try:
        if getattr(app, 'status', None) is not None:
            if changed > 0:
                app.status.set(f"Legend settings synced for '{tab_name}' across loaded files/channels.")
            else:
                app.status.set(f"Legend settings saved for '{tab_name}' and will be used on other files/channels when opened.")
    except Exception:
        pass

def _fiberlyse_legend_sync_apply_saved_style(plot_tab) -> None:
    app = _fiberlyse_legend_sync_find_app(plot_tab)
    if app is None:
        return
    try:
        styles = getattr(app, '_fiberlyse_synced_legend_styles_by_tab', {})
        if not isinstance(styles, dict):
            return
        tab_name = str(getattr(plot_tab, 'tab_name', '') or '')
        snapshot = styles.get(tab_name)
        if snapshot is None:
            return
        setattr(plot_tab, '_fiberlyse_legend_sync_suppress', True)
        _fiberlyse_legend_sync_apply_to_plot(plot_tab, snapshot, redraw=False)
    except Exception:
        pass
    finally:
        try:setattr(plot_tab, '_fiberlyse_legend_sync_suppress', False)
        except Exception:pass

_FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_LEGEND_SYNC = MainAppTk.__init__
def _fiberlyse_mainapptk_init_with_legend_sync(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_MAINAPPTK_INIT_FOR_LEGEND_SYNC(self, *args, **kwargs)
    try:setattr(self.root, '_fiberlyse_app', self)
    except Exception:pass
    try:
        if not isinstance(getattr(self, '_fiberlyse_synced_legend_styles_by_tab', None), dict):
            self._fiberlyse_synced_legend_styles_by_tab = {}
    except Exception:
        pass
MainAppTk.__init__ = _fiberlyse_mainapptk_init_with_legend_sync

_FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_LEGEND_SYNC = PlotTabTk.__init__
def _fiberlyse_plottabtk_init_with_legend_sync(self, *args, **kwargs):
    _FIBERLYSE_ORIGINAL_PLOTTABTK_INIT_FOR_LEGEND_SYNC(self, *args, **kwargs)
    try:_fiberlyse_legend_sync_apply_saved_style(self)
    except Exception:pass
PlotTabTk.__init__ = _fiberlyse_plottabtk_init_with_legend_sync

_FIBERLYSE_ORIGINAL_LEGEND_BUTTON_RELEASE_FOR_SYNC = PlotTabTk._on_legend_button_release
def _fiberlyse_plot_on_legend_button_release_with_sync(self, event):
    try:was_dragging = bool(getattr(self, '_legend_drag_state', None))
    except Exception:was_dragging = False
    result = _FIBERLYSE_ORIGINAL_LEGEND_BUTTON_RELEASE_FOR_SYNC(self, event)
    if was_dragging:
        try:_fiberlyse_legend_sync_publish(self, reason='drag')
        except Exception:pass
    return result
PlotTabTk._on_legend_button_release = _fiberlyse_plot_on_legend_button_release_with_sync

_FIBERLYSE_ORIGINAL_SHOW_LEGEND_BOX_DIALOG_FOR_SYNC = PlotTabTk.show_legend_box_dialog
def _fiberlyse_plot_show_legend_box_dialog_with_sync(self, *args, **kwargs):
    had_legend = _fiberlyse_legend_sync_has_legend(self)
    result = _FIBERLYSE_ORIGINAL_SHOW_LEGEND_BOX_DIALOG_FOR_SYNC(self, *args, **kwargs)
    try:
        if had_legend or _fiberlyse_legend_sync_has_legend(self):
            _fiberlyse_legend_sync_publish(self, reason='dialog')
    except Exception:
        pass
    return result
PlotTabTk.show_legend_box_dialog = _fiberlyse_plot_show_legend_box_dialog_with_sync

_FIBERLYSE_ORIGINAL_ON_MPL_BUTTON_PRESS_FOR_LEGEND_SYNC = PlotTabTk._on_mpl_button_press
def _fiberlyse_plot_on_mpl_button_press_with_legend_sync(self, event):
    try:
        in_legend = _fiberlyse_legend_contains_event(self, event) is not None
    except Exception:
        in_legend = False
    result = _FIBERLYSE_ORIGINAL_ON_MPL_BUTTON_PRESS_FOR_LEGEND_SYNC(self, event)
    if in_legend:
        try:_fiberlyse_legend_sync_publish(self, reason='legend_text_or_color')
        except Exception:pass
    return result
PlotTabTk._on_mpl_button_press = _fiberlyse_plot_on_mpl_button_press_with_legend_sync

# Refresh existing lazy-created channel widgets with any saved legend style after creation.
_FIBERLYSE_ORIGINAL_ENSURE_MOUSE_WIDGET_FOR_LEGEND_SYNC = MainAppTk._ensure_mouse_widget
def _fiberlyse_ensure_mouse_widget_with_legend_sync(self, mid: str) -> None:
    existed = False
    try:existed = mid in getattr(self, '_channel_widgets', {})
    except Exception:existed = False
    _FIBERLYSE_ORIGINAL_ENSURE_MOUSE_WIDGET_FOR_LEGEND_SYNC(self, mid)
    if existed:
        return
    try:
        widget = getattr(self, '_channel_widgets', {}).get(mid)
    except Exception:
        widget = None
    if widget is None:
        return
    try:
        for tab in _fiberlyse_legend_sync_iter_plot_tabs(self):
            try:_fiberlyse_legend_sync_apply_saved_style(tab)
            except Exception:pass
    except Exception:
        pass
MainAppTk._ensure_mouse_widget = _fiberlyse_ensure_mouse_widget_with_legend_sync
# ---- End Fiberlyse synced legend settings across analysis files ----

if __name__ == '__main__':main()

"""Koopman diagnostic experiments on S&P 500 observables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eigenworld.manifold.diffusion import DiffusionMap
from eigenworld.koopman.edmd import KoopmanEDMD


def load_sp500(data_path: Path) -> pd.DataFrame:
    """Load S&P 500 data from the CSV file."""
    df = pd.read_csv(
        data_path,
        skiprows=3,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_return_series(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return log returns, absolute returns, volume pulse, and aligned dates."""
    close = df["Close"].astype(float).to_numpy()
    volume = df["Volume"].astype(float).to_numpy()
    dates = pd.to_datetime(df["Date"]).to_numpy()

    returns = np.log(close[1:] / close[:-1])
    volume_safe = np.where(volume <= 0.0, np.nan, volume)
    volume_pulse = np.log(volume_safe[1:] / volume_safe[:-1])
    vol = np.abs(returns)
    dates = dates[1:]

    mask = np.isfinite(returns) & np.isfinite(volume_pulse)
    return returns[mask], vol[mask], volume_pulse[mask], dates[mask]


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the 7-feature matrix and aligned dates."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    log_return = np.log(close / close.shift(1))
    rolling_vol_21 = log_return.rolling(21).std()
    rolling_vol_63 = log_return.rolling(63).std()
    vol_ratio = rolling_vol_21 / rolling_vol_63
    rolling_return_21 = log_return.rolling(21).sum()
    rolling_return_63 = log_return.rolling(63).sum()
    vol_mean_63 = volume.rolling(63).mean()
    vol_std_63 = volume.rolling(63).std()
    volume_zscore = (volume - vol_mean_63) / vol_std_63

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "rolling_vol_21": rolling_vol_21,
            "rolling_vol_63": rolling_vol_63,
            "vol_ratio": vol_ratio,
            "rolling_return_21": rolling_return_21,
            "rolling_return_63": rolling_return_63,
            "volume_zscore": volume_zscore,
        }
    ).dropna()

    dates = pd.to_datetime(df.loc[features.index, "Date"]).to_numpy()
    return features.to_numpy(), dates


def build_lag_embedding(
    series: np.ndarray, lags: list[int]
) -> tuple[np.ndarray, int]:
    """Build Takens delay embedding with explicit lag set."""
    series = np.asarray(series, dtype=float).ravel()
    lags_sorted = sorted(int(lag) for lag in lags)
    if lags_sorted[0] < 1:
        raise ValueError("Lags must be positive integers.")
    max_lag = lags_sorted[-1]
    if series.size <= max_lag:
        raise ValueError("Not enough samples for requested lags.")

    n_rows = series.size - max_lag
    X = np.empty((n_rows, 1 + len(lags_sorted)), dtype=float)
    for i in range(n_rows):
        idx = i + max_lag
        X[i, 0] = series[idx]
        for j, lag in enumerate(lags_sorted, start=1):
            X[i, j] = series[idx - lag]
    return X, max_lag


def build_augmented_embedding(
    returns: np.ndarray, vol: np.ndarray
) -> tuple[np.ndarray, int]:
    """Build returns + absolute returns embedding with 4 lags."""
    returns = np.asarray(returns, dtype=float).ravel()
    vol = np.asarray(vol, dtype=float).ravel()
    if returns.size != vol.size:
        raise ValueError("Returns and vol arrays must have the same length.")

    max_lag = 4
    if returns.size <= max_lag:
        raise ValueError("Not enough samples for augmented embedding.")

    n_rows = returns.size - max_lag
    X = np.empty((n_rows, 10), dtype=float)
    for i in range(n_rows):
        idx = i + max_lag
        X[i, 0:5] = [
            returns[idx],
            returns[idx - 1],
            returns[idx - 2],
            returns[idx - 3],
            returns[idx - 4],
        ]
        X[i, 5:10] = [
            vol[idx],
            vol[idx - 1],
            vol[idx - 2],
            vol[idx - 3],
            vol[idx - 4],
        ]
    return X, max_lag


def build_vol_pulse_embedding(
    returns: np.ndarray, vol: np.ndarray, volume_pulse: np.ndarray
) -> tuple[np.ndarray, int]:
    """Build returns + absolute returns + volume pulse embedding with 2 lags."""
    returns = np.asarray(returns, dtype=float).ravel()
    vol = np.asarray(vol, dtype=float).ravel()
    volume_pulse = np.asarray(volume_pulse, dtype=float).ravel()
    if not (returns.size == vol.size == volume_pulse.size):
        raise ValueError("Input arrays must have the same length.")

    max_lag = 2
    if returns.size <= max_lag:
        raise ValueError("Not enough samples for volume pulse embedding.")

    n_rows = returns.size - max_lag
    X = np.empty((n_rows, 7), dtype=float)
    for i in range(n_rows):
        idx = i + max_lag
        X[i, :] = [
            returns[idx],
            returns[idx - 1],
            returns[idx - 2],
            vol[idx],
            vol[idx - 1],
            vol[idx - 2],
            volume_pulse[idx],
        ]
    return X, max_lag


def extract_windows(
    X: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    target_date: np.datetime64,
) -> dict[str, np.ndarray]:
    """Extract stable, pre-crash, and post-crash windows by date."""
    if X.shape[0] < window_size:
        raise ValueError("Not enough rows for requested window size.")

    stable_window = X[:window_size]

    pre_crash_window = extract_window_ending_at(
        X,
        dates,
        window_size,
        target_date,
        label="Pre-crash",
    )

    post_crash_window = X[-window_size:]

    return {
        "Stable": stable_window,
        "Pre-crash": pre_crash_window,
        "Post-crash": post_crash_window,
    }


def extract_window_ending_at(
    X: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    target_date: np.datetime64,
    label: str,
) -> np.ndarray:
    """Extract a window ending on the row closest to target_date."""
    start, end = window_indices_ending_at(dates, window_size, target_date, label)
    window = X[start:end]
    if window.shape[0] != window_size:
        raise ValueError(f"{label} window must match window size.")
    return window


def extract_window_ending_at_with_dates(
    X: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    target_date: np.datetime64,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a window ending on target_date and return window dates."""
    start, end = window_indices_ending_at(dates, window_size, target_date, label)
    window = X[start:end]
    window_dates = dates[start:end]
    if window.shape[0] != window_size:
        raise ValueError(f"{label} window must match window size.")
    if window_dates.shape[0] != window_size:
        raise ValueError(f"{label} window dates must match window size.")
    return window, window_dates


def window_indices_ending_at(
    dates: np.ndarray,
    window_size: int,
    target_date: np.datetime64,
    label: str,
) -> tuple[int, int]:
    """Compute window indices ending on the row closest to target_date."""
    idx = int(np.argmin(np.abs(dates - target_date)))
    end = idx + 1
    start = end - window_size
    if start < 0:
        raise ValueError(f"{label} window start is before the data range.")
    return start, end


def extract_window_starting_at(
    X: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    target_date: np.datetime64,
    label: str,
) -> np.ndarray:
    """Extract a window starting on the row closest to target_date."""
    idx = int(np.argmin(np.abs(dates - target_date)))
    start = idx
    end = start + window_size
    if end > X.shape[0]:
        raise ValueError(f"{label} window end is beyond the data range.")
    window = X[start:end]
    if window.shape[0] != window_size:
        raise ValueError(f"{label} window must match window size.")
    return window


def extract_four_windows(
    X: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    pre_crisis_date: np.datetime64,
    pre_crash_date: np.datetime64,
) -> dict[str, np.ndarray]:
    """Extract stable, pre-crisis, pre-crash, and post-crash windows."""
    if X.shape[0] < window_size:
        raise ValueError("Not enough rows for requested window size.")

    stable_window = X[:window_size]
    pre_crisis_window = extract_window_ending_at(
        X,
        dates,
        window_size,
        pre_crisis_date,
        label="Pre-crisis",
    )
    pre_crash_window = extract_window_ending_at(
        X,
        dates,
        window_size,
        pre_crash_date,
        label="Pre-crash",
    )
    post_crash_window = X[-window_size:]

    return {
        "Stable": stable_window,
        "Pre-crisis": pre_crisis_window,
        "Pre-crash": pre_crash_window,
        "Post-crash": post_crash_window,
    }


def format_date_range(dates: np.ndarray) -> str:
    """Format a start/end date range for display."""
    start = pd.to_datetime(dates[0]).date().isoformat()
    end = pd.to_datetime(dates[-1]).date().isoformat()
    return f"{start} -> {end}"


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "sp500.csv"
    df = load_sp500(data_path)
    features, feature_dates = build_feature_matrix(df)
    pre_crash_date = np.datetime64("2008-08-27")
    stable_start_date = np.datetime64("2005-01-03")

    for window_size in (63, 126, 252):
        n_components = min(10, window_size // 10)
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")

        stable_window = extract_window_starting_at(
            features,
            feature_dates,
            window_size,
            stable_start_date,
            label="Stable",
        )
        pre_crash_window = extract_window_ending_at(
            features,
            feature_dates,
            window_size,
            pre_crash_date,
            label="Pre-crash",
        )

        for name, X_window in ("Stable", stable_window), ("Pre-crash", pre_crash_window):
            X_scaled = StandardScaler().fit_transform(X_window)
            dmap = DiffusionMap(n_components=n_components, epsilon_method="median")
            diff_coords = dmap.fit_transform(X_scaled)
            model = KoopmanEDMD(rank=n_components)
            model.fit(diff_coords)
            summary = model.stability_summary()
            radius = summary["spectral_radius"]
            margin = 1.0 - radius
            print(
                f"Exp5 size={window_size} Window={name:<9} | radius={radius:.6f} "
                f"| n_unstable={summary['n_unstable']} | margin={margin:.4f}"
            )

    print("Experiment 6: Diffusion/Koopman window diagnostics")
    targets = [
        ("2008-03-31", "Spike"),
        ("2007-10-26", "Pre-crisis"),
        ("2008-08-27", "Pre-crash"),
        ("2009-06-29", "Post-crash"),
    ]
    for date_str, label in targets:
        target_date = np.datetime64(date_str)
        x_window, date_window = extract_window_ending_at_with_dates(
            features,
            feature_dates,
            window_size=500,
            target_date=target_date,
            label=label,
        )
        x_scaled = StandardScaler().fit_transform(x_window)
        dmap = DiffusionMap(n_components=10, epsilon_method="median")
        diff_coords = dmap.fit_transform(x_scaled)
        koopman_input = diff_coords[-63:]
        koopman_dates = date_window[-63:]

        edmd = KoopmanEDMD(rank=10)
        edmd.fit(koopman_input)
        summary = edmd.stability_summary()

        print(f"Exp6 target={date_str} label={label}")
        print(f"Diffusion window dates: {format_date_range(date_window)}")
        print(f"Koopman window dates: {format_date_range(koopman_dates)}")
        print(f"Diffusion coords shape: {diff_coords.shape}")
        print(f"Koopman input shape: {koopman_input.shape}")
        print(f"Stability summary: {summary}")
        print(f"Eigenvalue moduli: {summary['eigenvalue_moduli']}")
        print("")

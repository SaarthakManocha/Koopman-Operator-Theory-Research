"""Sliding-window diffusion map pipeline for manifold metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eigenworld.manifold.diffusion import DiffusionMap
from eigenworld.manifold.metrics import manifold_summary


class ManifoldPipeline:
	"""Compute diffusion-map metrics over rolling windows of OHLCV data."""

	def __init__(self, n_components: int = 10, window_size: int = 500, step_size: int = 21) -> None:
		"""Store pipeline parameters without creating any fitted state.

		Math:
			The window size and step size define a sliding subsequence of the
			feature matrix used to construct the Markov kernel in each window.
		"""
		self.n_components = int(n_components)
		self.window_size = int(window_size)
		self.step_size = int(step_size)
		self._feature_dates_ = None

	def _build_features(self, df: pd.DataFrame) -> np.ndarray:
		"""Engineer feature matrix from raw OHLCV dataframe.

		Math:
			Compute log returns, rolling volatilities, return sums, and a
			volume z-score to expose multi-scale dynamics. Drop rows with
			NaNs produced by rolling windows, then return the feature matrix.
		"""
		required = {"Close", "Volume"}
		missing = required - set(df.columns)
		if missing:
			raise ValueError(f"Missing required columns: {sorted(missing)}")

		close = df["Close"].astype(float)
		volume = df["Volume"].astype(float)

		log_return = np.log(close / close.shift(1))
		rolling_vol_21 = log_return.rolling(21).std()
		rolling_vol_63 = log_return.rolling(63).std()
		vol_ratio = rolling_vol_21 / rolling_vol_63
		rolling_return_21 = log_return.rolling(21).sum()
		rolling_return_63 = log_return.rolling(63).sum()
		volume_mean_63 = volume.rolling(63).mean()
		volume_std_63 = volume.rolling(63).std()
		volume_zscore = (volume - volume_mean_63) / volume_std_63

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
		)

		features = features.dropna()
		if len(features) < self.window_size:
			raise ValueError("Not enough rows after feature engineering.")

		if "Date" in df.columns:
			dates = pd.to_datetime(df.loc[features.index, "Date"])
		else:
			dates = pd.to_datetime(df.index)
		self._feature_dates_ = dates.to_numpy()
		return features.to_numpy()

	def run(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Run sliding window diffusion map over the full dataframe.

		Math:
			Each window is standardized, mapped with diffusion coordinates, and
			converted into spectral metrics that summarize manifold dynamics.
		"""
		if self.window_size > 2000:
			raise ValueError("window_size must not exceed 2000.")

		X = self._build_features(df)
		dates = np.asarray(self._feature_dates_)
		n_rows = X.shape[0]
		if n_rows < self.window_size:
			raise ValueError("Not enough rows for the requested window size.")

		total = ((n_rows - self.window_size) // self.step_size) + 1
		records = []

		for i, start in enumerate(range(0, n_rows - self.window_size + 1, self.step_size), start=1):
			end = start + self.window_size
			X_window = X[start:end]
			X_scaled = StandardScaler().fit_transform(X_window)
			dmap = DiffusionMap(n_components=self.n_components, epsilon_method="median")
			dmap.fit(X_scaled)
			metrics = manifold_summary(dmap)
			date = pd.to_datetime(dates[end - 1])
			records.append(
				{
					"date": date,
					"spectral_entropy": metrics["spectral_entropy"],
					"spectral_gap": metrics["spectral_gap"],
					"eigenvalue_decay": metrics["eigenvalue_decay"],
					"epsilon": metrics["epsilon"],
				}
			)

			if i % 10 == 0:
				print(
					f"Window {i}/{total} | date={date.date()} | "
					f"entropy={metrics['spectral_entropy']:.3f}"
				)

		result = pd.DataFrame.from_records(records).set_index("date")
		return result


if __name__ == "__main__":
	data_path = Path(__file__).resolve().parents[1] / "data" / "sp500.csv"
	df = pd.read_csv(
		data_path,
		skiprows=3,
		names=["Date", "Close", "High", "Low", "Open", "Volume"],
	)
	df["Date"] = pd.to_datetime(df["Date"])

	pipeline = ManifoldPipeline(n_components=10, window_size=500, step_size=21)
	result = pipeline.run(df)
	print(result.head(3))
	print(result.tail(3))
	print("pipeline.py verified")

	output_path = Path(__file__).resolve().parents[1] / "data" / "manifold_metrics.csv"
	result.to_csv(output_path)

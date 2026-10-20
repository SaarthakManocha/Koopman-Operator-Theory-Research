"""Two-stage diffusion-map + Koopman EDMD pipeline for S&P 500 features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eigenworld.koopman.edmd import KoopmanEDMD
from eigenworld.manifold.diffusion import DiffusionMap


class KoopmanPipeline:
	"""Two-stage sliding window pipeline."""

	def __init__(
		self,
		n_components: int = 6,
		rank: int = 6,
		diff_window: int = 500,
		koopman_window: int = 63,
		step_size: int = 21,
	) -> None:
		"""Initialize pipeline parameters and validate constraints."""
		self.n_components = int(n_components)
		self.rank = int(rank)
		self.diff_window = int(diff_window)
		self.koopman_window = int(koopman_window)
		self.step_size = int(step_size)
		if self.diff_window > 2000:
			raise ValueError("diff_window must not exceed 2000.")

	def run(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Two-stage sliding window pipeline.

		Stage 1 — Diffusion Map (large window):
			Fit DiffusionMap on feature windows to extract manifold coordinates.

		Stage 2 — Koopman EDMD (small window):
			Run KoopmanEDMD on the last koopman_window rows of diffusion coords.
		"""
		required = {"Date", "Close", "Volume"}
		if not required.issubset(df.columns):
			missing = ", ".join(sorted(required - set(df.columns)))
			raise ValueError(f"Input dataframe missing columns: {missing}.")

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
		values = features.to_numpy()

		if values.shape[0] < self.diff_window:
			raise ValueError("Not enough data for the requested diff_window.")

		n_rows = values.shape[0]
		total = ((n_rows - self.diff_window) // self.step_size) + 1
		records = []

		for i, start in enumerate(
			range(0, n_rows - self.diff_window + 1, self.step_size), start=1
		):
			end = start + self.diff_window
			x_window = values[start:end]
			x_scaled = StandardScaler().fit_transform(x_window)
			dmap = DiffusionMap(n_components=self.n_components, epsilon_method="median")
			diff_coords = dmap.fit_transform(x_scaled)
			koopman_input = diff_coords[-self.koopman_window :]

			edmd = KoopmanEDMD(rank=self.rank)
			edmd.fit(koopman_input)
			summary = edmd.stability_summary()
			date = pd.to_datetime(dates[end - 1])
			margin = 1.0 - summary["spectral_radius"]
			records.append(
				{
					"date": date,
					"spectral_radius": summary["spectral_radius"],
					"n_unstable": summary["n_unstable"],
					"dominant_frequency": summary["dominant_frequency"],
					"stability_margin": margin,
				}
			)

			if i % 5 == 0:
				print(
					f"Window {i}/{total} | date={date.date()} | "
					f"radius={summary['spectral_radius']:.4f} | margin={margin:.4f}"
				)

		return pd.DataFrame.from_records(records).set_index("date")


if __name__ == "__main__":
	data_path = Path(__file__).resolve().parents[1] / "data" / "sp500.csv"
	df = pd.read_csv(
		data_path,
		skiprows=3,
		names=["Date", "Close", "High", "Low", "Open", "Volume"],
	)
	df["Date"] = pd.to_datetime(df["Date"])

	pipeline = KoopmanPipeline()
	result = pipeline.run(df)
	print(result.head(3))
	print(result.tail(3))

	output_path = Path(__file__).resolve().parents[1] / "data" / "koopman_metrics.csv"
	result.to_csv(output_path)
	print("koopman pipeline verified")

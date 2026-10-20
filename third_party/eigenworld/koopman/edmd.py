"""Koopman EDMD implementation using only numpy and scipy."""

from __future__ import annotations

import numpy as np
from scipy import linalg as scipy_linalg


class KoopmanEDMD:
	"""Extended Dynamic Mode Decomposition for scalar time series."""

	def __init__(self, n_delays: int = 10, rank: int = 30) -> None:
		"""Initialize EDMD hyperparameters.

		Math:
			n_delays controls the delay-embedding dimension; rank truncates the
			SVD of the Gram matrix for stability.
		"""
		self.n_delays = int(n_delays)
		self.rank = int(rank)
		if self.rank > 50:
			raise ValueError("rank must not exceed 50.")
		self._is_fitted = False

	def _build_hankel(self, x: np.ndarray) -> np.ndarray:
		"""Build delay-embedding (Hankel) matrix from 1D time series.

		Math:
			Given scalar time series x of length T, build matrix H where:
			H[i] = [x[i], x[i-1], x[i-2], ..., x[i-n_delays+1]]

			Result shape: (T - n_delays + 1, n_delays)
			Each row is the state at time i, embedding the past n_delays steps.
			This lifts the scalar series into a proper n_delays-dimensional
			state space, recovering attractor geometry via Takens theorem.
		"""
		x = np.asarray(x, dtype=float).ravel()
		if x.size < self.n_delays:
			raise ValueError("Time series length must be >= n_delays.")
		T = x.size
		N = T - self.n_delays + 1
		H = np.empty((N, self.n_delays), dtype=float)
		for k in range(self.n_delays):
			H[:, k] = x[self.n_delays - 1 - k : T - k]
		return H

	def fit(self, x: np.ndarray) -> "KoopmanEDMD":
		"""Fit Koopman operator via EDMD on scalar time series x.

		Math:
			1. Build Hankel matrix H = _build_hankel(x)
			   Shape: (N, n_delays) where N = len(x) - n_delays + 1

			2. Split into current and next state:
			   X  = H[:-1]   shape (N-1, n_delays)
			   Xp = H[1:]    shape (N-1, n_delays)

			3. Compute gram and cross-correlation matrices:
			   G = X.T @ X   shape (n_delays, n_delays)
			   A = X.T @ Xp  shape (n_delays, n_delays)

			4. Truncated SVD of G for numerical stability:
			   U, s, Vt = svd(G, full_matrices=False)
			   Keep only top self.rank singular values.
			   G_pinv = V @ diag(1/s_r) @ U_r.T

			5. Koopman matrix:
			   K = G_pinv @ A

			6. Eigendecompose K:
			   eigenvalues, eigenvectors = eig(K)
			   Sort by |eigenvalue| descending.
		"""
		x = np.asarray(x, dtype=float)
		if x.ndim == 1:
			x = x.ravel()
			if x.size > 2000:
				raise ValueError("EDMD cap exceeded: max 2000 samples.")
			self.x_mean_ = np.mean(x)
			self.x_std_ = np.std(x)
			if self.x_std_ < 1e-10:
				raise ValueError("Input signal has near-zero variance.")
			x_norm = (x - self.x_mean_) / self.x_std_
			H = self._build_hankel(x_norm)
			X = H[:-1]
			Xp = H[1:]
		elif x.ndim == 2:
			if x.shape[0] > 2000:
				raise ValueError("EDMD cap exceeded: max 2000 samples.")
			X = x[:-1]
			Xp = x[1:]
			self.x_mean_ = X.mean(axis=0)
			self.x_std_ = X.std(axis=0)
			self.x_std_[self.x_std_ < 1e-10] = 1.0
			X = (X - self.x_mean_) / self.x_std_
			Xp = (Xp - self.x_mean_) / self.x_std_
		else:
			raise ValueError("Input must be 1D or 2D array.")

		G = X.T @ X
		A = X.T @ Xp

		U, s, Vt = scipy_linalg.svd(G, full_matrices=False)
		threshold = s[0] * 1e-10
		r = min(self.rank, int(np.sum(s > threshold)))
		if r == 0:
			raise ValueError("SVD returned no singular values.")
		U_r = U[:, :r]
		s_r = s[:r]
		Vt_r = Vt[:r, :]
		G_pinv = Vt_r.T @ np.diag(1.0 / s_r) @ U_r.T

		K = G_pinv @ A
		eigenvalues, eigenvectors = scipy_linalg.eig(K)
		moduli = np.abs(eigenvalues)
		order = np.argsort(moduli)[::-1]
		self.K_ = K
		self.eigenvalues_ = eigenvalues[order]
		self.eigenvectors_ = eigenvectors[:, order]
		self.eigenvalue_moduli_ = moduli[order]
		self._is_fitted = True
		return self

	def predict(self, x: np.ndarray, n_steps: int = 1) -> np.ndarray:
		"""Predict n_steps ahead using the fitted Koopman operator.

		Math:
			Take the last state vector (last row of Hankel matrix),
			apply K n_steps times:
			state_future = K^n_steps @ state_current
			Return the first element of state_future as the prediction.
		"""
		if not self._is_fitted:
			raise ValueError("KoopmanEDMD is not fitted.")
		if n_steps < 1:
			raise ValueError("n_steps must be >= 1.")
		x = np.asarray(x, dtype=float).ravel()
		x_norm = (x - self.x_mean_) / self.x_std_
		H = self._build_hankel(x_norm)
		state_current = H[-1]
		K_power = np.linalg.matrix_power(self.K_, int(n_steps))
		state_future = K_power @ state_current
		return np.asarray(state_future[0] * self.x_std_ + self.x_mean_)

	def stability_summary(self) -> dict:
		"""Return stability metrics from the eigenvalue spectrum.

		Returns dict with:
			'max_modulus': float — largest |eigenvalue|,
			               >1.0 means system has unstable mode
			'n_unstable': int — number of eigenvalues with |lambda| > 1.0
			'spectral_radius': float — same as max_modulus (standard term)
			'dominant_frequency': float — angle of largest eigenvalue
			                      in radians, encodes oscillation frequency
			'eigenvalue_moduli': list — all |eigenvalues| sorted descending
		"""
		if not self._is_fitted:
			raise ValueError("KoopmanEDMD is not fitted.")
		max_modulus = float(self.eigenvalue_moduli_[0])
		n_unstable = int(np.sum(self.eigenvalue_moduli_ > 1.0))
		dominant_frequency = float(np.angle(self.eigenvalues_[0]))
		return {
			"max_modulus": max_modulus,
			"n_unstable": n_unstable,
			"spectral_radius": max_modulus,
			"dominant_frequency": dominant_frequency,
			"eigenvalue_moduli": [float(val) for val in self.eigenvalue_moduli_.tolist()],
		}


if __name__ == "__main__":
	t = np.linspace(0.0, 10.0 * np.pi, 500)
	x_stable = np.sin(t) * np.exp(-0.1 * t)
	x_unstable = np.sin(t) * np.exp(0.1 * t)

	stable_model = KoopmanEDMD(n_delays=10, rank=30).fit(x_stable)
	unstable_model = KoopmanEDMD(n_delays=10, rank=30).fit(x_unstable)
	X_multi = np.column_stack([
		np.sin(t),
		np.cos(t),
		np.sin(2.0 * t),
	])
	model_2d = KoopmanEDMD(rank=3)
	model_2d.fit(X_multi)

	print("Stable summary:", stable_model.stability_summary())
	print("Unstable summary:", unstable_model.stability_summary())
	print("2D summary:", model_2d.stability_summary())

	assert stable_model.stability_summary()["max_modulus"] < 1.05
	assert unstable_model.stability_summary()["max_modulus"] > 1.0
	assert model_2d.stability_summary()["spectral_radius"] >= 0.0
	print("edmd.py verified")

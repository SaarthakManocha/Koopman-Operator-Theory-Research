"""Diffusion maps implementation using only numpy and scipy."""

from __future__ import annotations

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as sparse_linalg


class DiffusionMap:
	"""Compute diffusion map coordinates from data using a Gaussian kernel.

	Math:
		Given data points {x_i}, form a kernel W_ij = exp(-||x_i - x_j||^2 / epsilon).
		Normalize rows to a Markov matrix M = D^{-1} W, where D_ii = sum_j W_ij.
		The leading right eigenvectors of M parameterize the diffusion coordinates.

		Fixed epsilon fails when data distributions shift between windows
		(e.g., normal vs crisis market regimes). Adaptive epsilon rescales
		the kernel to the intrinsic geometry of each window independently.
	"""

	def __init__(self, n_components: int = 2, epsilon: float = 1.0, epsilon_method: str = "median") -> None:
		"""Initialize diffusion map hyperparameters.

		Math:
			epsilon controls the Gaussian kernel width in W_ij = exp(-||x_i - x_j||^2 / epsilon),
			and n_components selects how many nontrivial eigenvectors are retained.

			Fixed epsilon fails when data distributions shift between windows
			(e.g., normal vs crisis market regimes). Adaptive epsilon rescales
			the kernel to the intrinsic geometry of each window independently.
		"""
		self.n_components = int(n_components)
		self.epsilon = float(epsilon)
		self.epsilon_method = epsilon_method
		self._is_fitted = False
		self._last_progress_pct = -1

	def fit(self, X: np.ndarray) -> "DiffusionMap":
		"""Fit diffusion maps by building M and eigendecomposing it.

		Math:
			Build W_ij = exp(-||x_i - x_j||^2 / epsilon), then M = D^{-1} W with
			D_ii = sum_j W_ij. Solve M psi_k = lambda_k psi_k and keep the top
			nontrivial eigenpairs as diffusion coordinates.
		"""
		X = self._validate_input(X, context="fit")
		n_samples = X.shape[0]
		if n_samples > 2000:
			raise ValueError("DiffusionMap.fit cap exceeded: max 2000 rows.")
		if self.n_components < 1:
			raise ValueError("n_components must be >= 1.")
		if self.n_components + 1 >= n_samples:
			raise ValueError("n_components must be < n_samples - 1 to drop the trivial mode.")

		if self.epsilon_method == "median":
			self.epsilon_ = self._estimate_median_epsilon(X)
		elif self.epsilon_method == "fixed":
			self.epsilon_ = float(self.epsilon)
		else:
			raise ValueError("epsilon_method must be 'fixed' or 'median'.")
		if self.epsilon_ <= 0.0:
			raise ValueError("epsilon must be positive.")

		self._reset_progress()
		W = self._compute_kernel(X, X, label="Kernel (train)")
		M = self._row_normalize(W)
		eigenvalues, eigenvectors = self._eigendecompose(M)

		# Drop the trivial eigenvector (lambda ~ 1) and keep the next components.
		self.eigenvalues_ = eigenvalues[1 : self.n_components + 1]
		self.eigenvectors_ = eigenvectors[:, 1 : self.n_components + 1]
		self.diffusion_coords_ = self.eigenvectors_.copy()
		self.X_fit_ = X
		self._is_fitted = True
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		"""Map data into diffusion coordinates using Nystrom extension.

		Math:
			For a new point x, compute w_j = exp(-||x - x_j||^2 / epsilon),
			normalize p_j = w_j / sum_j w_j, then extend each eigenvector via
			psi_k(x) = (1 / lambda_k) * sum_j p_j * psi_k(x_j).
		"""
		if not self._is_fitted:
			raise RuntimeError("DiffusionMap.transform called before fit().")
		X = self._validate_input(X, context="transform")
		if X.shape[0] > 2000:
			raise ValueError("DiffusionMap.transform cap exceeded: max 2000 rows.")

		if X is self.X_fit_:
			return self.diffusion_coords_.copy()

		self._reset_progress()
		W = self._compute_kernel(X, self.X_fit_, label="Kernel (transform)")
		row_sums = W.sum(axis=1)
		if np.any(row_sums <= 0.0):
			raise ValueError("Encountered nonpositive kernel row sum in transform.")
		P = W / row_sums[:, None]

		coords = P @ self.eigenvectors_
		coords = coords / self.eigenvalues_[None, :]
		return coords

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		"""Fit diffusion maps and return the diffusion coordinates.

		Math:
			This is equivalent to fit(X) followed by transform(X), returning
			the eigenvector coordinates of the Markov matrix M.
		"""
		self.fit(X)
		return self.diffusion_coords_.copy()

	def _validate_input(self, X: np.ndarray, context: str) -> np.ndarray:
		"""Validate that X is a finite 2D array for diffusion map operations.

		Math:
			Diffusion maps assume points in R^d; invalid shapes or NaNs break
			the kernel W_ij = exp(-||x_i - x_j||^2 / epsilon).
		"""
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError(f"{context}: X must be a 2D array.")
		if X.shape[0] == 0 or X.shape[1] == 0:
			raise ValueError(f"{context}: X must be non-empty.")
		if not np.isfinite(X).all():
			raise ValueError(f"{context}: X must contain only finite values.")
		return X

	def _estimate_median_epsilon(self, X: np.ndarray) -> float:
		"""Estimate epsilon using the median heuristic on sampled pairwise distances.

		Math:
			Fixed epsilon fails when data distributions shift between windows
			(e.g., normal vs crisis market regimes). Adaptive epsilon rescales
			the kernel to the intrinsic geometry of each window independently.
		"""
		n_samples = X.shape[0]
		n_pairs = min(500, n_samples)
		rng = np.random.default_rng()
		idx = rng.integers(0, n_samples, size=(n_pairs, 2))
		diffs = X[idx[:, 0]] - X[idx[:, 1]]
		dist2 = np.sum(diffs * diffs, axis=1)
		return float(np.median(dist2))

	def _compute_kernel(self, X_left: np.ndarray, X_right: np.ndarray, label: str) -> np.ndarray:
		"""Compute Gaussian kernel matrix W_ij = exp(-||x_i - y_j||^2 / epsilon_).

		Math:
			Use the identity ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y to build
			the squared distances in blocks, then apply the Gaussian kernel.
		"""
		n_left = X_left.shape[0]
		n_right = X_right.shape[0]
		W = np.empty((n_left, n_right), dtype=float)
		left_sq = np.sum(X_left * X_left, axis=1)
		right_sq = np.sum(X_right * X_right, axis=1)

		chunk = max(64, min(512, n_left))
		for start in range(0, n_left, chunk):
			end = min(start + chunk, n_left)
			dot = X_left[start:end] @ X_right.T
			dist2 = left_sq[start:end, None] + right_sq[None, :] - 2.0 * dot
			np.maximum(dist2, 0.0, out=dist2)
			W[start:end] = np.exp(-dist2 / self.epsilon_)
			self._progress_update(end, n_left, label)
		return W

	def _row_normalize(self, W: np.ndarray) -> np.ndarray:
		"""Normalize kernel rows to a Markov matrix M = D^{-1} W.

		Math:
			D_ii = sum_j W_ij, then M_ij = W_ij / D_ii so each row sums to 1.
		"""
		row_sums = W.sum(axis=1)
		if np.any(row_sums <= 0.0):
			raise ValueError("Encountered nonpositive kernel row sum during fit.")
		return W / row_sums[:, None]

	def _eigendecompose(self, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Compute leading eigenpairs of the Markov matrix.

		Math:
			Solve M psi_k = lambda_k psi_k and sort eigenpairs by lambda_k
			descending; the leading eigenvectors parameterize diffusion space.
		"""
		n = M.shape[0]
		k = self.n_components + 1
		if k >= n:
			eigenvalues, eigenvectors = scipy_linalg.eig(M)
		else:
			try:
				eigenvalues, eigenvectors = sparse_linalg.eigs(
					M,
					k=k,
					which="LR",
					tol=1e-6,
					maxiter=20000,
				)
			except Exception as exc:
				if exc.__class__.__name__ != "ArpackNoConvergence":
					raise
				print("ARPACK did not converge; falling back to dense eigendecomposition.", flush=True)
				eigenvalues, eigenvectors = scipy_linalg.eig(M)

		eigenvalues, eigenvectors = self._force_real(eigenvalues, eigenvectors)
		order = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[order]
		eigenvectors = eigenvectors[:, order]
		return eigenvalues, eigenvectors

	def _force_real(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Ensure eigenpairs are real within numerical tolerance.

		Math:
			For a positive Markov matrix M, eigenpairs should be real; small
			imaginary parts from numerical error are discarded.
		"""
		if np.iscomplexobj(eigenvalues) or np.iscomplexobj(eigenvectors):
			max_imag = max(np.max(np.abs(np.imag(eigenvalues))), np.max(np.abs(np.imag(eigenvectors))))
			if max_imag > 1e-6:
				raise ValueError("Eigenpairs have large imaginary components.")
			eigenvalues = np.real(eigenvalues)
			eigenvectors = np.real(eigenvectors)
		return eigenvalues, eigenvectors

	def _reset_progress(self) -> None:
		"""Reset progress indicator state before heavy computations.

		Math:
			Progress updates are based on the fraction of rows processed in
			the kernel build, which is the dominant O(n^2) step.
		"""
		self._last_progress_pct = -1

	def _progress_update(self, processed: int, total: int, label: str) -> None:
		"""Print coarse progress updates during O(n^2) kernel construction.

		Math:
			Report the fraction of processed rows so the quadratic kernel build
			provides visible progress as it advances.
		"""
		if total < 300:
			return
		pct = int(100 * processed / total)
		if pct != self._last_progress_pct:
			print(f"{label}: {pct}% ({processed}/{total})", flush=True)
			self._last_progress_pct = pct


if __name__ == "__main__":
	rng = np.random.default_rng(7)
	X_demo = rng.normal(size=(120, 5))
	dmap = DiffusionMap(n_components=2, epsilon=1.0)
	coords = dmap.fit_transform(X_demo)
	print("Diffusion coords shape:", coords.shape)
	new_coords = dmap.transform(X_demo[:5])
	print("Out-of-sample coords shape:", new_coords.shape)

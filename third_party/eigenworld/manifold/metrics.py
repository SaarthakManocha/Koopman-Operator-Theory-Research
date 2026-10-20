"""Geometric metrics derived from a fitted diffusion map."""

from __future__ import annotations

import numpy as np

from eigenworld.manifold.diffusion import DiffusionMap


def _require_fitted(dmap: DiffusionMap) -> None:
	"""Ensure the diffusion map has been fit before computing metrics.

	Math:
		All metrics depend on the fitted eigenvalue spectrum; without a fit,
		no meaningful spectral quantities exist.
	"""
	if not getattr(dmap, "_is_fitted", False):
		raise ValueError("DiffusionMap is not fitted.")


def spectral_entropy(dmap: DiffusionMap) -> float:
	"""Compute spectral entropy of the Markov matrix eigenvalue spectrum.

	Math:
		weights = eigenvalues / sum(eigenvalues)
		entropy = -sum(weights * log(weights + 1e-12))

		Low entropy: one eigenvalue dominates, system locked in single mode.
		This is the crisis signature — the manifold has collapsed.

		High entropy: eigenvalues evenly distributed, system exploring many
		modes. This is healthy complex dynamics.

		The 1e-12 offset prevents log(0) on near-zero eigenvalues.
	"""
	_require_fitted(dmap)
	eigenvalues = np.asarray(dmap.eigenvalues_, dtype=float)
	if eigenvalues.size == 0:
		raise ValueError("No eigenvalues available for spectral entropy.")
	denom = np.sum(eigenvalues)
	if denom <= 0.0:
		raise ValueError("Eigenvalues must sum to a positive value.")
	weights = eigenvalues / denom
	entropy = -np.sum(weights * np.log(weights + 1e-12))
	return float(entropy)


def spectral_gap(dmap: DiffusionMap) -> float:
	"""Compute the gap between the first two nontrivial eigenvalues.

	Math:
		For ordered eigenvalues {lambda_1, lambda_2, ...}, the spectral gap
		is lambda_1 - lambda_2. A large gap implies dominance of one mode.
	"""
	_require_fitted(dmap)
	eigenvalues = np.asarray(dmap.eigenvalues_, dtype=float)
	if eigenvalues.size < 2:
		raise ValueError("At least two eigenvalues are required for spectral gap.")
	gap = eigenvalues[0] - eigenvalues[1]
	return float(gap)


def manifold_summary(dmap: DiffusionMap) -> dict:
	"""Summarize manifold geometry into a single metrics dictionary.

	Math:
		Combines participation ratio, spectral gap, eigenvalue decay, and
		fitted kernel bandwidth into a compact diagnostic profile.
	"""
	_require_fitted(dmap)
	eigenvalues = np.asarray(dmap.eigenvalues_, dtype=float)
	if eigenvalues.size == 0:
		raise ValueError("No eigenvalues available for summary.")
	decay = float(eigenvalues[-1] / eigenvalues[0])
	return {
		"spectral_entropy": spectral_entropy(dmap),
		"spectral_gap": spectral_gap(dmap),
		"n_eigenvalues": int(eigenvalues.size),
		"eigenvalue_decay": decay,
		"epsilon": float(dmap.epsilon_),
	}


if __name__ == "__main__":
	rng = np.random.default_rng(7)
	X_demo = rng.normal(size=(300, 5))
	dmap = DiffusionMap(n_components=5)
	dmap.fit(X_demo)
	summary = manifold_summary(dmap)
	print(summary)
	assert spectral_entropy(dmap) > 0.0
	assert spectral_gap(dmap) >= 0.0
	print("metrics.py verified")

"""Advanced pure-CV image-quality features for CineInfini.

Implements feature extractors that have NO HuggingFace / NO download
dependencies — pure numpy + scipy + cv2 + skimage. These features push
the pure-CV ceiling significantly higher than the basic photometric
metrics in v0.4.10.

Features implemented:
  * BRISQUE NSS (Natural Scene Statistics) — 36 features per frame.
  * LBP (Local Binary Pattern) histogram entropy — texture diversity.
  * HOG (Histogram of Oriented Gradients) cell variance — structure
    richness.
  * Edge density (Canny) — perceptual sharpness.
  * DCT block energy — frequency-domain compactness (compression
    artifact proxy).
  * Shannon entropy on luminance — information content.
  * Multi-scale Sobel (3 scales) — multi-resolution detail.

Each function returns a single float per video, computed by averaging
across N evenly-sampled frames.

Reference: Mittal et al. (2012) "No-Reference Image Quality Assessment
in the Spatial Domain" (BRISQUE).
"""
from __future__ import annotations
import math

import cv2
import numpy as np
from scipy import signal
from scipy.special import gamma


# -----------------------------------------------------------------------------
#  BRISQUE NSS (no SVR — we only return the 36 features for downstream Ridge)
# -----------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """Normalised 2-D Gaussian kernel."""
    k = signal.windows.gaussian(size, sigma).reshape(-1, 1)
    k2 = k @ k.T
    return k2 / k2.sum()


def _mscn(img_gray: np.ndarray, kernel_size: int = 7,
          sigma: float = 7 / 6) -> np.ndarray:
    """Mean-subtracted contrast-normalised coefficients."""
    img = img_gray.astype(np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    k = _gaussian_kernel_2d(kernel_size, sigma)
    mu = signal.convolve2d(img, k, mode="same", boundary="symm")
    sigma_map = np.sqrt(np.abs(
        signal.convolve2d(img * img, k, mode="same", boundary="symm") - mu * mu
    ))
    return (img - mu) / (sigma_map + 1.0 / 255)


def _aggd_params(x: np.ndarray) -> tuple[float, float, float, float]:
    """Asymmetric Generalized Gaussian Distribution (AGGD) parameters.

    Returns (alpha, beta_left, beta_right, mean_aggd).
    Reference: Lasmar et al. (2009).
    """
    x = x.astype(np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    left = x[x < 0]
    right = x[x > 0]
    if left.size == 0 or right.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    sigma_l = np.sqrt((left * left).mean())
    sigma_r = np.sqrt((right * right).mean())
    if sigma_l == 0 or sigma_r == 0:
        return 0.0, 0.0, 0.0, 0.0
    gamma_hat = sigma_l / sigma_r
    r_hat = (np.abs(x).mean() ** 2) / (x * x).mean()
    R_hat = (r_hat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / \
            ((gamma_hat ** 2 + 1) ** 2)

    # Search for alpha that matches R_hat
    alphas = np.arange(0.2, 10.001, 0.001)
    rhos = (gamma(2.0 / alphas) ** 2) / (gamma(1.0 / alphas) * gamma(3.0 / alphas))
    diffs = np.abs(rhos - R_hat)
    alpha = float(alphas[np.argmin(diffs)])

    constant = math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    beta_l = float(sigma_l * constant)
    beta_r = float(sigma_r * constant)
    mean_aggd = float((beta_r - beta_l) * (gamma(2.0 / alpha) / gamma(1.0 / alpha)))
    return alpha, beta_l, beta_r, mean_aggd


def _ggd_params(x: np.ndarray) -> tuple[float, float]:
    """Generalized Gaussian Distribution (alpha, sigma)."""
    x = x.astype(np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0
    sigma_sq = (x * x).mean()
    if sigma_sq == 0:
        return 0.0, 0.0
    rho = sigma_sq / (np.abs(x).mean() ** 2 + 1e-12)
    alphas = np.arange(0.2, 10.001, 0.001)
    rhos = gamma(1.0 / alphas) * gamma(3.0 / alphas) / (gamma(2.0 / alphas) ** 2)
    alpha = float(alphas[np.argmin(np.abs(rhos - rho))])
    return alpha, float(math.sqrt(sigma_sq))


def brisque_features(img_gray: np.ndarray) -> np.ndarray:
    """Return the 36-d BRISQUE NSS feature vector for one (grayscale) image.

    18 features at native scale + 18 at half scale.
    """
    feats = []
    for scale in (1.0, 0.5):
        if scale != 1.0:
            new_size = (int(img_gray.shape[1] * scale),
                        int(img_gray.shape[0] * scale))
            im = cv2.resize(img_gray, new_size, interpolation=cv2.INTER_AREA)
        else:
            im = img_gray
        m = _mscn(im)
        # GGD on MSCN itself
        alpha, sig = _ggd_params(m)
        feats.extend([alpha, sig * sig])
        # AGGD on the four pairwise products
        pairs = [
            m[:, :-1] * m[:, 1:],          # horizontal
            m[:-1, :] * m[1:, :],          # vertical
            m[:-1, :-1] * m[1:, 1:],       # main diagonal
            m[1:, :-1] * m[:-1, 1:],       # secondary diagonal
        ]
        for p in pairs:
            a, bl, br, eta = _aggd_params(p)
            feats.extend([a, eta, bl, br])
    return np.array(feats, dtype=np.float64)  # length 2 + 4*4 = 18 per scale, 36 total


# -----------------------------------------------------------------------------
#  Other advanced pure-CV features
# -----------------------------------------------------------------------------

def lbp_entropy(img_gray: np.ndarray, P: int = 8, R: int = 1) -> float:
    """Shannon entropy of the LBP histogram (texture richness)."""
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img_gray, P, R, method="uniform")
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins),
                            density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def hog_cell_var(img_gray: np.ndarray) -> float:
    """Variance of HOG cell magnitudes (structural diversity)."""
    from skimage.feature import hog
    feats = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True)
    return float(np.var(feats))


def canny_edge_density(img_gray: np.ndarray, t1: int = 80, t2: int = 200) -> float:
    """Fraction of pixels classified as edges by Canny."""
    edges = cv2.Canny(img_gray, t1, t2)
    return float((edges > 0).sum() / edges.size)


def dct_block_energy(img_gray: np.ndarray, block: int = 8) -> float:
    """Mean of high-frequency DCT block energy (compression artifact proxy).

    Splits image into 8×8 blocks, takes DCT, drops the DC, returns the
    mean total energy of the AC coefficients. Lower = more compressed.
    """
    h, w = img_gray.shape
    h2 = h - h % block
    w2 = w - w % block
    img = img_gray[:h2, :w2].astype(np.float32)
    energies = []
    for y in range(0, h2, block):
        for x in range(0, w2, block):
            blk = img[y:y + block, x:x + block]
            d = cv2.dct(blk)
            d[0, 0] = 0  # zero out DC
            energies.append(np.sum(d * d))
    return float(np.mean(energies))


def shannon_entropy(img_gray: np.ndarray) -> float:
    """Per-pixel Shannon entropy of luminance."""
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256),
                            density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def multi_scale_sobel(img_gray: np.ndarray) -> tuple[float, float, float]:
    """Sobel gradient magnitude at 3 scales (1×, 0.5×, 0.25×)."""
    out = []
    for scale in (1.0, 0.5, 0.25):
        if scale != 1.0:
            new_size = (max(8, int(img_gray.shape[1] * scale)),
                        max(8, int(img_gray.shape[0] * scale)))
            im = cv2.resize(img_gray, new_size, interpolation=cv2.INTER_AREA)
        else:
            im = img_gray
        gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        out.append(float(np.mean(np.sqrt(gx * gx + gy * gy))))
    return tuple(out)


def all_advanced_features(frames_bgr: list[np.ndarray]) -> dict:
    """Compute all advanced features by averaging over the given frames.

    Args:
        frames_bgr: list of BGR uint8 frames.

    Returns dict with all metric values (single float each).
    """
    if not frames_bgr:
        return {}
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    # BRISQUE per-frame, then mean across frames + 36 features
    brisque_stack = np.stack([brisque_features(g) for g in grays])
    brisque_mean = brisque_stack.mean(axis=0)
    out = {f"brisque_f{i:02d}": float(brisque_mean[i]) for i in range(36)}
    out["brisque_alpha_native"] = float(brisque_mean[0])
    out["brisque_sigma2_native"] = float(brisque_mean[1])
    out["brisque_alpha_halfscale"] = float(brisque_mean[18])
    out["brisque_sigma2_halfscale"] = float(brisque_mean[19])

    # The other features
    out["lbp_entropy"] = float(np.mean([lbp_entropy(g) for g in grays]))
    out["hog_cell_var"] = float(np.mean([hog_cell_var(g) for g in grays]))
    out["canny_edge_density"] = float(np.mean([canny_edge_density(g) for g in grays]))
    out["dct_block_energy"] = float(np.mean([dct_block_energy(g) for g in grays]))
    out["shannon_entropy_y"] = float(np.mean([shannon_entropy(g) for g in grays]))
    s1, s2, s3 = zip(*[multi_scale_sobel(g) for g in grays])
    out["sobel_scale_1"] = float(np.mean(s1))
    out["sobel_scale_half"] = float(np.mean(s2))
    out["sobel_scale_quarter"] = float(np.mean(s3))
    return out

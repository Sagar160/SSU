import numpy as np
from scipy.spatial import cKDTree

def evaluate_1d_metrics(pred, actual, tau=0.01):
    pred = np.array(pred).reshape(-1)
    actual = np.array(actual).reshape(-1)

    # L1 distance (mean absolute error)
    l1 = np.mean(np.abs(pred - actual))

    # L2 distance (root mean squared error)
    l2 = np.sqrt(np.mean((pred - actual) ** 2))

    # FID (Fréchet distance) for 1D: between two Gaussians
    mu1, mu2 = np.mean(pred), np.mean(actual)
    sigma1, sigma2 = np.var(pred), np.var(actual)
    fid = (mu1 - mu2) ** 2 + sigma1 + sigma2 - 2 * np.sqrt(sigma1 * sigma2)

    # Precision and Recall for F1 (using tau)
    precision = np.mean(np.abs(pred - actual) < tau)
    recall = np.mean(np.abs(actual - pred) < tau)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Value sign check
    same_sign = (np.sign(pred) == np.sign(actual)).sum() / len(pred)

    return l1, l2, fid, f1, same_sign

# # Example usage:
# pred = [0.1, 0.4, 0.5]
# actual = [0.15, 0.45, 0.55]
# cd1, cd2, f1, ss = evaluate_1d_metrics(pred, actual, tau=0.05)
# print(f"Chamferl1: {cd1}, Chamferl2: {cd2}, F1: {f1}, SS: {ss}")

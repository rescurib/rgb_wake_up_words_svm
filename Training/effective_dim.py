"""Estimate the effective PCA dimensionality for the MFCC dataset.

This script loads the same MFCC files used by the training pipeline
(`mfcc_<label>.npy`), removes the first noisy sample from each file,
computes PCA with all components, and reports the participation ratio
plus the number of components needed for common explained variance
thresholds.
"""

import argparse
import glob
import sys

import numpy as np
from sklearn.decomposition import PCA

def load_mfcc_dataset(pattern="mfcc_*.npy", drop_first_sample=True):
    npy_files = sorted([
        f for f in glob.glob(pattern)
        if not f.endswith("_files.npy")
    ])
    if not npy_files:
        raise FileNotFoundError(f"No files matching '{pattern}' were found.")

    all_X = []
    labels = []
    for data_file in npy_files:
        label = data_file.replace("mfcc_", "").replace(".npy", "")
        X = np.load(data_file)
        if drop_first_sample and X.shape[0] > 1:
            X = X[1:]
        all_X.append(X)
        labels.extend([label] * len(X))
        print(f"Loaded {X.shape[0]} samples for label '{label}'.")

    X_combined = np.vstack(all_X)
    return X_combined, np.array(labels)


def participation_ratio(eigenvalues):
    return (eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2)


def components_for_explained_variance(ratio, threshold):
    cumulative = np.cumsum(ratio)
    return int(np.searchsorted(cumulative, threshold, side="right") + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate effective PCA dimensionality for MFCC datasets."
    )
    parser.add_argument(
        "--pattern",
        default="mfcc_*.npy",
        help="Glob pattern for MFCC NumPy files. Default: mfcc_*.npy",
    )
    parser.add_argument(
        "--no-drop-first",
        dest="drop_first",
        action="store_false",
        help="Do not remove the first sample from each file.",
    )
    parser.add_argument(
        "--variance-thresholds",
        nargs="+",
        type=float,
        default=[0.90, 0.95, 0.99],
        help="Explained variance thresholds to report as fractions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        X_combined, labels = load_mfcc_dataset(args.pattern, drop_first_sample=args.drop_first)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    n_samples, n_features = X_combined.shape
    print(f"Total samples: {n_samples}, features: {n_features}")

    pca_full = PCA(n_components=None)
    pca_full.fit(X_combined)

    lambdas = pca_full.explained_variance_
    ratio = pca_full.explained_variance_ratio_
    pr = participation_ratio(lambdas)
    pr_components = int(round(pr))

    print(f"Participation ratio: {pr:.2f}")
    print(f"Suggested n_components from participation ratio: {pr_components}")
    print("")

    print("Explained variance summary:")
    for threshold in sorted(set(args.variance_thresholds)):
        count = components_for_explained_variance(ratio, threshold)
        print(f"  {threshold * 100:.0f}% variance -> {count} components")

    print("")
    print("Top PCA eigenvalues and cumulative variance:")
    cumulative = np.cumsum(ratio)
    for idx in range(min(10, len(ratio))):
        print(
            f"  PC {idx + 1:2d}: variance={ratio[idx] * 100:5.2f}%  "
            f"cumulative={cumulative[idx] * 100:5.2f}%"
        )

    print("")
    print("Run again with --variance-thresholds 0.90 0.95 0.99 to adjust targets.")


if __name__ == "__main__":
    main()


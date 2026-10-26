"""Shared helpers: data loading and table printing."""
import numpy as np
import pandas as pd


def load_heights(path="GaltonFamilies.csv", parent="father"):
    """Return (parent_heights, child_heights). Parents are repeated once per child in the
    raw file, so we keep one row per family to avoid counting the same parent many times."""
    df = pd.read_csv(path)
    parents = df.drop_duplicates("family")[parent].to_numpy()
    children = df["childHeight"].to_numpy()
    return parents, children


def print_tracking_table(history, rows=(0, 1, 2)):
    head = f"{'Iter':<6}{'mu1':>9}{'mu2':>9}{'var1':>9}{'var2':>9}{'pi1':>9}{'pi2':>9}{'LogLik':>12}"
    print(head); print("-" * len(head))
    for h in history:
        if h["iter"] in rows:
            print(f"{h['iter']:<6}{h['mu'][0]:>9.4f}{h['mu'][1]:>9.4f}{h['var'][0]:>9.4f}"
                  f"{h['var'][1]:>9.4f}{h['pi'][0]:>9.4f}{h['pi'][1]:>9.4f}{h['ll']:>12.4f}")


def load_labeled_heights(path="GaltonFamilies.csv", parent="father"):
    """Return (x, is_male): pooled heights (children first, then one row per family for the
    fathers) and the hidden sex label of each height, both built from the SAME filtered rows
    in the SAME order, so is_male[i] always describes x[i]. Used only to EVALUATE the model;
    the labels are never given to EM."""
    df = pd.read_csv(path)
    kids = df[["childHeight", "gender"]].to_numpy()
    dads = df.drop_duplicates("family")[parent].to_numpy()
    x = np.concatenate([kids[:, 0].astype(float), dads.astype(float)])
    is_male = np.concatenate([kids[:, 1] == "male", np.ones(len(dads), dtype=bool)])
    assert len(x) == len(is_male)
    return x, is_male

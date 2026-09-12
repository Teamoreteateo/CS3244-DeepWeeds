"""Generate and freeze fold assignments. Run ONCE. Commit the output."""
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit)
from data import build_manifest, validate, SPLITS, SEED

N_FOLDS, TEST_FRAC = 5, 0.2


def main():
    df = build_manifest()
    for k, v in validate(df).items():
        print(f"  {k}: {v}")

    df = df[df["exists"] & df["session"].notna()].reset_index(drop=True)
    print(f"usable rows: {len(df)}")

    # Hold out test set BY SESSION so the grouped experiment is clean throughout
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    _, te = next(gss.split(df, df["Label"], groups=df["session"]))
    df["is_test"] = False
    df.loc[df.index[te], "is_test"] = True
    pool = df[~df["is_test"]].reset_index(drop=True)
    print(f"pool={len(pool)}  test={int(df['is_test'].sum())}")

    pool["fold_random"] = -1
    for k, (_, v) in enumerate(
            StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
            .split(pool, pool["Label"])):
        pool.loc[v, "fold_random"] = k

    pool["fold_grouped"] = -1
    for k, (_, v) in enumerate(
            StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
            .split(pool, pool["Label"], groups=pool["session"])):
        pool.loc[v, "fold_grouped"] = k

    df = df.merge(pool[["Filename", "fold_random", "fold_grouped"]],
                  on="Filename", how="left")
    df[["fold_random", "fold_grouped"]] = (
        df[["fold_random", "fold_grouped"]].fillna(-1).astype(int))

    assert_no_leakage(df)

    SPLITS.mkdir(exist_ok=True)
    cols = ["Filename", "Label", "Species", "date", "instrument", "session",
            "is_test", "fold_random", "fold_grouped"]
    df[cols].to_csv(SPLITS / "folds.csv", index=False)
    print(f"wrote {SPLITS/'folds.csv'}")

    pool2 = df[~df["is_test"]]
    print("\nclass x grouped-fold:\n", pd.crosstab(pool2["Species"], pool2["fold_grouped"]))
    print("\nclass x random-fold:\n", pd.crosstab(pool2["Species"], pool2["fold_random"]))


def assert_no_leakage(df):
    pool = df[~df["is_test"]]
    for k in range(N_FOLDS):
        tr = set(pool.loc[pool["fold_grouped"] != k, "session"])
        va = set(pool.loc[pool["fold_grouped"] == k, "session"])
        assert not (tr & va), f"GROUPED FOLD {k} LEAKS {len(tr & va)} SESSIONS"
    assert not (set(df.loc[df["is_test"], "session"]) &
                set(df.loc[~df["is_test"], "session"])), "TEST SET LEAKS SESSIONS"
    print("leakage assertions passed")


if __name__ == "__main__":
    main()
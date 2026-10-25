import numpy as np
import pandas as pd

from src import config
from src.eval.splits import make_splits, mixed_tasks


def _meta(n_tasks=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        k = rng.integers(1, 6)
        p = 0.5 if t % 3 == 0 else 0.0          # every third task can be mixed
        for r in range(k):
            rows.append((f"t{t}r{r}", f"task{t}", bool(rng.random() < p)))
    return pd.DataFrame(rows, columns=["uid", "instance_id", "resolved"])


def test_tasks_never_cross_dev_and_test():
    df = make_splits(_meta())
    assert (df.groupby("instance_id")["split"].nunique() == 1).all()


def test_test_split_takes_its_share_of_mixed_and_other_tasks():
    meta = _meta()
    df = make_splits(meta)
    mixed = mixed_tasks(meta)
    test = set(df.loc[df["split"] == "test", "instance_id"])
    others = set(meta["instance_id"]) - mixed
    assert len(test & mixed) == round(config.TEST_FRAC * len(mixed))
    assert len(test & others) == round(config.TEST_FRAC * len(others))


def test_folds_cover_dev_only_and_group_by_task():
    df = make_splits(_meta())
    assert (df.loc[df["split"] == "test", "fold"] == -1).all()
    dev = df[df["split"] == "dev"]
    assert set(dev["fold"]) == set(range(config.N_FOLDS))
    assert (dev.groupby("instance_id")["fold"].nunique() == 1).all()


def test_judge_sample_is_balanced_pairs_from_test():
    df = make_splits(_meta())
    js = df[df["judge_sample"]]
    assert len(js) == config.LLM_JUDGE_SAMPLE
    assert (js["split"] == "test").all()
    assert (js.groupby("instance_id")["resolved"].agg(["size", "sum"]) == [2, 1]).all().all()


def test_existing_judge_sample_is_kept_and_pinned_to_test():
    meta = _meta()
    base = make_splits(meta)
    dev_mixed = sorted(mixed_tasks(meta) & set(base.loc[base["split"] == "dev", "instance_id"]))[:5]
    keep = base[base["instance_id"].isin(dev_mixed)].groupby(["instance_id", "resolved"])["uid"].first().tolist()
    df = make_splits(meta, keep)
    assert set(df.loc[df["judge_sample"], "uid"]) == set(keep)
    assert (df.loc[df["judge_sample"], "split"] == "test").all()
    assert len(set(df.loc[df["split"] == "test", "instance_id"]) & mixed_tasks(meta)) == \
        round(config.TEST_FRAC * len(mixed_tasks(meta)))


def test_deterministic():
    meta = _meta()
    pd.testing.assert_frame_equal(make_splits(meta), make_splits(meta))

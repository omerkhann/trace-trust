import numpy as np
import pytest
from scipy.stats import rankdata

from src.calibration import Calibrator, prior_shift
from src.eval.metrics import ece


def test_prior_shift_matches_bayes_and_is_identity_at_same_rate():
    p = np.array([0.1, 0.5, 0.9])
    assert prior_shift(p, 0.3, 0.3) == pytest.approx(p)
    # p = 0.5 trained at a 50% rate, used at a 10% rate -> 0.1
    assert prior_shift(np.array([0.5]), 0.5, 0.1)[0] == pytest.approx(0.1)


def test_platt_fixes_a_shifted_rate_and_keeps_ranking():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.6, 4000)
    y = rng.random(4000) < true_p
    biased = prior_shift(true_p, 0.1, 0.5)          # scores as if trained at a 50% rate
    cal = Calibrator("platt").fit(biased, y)
    q = cal.transform(biased)
    assert ece(y, q) < ece(y, biased) / 3
    assert (rankdata(q) == rankdata(biased)).all()


def test_isotonic_output_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    y = rng.random(500) < p ** 2
    q = Calibrator("isotonic").fit(p, y).transform(np.sort(p))
    assert np.all(np.diff(q) >= -1e-12) and q.min() >= 0 and q.max() <= 1


def test_none_is_identity_and_unknown_method_rejected():
    p = np.array([0.2, 0.7])
    assert Calibrator("none").fit(p, [0, 1]).transform(p) == pytest.approx(p)
    with pytest.raises(ValueError):
        Calibrator("temperature")

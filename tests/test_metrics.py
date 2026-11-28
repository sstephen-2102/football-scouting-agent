"""Member B — the locked regression yardstick."""

import pytest

from src.common.exceptions import InvalidParameterError
from src.modeling.metrics import regression_metrics


def test_hand_checked_example() -> None:
    result = regression_metrics([100, 200], [110, 180])
    assert result["mae_eur"] == pytest.approx(15.0)
    assert result["rmse_eur"] == pytest.approx(15.8114, abs=1e-4)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["n"] == 2


def test_negative_predictions_are_clipped_to_zero() -> None:
    clipped = regression_metrics([100, 200], [-50, 180])
    zeroed = regression_metrics([100, 200], [0, 180])
    assert clipped == zeroed


def test_keys_match_the_spec() -> None:
    result = regression_metrics([1, 2, 3], [1, 2, 3])
    assert set(result) == {"mae_eur", "rmse_eur", "mae_log", "rmse_log", "spearman", "n"}


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(InvalidParameterError):
        regression_metrics([1, 2], [1])
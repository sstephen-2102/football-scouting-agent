"""Synthetic educational Simplex tests."""

from src.optimize.simplex import simplex


def test_two_variable_known_optimum() -> None:
    result = simplex(
        objective=[3.0, 2.0],
        constraints_a=[
            [1.0, 1.0],  # x + y <= 4
            [1.0, 0.0],  # x <= 2
            [0.0, 1.0],  # y <= 3
        ],
        constraints_b=[4.0, 2.0, 3.0],
        maximize=True,
    )
    assert result["status"] == "optimal"
    assert abs(result["x"][0] - 2.0) < 1e-6
    assert abs(result["x"][1] - 2.0) < 1e-6
    assert abs(result["objective_value"] - 10.0) < 1e-6
    assert len(result["tableau_history"]) >= 1
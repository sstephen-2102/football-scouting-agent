"""Write the starter APA 7 reference list as Markdown (no notebooks)."""

from __future__ import annotations

from pathlib import Path

from src.common.config import PROJECT_ROOT

REFERENCES = [
    (
        "*Football data from Transfermarkt* [Data set]. (2026). Kaggle. "
        "https://www.kaggle.com/datasets/davidcariboo/player-scores"
    ),
    (
        "Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful "
        "seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on "
        "Discrete Algorithms* (pp. 1027–1035)."
    ),
    (
        "Dantzig, G. B. (1963). *Linear programming and extensions*. Princeton "
        "University Press."
    ),
    (
        "Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the "
        "heuristic determination of minimum cost paths. *IEEE Transactions on Systems "
        "Science and Cybernetics, 4*(2), 100–107."
    ),
    (
        "He, M., Cachucho, R., & Knobbe, A. (2015). Football player’s performance and "
        "market value. In *Proceedings of the 2nd Workshop on Sports Analytics* "
        "(pp. 1–9)."
    ),
    (
        "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and "
        "practice* (3rd ed.). OTexts. https://otexts.com/fpp3/"
    ),
    (
        "Müller, O., Simons, A., & Weinmann, M. (2017). Beyond crowd judgments: "
        "Data-driven estimation of market value in association football. "
        "*European Journal of Operational Research, 263*(2), 611–624. "
        "https://doi.org/10.1016/j.ejor.2017.05.005"
    ),
    (
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
        "Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., "
        "Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., "
        "& Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. "
        "*Journal of Machine Learning Research, 12*, 2825–2830."
    ),
    (
        "Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern "
        "approach* (4th ed.). Pearson."
    ),
    (
        "Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An "
        "introduction* (2nd ed.). MIT Press."
    ),
]


def write_references(destination: Path | None = None) -> Path:
    """Write `docs/references.md` from the curated list."""
    destination = destination or (PROJECT_ROOT / "docs" / "references.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = "# References\n\n" + "\n\n".join(REFERENCES) + "\n"
    destination.write_text(body, encoding="utf-8")
    return destination


def main() -> None:
    path = write_references()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

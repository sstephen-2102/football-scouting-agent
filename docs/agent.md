# Scouting agent (Module 1)

The live PEAS payload and the ten environment dimensions are `src/agent/peas.py`, served at `GET /v1/agent`. Fill teammate names after Week 1.

## PEAS

| Component | This project |
|---|---|
| **Performance** | Test-set Spearman of predicted vs actual next value; PR AUC for “undervalued”; expected residual captured under budget *B* |
| **Environment** | Transfermarkt snapshot to 6 July 2026; Big-5 player-seasons; other clubs and agents unobserved |
| **Actuators** | Assign archetype; emit value score; propose XI; allocate budget; bid or pass |
| **Sensors** | Appearances, games, historical valuations, transfers, club and player attributes dated on or before the cutoff |

## Environment sketch (ten dimensions)

The function `ten_environment_dimensions()` in `src/agent/peas.py` is the source of truth. Summary: partially observable (wages, injuries, and true bids are hidden); single-agent for the course; stochastic; sequential; dynamic in live football but simulated on a frozen snapshot; mixed discrete slots and continuous EUR; known search/MIP rules with an unknown value process; competitive market treated as a planning problem; finite window under budget *B*; delayed noisy valuation feedback.

## Disclaimer

Transfermarkt market values are crowd-informed estimates, not clearing prices. Outputs are research scores, not transfer advice.

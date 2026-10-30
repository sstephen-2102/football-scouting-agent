# Scouting agent (Module 1)

Fill names and the 10 course dimensions after Week 1. The live PEAS payload is `src/agent/peas.py`, served at `GET /v1/agent`.

## PEAS

| Component | This project |
|---|---|
| **Performance** | Test-set Spearman of predicted vs actual next value; PR AUC for “undervalued”; expected residual captured under budget *B* |
| **Environment** | Transfermarkt snapshot to 6 July 2026; Big-5 player-seasons; other clubs and agents unobserved |
| **Actuators** | Assign archetype; emit value score; propose XI; allocate budget; bid or pass |
| **Sensors** | Appearances, games, historical valuations, transfers, club and player attributes dated on or before the cutoff |

## Environment sketch (complete with the course’s 10 dimensions in Week 1)

Partially observable (we do not see wages, injuries, or true bids). Stochastic (injuries, form). Sequential (budget spent now is gone later). Dynamic in real football, but we simulate on a frozen snapshot. Mixed discrete (position slots) and continuous (EUR). Single-agent for the course (we do not model competing clubs as agents).

## Disclaimer

Transfermarkt market values are crowd-informed estimates, not clearing prices. Outputs are research scores, not transfer advice.

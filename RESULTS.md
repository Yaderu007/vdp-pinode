# Results (run on my own PC)

System: see `results/system_info.txt`. Raw data: `results/<plan>.csv` (one row per run),
tables: `results/<plan>_summary.csv`, plots: `figures/`. Success = error ≤ 10 % and data error ≤ 2σ².

## 1. Kumar's setting (µ 1.5, 100 points, σ 0.1, 10 windows), 50 seeds

| method | successes | median error | spread (sd of µ) | median time |
|---|---|---|---|---|
| least squares | 50/50 | 0.38 % | 0.008 | 0.4 s |
| MLP (1,218 weights) | 50/50 | 1.13 % | 0.023 | 30 s |
| KAN (400 weights) | 50/50 | 0.99 % | 0.020 | 95 s |

All medians are below Kumar's reported 1.67 %.

## 2. µ × noise (50 windows), 50 seeds each: successes · median error

| µ | σ | least squares | MLP | KAN |
|---|---|---|---|---|
| 0.5 | 0.01 | 50 · 0.22 % | 50 · 0.42 % | 50 · 0.38 % |
| 0.5 | 0.1 | 50 · 2.0 % | 49 · 3.0 % | 49 · 2.9 % |
| 0.5 | 0.3 | 34 · 5.9 % | 31 · 8.3 % | 30 · 8.2 % |
| 1.5 | 0.01 | 50 · 0.04 % | 50 · 0.15 % | 50 · 0.13 % |
| 1.5 | 0.1 | 50 · 0.38 % | 50 · 1.2 % | 50 · 1.3 % |
| 1.5 | 0.3 | 50 · 1.2 % | 43 · 5.0 % | 42 · 5.3 % |
| 4 | 0.01 | 50 · 0.01 % | 3 · 1.0 % | 20 · 1.8 % |
| 4 | 0.1 | 38 · 0.06 % | 46 · 1.5 % | 38 · 2.4 % |
| 4 | 0.3 | 32 · 0.50 % | 45 · 3.9 % | 29 · 7.9 % |

At µ = 4 least squares (started from µ = 1.0) can lock onto a wrong answer (worst 177 % off);
the MLP with short windows is the most reliable. At σ = 0.01 the networks get µ close but miss the strict fit rule.

## 3. Data density (σ 0.1), 20 seeds: successes for 50 · 100 · 400 points

| µ | least squares | MLP | KAN |
|---|---|---|---|
| 0.5 | 20 · 20 · 20 | 18 · 19 · 20 | 19 · 19 · 20 |
| 1.5 | 20 · 20 · 20 | 20 · 20 · 20 | 20 · 20 · 20 |
| 4 | 13 · 18 · 19 | 8 · 18 · 18 | 0 · 14 · 20 |

## 4. Robustness (µ 1.5, σ 0.1), 20 seeds

Other start states (0.5, 0.5) and (−1, 1.5), Laplace noise: every method 20/20.
Small MLP (2-16-16-2, 354 weights): 20/20, median 1.23 % (same as the 1,218-weight MLP).

## 5. Duffing (recover β = 0.3), 20 seeds: successes for σ 0.01 · 0.1 · 0.3

least squares 20 · 20 · 19 · KAN 20 · 20 · 17 · MLP 7 · 12 · 15 (β biased low).

## Answer

- Equation fully known: least squares is the most accurate and ~100× faster, but can get stuck at large µ.
- Hard, jumpy motion (µ = 4): the MLP Neural ODE with short windows was the most reliable.
- MLP vs KAN: equal on Van der Pol up to µ 1.5, KAN ~3× slower; on Duffing the KAN clearly beats the MLP.
- More noise, small µ and few points make the problem harder for every method.

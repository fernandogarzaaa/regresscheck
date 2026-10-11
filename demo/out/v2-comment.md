<!-- regresscheck -->
### ⚠️ regresscheck: **WARN** for `v2`

Baseline `v1` -> candidate `v2` on 80 golden cases.

| Metric | Baseline | Candidate | Delta |
|---|---|---|---|
| Category accuracy | 66.2% | 87.5% | +21.2% |
| Mean summary score (1-5) | 3.62 | 3.89 | +0.26 |
| Regressions / improvements | | | 4 / 21 |

- general accuracy dropped 20.0% (category warn > 10%)

Per-category: billing +20.0%, technical +55.0%, account +30.0%, general -20.0%

<details><summary>Regressed cases</summary>

| Case | Expected | Was | Now |
|---|---|---|---|
| `gen-012` | general | general | technical |
| `gen-015` | general | general | technical |
| `gen-017` | general | general | billing |
| `gen-019` | general | general | technical |

</details>

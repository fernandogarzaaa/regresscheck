<!-- regresscheck -->
### ❌ regresscheck: **CRITICAL** for `v3-bad`

Baseline `v2` -> candidate `v3-bad` on 80 golden cases.

| Metric | Baseline | Candidate | Delta |
|---|---|---|---|
| Category accuracy | 87.5% | 77.5% | -10.0% |
| Mean summary score (1-5) | 3.89 | 3.26 | -0.62 |
| Regressions / improvements | | | 12 / 4 |

- accuracy dropped 10.0% (critical > 8%)
- summary score dropped 15.6% of scale (critical > 8%)
- technical accuracy dropped 55.0% (category warn > 10%)

Per-category: billing +5.0%, technical -55.0%, account -5.0%, general +15.0%

<details><summary>Regressed cases</summary>

| Case | Expected | Was | Now |
|---|---|---|---|
| `tec-001` | technical | technical | billing |
| `tec-004` | technical | technical | general |
| `tec-007` | technical | technical | general |
| `tec-008` | technical | technical | general |
| `tec-009` | technical | technical | general |
| `tec-013` | technical | technical | general |
| `tec-014` | technical | technical | general |
| `tec-015` | technical | technical | general |
| `tec-016` | technical | technical | general |
| `tec-017` | technical | technical | general |
| ... | 2 more | | |

</details>

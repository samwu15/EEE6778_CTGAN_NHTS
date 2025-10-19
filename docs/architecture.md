
# System Architecture (Planned)

```
NHTS 2017 CSVs  ──>  Preprocessing (rules 2,4,5)  ──>  CTGAN Training  ──>  Synthetic Households
                       |                                    |                   |
                       |                                    v                   v
                       └──────────────>  Evaluation (KS/MMD/Correlation)  ──>  Plots & Report
                                                                             |
                                                                             v
                                                                  Streamlit Interactive UI
```

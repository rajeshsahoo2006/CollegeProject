# Final Project — Wells Fargo Reputational Risk (Sept 2016 Fake-Accounts Scandal)

Python pipeline that quantifies the reputational-risk component of Wells Fargo's
daily return volatility during the September 2016 fake-accounts scandal.

## What the script does

`final_project.py` implements the full assignment end-to-end:

1. **Event extraction (Part a)** — loads social-media posts, strips HTML / URLs /
   non-ASCII, and filters to posts containing both a Wells Fargo identifier and
   at least one scandal token (`fake`, `unauthorized`, `scandal`, `fraud`,
   `fine`, `cross-sell`, `stumpf`, `cfpb`).
2. **Sentiment scoring** — VADER `compound` polarity in `[-1, +1]` per post,
   aggregated to a daily average.
3. **Return computation (Part b)** — daily simple returns for Wells Fargo
   (`close.WF`) and the market benchmark (`close.MK`).
4. **Correlation + scatterplot (Part c)** — `ρ(return, sentiment)` and
   `ρ(return, market)`; scatter of daily close vs sentiment and a dual-axis
   timeline plot.
5. **Risk decomposition (Part d)** — total daily volatility split into
   systematic (`ρλ,m · σλ`), reputational (`ρλ,s · σλ`), and residual (`ε`)
   components; market beta; historical VaR and CVaR at 95%, 99%, 99.9%.

## Inputs (not committed)

Place these two files in the same folder as `final_project.py`:

- `Final_WFC_SocialMedia_data (1).csv`
- `Final_WFC_stockPrices (1).xlsx`

## Run

```bash
pip install pandas numpy matplotlib openpyxl vaderSentiment
python3 final_project.py
```

Outputs land in `output/`:

- `cleaned_event_posts.csv` — filtered posts with VADER scores.
- `stock_with_sentiment.csv` — daily joined prices, returns, sentiment.
- `systematic_reputational_risks.csv` — Table 3 values.
- `expected_returns_and_var.csv` — Table 2 values.
- `largest_drops.csv` — top five single-day losses.
- `scatter_close_vs_sentiment.png` — Part (c) scatterplot.
- `timeline_price_sentiment.png` — dual-axis price / sentiment.

## Results summary (Aug 24 – Oct 10, 2016)

| Metric | Value |
|---|---|
| Total volatility σλ | 1.329% daily |
| ρ(return, sentiment) | +0.0387 |
| ρ(return, market) | +0.4519 |
| Systematic risk | 0.006004 |
| Reputational risk | 0.000514 |
| Residual risk (ε) | 0.006768 |
| Market beta β | 0.7881 |
| VaR 95% / 99% / 99.9% | 2.21% / 2.98% / 3.23% |
| CVaR 95% / 99.9% | 2.81% / 3.26% |
| Worst daily loss | −3.26% on 2016-09-13 |

Full write-up in APA 7 format: `FinalProject_WellsFargo_APA7.docx` (submitted separately).

"""
Final Project: Wells Fargo Reputational Risk Analysis
September 2016 fake-accounts scandal.

Pipeline:
  1. Clean social media text (strip HTML, URLs, non-ASCII, whitespace).
  2. Filter posts to scandal-related tokens.
  3. Sentiment scoring with VADER (compound score in [-1, 1]).
  4. Daily average sentiment -> merge with stock prices.
  5. Daily expected returns for Wells Fargo and market.
  6. Correlations rho_{lambda,s} and rho_{lambda,m}.
  7. Systematic risk, reputational risk, residual risk.
  8. Historical VaR / CVaR at 95% and 99.9%.
  9. Scatterplot: daily close price vs daily sentiment.
 10. Write summary tables + cleaned CSVs.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
SM_FILE = HERE / "Final_WFC_SocialMedia_data (1).csv"
SP_FILE = HERE / "Final_WFC_stockPrices (1).xlsx"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

EVENT_START = pd.Timestamp("2016-08-24")
EVENT_END = pd.Timestamp("2016-10-10")
TOKENS = [
    "wells fargo", "wellsfargo",
    "fake account", "unauthorized account",
    "scandal", "fraud", "fine", "cross-sell",
    "stumpf", "cfpb", "fake", "unauthorized",
]


# ---------- 1. CLEANING ------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+|tinyurl\.com/\S+")
_NONPRINT = re.compile(r"[^\x20-\x7E]")
_BRACES = re.compile(r"[\[\]\{\}]")
_MULTI_WS = re.compile(r"\s+")


def clean_text(raw) -> str:
    if not isinstance(raw, str):
        return ""
    s = _HTML_TAG.sub(" ", raw)
    s = _URL.sub(" ", s)
    s = _BRACES.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    s = _NONPRINT.sub(" ", s)
    s = _MULTI_WS.sub(" ", s).strip().strip("'\"")
    return s


def matches_event(text: str) -> bool:
    t = text.lower()
    has_wf = "wells fargo" in t or "wellsfargo" in t or "wfc" in t
    has_scandal = any(tok in t for tok in [
        "fake", "unauthorized", "scandal", "fraud", "fine", "cross-sell",
        "stumpf", "cfpb",
    ])
    return has_wf and has_scandal


# ---------- 2. LOAD & CLEAN SOCIAL MEDIA -------------------------------------
print("[1/7] Loading social media data...")
sm = pd.read_csv(SM_FILE, encoding="latin-1")
sm["date"] = pd.to_datetime(sm["date"], errors="coerce")
print(f"  raw rows: {len(sm):,}")

sm["plainText"] = sm["messageText"].apply(clean_text)
sm = sm[sm["plainText"].str.len() > 0].copy()
sm = sm[(sm["date"] >= EVENT_START) & (sm["date"] <= EVENT_END)].copy()
print(f"  after cleaning + date filter: {len(sm):,}")

sm["is_event"] = sm["plainText"].apply(matches_event)
event_posts = sm[sm["is_event"]].copy()
print(f"  posts matching scandal tokens: {len(event_posts):,}")


# ---------- 3. SENTIMENT -----------------------------------------------------
print("[2/7] Scoring sentiment (VADER)...")
sia = SentimentIntensityAnalyzer()
event_posts["sentiment_score"] = event_posts["plainText"].apply(
    lambda t: sia.polarity_scores(t)["compound"]
)

event_posts.to_csv(OUT_DIR / "cleaned_event_posts.csv", index=False)


# ---------- 4. DAILY AGG + MERGE WITH STOCK ----------------------------------
print("[3/7] Aggregating daily sentiment...")
daily_sent = (
    event_posts.groupby(event_posts["date"].dt.date)["sentiment_score"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "sentiment_score", "count": "n_posts"})
    .reset_index()
)
daily_sent["date"] = pd.to_datetime(daily_sent["date"])

sp = pd.read_excel(SP_FILE)
sp["date"] = pd.to_datetime(sp["date"])
sp = sp[(sp["date"] >= EVENT_START) & (sp["date"] <= EVENT_END)].copy()

# Recompute daily returns cleanly from close prices.
sp = sp.sort_values("date").reset_index(drop=True)
sp["ret_WF"] = sp["close.WF"].pct_change()
sp["ret_MK"] = sp["close.MK"].pct_change()

merged = sp.merge(daily_sent, on="date", how="left")
merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
merged["n_posts"] = merged["n_posts"].fillna(0).astype(int)
merged.to_csv(OUT_DIR / "stock_with_sentiment.csv", index=False)
print(f"  trading days in window: {len(merged)}")


# ---------- 5. RISK METRICS --------------------------------------------------
print("[4/7] Computing risk metrics...")
ret_wf = merged["ret_WF"].dropna()
ret_mk = merged["ret_MK"].dropna()

sigma_wf = ret_wf.std(ddof=1)            # total stock volatility
sigma_mk = ret_mk.std(ddof=1)            # market volatility
mean_wf = ret_wf.mean()                   # avg daily return WF

# Correlations
valid = merged.dropna(subset=["ret_WF", "ret_MK"])
rho_lm = valid["ret_WF"].corr(valid["ret_MK"])            # market
rho_ls = valid["ret_WF"].corr(valid["sentiment_score"])   # sentiment

systematic_risk = rho_lm * sigma_wf          # rho_{lambda,m} * sigma_lambda
reputational_risk = rho_ls * sigma_wf        # rho_{lambda,s} * sigma_lambda
residual_risk = sigma_wf - systematic_risk - reputational_risk

# Market beta
beta = valid["ret_WF"].cov(valid["ret_MK"]) / valid["ret_MK"].var(ddof=1)


# ---------- 6. VAR / CVAR ----------------------------------------------------
def hist_var(returns: pd.Series, conf: float) -> float:
    """Historical VaR: the loss at the (1-conf) quantile. Returned as positive loss."""
    q = returns.quantile(1 - conf)
    return -q


def hist_cvar(returns: pd.Series, conf: float) -> float:
    """Historical CVaR: average loss beyond VaR. Positive."""
    q = returns.quantile(1 - conf)
    tail = returns[returns <= q]
    return -tail.mean() if len(tail) else float("nan")


var_95 = hist_var(ret_wf, 0.95)
var_99 = hist_var(ret_wf, 0.99)
var_999 = hist_var(ret_wf, 0.999)
cvar_95 = hist_cvar(ret_wf, 0.95)
cvar_999 = hist_cvar(ret_wf, 0.999)

# Biggest single-day drops
drops = merged.dropna(subset=["ret_WF"]).nsmallest(5, "ret_WF")[
    ["date", "close.WF", "ret_WF"]
]


# ---------- 7. OUTPUT TABLES + PLOT ------------------------------------------
print("[5/7] Writing output tables...")
systematic_tbl = pd.DataFrame([{
    "Bank": "Wells Fargo",
    "Correlation (rho_lambda,s)": round(rho_ls, 4),
    "Market/Systematic Risk (rho_lambda,m * sigma_lambda)": round(systematic_risk, 6),
    "Reputational Risk (rho_lambda,s * sigma_lambda)": round(reputational_risk, 6),
    "Residual Risk (epsilon)": round(residual_risk, 6),
    "Sentiment Analysis Period": f"{EVENT_START.date()} to {EVENT_END.date()}",
}])

returns_tbl = pd.DataFrame([{
    "Bank": "Wells Fargo",
    "Date Stock Dropped (>=5% loss examples)": ", ".join(
        drops["date"].dt.strftime("%Y-%m-%d").tolist()
    ),
    "Biggest Drop (%)": round(drops["ret_WF"].iloc[0] * 100, 4),
    "VaR_95 (%)": round(var_95 * 100, 4),
    "CVaR_95 (%)": round(cvar_95 * 100, 4),
    "CVaR_99.9 (%)": round(cvar_999 * 100, 4),
    "Avg Daily Return WF (%)": round(mean_wf * 100, 4),
    "Sigma WF (%)": round(sigma_wf * 100, 4),
    "Sigma MK (%)": round(sigma_mk * 100, 4),
    "Beta": round(beta, 4),
    "VaR_99 (%)": round(var_99 * 100, 4),
    "VaR_99.9 (%)": round(var_999 * 100, 4),
}])

systematic_tbl.to_csv(OUT_DIR / "systematic_reputational_risks.csv", index=False)
returns_tbl.to_csv(OUT_DIR / "expected_returns_and_var.csv", index=False)
drops.to_csv(OUT_DIR / "largest_drops.csv", index=False)

# Scatterplot: daily close vs daily sentiment (Part c)
print("[6/7] Building scatterplot...")
fig, ax = plt.subplots(figsize=(9, 6))
plot_df = merged.dropna(subset=["close.WF"])
sc = ax.scatter(
    plot_df["sentiment_score"],
    plot_df["close.WF"],
    c=plot_df["date"].astype("int64"),
    cmap="viridis",
    s=60,
    edgecolor="black",
)
ax.set_xlabel("Daily Average Sentiment Score (VADER compound)")
ax.set_ylabel("Wells Fargo Closing Price (USD)")
ax.set_title(
    f"WFC Close vs Sentiment  (Aug 24 - Oct 10, 2016)\n"
    f"rho(return, sentiment) = {rho_ls:+.4f}"
)
ax.grid(alpha=0.3)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Date")
cbar.ax.set_yticklabels(
    pd.to_datetime(cbar.ax.get_yticks()).strftime("%m-%d")
)
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_close_vs_sentiment.png", dpi=150)
plt.close()

# Secondary plot: price + sentiment over time
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(merged["date"], merged["close.WF"], "b-", label="WFC Close", linewidth=2)
ax1.set_xlabel("Date"); ax1.set_ylabel("WFC Close (USD)", color="b")
ax1.tick_params(axis="y", labelcolor="b")
ax2 = ax1.twinx()
ax2.plot(merged["date"], merged["sentiment_score"], "r-", alpha=0.7, label="Sentiment")
ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax2.set_ylabel("Daily Avg Sentiment", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax1.axvline(pd.Timestamp("2016-09-08"), color="orange", linestyle=":",
            alpha=0.7, label="CFPB announcement 9/8")
plt.title("Wells Fargo Price and Sentiment — Fake Accounts Scandal")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT_DIR / "timeline_price_sentiment.png", dpi=150)
plt.close()


# ---------- 8. PRINT REPORT --------------------------------------------------
print("\n" + "=" * 72)
print(" WELLS FARGO REPUTATIONAL RISK  —  FINAL PROJECT REPORT")
print("=" * 72)
print(f" Window:                        {EVENT_START.date()} to {EVENT_END.date()}")
print(f" Trading days:                  {len(merged)}")
print(f" Event-related posts scored:    {len(event_posts):,}")
print()
print(" ─── Part (b) Expected Returns ──────────────────────────────────────")
print(f" Avg daily return (WF):         {mean_wf*100:+.4f}%")
print(f" Avg daily return (Market):     {ret_mk.mean()*100:+.4f}%")
print(f" sigma_WF (vol):                {sigma_wf*100:.4f}%")
print(f" sigma_MK (vol):                {sigma_mk*100:.4f}%")
print(f" Beta (WF vs MK):               {beta:.4f}")
print()
print(" ─── Part (c) Correlations ──────────────────────────────────────────")
print(f" rho(return_WF, sentiment)      {rho_ls:+.4f}")
print(f" rho(return_WF, return_MK)      {rho_lm:+.4f}")
print()
print(" ─── Part (d) Risk Decomposition ────────────────────────────────────")
print(f" Systematic Risk   rho_lm*σ  =  {systematic_risk:+.6f}")
print(f" Reputational Risk rho_ls*σ  =  {reputational_risk:+.6f}")
print(f" Residual Risk     epsilon   =  {residual_risk:+.6f}")
print(f" Total Vol         σ_λ        =  {sigma_wf:+.6f}")
print()
print(" ─── Historical VaR / CVaR (daily returns) ─────────────────────────")
print(f" VaR 95%:    {var_95*100:.4f}%     CVaR 95%:    {cvar_95*100:.4f}%")
print(f" VaR 99%:    {var_99*100:.4f}%")
print(f" VaR 99.9%:  {var_999*100:.4f}%   CVaR 99.9%:  {cvar_999*100:.4f}%")
print()
print(" ─── Top 5 single-day drops ─────────────────────────────────────────")
for _, r in drops.iterrows():
    print(f"   {r['date'].date()}   close=${r['close.WF']:.2f}   "
          f"return={r['ret_WF']*100:+.4f}%")
print()
print(f" Outputs written to: {OUT_DIR}")
print("=" * 72)
print("[7/7] Done.")

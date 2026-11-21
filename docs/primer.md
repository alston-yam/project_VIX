# Learn the VIX premium before the first meeting

The short answer to the pitch question is "usually yes, and that is the point". The literature says the VIX is too pessimistic on average. Since 1990 it has averaged about **19.6** against about **15.5** for the S&P 500 volatility that actually followed, a gap of roughly **4 volatility points** ([CFA Institute](https://blogs.cfainstitute.org/blog/2024/07/31/how-well-does-the-market-predict-volatility)). That gap is not a forecasting mistake that anyone can harvest for free. It works like an insurance premium. People who sell variance collect small gains most months and take rare, very large losses in crashes, which is exactly when the rest of a portfolio is falling too. To contribute properly to the project, a beginner needs to understand five things: why we measure wobble in *variance* rather than volatility, what the VIX actually prices, how to line up the dates without peeking into the future, why a t-stat above 2 can still mislead when returns have fat tails, and which sign convention each paper uses. The team's own monthly 1990–2025 statistics (win rate, skew, worst month) **have not been computed by anyone yet**. Every number in this report that is labelled "expected" is a prior to check against the team's own run, not a result. A null or weak result is a perfectly good finding to present.

## From daily returns to the VIX in six steps

### Returns and why we use logs

A **return** is the percentage change in price. If the S&P goes from 100 to 101, the simple return is (101 − 100)/100 = +1%. A **log return** is ln(101/100) ≈ 0.995%. For small daily moves the two are almost identical. We use log returns because they add up over time: going 100 → 150 → 75 is +50% then −50%, but the true total is −25%, and the log returns ln(1.5) ≈ 0.405 and ln(0.5) ≈ −0.693 sum to −0.288 = ln(0.75), which is exactly right ([Gundersen](https://www.gregorygundersen.com/blog/2022/02/06/log-returns/)).

### Volatility, variance, and the square-root-of-time rule

**Volatility** is the standard deviation of returns. It measures how much prices wobble, not which way they go. Think of throwing darts: volatility measures how spread out the darts land, not where the centre is. **Variance** is volatility squared.

The single most important idea in this project is this: **variance adds up over time, volatility does not.** If daily returns are independent, the variance of a 252-day total is 252 × the daily variance. Volatility is the square root of that, so it grows with √time. A 1% daily standard deviation therefore becomes about 1% × √252 ≈ 1% × 15.87 ≈ **15.9% a year**. Traders use the "rule of 16" (√256 = 16) as a shortcut: annual vol ÷ 16 ≈ the typical daily move. To get from annual to monthly, divide by √12 ≈ 3.46.

**Realised volatility** is the wobble that actually happened, measured after the fact from real returns. **Implied volatility** is the wobble that option prices "assume". It looks forward and is backed out of today's option prices. Implied volatility predicts "only the size of potential price swings", not their direction ([Charles Schwab](https://www.schwab.com/learn/story/what-is-implied-volatility-basics-examples)). It is also not a pure forecast. It is a *price* for future wobble, and that price includes a risk premium.

### Options are insurance whose price reveals expected wobble

A **call option** is the right, but not the obligation, to buy an asset at a fixed price (the **strike**, K) by an expiry date. Its payoff is max(S − K, 0), where S is the price at expiry. A **put option** is the right to sell at K, with payoff max(K − S, 0). The upfront price is the **premium**. An option is **out of the money (OTM)** if it would be worthless if exercised right now: a put with a strike below today's price, or a call with a strike above it.

A put on a stock portfolio works like home insurance. You pay a premium, and you are paid out if the bad event happens. The key fact is that **the more the market might move, the more an option is worth**, because the downside is capped at the premium while the payoff can be large. In the Black–Scholes model, everything that goes into an option price is observable except volatility. So the market price of the option pins down the volatility, and that number is the implied volatility. Higher implied volatility makes both calls and puts more expensive ([Schwab](https://www.schwab.com/learn/story/what-is-implied-volatility-basics-examples)).

### The VIX is the price of 30 days of S&P variance

The **VIX** measures the market's expected 30-day volatility of the S&P 500. It is built from SPX option prices and US Treasury rates, and quoted as σ × 100 ([Cboe VIX Methodology](https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf)). In plain English, the recipe is:

1. Take every OTM SPX put and call for the two expiries either side of 30 days.
2. Weight each price by 1/K², which gives cheap far-OTM puts extra weight.
3. Add them up. The sum is the cost of a portfolio whose value tracks variance.
4. Blend the two expiries to exactly 30 calendar days.
5. Take the square root and multiply by 100.

This approach is called "model-free" because it uses option prices directly, without Black–Scholes. The formula comes from variance-swap replication theory in Demeterfi, Derman, Kamal & Zou (1999), which the Cboe white paper footnotes ([Cboe VIX White Paper](https://www.optionseducation.org/getmedia/5d6a69e4-8a3a-449d-8fef-be10638919c7/vixwhite.pdf)).

**What a VIX of 20 means.** It implies a one-standard-deviation move of about 20% a year, 20/3.46 ≈ **5.8% a month**, or 20/15.87 ≈ **1.26% a day**. The VIX is nicknamed the "fear gauge" because demand for crash protection (OTM puts) jumps when markets fall. That makes the VIX move sharply opposite to the S&P.

**What the levels mean.** As a rough rule, below about 15 is calm, 15–20 is normal, 20–30 is stressed, above 30 is a crisis, and above 50 is rare panic. On 19 November 2026 FRED showed **28.44** ([FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)). The record close is **82.69 on 16 March 2020**, and the record intraday high is **89.53 on 24 October 2008** ([Macroption](https://www.macroption.com/vix-all-time-high/)). The biggest one-day spike happened on **5 August 2024**, when the VIX rose about 180% to almost 66 before the US open. That move was partly an artefact of dealer quoting and wide put bid-ask spreads, because the VIX is "based on quotes instead of trades" ([BIS Bulletin 95](https://www.bis.org/publ/bisbull95.htm)).

### Variance swaps turn the VIX into a tradable bet

A **variance swap** is a contract that pays N × (realised variance − strike) to the buyer ([Demeterfi et al. 1999](https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf)). The fair strike is a weighted strip of OTM option prices, which is the same structure as the VIX formula. So **VIX² ≈ the fair price of 30-day variance**. It is only approximate, because of discrete strikes, cut-off far strikes, and jumps. The replication holds exactly only "as long as stock prices evolve without jumps" ([Demeterfi et al.](https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf)).

That is why our seller's payoff, implied variance − realised variance, is the profit and loss of a real (if idealised) trade: selling a one-month variance swap and holding it to expiry.

There are three reasons to compare variances rather than vols:

- **Variance is what options replicate.** "Variance is the primary underlyer", and volatility swaps cannot be replicated statically ([Demeterfi et al.](https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf)).
- **Variance adds across time.** Vol does not.
- **Squaring makes losses accelerate.** This is called **convexity**. Suppose realised vol is either 10 or 30 with equal odds. Average vol is 20, but average variance is (100 + 900)/2 = 500, which is about 22.4². A seller who prices off average vol underprices variance.

The crash case shows why this matters. Take a swap with $100k of "vega notional" (dollars per vol point) struck at 20. That is $2,500 per variance point. If realised vol comes in at 15, the seller gains 2,500 × (400 − 225) = **$437,500**. If realised vol comes in at 80, the seller loses 2,500 × (6,400 − 400) = **$15 million**. A plain vol bet of the same size would lose only $6m. This is the "squared" effect your scatter plot will show.

### The variance risk premium and its two sign conventions

The **variance risk premium (VRP)** is the gap between implied and later realised variance. Take the pitch's calm example: VIX = 20 gives implied variance 0.20² = 0.04, and realised vol of 15% gives 0.0225. The raw payoff is 0.0175 (annualised variance units). The **scaled payoff** is 0.0175/0.04 = **+44%**. In the crash example, realised vol of 40% gives 0.16, so the payoff is 0.04 − 0.16 = −0.12, and the scaled payoff is −0.12/0.04 = **−300%**.

Note that the scaled payoff equals 1 − RV/IV. It can never exceed **+100%** (realised variance cannot be negative), but it has **no lower limit**. That asymmetry is the whole story of the pitch in one line.

Papers define the VRP with opposite signs, so always say which one you use:

| Convention | Formula | Sign for S&P 500 | Meaning |
|---|---|---|---|
| Bollerslev, Tauchen & Zhou style (our pitch) | implied − realised (VIX² − RV) | Positive | Insurance buyers overpay |
| Carr & Wu | realised − implied (RV − SW) | Negative | Long variance loses on average |

Both describe the same economic fact. Our pitch uses the seller's view, implied − realised.

## Five statistics decide whether the premium is real

### t-stat and why Newey–West matters

The **t-statistic** asks whether an average is really different from zero or could be luck: t = mean ÷ (standard deviation ÷ √n). As a rule of thumb, |t| > 2 means "significant at about the 5% level". With about **432 months** (36 years × 12), even a modest average gap can clear 2.

There is a trap, though. The plain t-stat assumes each observation is independent. If you used *daily* overlapping 21-day windows, neighbouring windows would share 20 of their 21 returns. That would inflate the naive t-stat by roughly √21 ≈ 4.6 times. **Newey–West (HAC) standard errors** correct for this kind of overlap and autocorrelation. Carr & Wu use Newey–West with 30 lags for overlapping daily data and 6 lags for monthly data ([Carr & Wu 2009](https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf)). In statsmodels this is `fit(cov_type='HAC', cov_kwds={'maxlags': L})`, where `maxlags` is required and the default kernel is Bartlett, i.e. Newey–West ([statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html)). Our monthly first-of-month windows overlap only slightly, so plain t-stats are fine, with a Newey–West version using a few lags as a cheap robustness check.

### Win rate, skew, kurtosis and the worst month

The **win rate** is the share of months where the seller made money. **Skewness** measures lopsidedness. Negative skew means many small gains and a few huge losses: "picking up pennies in front of a steamroller". **Kurtosis** measures fat tails. A normal bell curve has kurtosis 3 (excess kurtosis 0), and higher values mean extreme months are far more common than a bell curve predicts.

Fat tails also make the t-stat overconfident. A 36-year sample can look highly significant, but the mean is driven by a handful of crash months, and one more crash could change it a lot. That is why the pitch should show the mean **with and without the worst 5 months**, the median, and the ratio of the worst loss to the average gain.

### Why the Sharpe ratio flatters short volatility

The **Sharpe ratio** is (average return − risk-free rate) ÷ standard deviation, annualised by multiplying a monthly Sharpe by √12. It uses only the mean and the spread, so it ignores skew and crash risk. The literature warns that "the Sharpe ratio and other related reward-to-risk measures may be manipulated with option-like strategies", and that strategies with regular modest profits punctuated by occasional crashes are high-Sharpe by construction ([Goetzmann, Ingersoll, Spiegel & Welch, NBER w9116](https://www.nber.org/papers/w9116)).

Here is a worked example. A strategy earns +2% for 59 months and −40% in one month. The mean is 1.3% a month and the standard deviation is about 5.4%. That gives a monthly Sharpe of about 0.24, or about 0.83 annualised. Drop the crash month from the backtest window and the Sharpe looks almost infinite. A useful rule of thumb links the two measures: t ≈ annual Sharpe × √(years).

## The literature agrees on the sign but not on the cause

**Carr & Wu (2009)**, "Variance Risk Premiums", *Review of Financial Studies* 22(3):1311–1341 ([IDEAS/RePEc](https://ideas.repec.org/a/oup/rfinst/v22y2009i3p1311-1341..html)), is the anchor paper. They measure the premium as **realised minus a synthetic variance swap rate**. For the S&P 500 it is strongly negative, and the excess return on a long variance swap is "over −50 percent per month" for the two S&P indexes and the Dow. Market beta "can only explain a small portion" of it, and the Fama–French factors "cannot account for" it either. Crucially, their sample is **January 1996 to February 2003** of OptionMetrics data, not 1990–2025 ([Carr & Wu PDF](https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf)).

**Bollerslev, Tauchen & Zhou (2009)**, *RFS* 22(11):4463–4492, define the VRP as **implied minus realised** over 1990–2007. They report a mean of **18.30 (percent squared, monthly)**. They also find that a high VRP predicts higher future stock returns, strongest at the quarterly horizon (robust t = 2.86, adjusted R² 6.82%) ([BTZ working paper](https://repec.econ.au.dk/repec/creates/rp/08/rp08_48.pdf); [EconPapers](https://econpapers.repec.org/RePEc:oup:rfinst:v:22:y:2009:i:11:p:4463-4492)). Their realised variance comes from 5-minute returns and is designed to be known at the time of the forecast. Ours uses the *next* 21 days, so our number is the seller's realised payoff, not their predictor. Do not compare the two numbers directly.

Several other papers point the same way:

- **Coval & Shumway (2001)**, *Journal of Finance*: zero-beta at-the-money S&P straddles lose about **3% per week** ([EconPapers](https://econpapers.repec.org/RePEc:bla:jfinan:v:56:y:2001:i:3:p:983-1009)).
- **Bakshi & Kapadia (2003)**: delta-hedged S&P option buyers lose on average, more so when vol is high ([IDEAS](https://ideas.repec.org/a/oup/rfinst/v16y2003i2p527-566.html)).
- **Todorov (2010)**: fear of jumps, and willingness to pay for jump protection, rise sharply right after jumps ([IDEAS](https://ideas.repec.org/a/oup/rfinst/v23y2010i1p345-383.html)).

For a strategy-level view, the Cboe PUT index (selling S&P puts) returned +1835% cumulatively from 1986 to 2018, with 9.95% volatility against 14.93% for the S&P. Its skewness was **−2.09** against −0.81 for the S&P ([Cboe](https://www.cboe.com/insights/posts/white-paper-shows-volatility-risk-premium-facilitated-higher-risk-adjusted-returns-for-put-index)). That is the cleanest one-number picture of the steamroller.

**Is it crash-risk compensation?** Partly. Long variance pays out exactly when stocks fall, so buyers accept a loss on it the way homeowners accept a loss on insurance ([Carr & Wu](https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf)). But beta explains little of it, and Bondarenko finds that no canonical model explains how expensive S&P puts have been ([Bondarenko 2014](https://ideas.repec.org/a/wsi/qjfxxx/v04y2014i03ns2010139214500153.html)). So the honest slide wording is "crash compensation plus an unexplained extra". Real blow-ups make the point vivid. On 5 February 2018, XIV lost 96.3% of its value and was shut down ([ETF.com](https://www.etf.com/sections/news/inverse-vix-etn-shuts-down)), and LJM Partners fell from $812m to $14m ([Reuters via Yahoo](https://www.yahoo.com/news/exclusive-ljm-partners-shutting-doors-001311723.html)).

### Corrections and cautions for the team's summary

| Claim in team notes | What to say instead |
|---|---|
| Carr & Wu show a positive VRP over our period | Their sample is **1996–2003**, and their sign is **realised − implied** (negative). Restate it in our sign or say so explicitly. |
| Quote Carr & Wu's S&P mean and t-stat from Table 5 | The table did not extract. **Don't quote Table 5 numbers**. Use the quoted "over −50% per month" instead. |
| "2008 record close 80.86" | The sources conflict: 80.86 on 20 Nov ([Macroption](https://www.macroption.com/vix-all-time-high/)) vs 80.74 on 21 Nov ([CNBC](https://www.cnbc.com/2020/03/16/wall-streets-fear-gauge-hits-highest-level-ever.html)). Either way, **2020's 82.69 is the record close**. Check against FRED. |
| VIX data 1990–2002 is what traders saw | Today's VIX launched on 22 Sept 2003. **Pre-2003 values are back-calculated**. The original 1993 VIX (now VXO) used at-the-money S&P 100 options ([Cboe Methodology](https://cdn.cboe.com/resources/indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf)). |
| "Win rate ~80–90%", "worst months 2008/2020" | These are **unverified priors**. No source gave them. Compute them. |
| The payoff is a tradable strategy | It is an idealised variance swap. The VIX index itself is not tradable, and VIX futures and ETPs earn a different (roll/contango) premium ([IBKR Campus](https://ibkrcampus.com/campus/traders-insight/securities/options/trading-volatility-is-difficult/)). |

## One Colab script produces every number and chart

The script below combines the research notes' snippets into one script that runs end to end. It fixes one bug from the notes: the demeaned realised-variance line multiplied by 252/(H−1), but `.var()` already divides by H−1, so the correct scaling is ×252. FRED's CSV needs no API key. yfinance returns MultiIndex columns by default, and Yahoo rate-limits heavy users, sometimes for hours ([yfinance #2431](https://github.com/ranaroussi/yfinance/discussions/2431)), so the script **caches the downloads**.

```python
# !pip install -q -U yfinance statsmodels
import numpy as np, pandas as pd, yfinance as yf, statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

START, END, H = "1989-12-01", "2026-03-01", 21   # pad end: need 21 days after Dec-2025

# 1. Data (cached)
vc = Path("vix_cache.csv")
if vc.exists():
    vix = pd.read_csv(vc, index_col=0, parse_dates=True).iloc[:, 0]
else:
    vix = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", na_values=".")
    vix.columns = ["date", "vix"]; vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date")["vix"].dropna(); vix.to_csv(vc)

sc = Path("gspc_cache.csv")
if sc.exists():
    spx = pd.read_csv(sc, index_col=0, parse_dates=True)["Close"]
else:
    d = yf.download("^GSPC", start=START, end=END, auto_adjust=False,
                    multi_level_index=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    d.to_csv(sc); spx = d["Close"]
spx.index = pd.to_datetime(spx.index).tz_localize(None)

# 2. Build implied and FORWARD realised variance (no look-ahead in IV)
df = pd.concat({"spx": spx, "vix": vix}, axis=1).dropna()       # inner join on dates
df["r"]  = np.log(df["spx"]).diff()
df["iv"] = (df["vix"] / 100) ** 2
df["rv"] = (252 / H) * df["r"].pow(2)[::-1].rolling(H).sum()[::-1].shift(-1)   # r_{t+1..t+21}
df["rv_dm"] = 252 * df["r"].shift(-1)[::-1].rolling(H).var()[::-1]           # robustness
t = df.index[100]
assert np.isclose(df.loc[t, "rv"], 252/H * (df["r"].iloc[101:101+H]**2).sum())  # alignment check

df["payoff"]    = df["iv"] - df["rv"]          # seller's payoff, annualised variance
df["payoff_sc"] = df["payoff"] / df["iv"]      # = 1 - RV/IV
df["iv_vol"], df["rv_vol"] = np.sqrt(df["iv"]), np.sqrt(df["rv"])
df["spx_fwd"] = df["spx"].shift(-H) / df["spx"] - 1

# 3. Monthly, (near) non-overlapping: first trading day of each month
m = df.groupby(df.index.to_period("M")).head(1).loc["1990-01":"2025-12"].dropna(subset=["rv"])
print("months:", len(m))                        # expect ~432

# 4. Statistics
def mean_t(x, lags=0):
    x = x.dropna()
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC",
            cov_kwds={"maxlags": lags, "use_correction": True})
    return round(r.params[0], 5), round(r.tvalues[0], 2)

for col in ["payoff", "payoff_sc"]:
    p = m[col]
    print(f"\n== {col} ==")
    print("mean,t iid:", mean_t(p), "| NW6:", mean_t(p, 6), "| median:", p.median())
    print("win rate:", (p > 0).mean(), "| skew:", stats.skew(p), "| ex-kurt:", stats.kurtosis(p))
    print("worst:", p.idxmin().date(), p.min(), "| avg gain:", p[p > 0].mean(),
          "| worst/avg gain:", p.min() / p[p > 0].mean())
    print("mean excl. worst 5:", p.drop(p.nsmallest(5).index).mean())
print("\nmean IV vol, RV vol:", m["iv_vol"].mean(), m["rv_vol"].mean())
print("daily overlapping, NW21:", mean_t(df.loc["1990":"2025", "payoff"], 21))
print(m["payoff_sc"].nsmallest(10))

# 5. Do losses cluster in sell-offs? (crash-risk test)
print("corr(payoff, SPX fwd):", m["payoff"].corr(m["spx_fwd"]))
dec = m["spx_fwd"] <= m["spx_fwd"].quantile(0.10)
for name, mask in [("SPX < -5%", m["spx_fwd"] < -0.05), ("worst decile", dec), ("other 90%", ~dec)]:
    print(name, mask.sum(), "mean:", m.loc[mask, "payoff_sc"].mean(),
          "win:", (m.loc[mask, "payoff_sc"] > 0).mean())
loss = m["payoff"].clip(upper=0)
print("share of total losses in worst-decile SPX months:", loss[dec].sum() / loss.sum())

# 6. Robustness: post-2003 only (real-time VIX), demeaned RV
print("2004-2025 mean,t:", mean_t(m.loc["2004":, "payoff"]))
print("demeaned RV mean,t:", mean_t(m["iv"] - m["rv_dm"]))

# 7. Charts for slides 5 and 6
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(m["iv_vol"], m["rv_vol"], s=8, alpha=.6)
lim = [0, max(m["iv_vol"].max(), m["rv_vol"].max())]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set(xlabel="Implied vol (VIX/100)", ylabel="Realised vol, next 21 days",
          title="Below the line = VIX too pessimistic")
m["payoff_sc"].cumsum().plot(ax=ax[1], title="Cumulative scaled seller payoff")
for d in m["payoff_sc"].nsmallest(3).index:
    ax[1].axvline(d, color="r", lw=.6)
plt.tight_layout(); plt.savefig("charts.png", dpi=200)
```

The step-by-step logic matters more than the syntax. Data from both sources is **inner-joined on dates**, so a holiday gap in one series cannot shift the windows. Implied variance uses only the close on day t. Realised variance uses only returns from t+1 to t+21; the reverse-rolling sum followed by `shift(-1)` enforces this, and the `assert` checks it on one row (check two or three more by hand). The script then samples once a month, computes both the raw and the scaled payoff, and runs the significance, distribution and crash-clustering tests the brief asks for. The `cumsum` of scaled payoffs is a simple sum of monthly returns, not a compounded equity curve. Label it that way on the slide.

## What would break the result, and what to expect

The biggest threat is **look-ahead bias**: accidentally letting the realised window include day t, or using a VIX value from after the window starts. The assert guards against the first. The second is avoided by using the close on day t.

There is also a **horizon mismatch**. The VIX covers 30 *calendar* days, measured in minutes on a 525,600-minute year ([Cboe Methodology](https://cdn.cboe.com/resources/indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf)). Our window is 21 *trading* days. Both are annualised, so the comparison is consistent, but test H = 22 as a robustness check.

**Pre-2003 VIX values are back-calculated**, so rerun the analysis on 2004–2025 (the script does). **Measure choice changes the story.** Raw variance payoffs are dominated by a few crash months, while the scaled payoff and the vol-point spread are steadier. Pick one headline measure and state it on every slide. Finally, **sampling date matters**. With first-of-month sampling, a crash is credited to whichever month's start date opens the window that contains it, so check which window holds October 2008 and March 2020.

**Sanity checks (expected, unverified):**

- **Mean vols.** Mean VIX should be about 19–20 and mean realised vol about 15–16. These are consistent with the published 19.59 vs 15.50 for 1990–2024 ([CFA Institute](https://blogs.cfainstitute.org/blog/2024/07/31/how-well-does-the-market-predict-volatility)) and 19.3 vs 15.1 for 1990–2018 ([Cboe](https://www.cboe.com/insights/posts/white-paper-shows-volatility-risk-premium-facilitated-higher-risk-adjusted-returns-for-put-index)). If yours are far off, suspect a units bug (forgetting /100, or ×252 vs ×12).
- **Month count.** About 432 months.
- **Unsourced priors.** These should be clearly flagged if quoted before you compute them: a win rate of perhaps 80–90%, strongly negative skew, worst months around late 2008 and March 2020 with Aug 2011, Aug 2015 and Feb 2018 also near the bottom, and a clearly positive correlation between the payoff and the S&P's forward return.
- **Fragile t-stat.** A raw-variance t-stat that drops sharply once the worst 5 months are removed is not a failure. It *is* the crash-risk finding.

## Five people, twenty hours, seven slides

With about 4 hours each, split the work by output, not by topic, so that everyone owns a slide:

| Person | Task (~4 h) | Owns slide |
|---|---|---|
| 1. Data lead | Run sections 1–3 of the script, cache the CSVs, hand-check alignment on 3 dates, confirm the 2008/2020 record closes from FRED | 3 (data) |
| 2. Stats lead | Sections 4 and 6: t-stats (iid and NW), win rate, skew, kurtosis, mean excluding worst 5, post-2004 rerun | 4 (method) |
| 3. Crash analyst | Section 5: sell-off conditioning, loss share, list of the worst 10 months with the news event behind each | 6 (cumulative/worst months) |
| 4. Charts and literature | Section 7 charts styled for slides; one-line literature box with correct sign conventions | 5 (scatter), 2 (idea examples) |
| 5. Story and presenter | Title, conclusion, consistent units and sign across slides, 5-minute script, rehearsal | 1, 7 |

A beginner contributes best by taking role 1 or 3. Both need careful checking more than deep theory, and both catch the errors that sink pitches.

### Glossary

| Term | Plain meaning |
|---|---|
| Log return | ln(today's price ÷ yesterday's price); adds up over time |
| Volatility / variance | How much prices wobble (standard deviation) / that number squared |
| Implied vs realised | Wobble priced into options vs wobble that actually happened |
| VIX | 30-day implied vol of the S&P 500, in annual % |
| Variance swap | A bet paying realised variance minus a fixed strike (≈ VIX²) |
| VRP | Implied minus realised variance (our sign); the insurance premium |
| Scaled payoff | (IV − RV)/IV; at most +100%, no lower limit |
| t-stat | Mean ÷ standard error; above 2 ≈ significant |
| Newey–West | Standard errors that correct for overlapping/autocorrelated data |
| Skew / kurtosis | Lopsidedness / fat-tailedness of outcomes |
| Look-ahead bias | Using information you would not have had at the time |

### Must-know before the first meeting

| Check | You can say it in one line |
|---|---|
| Why variance, not vol | Variance adds over time and is what option strips replicate |
| VIX 20 in monthly terms | About ±5.8% a month, ±1.26% a day |
| Calm vs crash example | (0.04 − 0.0225)/0.04 = +44%; (0.04 − 0.16)/0.04 = −300% |
| Our sign convention | Implied − realised (positive = seller wins); Carr & Wu is the reverse |
| Carr & Wu sample | 1996–2003; do not quote Table 5 |
| Pre-2003 VIX | Back-calculated, so rerun on 2004+ |
| No look-ahead | IV from day t close; RV from t+1 to t+21 only |
| Why Sharpe misleads | It ignores skew; crash months dominate |

## Conclusion

The real deliverable is not proving that the VIX overshoots, since that is long established. It is showing *how* the premium is earned. If the team's data shows a strongly positive mean, a high win rate, a deeply negative skew, and losses concentrated in the worst S&P months, the answer to "is the VIX too pessimistic?" becomes: "on average yes, but it is pessimistic about exactly the months that hurt most, so it is fair pay for insurance, not a mistake". That framing turns a fragile t-stat or a null result on the post-2003 sample into an insight rather than a failure.

The most common way beginners damage this kind of pitch is inconsistency: mixing vol points and variance, flipping the sign convention between slides, or citing a paper's numbers for a period it never studied. One headline measure, one sign, and every number traced to either the team's own code or a cited source is what will make the deck credible.

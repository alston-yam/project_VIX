# Key market events

> **Auto-generated** by `python -m vrp.pipeline` (`src/vrp/events.py`). Do not edit by hand: change the event list in `events.py` and re-run `make data`.

Only the event windows below were chosen by hand. Every statistic is computed from our processed data (VIX closes from FRED, S&P 500 closes from Yahoo Finance).

How to read the columns:

- **Max VIX close**: highest daily *closing* VIX inside the window.
- **Worst S&P day**: most negative daily log return inside the window.
- **S&P drawdown**: largest peak-to-trough fall of the S&P close, starting from the last close *before* the window (so a first-day crash counts).

## Event table

| event_id | name | start_date | end_date | category | description |
|---|---|---|---|---|---|
| gulf_war_1990 | Gulf War / 1990 recession | 1990-08-01 | 1990-10-31 | geopolitical | Iraq invades Kuwait (2 Aug 1990); oil spike during the 1990–91 NBER recession. |
| asian_crisis_1997 | Asian financial crisis | 1997-10-01 | 1997-10-31 | financial crisis | Contagion from Asian currency crises; 27 Oct 1997 US sell-off and trading halts. |
| ltcm_1998 | Russia default / LTCM | 1998-08-01 | 1998-10-31 | financial crisis | Russia defaults (17 Aug 1998); hedge fund LTCM rescued (23 Sep 1998). |
| dotcom_bust | Dot-com bust | 2000-03-24 | 2002-10-09 | bear market | S&P 500 falls from its Mar 2000 peak to its Oct 2002 low. |
| sep11_2001 | 9/11 attacks | 2001-09-11 | 2001-09-28 | geopolitical | Terrorist attacks; NYSE closed 11–14 Sep 2001 and reopened 17 Sep. |
| worldcom_2002 | WorldCom / accounting scandals | 2002-06-25 | 2002-07-31 | financial crisis | WorldCom reveals accounting fraud (25 Jun 2002), files for bankruptcy (21 Jul 2002). |
| gfc_2008 | Global Financial Crisis | 2008-09-15 | 2009-03-09 | financial crisis | Lehman bankruptcy (15 Sep 2008); peak stress Oct–Nov 2008; S&P low on 9 Mar 2009. |
| flash_crash_2010 | Flash Crash | 2010-05-06 | 2010-05-28 | market structure | Intraday collapse and rebound on 6 May 2010; euro-area worries in the following weeks. |
| us_downgrade_2011 | US downgrade / euro crisis | 2011-08-01 | 2011-08-31 | macro / policy | S&P downgrades US debt (5 Aug 2011); euro sovereign debt stress. |
| china_deval_2015 | China devaluation | 2015-08-11 | 2015-08-31 | macro / policy | PBoC devalues the yuan (11 Aug 2015); 24 Aug 2015 opening crash. |
| volmageddon_2018 | Volmageddon (XIV collapse) | 2018-02-01 | 2018-02-09 | market structure | Short-vol products blow up on 5 Feb 2018; XIV loses ~96% and is terminated. |
| q4_2018 | Q4 2018 sell-off | 2018-10-01 | 2018-12-31 | macro / policy | Fed tightening and trade-war fears; S&P low on 24 Dec 2018. |
| covid_2020 | COVID crash | 2020-02-19 | 2020-03-31 | pandemic | Pandemic sell-off from the 19 Feb 2020 peak; record VIX close on 16 Mar 2020. |
| bear_2022 | 2022 bear market | 2022-01-03 | 2022-10-12 | bear market | Inflation and rapid Fed hikes; S&P low on 12 Oct 2022. |
| yen_carry_2024 | Yen carry-trade unwind | 2024-08-01 | 2024-08-09 | market structure | BoJ hike and yen carry unwind; VIX spikes on 5 Aug 2024. |
| tariff_2025 | 2025 tariff shock | 2025-04-02 | 2025-04-30 | macro / policy | US 'Liberation Day' tariff announcement (2 Apr 2025) and sell-off. |

## What the data shows

| Event | Start | End | Max VIX close | Date | Worst S&P day (log) | Date  | S&P drawdown |
|---|---|---|---|---|---|---|---|
| Gulf War / 1990 recession | 1990-08-01 | 1990-10-31 | 36.47 | 1990-08-23 | -3.07% | 1990-08-06 | -17.0% |
| Asian financial crisis | 1997-10-01 | 1997-10-31 | 38.20 | 1997-10-30 | -7.11% | 1997-10-27 | -10.8% |
| Russia default / LTCM | 1998-08-01 | 1998-10-31 | 45.74 | 1998-10-08 | -7.04% | 1998-08-31 | -14.6% |
| Dot-com bust | 2000-03-24 | 2002-10-09 | 45.08 | 2002-08-05 | -6.00% | 2000-04-14 | -49.1% |
| 9/11 attacks | 2001-09-11 | 2001-09-28 | 43.74 | 2001-09-20 | -5.05% | 2001-09-17 | -11.6% |
| WorldCom / accounting scandals | 2002-06-25 | 2002-07-31 | 44.92 | 2002-07-23 | -3.91% | 2002-07-19 | -19.6% |
| Global Financial Crisis | 2008-09-15 | 2009-03-09 | 80.86 | 2008-11-20 | -9.47% | 2008-10-15 | -46.1% |
| Flash Crash | 2010-05-06 | 2010-05-28 | 45.79 | 2010-05-20 | -3.98% | 2010-05-20 | -8.9% |
| US downgrade / euro crisis | 2011-08-01 | 2011-08-31 | 48.00 | 2011-08-08 | -6.90% | 2011-08-08 | -13.4% |
| China devaluation | 2015-08-11 | 2015-08-31 | 40.74 | 2015-08-24 | -4.02% | 2015-08-24 | -11.2% |
| Volmageddon (XIV collapse) | 2018-02-01 | 2018-02-09 | 37.32 | 2018-02-05 | -4.18% | 2018-02-05 | -8.6% |
| Q4 2018 sell-off | 2018-10-01 | 2018-12-31 | 36.07 | 2018-12-24 | -3.34% | 2018-10-10 | -19.6% |
| COVID crash | 2020-02-19 | 2020-03-31 | 82.69 | 2020-03-16 | -12.77% | 2020-03-16 | -33.9% |
| 2022 bear market | 2022-01-03 | 2022-10-12 | 36.45 | 2022-03-07 | -4.42% | 2022-09-13 | -25.4% |
| Yen carry-trade unwind | 2024-08-01 | 2024-08-09 | 38.57 | 2024-08-05 | -3.04% | 2024-08-05 | -6.1% |
| 2025 tariff shock | 2025-04-02 | 2025-04-30 | 52.33 | 2025-04-08 | -6.16% | 2025-04-04 | -12.1% |

## Monthly-sample rows that overlap each event

A monthly row overlaps an event if its forward window (t+1 … t+21) intersects the event window. `payoff_scaled` = 1 − RV/IV (positive = variance seller gained).

| Event | Rows | month:payoff_scaled | Worst payoff_scaled |
|---|---|---|---|
| Gulf War / 1990 recession | 4 | 1990-07:+0.415; 1990-08:-0.471; 1990-09:+0.593; 1990-10:+0.442 | -0.471 |
| Asian financial crisis | 2 | 1997-09:+0.660; 1997-10:-1.256 | -1.256 |
| Russia default / LTCM | 3 | 1998-08:-0.983; 1998-09:+0.162; 1998-10:+0.735 | -0.983 |
| Dot-com bust | 32 | 2000-03:-0.641; 2000-04:-0.844; 2000-05:+0.095; 2000-06:+0.530; 2000-07:+0.390; 2000-08:+0.742; 2000-09:+0.359; 2000-10:-0.471; 2000-11:+0.335; 2000-12:-0.346; 2001-01:+0.457; 2001-02:+0.337; 2001-03:-0.105; 2001-04:+0.096; 2001-05:+0.521; 2001-06:+0.588; 2001-07:+0.038; 2001-08:+0.374; 2001-09:-0.425; 2001-10:+0.628; 2001-11:+0.802; 2001-12:+0.668; 2002-01:+0.482; 2002-02:+0.072; 2002-03:+0.526; 2002-04:+0.204; 2002-05:-0.186; 2002-06:+0.116; 2002-07:-1.403; 2002-08:+0.255; 2002-09:+0.413; 2002-10:-0.011 | -1.403 |
| 9/11 attacks | 1 | 2001-09:-0.425 | -0.425 |
| WorldCom / accounting scandals | 2 | 2002-06:+0.116; 2002-07:-1.403 | -1.403 |
| Global Financial Crisis | 7 | 2008-09:-5.101; 2008-10:-3.320; 2008-11:-0.972; 2008-12:+0.697; 2009-01:+0.110; 2009-02:+0.245; 2009-03:+0.237 | -5.101 |
| Flash Crash | 1 | 2010-05:-1.618 | -1.618 |
| US downgrade / euro crisis | 2 | 2011-07:-0.105; 2011-08:-3.189 | -3.189 |
| China devaluation | 1 | 2015-08:-4.156 | -4.156 |
| Volmageddon (XIV collapse) | 2 | 2018-01:+0.116; 2018-02:-2.840 | -2.840 |
| Q4 2018 sell-off | 4 | 2018-09:+0.833; 2018-10:-2.758; 2018-11:+0.090; 2018-12:-2.741 | -2.758 |
| COVID crash | 2 | 2020-02:-2.502; 2020-03:-6.812 | -6.812 |
| 2022 bear market | 10 | 2022-01:-0.254; 2022-02:-0.061; 2022-03:+0.542; 2022-04:-0.575; 2022-05:+0.098; 2022-06:-0.337; 2022-07:+0.481; 2022-08:+0.238; 2022-09:-0.042; 2022-10:+0.233 | -0.575 |
| Yen carry-trade unwind | 1 | 2024-08:-0.044 | -0.044 |
| 2025 tariff shock | 1 | 2025-04:-3.798 | -3.798 |

## Verification notes

- **2008 record close.** Published sources conflict (80.86 on 20 Nov 2008 vs 80.74 on 21 Nov 2008). FRED shows **80.86** on 2008-11-20 and **72.67** on 2008-11-21. We use FRED.
- **Highest close in the sample.** 82.69 on 2020-03-16 (expected 82.69 on 2020-03-16; checked in `validate.py`).
- **5 Aug 2024.** The widely quoted VIX of about 65 was an *intraday* print before the US open. FRED records closes only: **38.57** on 2024-08-05.
- **9/11.** US equity markets were closed 11–14 Sep 2001, so the window's first trading day is 17 Sep 2001 and the drawdown starts from the 10 Sep close.
- **Pre-2003 VIX values** (1990, 1997, 1998, 2001, 2002 events) are back-calculated by Cboe with today's method; they are not what traders saw in real time.

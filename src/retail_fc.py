# -*- coding: utf-8 -*-
"""
retail_fc - small helper library for the JKU Linz T2 session
"Demand Forecasting & Inventory Analytics" (Digital Analytics in Retail).

Everything a student needs in one readable file: loading the course data,
classifying demand patterns, forecast accuracy metrics and their blind spots,
backtesting splitters, simple univariate forecasters (including Croston and
TSB for intermittent demand), feature building for a global LightGBM model,
and the inventory formulas that turn a forecast into a stock decision.

Each function is short on purpose. Read them; that is part of the course.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- data
def load_data(data_dir="../data"):
    """Return dict(demand, products, promotions, calendar). Dates parsed."""
    d = Path(data_dir)
    demand = pd.read_csv(d / "demand_weekly.csv", parse_dates=["week_start"])
    products = pd.read_csv(d / "products.csv")
    promotions = pd.read_csv(d / "promotions.csv", parse_dates=["week_start"])
    calendar = pd.read_csv(d / "calendar.csv", parse_dates=["week_start"])
    demand = demand.merge(products[["product_id", "product_name", "category"]], on="product_id")
    demand["series_id"] = demand["product_id"] + "_" + demand["channel"]
    demand = demand.sort_values(["series_id", "week_start"]).reset_index(drop=True)
    return dict(demand=demand, products=products, promotions=promotions, calendar=calendar)


def to_wide(demand, value="units"):
    """Long table -> one column per series (product_id_channel), one row per week."""
    return demand.pivot(index="week_start", columns="series_id", values=value).sort_index()


# ------------------------------------------------------- demand pattern (Syntetos-Boylan)
ADI_CUT, CV2_CUT = 1.32, 0.49


def adi_cv2(y):
    """Average Demand Interval and squared coefficient of variation of non-zero sizes."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    if nz.size == 0:
        return np.inf, np.nan
    adi = y.size / nz.size
    cv2 = float((nz.std(ddof=0) / nz.mean()) ** 2) if nz.size > 1 else 0.0
    return float(adi), cv2


def classify_sbc(adi, cv2, adi_cut=ADI_CUT, cv2_cut=CV2_CUT):
    """Syntetos-Boylan-Croston quadrant: smooth / erratic / intermittent / lumpy."""
    if not np.isfinite(adi):
        return "no demand"
    if adi < adi_cut:
        return "smooth" if cv2 < cv2_cut else "erratic"
    return "intermittent" if cv2 < cv2_cut else "lumpy"


def pattern_table(wide):
    """ADI, CV^2, zero share and SBC class for every column of a wide table."""
    rows = []
    for col in wide.columns:
        y = wide[col].to_numpy()
        adi, cv2 = adi_cv2(y)
        rows.append(dict(series_id=col, adi=adi, cv2=cv2, zero_share=float((y == 0).mean()),
                         mean_units=float(y.mean()), pattern=classify_sbc(adi, cv2)))
    return pd.DataFrame(rows).set_index("series_id")


# --------------------------------------------------------------------------- metrics
def mae(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def bias(y, f):
    """Mean error (forecast - actual). Positive = over-forecast."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(f - y))


def mape(y, f):
    """Mean absolute percentage error. Undefined (inf) if any actual is zero:
    that is the point, not a bug to hide."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    if np.any(y == 0):
        return float("inf")
    return float(np.mean(np.abs(y - f) / np.abs(y)) * 100)


def mape_nonzero(y, f):
    """MAPE computed only on periods with non-zero actuals (what most tools silently do)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = y != 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[m] - f[m]) / np.abs(y[m])) * 100)


def smape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = (np.abs(y) + np.abs(f))
    r = np.where(den == 0, 0.0, 2 * np.abs(y - f) / np.where(den == 0, 1, den))
    return float(np.mean(r) * 100)


def wape(y, f):
    """Weighted APE = sum|error| / sum|actual|. Scale-free, zero-safe, weights big items more."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    s = np.sum(np.abs(y))
    return float("inf") if s == 0 else float(np.sum(np.abs(y - f)) / s * 100)


def mase(y, f, y_train, m=1):
    """MAE scaled by the in-sample MAE of the seasonal naive (m=1: plain naive)."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float("inf") if scale == 0 else mae(y, f) / scale


def rmsse(y, f, y_train, m=1):
    """RMSE scaled by in-sample RMSE of the (seasonal) naive. Used in the M5 competition."""
    y_train = np.asarray(y_train, float)
    scale = np.sqrt(np.mean((y_train[m:] - y_train[:-m]) ** 2))
    return float("inf") if scale == 0 else rmse(y, f) / scale


def pinball(y, q_pred, q):
    """Pinball (quantile) loss for a forecast of quantile q."""
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(y, f, y_train=None, m=1):
    out = dict(MAE=mae(y, f), RMSE=rmse(y, f), Bias=bias(y, f), MAPE=mape(y, f),
               MAPE_nonzero=mape_nonzero(y, f), sMAPE=smape(y, f), WAPE=wape(y, f))
    if y_train is not None:
        out["MASE"] = mase(y, f, y_train, m)
        out["RMSSE"] = rmsse(y, f, y_train, m)
    return out


# ---------------------------------------------------------------------- splitters
def fixed_split(n, test_size):
    """One split: train = [0, n-test_size), test = [n-test_size, n)."""
    yield np.arange(0, n - test_size), np.arange(n - test_size, n)


def rolling_origin(n, min_train, horizon, step=1, window=None):
    """Rolling-origin evaluation. window=None -> expanding (walk-forward);
    window=int -> sliding window of fixed length. Yields (train_idx, test_idx)."""
    origin = min_train
    while origin + horizon <= n:
        start = 0 if window is None else max(0, origin - window)
        yield np.arange(start, origin), np.arange(origin, origin + horizon)
        origin += step


def blocked_kfold(n, k, gap=0):
    """k contiguous blocks; fold i tests on block i and trains on everything BEFORE it
    (minus a gap). Fold 0 has no past, so it is skipped. Time order is never broken."""
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(1, k):
        yield np.arange(0, max(0, edges[i] - gap)), np.arange(edges[i], edges[i + 1])


# ---------------------------------------------------------- univariate forecasters
def naive(y, h):
    return np.repeat(float(np.asarray(y)[-1]), h)


def seasonal_naive(y, h, m=52):
    y = np.asarray(y, float)
    if len(y) < m:
        return naive(y, h)
    return np.array([y[-m + (i % m)] for i in range(h)])


def moving_average(y, h, window=8):
    y = np.asarray(y, float)
    return np.repeat(float(y[-window:].mean()), h)


def ses(y, h, alpha=None):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(np.asarray(y, float), initialization_method="estimated").fit(
            smoothing_level=alpha, optimized=alpha is None)
    return np.asarray(fit.forecast(h))


def holt_winters(y, h, m=52, trend="add", seasonal="add", damped=True):
    """ETS-style Holt-Winters. Falls back to Holt (no seasonality) on short histories."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = np.asarray(y, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(y) >= 2 * m:
            fit = ExponentialSmoothing(y, trend=trend, damped_trend=damped, seasonal=seasonal,
                                       seasonal_periods=m, initialization_method="estimated").fit()
        else:
            fit = ExponentialSmoothing(y, trend=trend, damped_trend=damped,
                                       initialization_method="estimated").fit()
    return np.clip(np.asarray(fit.forecast(h)), 0, None)


def croston(y, h, alpha=0.1, variant="classic"):
    """Croston (1972): smooth the non-zero size z and the interval p separately,
    forecast z/p. variant='sba' applies the Syntetos-Boylan bias correction (1 - alpha/2)."""
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(h)
    z, p, q = y[nz[0]], float(nz[0] + 1), 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            p += alpha * (q - p)
            q = 1
        else:
            q += 1
    f = z / p
    if variant == "sba":
        f *= (1 - alpha / 2)
    return np.repeat(float(f), h)


def tsb(y, h, alpha=0.1, beta=0.1):
    """Teunter-Syntetos-Babai (2011): smooth the demand PROBABILITY every period (so the
    forecast decays during long zero runs, unlike Croston) and the size on demand periods."""
    y = np.asarray(y, float)
    p = float((y > 0).mean()) or 0.01
    z = float(y[y > 0].mean()) if (y > 0).any() else 0.0
    for v in y:
        if v > 0:
            p += beta * (1 - p)
            z += alpha * (v - z)
        else:
            p += beta * (0 - p)
    return np.repeat(p * z, h)


FORECASTERS = {
    "Naive": naive,
    "Seasonal naive (52)": seasonal_naive,
    "Moving average (8)": moving_average,
    "SES": ses,
    "Holt-Winters": holt_winters,
    "Croston": croston,
    "Croston SBA": lambda y, h: croston(y, h, variant="sba"),
    "TSB": tsb,
}


# ----------------------------------------------------------- global model features
LAGS = (1, 2, 3, 4, 8, 12, 26, 52)
ROLLS = (4, 8, 13, 26)


def make_features(demand, horizon=1, lags=LAGS, rolls=ROLLS):
    """Direct h-step features. The target is units at week t; every lag or rolling
    statistic is computed from data available at the forecast origin t - horizon.
    Promotion, price and calendar at week t are known in advance, so they enter as-is."""
    df = demand.sort_values(["series_id", "week_start"]).copy()
    g = df.groupby("series_id")["units"]
    for k in lags:
        df[f"lag_{k}"] = g.shift(horizon + k - 1)
    shifted = g.shift(horizon)
    for w in rolls:
        df[f"roll_mean_{w}"] = shifted.groupby(df["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    df["roll_std_8"] = shifted.groupby(df["series_id"]).transform(
        lambda s: s.rolling(8, min_periods=2).std())
    df["zero_share_13"] = shifted.groupby(df["series_id"]).transform(
        lambda s: (s == 0).rolling(13, min_periods=1).mean())
    # weeks since the last non-zero week, as seen from the origin
    def _since(s):
        out, c = np.empty(len(s)), 0
        for i, v in enumerate(s.to_numpy()):
            c = 0 if v > 0 else c + 1
            out[i] = c
        return pd.Series(out, index=s.index)
    df["weeks_since_sale"] = shifted.fillna(0).groupby(df["series_id"]).transform(_since)
    df["month"] = df["week_start"].dt.month
    df["iso_week"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["is_black_friday_week"] = (df["iso_week"] == 47).astype(int)
    df["is_christmas_window"] = df["iso_week"].between(48, 51).astype(int)
    df["horizon"] = horizon
    for c in ("product_id", "channel", "category"):
        df[c] = df[c].astype("category")
    df = df.dropna(subset=[f"lag_{max(lags)}"])
    return df


FEATURE_COLS = ([f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in ROLLS] +
                ["roll_std_8", "zero_share_13", "weeks_since_sale", "promo_flag", "discount_pct",
                 "price", "month", "iso_week", "is_black_friday_week", "is_christmas_window",
                 "product_id", "channel", "category"])


def lgbm_params(**over):
    p = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03,
             n_estimators=600, num_leaves=31, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=42)
    p.update(over)
    return p


# --------------------------------------------------------------------- inventory
from scipy.stats import norm  # noqa: E402


def lead_time_stats(weekly_forecast, sigma_week, lead_time):
    """Mean and standard deviation of demand over the lead time, assuming independent
    weekly errors with the same sigma. weekly_forecast may be a scalar or a vector."""
    f = np.atleast_1d(np.asarray(weekly_forecast, float))
    mean_lt = float(f[:lead_time].sum()) if f.size >= lead_time else float(f.mean() * lead_time)
    return mean_lt, float(sigma_week * np.sqrt(lead_time))


def safety_stock(sigma_lt, service_level):
    """z * sigma over the lead time, for a cycle service level (probability of no stock-out)."""
    return float(norm.ppf(service_level) * sigma_lt)


def reorder_point(mean_lt, ss):
    return float(mean_lt + ss)


def newsvendor_quantile(cost_under, cost_over):
    """Critical ratio: the quantile of demand to stock when a unit short costs cost_under
    and a unit left over costs cost_over."""
    return cost_under / (cost_under + cost_over)


def simulate_order_up_to(demand, forecast, sigma_week, lead_time, service_level, review=1):
    """Periodic-review order-up-to policy driven by a weekly forecast. Returns fill rate,
    stock-out weeks, average on-hand stock and lost units. Kept deliberately simple."""
    demand, forecast = np.asarray(demand, float), np.asarray(forecast, float)
    n = len(demand)
    on_hand, pipeline = forecast[:lead_time + review].sum(), []
    lost, on_hand_hist, so_weeks = 0.0, [], 0
    for t in range(n):
        # receive
        arrived = sum(q for (arr, q) in pipeline if arr == t)
        pipeline = [(a, q) for (a, q) in pipeline if a != t]
        on_hand += arrived
        # order up to S = forecast over (L + R) + safety stock
        if t % review == 0:
            horizon = forecast[t:t + lead_time + review]
            mean_lt = horizon.sum() if horizon.size else forecast[-1] * (lead_time + review)
            ss = safety_stock(sigma_week * np.sqrt(lead_time + review), service_level)
            position = on_hand + sum(q for _, q in pipeline)
            order = max(0.0, mean_lt + ss - position)
            if order > 0:
                pipeline.append((t + lead_time, order))
        # demand
        served = min(on_hand, demand[t])
        lost += demand[t] - served
        so_weeks += int(demand[t] > on_hand)
        on_hand -= served
        on_hand_hist.append(on_hand)
    total = demand.sum()
    return dict(fill_rate=float(1 - lost / total) if total > 0 else 1.0,
                stockout_weeks=so_weeks, avg_on_hand=float(np.mean(on_hand_hist)), lost_units=float(lost))

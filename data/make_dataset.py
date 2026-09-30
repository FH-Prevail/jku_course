# -*- coding: utf-8 -*-
"""
make_dataset.py - synthetic omnichannel retail demand for the JKU Linz T2 session
(Demand Forecasting & Inventory Analytics, 03.10.2026).

One row per product x channel x week, 156 weeks (2023-01-02 to 2025-12-29),
40 products in 6 categories, 2 channels (store, online) = 80 series.

The generator deliberately mixes the four Syntetos-Boylan demand patterns
(smooth, erratic, intermittent, lumpy), adds trend, yearly seasonality,
promotions with a post-promotion dip, Black Friday online spikes and two short
stock-outs, so that every topic of the session has something real to look at.

    python make_dataset.py          # writes the CSV files next to this script

The intended pattern of each product is written to generator_truth.csv for the
instructor only. Students derive the pattern from the data (ADI / CV^2).
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261003
OUT = Path(__file__).resolve().parent
N_WEEKS = 156
WEEKS = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")
CHANNELS = ["store", "online"]

# category: (n products, price range, gross margin, lead time weeks range,
#            seasonality profile, weekly promotion probability, pattern mix)
CATEGORIES = {
    "Electronics": dict(n=8, price=(49, 499), margin=0.25, lt=(2, 4), season="christmas",
                        promo=0.10, mix=["smooth"] * 2 + ["erratic"] * 3 + ["intermittent"] * 2 + ["lumpy"]),
    "Toys":        dict(n=6, price=(9, 79), margin=0.40, lt=(3, 6), season="christmas_strong",
                        promo=0.08, mix=["smooth"] * 2 + ["erratic"] + ["intermittent"] * 2 + ["lumpy"]),
    "Garden":      dict(n=6, price=(12, 249), margin=0.35, lt=(2, 5), season="summer",
                        promo=0.06, mix=["smooth"] + ["erratic"] * 2 + ["intermittent"] * 2 + ["lumpy"]),
    "Grocery":     dict(n=8, price=(1.5, 12), margin=0.20, lt=(1, 1), season="flat",
                        promo=0.15, mix=["smooth"] * 6 + ["erratic"] * 2),
    "Fashion":     dict(n=6, price=(19, 149), margin=0.55, lt=(4, 8), season="two_peaks",
                        promo=0.12, mix=["smooth"] * 2 + ["erratic"] * 2 + ["intermittent", "lumpy"]),
    "Spare Parts": dict(n=6, price=(5, 89), margin=0.45, lt=(2, 6), season="flat",
                        promo=0.0, mix=["intermittent"] * 3 + ["lumpy"] * 3),
}

NAMES = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Plug", "4K Streaming Stick",
                    "Robot Vacuum", "Gaming Headset", "E-Reader", "Soundbar"],
    "Toys": ["Building Blocks Set", "Plush Bear", "Puzzle 1000", "RC Car", "Board Game Classic",
             "Wooden Train Set"],
    "Garden": ["Garden Hose 20m", "Plant Fertilizer 1kg", "Patio Chair", "Hedge Trimmer",
               "Solar Path Light", "Barbecue Grill"],
    "Grocery": ["Espresso Beans 1kg", "Olive Oil 750ml", "Pasta 500g", "Oat Drink 1L",
                "Dark Chocolate 100g", "Mineral Water 6x1L", "Protein Bar", "Green Tea 50 bags"],
    "Fashion": ["Denim Jacket", "Running Shoes", "Wool Scarf", "Linen Shirt", "Rain Coat",
                "Leather Belt"],
    "Spare Parts": ["Vacuum Filter HEPA", "Coffee Machine Gasket", "Grill Igniter",
                    "Trimmer Blade", "Speaker Battery", "Hose Connector"],
}

# base weekly demand (both channels together) by intended pattern
BASE = {"smooth": (60, 260), "erratic": (12, 45), "intermittent": (0.6, 1.8), "lumpy": (0.5, 1.4)}


def season_profile(kind, woy, channel):
    """Multiplicative seasonal index for an ISO week-of-year array."""
    w = woy.astype(float)
    idx = np.ones_like(w)
    bump = lambda c, s, h: h * np.exp(-0.5 * ((w - c) / s) ** 2)
    if kind == "christmas":
        idx += bump(50, 2.5, 0.6) + bump(47, 1.0, 0.25)
        idx -= np.where(w <= 4, 0.15, 0.0)
    elif kind == "christmas_strong":
        idx += bump(50, 2.8, 1.5)
        idx -= np.where(w <= 6, 0.25, 0.0)
    elif kind == "summer":
        idx += bump(22, 6.0, 0.8) - bump(3, 6.0, 0.45) - bump(51, 4.0, 0.45)
    elif kind == "two_peaks":
        idx += bump(12, 3.0, 0.35) + bump(38, 3.0, 0.35) + bump(50, 2.0, 0.3)
    else:  # flat, mild yearly wave
        idx += 0.05 * np.sin(2 * np.pi * (w - 10) / 52)
    if channel == "online":
        # Black Friday week (ISO week 47) online spike for gift categories
        if kind in ("christmas", "christmas_strong", "two_peaks"):
            idx += bump(47, 0.7, 0.8)
    else:
        if kind.startswith("christmas"):
            idx += bump(51, 1.2, 0.2)   # last-minute store shopping
    return np.clip(idx, 0.15, None)


def main():
    rng = np.random.default_rng(SEED)
    woy = WEEKS.isocalendar().week.to_numpy()
    t = np.arange(N_WEEKS)

    products, demand_rows, promo_rows, truth = [], [], [], []
    pid = 0
    for cat, spec in CATEGORIES.items():
        for j in range(spec["n"]):
            pid += 1
            product_id = f"P{pid:03d}"
            pattern = spec["mix"][j]
            price = float(np.round(rng.uniform(*spec["price"]), 2))
            cost = float(np.round(price * (1 - spec["margin"]) * rng.uniform(0.9, 1.1), 2))
            lead = int(rng.integers(spec["lt"][0], spec["lt"][1] + 1))
            store_share = float(rng.uniform(0.45, 0.75))
            base_total = float(rng.uniform(*BASE[pattern]))
            products.append(dict(product_id=product_id, product_name=NAMES[cat][j], category=cat,
                                 unit_price=price, unit_cost=cost, lead_time_weeks=lead,
                                 holding_cost_rate=0.20))
            truth.append(dict(product_id=product_id, intended_pattern=pattern,
                              base_weekly_units=round(base_total, 2), store_share=round(store_share, 2)))

            for ch in CHANNELS:
                share = store_share if ch == "store" else 1 - store_share
                growth = rng.uniform(-0.03, 0.03) if ch == "store" else rng.uniform(0.06, 0.16)
                level = base_total * share * (1 + growth) ** (t / 52)
                lam = level * season_profile(spec["season"], woy, ch)

                # promotions: known in advance, never two weeks in a row
                promo = np.zeros(N_WEEKS, dtype=int)
                discount = np.zeros(N_WEEKS)
                if spec["promo"] > 0:
                    for k in range(1, N_WEEKS):
                        if promo[k - 1] == 0 and rng.random() < spec["promo"]:
                            promo[k] = 1
                            discount[k] = rng.choice([10, 15, 20, 25, 30])
                elasticity = rng.uniform(2.0, 3.5) * (1.15 if ch == "online" else 1.0)
                uplift = (1 - discount / 100) ** (-elasticity)
                dip = np.ones(N_WEEKS)
                dip[1:] = np.where(promo[:-1] == 1, 0.85, 1.0)   # post-promotion dip
                lam = lam * uplift * dip
                lam = lam * rng.lognormal(0, 0.08, N_WEEKS)      # week-to-week noise

                if pattern == "smooth":
                    k = rng.uniform(25, 60)                      # low dispersion
                    units = rng.negative_binomial(k, k / (k + lam))
                elif pattern == "erratic":
                    k = rng.uniform(0.9, 1.6)                    # high dispersion, rarely zero
                    units = rng.negative_binomial(k, k / (k + lam))
                elif pattern == "intermittent":
                    p = np.clip(lam / (lam + 1.2), 0.05, 0.95)   # occurrence probability
                    p = np.clip(p * (1 + 0.6 * promo), 0.0, 0.97)
                    occ = rng.random(N_WEEKS) < p
                    size = 1 + rng.poisson(rng.uniform(0.3, 1.2), N_WEEKS)   # small, regular sizes
                    units = occ * size
                else:  # lumpy: rare, and when it happens the size is all over the place
                    p = np.clip(lam / (lam + 1.6), 0.04, 0.6)
                    occ = rng.random(N_WEEKS) < p
                    size = np.round(rng.lognormal(rng.uniform(1.0, 1.8), 0.9, N_WEEKS)).astype(int)
                    size = np.clip(size, 1, None)
                    units = occ * size

                units = units.astype(int)
                eff_price = np.round(price * (1 - discount / 100), 2)
                for k in range(N_WEEKS):
                    demand_rows.append((WEEKS[k].date().isoformat(), product_id, ch, int(units[k]),
                                        float(eff_price[k]), int(discount[k]), int(promo[k]), 0))
                    if promo[k]:
                        promo_rows.append(dict(product_id=product_id, channel=ch,
                                               week_start=WEEKS[k].date().isoformat(),
                                               discount_pct=int(discount[k]),
                                               promo_type=rng.choice(["price cut", "newsletter", "bundle"])))

    demand = pd.DataFrame(demand_rows, columns=["week_start", "product_id", "channel", "units",
                                                "price", "discount_pct", "promo_flag", "stockout_flag"])

    # two short stock-outs on fast movers: recorded sales drop to zero although demand existed
    for prod, ch, start in [("P027", "store", "2024-05-06"), ("P004", "online", "2025-02-10")]:
        m = (demand.product_id == prod) & (demand.channel == ch)
        idx = demand.index[m & demand.week_start.isin(
            [d.date().isoformat() for d in pd.date_range(start, periods=2, freq="W-MON")])]
        demand.loc[idx, ["units", "stockout_flag"]] = [0, 1]

    calendar = pd.DataFrame({
        "week_start": [d.date().isoformat() for d in WEEKS],
        "year": WEEKS.year, "month": WEEKS.month, "iso_week": woy,
        "is_black_friday_week": (woy == 47).astype(int),
        "is_christmas_window": ((woy >= 48) & (woy <= 51)).astype(int),
        "is_january": (WEEKS.month == 1).astype(int),
        "is_summer": ((woy >= 18) & (woy <= 30)).astype(int),
    })

    pd.DataFrame(products).to_csv(OUT / "products.csv", index=False)
    demand.to_csv(OUT / "demand_weekly.csv", index=False)
    pd.DataFrame(promo_rows).to_csv(OUT / "promotions.csv", index=False)
    calendar.to_csv(OUT / "calendar.csv", index=False)
    pd.DataFrame(truth).to_csv(OUT / "generator_truth.csv", index=False)

    print(f"products:   {len(products)}")
    print(f"demand:     {len(demand):,} rows  ({demand.units.sum():,} units, "
          f"{(demand.units == 0).mean():.1%} zero weeks)")
    print(f"promotions: {len(promo_rows)} promo weeks")
    print(demand.groupby('channel').units.sum())


if __name__ == "__main__":
    main()

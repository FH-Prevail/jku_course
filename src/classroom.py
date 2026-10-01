"""The code behind the two T2 notebooks: load the course data, draw the pictures, compare the forecasts
and replay the stock rule.

scripts/build_notebooks.py copies these functions into the notebook cells, so the code students read in Colab
is exactly the code behind the slides. Nothing here trains a model: every forecast was computed in advance by
analysis/build_classroom.py, using only what was known at each forecast date.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display

# >>> settings (copied into the first cell of each notebook)
METHODS = ["Repeat last year", "Simple exponential smoothing", "Prophet", "LightGBM in global mode"]  # the four approaches
PRODUCTS = {"Plush Bear, store": "P010_store", "Plush Bear, online": "P010_online",   # name shown -> series in the data
            "Puzzle 1000, store": "P011_store", "Vacuum Filter, store": "P035_store",
            "Garden Hose, store": "P015_store"}
WINDOWS = {"Spring": "2025-03-24", "Summer": "2025-06-16", "Christmas": "2025-12-01"}   # first week of each 4-week test window
COLORS = {"Actual sales": "#262626", "Repeat last year": "#E66A1F", "Simple exponential smoothing": "#1B9E5A",
          "Prophet": "#D04F95", "LightGBM in global mode": "#0B5CAD"}                 # the same colour per method as on the slides
MARKERS = {"Repeat last year": "s", "Simple exponential smoothing": "^", "Prophet": "X", "LightGBM in global mode": "D"}
BLUE = "#0B5CAD"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 110})
# <<< settings

COMPARE = METHODS          # older name, still used by the slide builder

demand = products = comparisons = inventory = results = None
D = P = B = I = R = None   # short names for the slide builder and the checks


def load(folder):
    """Read the five course files from a folder or a web address."""
    global demand, products, comparisons, inventory, results, D, P, B, I, R
    folder = str(folder)
    demand = pd.read_csv(f"{folder}/demand.csv", parse_dates=["week_start"])
    products = pd.read_csv(f"{folder}/products.csv").set_index("product_id")
    comparisons = pd.read_csv(f"{folder}/comparisons.csv", parse_dates=["week_start", "test_start"])
    inventory = pd.read_csv(f"{folder}/inventory.csv", parse_dates=["week_start"])
    if folder.startswith(("http", "file:")):
        results = json.load(urllib.request.urlopen(f"{folder}/results.json"))
    else:
        results = json.loads(Path(folder, "results.json").read_text())
    D, P, B, I, R = demand, products, comparisons, inventory, results


def series_id(product):
    """Turn a product name such as 'Plush Bear, store' into its series in the data, 'P010_store'."""
    if product in PRODUCTS:
        return PRODUCTS[product]
    if product in PRODUCTS.values():
        return product
    raise ValueError("Choose one of the product names in the dropdown.")


# ------------------------------------------------------------------ notebook 1, section 1: read the demand

def totals():
    """Units sold per month in each channel, then the yearly totals."""
    weekly = demand.groupby(["week_start", "channel"])["units"].sum().unstack()
    monthly = weekly.resample("MS").sum()                       # add up the weeks of each month
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(monthly.index, monthly["store"], color=BLUE, lw=2, label="store")
    ax.plot(monthly.index, monthly["online"], color=COLORS["Repeat last year"], lw=2, label="online")
    ax.set(ylabel="Units sold per month", title="All products: store and online")
    ax.legend()
    fig.tight_layout()
    plt.show()
    yearly = demand.assign(year=demand["week_start"].dt.year).groupby(["year", "channel"])["units"].sum().unstack()
    return yearly.rename_axis("Year").rename(columns={"store": "Store units", "online": "Online units"})


def pattern_examples():
    """One product of each demand pattern: its weekly sales in 2025."""
    examples = [("P010_store", "Smooth: sales most weeks"),
                ("P011_store", "Erratic: frequent, variable amounts"),
                ("P035_store", "Intermittent: many weeks without sales"),
                ("P038_store", "Lumpy: gaps and variable amounts")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), layout="constrained")
    for ax, (sid, title) in zip(axes.flat, examples):
        sales = demand[(demand["series_id"] == sid) & (demand["week_start"].dt.year == 2025)]
        ax.bar(sales["week_start"], sales["units"], width=6, color=BLUE)
        ax.set(title=title, ylabel="Units")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
    plt.show()


def pattern_map():
    """Every series placed by its two numbers, ADI and CV²: the four demand patterns as one picture."""
    series = pd.DataFrame(results["patterns"]["series"])
    cuts, counts = results["patterns"]["cuts"], results["patterns"]["counts"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(series["cv2"].clip(upper=2), series["adi"].clip(upper=10), color=BLUE, s=30, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axvline(cuts["cv2"], color="#888888", ls="--", lw=1)     # the usual limits from the literature
    ax.axhline(cuts["adi"], color="#888888", ls="--", lw=1)
    ax.set(xlabel="CV²: how irregular the sale sizes are", ylabel="ADI: average weeks between sales", xlim=(0, 2.06), ylim=(0, 10.6))
    for name, x, y in [("Smooth", 0.03, 0.25), ("Erratic", 1.62, 0.25), ("Intermittent", 0.03, 9.8), ("Lumpy", 1.62, 9.8)]:
        ax.text(x, y, f"{name} ({counts[name.lower()]})", fontsize=11, color="#253746", weight="bold")
    fig.tight_layout()
    plt.show()
    print("Each dot is one product in one channel, classified on 2023 and 2024. Values beyond the edge are drawn on the edge.")
    print("ADI: the average number of weeks between sales (1 = a sale every week). CV²: how irregular the amounts are from one sale to the next.")


def pattern_of(product="Plush Bear, store"):
    """The two numbers of one product, ADI and CV², and the demand pattern they give (measured on 2023 and 2024)."""
    sid = series_id(product)
    row = next(s for s in results["patterns"]["series"] if s["series_id"] == sid)
    cuts = results["patterns"]["cuts"]
    print(f"{product}: ADI {row['adi']:.2f} (limit {cuts['adi']}), CV² {row['cv2']:.2f} (limit {cuts['cv2']})")
    print(f"Demand pattern: {row['pattern']}")


# ------------------------------------------------------------------ notebook 1, section 2: compare four approaches

def comparison_data(product="Plush Bear, store", window="Christmas"):
    """The four forecasts and the actual sales for one product and one 4-week test window."""
    if window not in WINDOWS:
        raise ValueError("Choose Spring, Summer or Christmas.")
    rows = comparisons[(comparisons["series_id"] == series_id(product)) &
                       (comparisons["test_start"] == pd.Timestamp(WINDOWS[window]))]
    return rows.copy()


def score_table(rows):
    """For each method: average miss in units, total miss as a share of demand, and average bias (forecast minus actual)."""
    table = []
    for method in METHODS:
        scored = rows[(rows["method"] == method) & (rows["scoreable"] == 1)]   # weeks with an empty shelf are left out
        actual = scored["actual"].to_numpy()
        forecast = scored["forecast"].to_numpy()
        total = actual.sum()
        table.append({"Method": method,
                      "Average miss, units": np.abs(forecast - actual).mean(),
                      "Total miss, % of demand": np.abs(forecast - actual).sum() / total * 100 if total else np.nan,
                      "Average bias, units": (forecast - actual).mean()})
    return pd.DataFrame(table).set_index("Method")


def compare(product="Plush Bear, store", window="Christmas"):
    """Chart the four forecasts against actual sales, then score them."""
    rows = comparison_data(product, window)
    actual = rows[rows["method"] == METHODS[0]].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(np.arange(4) - 0.12, actual["actual"], width=0.24, color="#A8B4BE", label="Actual sales")
    for method in METHODS:
        forecast = rows[rows["method"] == method].sort_values("week_start")
        ax.plot(range(4), forecast["forecast"], color=COLORS[method], marker=MARKERS[method], lw=2, label=method)
    ax.set(xticks=range(4), xticklabels=actual["week_start"].dt.strftime("%d %b"), ylabel="Units", title=f"{product} | {window}")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    fig.tight_layout()
    plt.show()
    made = (actual["week_start"].min() - pd.Timedelta(weeks=1)).date()
    print(f"Forecast made after {made}; all four weeks forecast together.")
    scores = score_table(rows)
    display(scores.style.format({"Average miss, units": "{:.1f}", "Total miss, % of demand": "{:.1f} %",
                                 "Average bias, units": "{:+.1f}"}, na_rep="Undefined: no demand"))
    winner = scores["Average miss, units"].idxmin()
    print(f"In this window, {winner} has the smallest average miss. This is evidence for this product and window only.")
    print("Bias is forecast minus actual: positive means too high; negative means too low.")


def forecast_widget():
    """Two dropdowns, product and window, that redraw the comparison."""
    from ipywidgets import interactive, Dropdown
    controls = interactive(compare,
                           product=Dropdown(options=list(PRODUCTS), value="Plush Bear, store", description="Product"),
                           window=Dropdown(options=list(WINDOWS), value="Christmas", description="Window"))
    display(controls)
    return controls


# ------------------------------------------------------------------ notebook 1, sections 4 and 5

def backtest():
    """The backtest over 2025: total miss and bias of every method, all 80 series, 13 test dates."""
    table = []
    for method in METHODS:
        scored = comparisons[(comparisons["method"] == method) & (comparisons["scoreable"] == 1)]
        error = scored["forecast"] - scored["actual"]
        table.append([method, f"{error.abs().sum() / scored['actual'].sum() * 100:.1f} %",
                      f"{error.sum() / scored['actual'].sum() * 100:+.1f} %"])
    for method in ["Holt-Winters", "Croston SBA", "TSB", "Moving average"]:     # four more local methods, scored the same way
        score = results["backtest_all"][method]["all"]
        table.append([method, f"{score['wape']:.1f} %", f"{score['bias_pct']:+.1f} %"])
    print("13 test dates; four weeks ahead each; all 80 series. The same weeks and products for every method.")
    print("The four methods of the window comparison come first; four more local methods from the slides were scored the same way.")
    print("Flagged stockout observations are excluded from scoring because their demand is unknown.")
    return pd.DataFrame(table, columns=["Method", "WAPE: total miss / total demand", "Bias: net error / total demand"]).set_index("Method")


def zeros_demo():
    """A forecast of zero for a slow mover: right in three weeks, and still every unit missed."""
    actual = np.array([0, 0, 0, 4])
    forecast = np.zeros(4)
    print("Teaching example, not an extracted course product.")
    display(pd.DataFrame({"Week": [1, 2, 3, 4], "Actual units": actual, "Forecast units": forecast}))
    print("Three weeks exactly right, but all four units of demand missed.")
    print("MAPE is undefined because actual demand is zero in three weeks. WAPE is 100 %.")
    print("Do not choose a stock policy by counting the weeks a forecast gets right.")


# ------------------------------------------------------------------ notebook 2: from forecast to stock

def order_example(target=95):
    """The worked order from the slides, with round teaching numbers."""
    forecasts = np.array([180, 190, 210, 200, 220])     # the five weeks the stock must cover
    weekly_sigma, lead_time = 40.0, 4                    # error spread per week; supplier lead time in weeks
    safety_stock = int(np.ceil(norm.ppf(target / 100) * weekly_sigma * np.sqrt(lead_time + 1)))
    stock_target = int(forecasts.sum() + safety_stock)
    on_hand, on_order = 300, 500
    order = max(0, stock_target - on_hand - on_order)
    return {"weekly_forecasts": forecasts.tolist(), "forecast_total": int(forecasts.sum()), "weekly_sigma": weekly_sigma,
            "protection_weeks": lead_time + 1, "safety_stock": safety_stock, "stock_target": stock_target,
            "on_hand": on_hand, "on_order": on_order, "order": order}


def worked_order():
    """Show the worked order as a table."""
    e = order_example()
    print("Teaching scenario: five weekly forecasts, one order decision. These are round illustrative numbers.")
    display(pd.DataFrame({"Quantity": ["Forecast over five weeks", "Safety stock", "Stock target", "Already on the shelf", "Already on order", "Order now"],
                          "Units": [e["forecast_total"], e["safety_stock"], e["stock_target"], e["on_hand"], e["on_order"], e["order"]]}).set_index("Quantity"))
    print("We assume no backorders and arrivals as scheduled. The stock target is not the quantity to order.")


def profile(product="Plush Bear, store", method="LightGBM in global mode"):
    """One product's 2025 weeks for the stock replay: actual sales, the forecasts for the weeks the stock must cover,
    and the error spread measured on the first 25 weeks."""
    sid = series_id(product)
    if method not in METHODS:
        raise ValueError("Choose a method shown in the table.")
    weeks = inventory[(inventory["series_id"] == sid) & (inventory["method"] == method)].sort_values("week_start")
    if weeks.empty:
        raise ValueError("This product has no stock exercise.")
    actual = weeks["actual"].to_numpy(float)
    forecasts = weeks[[f"h{h}" for h in range(1, 8)]].to_numpy(float)     # forecasts 1 to 7 weeks ahead, made each week
    lead_time = int(products.loc[sid.split("_")[0], "lead_time_weeks"])
    forecasts = forecasts[:, :lead_time + 1]                                # keep the weeks until the next delivery
    first_half = 25                                                         # January to June: measure the error spread
    sigma = float(np.std(actual[:first_half] - forecasts[:first_half, 0], ddof=1))
    return weeks, actual, forecasts, sigma, lead_time, first_half


def simulate(product="Plush Bear, store", target=95, method="LightGBM in global mode", details=False):
    """Replay the ordering rule week by week over the second half of 2025 and measure what it achieved."""
    if not np.isfinite(target) or not 50 <= target <= 99:
        raise ValueError("Choose a target from 50 to 99 percent.")
    weeks, actual, forecasts, sigma, lead_time, start = profile(product, method)
    safety_stock = float(norm.ppf(target / 100) * sigma * np.sqrt(lead_time + 1))
    on_hand = float(np.ceil(forecasts[start].sum()))    # begin with one protection period of stock on the shelf
    pipeline = []                                        # orders on their way: (week they arrive, units)
    trace = []
    for t in range(start, len(actual)):
        on_hand += sum(units for arrives, units in pipeline if arrives == t)        # this week's deliveries arrive
        pipeline = [(arrives, units) for arrives, units in pipeline if arrives != t]
        stock_target = float(np.ceil(forecasts[t].sum() + safety_stock))           # forecast over the protection period + safety stock
        order = max(0.0, stock_target - on_hand - sum(units for _, units in pipeline))
        pipeline.append((t + lead_time, order))
        served = min(on_hand, actual[t])                 # we can only sell what is on the shelf
        lost = actual[t] - served                        # no backorders: unserved demand is lost
        on_hand -= served
        trace.append({"week": str(weeks["week_start"].iloc[t].date()), "actual": float(actual[t]),
                      "forecast_sum": float(forecasts[t].sum()), "level": stock_target, "order": order, "lost": lost, "stock": on_hand})
    replay = pd.DataFrame(trace)
    total = replay["actual"].sum()
    result = {"target": float(target), "safety_stock": float(np.ceil(safety_stock)),
              "average_stock": float(replay["stock"].mean()),
              "fill_rate": float(100 * (1 - replay["lost"].sum() / total)) if total else 100.0,
              "stockout_weeks": int((replay["lost"] > 0).sum()), "lost_units": float(replay["lost"].sum()),
              "weeks_stock": float(replay["stock"].mean() / replay["actual"].mean()) if total else 0.0,
              "weekly_sigma": sigma, "protection_weeks": lead_time + 1, "test_weeks": len(replay)}
    return (result, trace) if details else result


def stock_decision(product="Plush Bear, store", target=95):
    """What one service target achieved in the replay."""
    r = simulate(product, target)
    lead_time = profile(product)[4]
    print(f"{product}: cycle service target {target} %; supplier lead time {lead_time} weeks.")
    display(pd.DataFrame({"Observed result": [f"{r['average_stock']:.0f} units", f"{r['fill_rate']:.1f} %",
                                              f"{r['stockout_weeks']} of {r['test_weeks']} weeks", f"{r['lost_units']:.0f} units"]},
                         index=["Average stock on the shelf", "Fill rate: share of demand served", "Weeks with a stockout", "Demand not served"]))
    print(f"Safety stock allowance: about {r['safety_stock']:.0f} units. This is part of the stock target, not the order quantity.")
    print("A cycle service target is a planning probability; fill rate measures units served. They are different measures.")


def stock_widget():
    """A product dropdown and a target slider that rerun the replay."""
    from ipywidgets import interactive, Dropdown, IntSlider
    controls = interactive(stock_decision,
                           product=Dropdown(options=["Plush Bear, store", "Puzzle 1000, store", "Vacuum Filter, store"], description="Product"),
                           target=IntSlider(value=95, min=50, max=99, step=1, description="Target %", continuous_update=False))
    display(controls)
    return controls


def stock_table(product="Plush Bear, store", targets=(50, 80, 90, 95, 99)):
    """The replay at five service targets, side by side."""
    table = []
    for t in targets:
        r = simulate(product, t)
        table.append([f"{t} %", f"{r['average_stock']:.0f}", f"{r['fill_rate']:.1f} %", r["stockout_weeks"], f"{r['lost_units']:.0f}"])
    return pd.DataFrame(table, columns=["Cycle service target", "Average stock, units", "Fill rate", "Weeks with a stockout",
                                        "Units not served"]).set_index("Cycle service target")


def stock_compare():
    """Three products, the same 95 percent target."""
    table = []
    for product in ["Plush Bear, store", "Puzzle 1000, store", "Vacuum Filter, store"]:
        r = simulate(product, 95)
        table.append([product, f"{r['fill_rate']:.1f} %", r["stockout_weeks"], f"{r['weeks_stock']:.1f}"])
    return pd.DataFrame(table, columns=["Product", "Fill rate at 95 % target", "Weeks with a stockout",
                                        "Weeks of demand on shelf"]).set_index("Product")


def forecast_value(product="Plush Bear, store"):
    """The same 95 percent target with each forecast: stock, service and the yearly cost of holding the stock."""
    item = products.loc[series_id(product).split("_")[0]]
    table = []
    for method in METHODS:
        r = simulate(product, 95, method)
        cost = r["average_stock"] * item["unit_cost"] * item["holding_cost_rate"]
        table.append([method, f"{r['average_stock']:.0f}", f"{r['fill_rate']:.1f} %", f"EUR {cost:.0f}"])
    print(f"{product}. Same 95 % target, each method uses its own measured errors.")
    print(f"Unit cost EUR {item['unit_cost']:.2f}; annual holding rate {item['holding_cost_rate']:.0%}.")
    print("Annualised stock holding cost only: a scenario based on the replay average, excluding shortages and implementation.")
    return pd.DataFrame(table, columns=["Method", "Average stock, units", "Fill rate", "Annualised holding cost"]).set_index("Method")


def christmas_order(salvage=50):
    """One order for one selling week (the newsvendor): the order is a quantile of demand set by the critical ratio."""
    if not np.isfinite(salvage) or not 0 <= salvage <= 95:
        raise ValueError("Choose recovery from 0 to 95 percent.")
    weeks, actual, forecasts, sigma, lead_time, first_half = profile("Puzzle 1000, store")
    i = np.flatnonzero(weeks["week_start"] == pd.Timestamp("2025-12-15"))[0]
    item = products.loc["P011"]
    cost_short = float(item["unit_price"] - item["unit_cost"])             # lost margin on a unit we did not have
    cost_leftover = float(item["unit_cost"] * (1 - salvage / 100))          # purchase cost minus what we recover
    ratio = cost_short / (cost_short + cost_leftover)                       # the critical ratio
    order = max(0, int(np.ceil(forecasts[i, 0] + norm.ppf(ratio) * sigma)))
    return {"forecast": float(forecasts[i, 0]), "sigma": sigma, "under": cost_short, "over": cost_leftover,
            "ratio": ratio, "order": order, "actual": float(actual[i])}


def christmas_display(salvage=50):
    """Print the Christmas order in words."""
    r = christmas_order(salvage)
    print(f"Puzzle 1000, store, 15 December 2025: forecast {r['forecast']:.0f} units; actual {r['actual']:.0f}.")
    print(f"Recover {salvage} % of purchase cost on leftovers: a unit short costs EUR {r['under']:.2f}, a leftover costs EUR {r['over']:.2f}.")
    print(f"Critical ratio {r['ratio']:.2f}: choose that quantile of demand. Suggested order {r['order']} units.")
    print("A quantile is a demand level with a chosen probability of not being exceeded. Here we approximate demand with a bell curve.")
    print("This separate exercise assumes one order can arrive for the selling week; it does not simulate the supplier lead time.")


def christmas_widget():
    """A slider for the share of the purchase cost recovered on leftovers."""
    from ipywidgets import interactive, IntSlider
    controls = interactive(christmas_display,
                           salvage=IntSlider(value=50, min=0, max=95, step=5, description="Recovery %", continuous_update=False))
    display(controls)
    return controls

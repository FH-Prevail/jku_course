# Course data: omnichannel retail demand, weekly

Synthetic but realistic weekly demand for an omnichannel retailer, built for the
T2 session *Demand Forecasting & Inventory Analytics* (JKU Linz, 03.10.2026).
It stands on its own; it is not derived from the other sessions' material.

Regenerate at any time with `python make_dataset.py` (fixed random seed, so the
files are reproducible).

## Files

| File | Rows | One row is |
|---|---|---|
| `demand_weekly.csv` | 12,480 | one product in one channel in one week (80 series x 156 weeks) |
| `products.csv` | 40 | one product: category, price, cost, lead time |
| `promotions.csv` | 1,037 | one promotion week for one product in one channel |
| `calendar.csv` | 156 | one week: month, ISO week, Black Friday, Christmas window, summer flags |
| `generator_truth.csv` | 40 | instructor only: the demand pattern each product was generated with |
| `sample/sample_weekly_sales.csv` | 468 | one week of store sales for the Plush Bear, the Puzzle 1000 or the Vacuum Filter: the small file notebook 0 loads by link, upload or Google Drive |

Period: **2023-01-02 to 2025-12-22** (156 weeks), weeks starting on Monday.
Channels: `store` and `online`. Categories: Electronics, Toys, Garden, Grocery,
Fashion, Spare Parts.

## `demand_weekly.csv`

| Column | Type | Meaning |
|---|---|---|
| `week_start` | date | Monday of the week |
| `product_id` | text | `P001` to `P040`, joins to `products.csv` |
| `channel` | text | `store` or `online` |
| `units` | int | units sold in that week (the forecast target) |
| `price` | float | effective selling price in that week (list price minus discount) |
| `discount_pct` | int | promotion discount in percent, 0 if none |
| `promo_flag` | 0/1 | 1 if a promotion ran that week (known in advance) |
| `stockout_flag` | 0/1 | 1 in the four weeks where sales were zero because the shelf was empty, not because demand was zero |

## `products.csv`

| Column | Meaning |
|---|---|
| `product_id`, `product_name`, `category` | product master |
| `unit_price` | list price in EUR |
| `unit_cost` | purchase cost in EUR |
| `lead_time_weeks` | supplier lead time in weeks (1 to 6) |
| `holding_cost_rate` | annual holding cost as a share of unit cost (0.20) |

## What is in the data on purpose

- **Four demand patterns.** Fast movers (smooth), variable fast movers (erratic),
  slow movers with regular sizes (intermittent) and slow movers with wild sizes
  (lumpy). Classify them with ADI and CV² (Syntetos-Boylan); do not trust the
  category name.
- **Trend.** Online grows 6 to 16 percent a year; store is flat.
- **Seasonality.** Toys and Electronics peak before Christmas, Garden in early
  summer, Fashion in spring and autumn, Grocery is nearly flat. Online has a
  Black Friday spike (ISO week 47).
- **Promotions.** Discounts of 10 to 30 percent lift demand from about 1.4x at
  10 percent to 2.6x at 30 percent, more online than in store, followed by a dip
  the week after.
- **Two stock-outs** of two weeks each, on variable fast movers (erratic): four
  flagged weeks in all. Sales are not demand.
- **Zeros.** About a third of all rows are zero weeks; almost all of them belong
  to the slow movers.

## Licence

Free to use for teaching. No real company data is contained.

## Classroom evidence

`classroom/` contains the frozen demand, products, forecast comparisons, inventory
forecast profiles and `results.json`, shared by the deck and the two student
notebooks. Build it with `python3 analysis/build_classroom.py` from the repository
root. The first cell of each notebook reads these files straight from the
repository's main branch with pandas. Every forecast was computed in advance, so
nothing is trained live in class.

The classroom pipeline estimates the four flagged stockout observations from
preceding history for training and excludes censored targets from forecast
scoring. `analysis/build_classroom.py` holds the exact splits and settings.

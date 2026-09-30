# T2 · From Demand Forecasts To Replenishment Decisions

**Demand Forecasting & Inventory Analytics · UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi**

Course material for the T2 session: the slides, three Colab notebooks and one synthetic retail dataset. One question carries the afternoon: **what will we order, and what evidence makes that a reasonable choice?**

## Slides

[The slides as PDF](slides/T2_Demand_Forecasting_Inventory_Analytics.pdf)

## Notebooks

| Notebook | What it is for | Open in Colab |
|---|---|---|
| [0 · Your Own Data In Colab](notebooks/0_your_data_in_colab.ipynb) | Three ways to load a file (upload, direct link, Google Drive), tried with a small sample, then a first look at the table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/0_your_data_in_colab.ipynb) |
| [1 · Demand Patterns and Forecast Comparison](notebooks/1_see_and_forecast.ipynb) | Hands-on 1 (sections 1 to 3); sections 4 and 5 support the evaluation chapter | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/1_see_and_forecast.ipynb) |
| [2 · Safety Stock and Replenishment Policy](notebooks/2_forecast_to_stock.ipynb) | Hands-on 2 (sections 1 to 4); sections 5 and 6 are optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/2_forecast_to_stock.ipynb) |

If a button does not open, copy the address instead:

- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/0_your_data_in_colab.ipynb
- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/1_see_and_forecast.ipynb
- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/2_forecast_to_stock.ipynb

Or open Colab and choose **File, Open notebook, GitHub**, type `FH-Prevail/jku_course` and pick the notebook.

**Keep your work:** in Colab choose **Copy to Drive** (File, Save a copy in Drive) before you change anything. The copy goes to your own Google Drive; without it, your changes are lost when you close the tab. The notebooks here never change.

Notebooks 1 and 2 start with a setup cell that fetches the course data (`data/classroom/`) and the display helpers (`src/classroom.py`) from this repository and prints "Ready". Nothing is trained in Colab: every table and picture comes from forecasts computed in advance. Gemini, where your account offers it, can explain any output; check what it says against the numbers.

## Data

One synthetic dataset, built like a real retailer's: 40 products sold in store and online, 80 weekly series from January 2023 to December 2025, with seasons, planned promotions and four empty-shelf weeks. No real company data is contained. [data/README.md](data/README.md) describes every file; `data/sample/sample_weekly_sales.csv` is the small file notebook 0 loads.

## For your notes

[The decision sheet](docs/T2_Decision_Sheet.pdf) ([Word](docs/T2_Decision_Sheet.docx)): one forecast choice and one service target, each with a number and a limitation.

## How the numbers were made

`analysis/build_classroom.py` trains, backtests and freezes every number the notebooks show into `data/classroom/results.json`; `src/retail_fc.py` holds the forecasting and inventory mathematics and `src/classroom.py` the notebook displays. To rebuild on your own computer:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/build_classroom.py      # the course data behind the notebooks, about four minutes
python3 scripts/build_notebooks.py       # the three notebooks
```

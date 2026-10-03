# T2 · From Demand Forecasts To Replenishment Decisions

**Demand Forecasting & Inventory Analytics · UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi**

Course material for the T2 session: the slides, three Colab notebooks and one synthetic retail dataset. One question carries the afternoon: **what will we order, and what evidence makes that a reasonable choice?**

## Slides

[The slides as PDF](slides/T2_Demand_Forecasting_Inventory_Analytics.pdf)

## Notebooks

| Notebook | What it is for | Open in Colab |
|---|---|---|
| [0 · Your Own Data In Colab](notebooks/0_your_data_in_colab.ipynb) | Three ways to load a file (upload, direct link, Google Drive), tried with a small sample, then a first look at the table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/0_your_data_in_colab.ipynb) |
| [1 · Demand Patterns and Forecast Comparison](notebooks/1_see_and_forecast.ipynb) | Hands-on 1 (sections 1 to 3) and graded Tasks 1.1 to 1.3; sections 4 and 5 support the evaluation chapter | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/1_see_and_forecast.ipynb) |
| [2 · Safety Stock and Replenishment Policy](notebooks/2_forecast_to_stock.ipynb) | Hands-on 2 (sections 1 to 4) and graded Tasks 2.1 to 2.3; sections 5 and 6 are optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/2_forecast_to_stock.ipynb) |

If a button does not open, copy the address instead:

- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/0_your_data_in_colab.ipynb
- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/1_see_and_forecast.ipynb
- https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/2_forecast_to_stock.ipynb

Or open Colab and choose **File, Open notebook, GitHub**, type `FH-Prevail/jku_course` and pick the notebook.

**Keep your work:** in Colab choose **Copy to Drive** (File, Save a copy in Drive) before you change anything. The copy goes to your own Google Drive; without it, your changes are lost when you close the tab. The notebooks here never change.

Every cell shows its code above its result: the same code that produced the numbers on the slides. The first cell of notebooks 1 and 2 loads the course data (`data/classroom/`) from this repository and prints "Ready". Nothing is trained in Colab for the tasks: every table and picture comes from forecasts computed in advance. An optional extra at the end of notebook 1 trains Prophet and LightGBM in global mode in a few seconds, if you want to see how a forecast is made. Gemini, where your account offers it, can explain or change any cell; check what it says against the numbers.

## Data

One synthetic dataset, built like a real retailer's: 40 products sold in store and online, 80 weekly series from January 2023 to December 2025, with seasons, planned promotions and four empty-shelf weeks. No real company data is contained. [data/README.md](data/README.md) describes every file; `data/sample/sample_weekly_sales.csv` is the small file notebook 0 loads.

## Assignment (graded)

The T2 assignment is the tasks inside notebooks 1 and 2, done individually. Each notebook has three tasks and 10 points; each notebook is half of the assignment. Together they count **25 % of the course grade**, plus 5 % for taking part in the session.

1. Open the notebook in Colab and choose **Copy to Drive**, so your answers are saved.
2. Write your name and student number in the first cell, and your answers in the cells marked **Your answer**.
3. **Runtime, Run all**, then **File, Download, Download .ipynb**. Name the files `T2_Notebook1_Lastname_Firstname.ipynb` and `T2_Notebook2_Lastname_Firstname.ipynb`.
4. Upload both files to the T2 assignment on MS Teams by **Saturday 10 October 2026, 23:59**.

You may discuss with others, but the answers you submit are your own. Gemini is allowed: say in your answer where you used it, and check it against the numbers.

The course grade across the three sessions: T1, a group assignment (35 %) and participation (5 %); T2 and T3, an individual assignment (25 %) and participation (5 %) each. All assignments are submitted on MS Teams.

## How the numbers were made

`analysis/build_classroom.py` trains, backtests and freezes every number the notebooks show into `data/classroom/results.json`; `src/retail_fc.py` holds the forecasting and inventory mathematics and `src/classroom.py` the code shown in the notebook cells (`scripts/build_notebooks.py` copies it in). To rebuild on your own computer:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/build_classroom.py      # the course data behind the notebooks, about four minutes
python3 scripts/build_notebooks.py       # the three notebooks
```

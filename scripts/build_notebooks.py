"""Create the two student notebooks. Their setup cell fetches data/classroom and src/classroom.py from the
course repository on GitHub (or from a local folder given in the environment variable T2_SOURCE, for testing
before a push). Nothing is trained in the notebooks; every table comes from the frozen classroom evidence."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
REPO_RAW='https://raw.githubusercontent.com/FH-Prevail/jku_course/main'
FILES=['src/classroom.py','data/classroom/results.json','data/classroom/products.csv','data/classroom/demand.csv',
       'data/classroom/comparisons.csv','data/classroom/inventory.csv']
setup='''#@title Start here: run this cell once. It fetches the course data from GitHub. Nothing to edit.
import os, sys, pathlib, shutil, urllib.request, importlib, importlib.util, subprocess
required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy", "ipywidgets": "ipywidgets"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
SOURCE = os.environ.get("T2_SOURCE") or "'''+REPO_RAW+'''"
FILES = '''+repr(FILES)+'''
folder = pathlib.Path("t2_classroom")
folder.mkdir(exist_ok=True)
try:
    for f in FILES:
        target = folder / pathlib.Path(f).name
        if SOURCE.startswith("http"):
            urllib.request.urlretrieve(f"{SOURCE}/{f}", target)
        else:
            shutil.copy(pathlib.Path(SOURCE) / f, target)
except Exception as e:
    raise SystemExit(f"Could not fetch the course files ({e}). Check the internet connection, then Runtime > Run all again.")
sys.path.insert(0, str(folder.resolve()))
get_ipython().run_line_magic("matplotlib", "inline")
import classroom as lesson
importlib.reload(lesson)
lesson.load(folder)
print("Ready. The course data is loaded.")
print("The forecasts were computed in advance, using only what was known at each forecast date.")
'''
intro='''*UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi, Logistikum*

Choose **Copy to Drive** (File, Save a copy in Drive), then **Runtime → Run all**. Work in pairs. You do not write code: read the tables and use the controls.

The first cell fetches the course data, the precomputed forecasts and the display helpers from the course repository, github.com/FH-Prevail/jku_course. It does not train a model, and it takes a few seconds. The cell can stay collapsed.

**If a control is missing:** use the static example directly above it. If you edited a cell by accident, Undo, then run it again; otherwise reopen a fresh copy from the repository. Restarting does not undo edits.

**Optional Gemini help:** select an output and ask “Explain this table in plain English. Separate what it shows from what it does not prove.” Check the answer against the numbers. If Gemini is unavailable, ask your partner or lecturer. Use only the synthetic course data.
'''
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s,hidden=False):
 c=nbf.v4.new_code_cell(s)
 if hidden:c.metadata={'cellView':'form','jupyter':{'source_hidden':True}}
 return c

def colab_badge(name):
 url=f'https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/{name}'
 # The official badge: target="_parent" lets the link leave GitHub's sandboxed notebook frame, where a plain link can be blocked.
 return (f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
         f'\n\nIf the button does not open, use this link, or copy it into the browser: <a href="{url}" target="_parent">{url}</a>\n'
         '\nOr in Colab: **File, Open notebook, GitHub**, type `FH-Prevail/jku_course` and choose the notebook.')

def notebook(name,title,cells,lead=None):
 """A course notebook: badge, intro and the hidden setup cell, then the cells. With `lead`, a stand-alone
 notebook instead: badge and that text, no setup cell (the data-loading demo)."""
 head=[md('# '+title+'\n\n'+colab_badge(name)+'\n\n'+intro),code(setup,True)] if lead is None else [md('# '+title+'\n\n'+colab_badge(name)+'\n\n'+lead)]
 n=nbf.v4.new_notebook(cells=[*head,*cells])
 n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':name,'provenance':[]}}
 nbf.write(n,ROOT/'notebooks'/name)

SAMPLE=REPO_RAW+'/data/sample/sample_weekly_sales.csv'
notebook('0_your_data_in_colab.ipynb','0 · Your Own Data In Colab',[
 md('''## 1. From a direct link

A direct link returns the file itself, not a web page about it. On GitHub, open the file and copy the address of its **Raw** button. A Google Drive share link is not a direct link. The two course notebooks load their data this way.'''),
 code('''import pandas as pd
url = "'''+SAMPLE+'''"
df = pd.read_csv(url)
df.head()'''),
 md('''## 2. Upload a file

Click the **folder icon** on the left to open the Files panel, then the **upload** icon, and pick the file: the sample `sample_weekly_sales.csv` (in the repository under `data/sample/`) or a file of your own. It lands in the session's folder. **Uploaded files are deleted when the session ends.**'''),
 code('''import pandas as pd
df = pd.read_csv("sample_weekly_sales.csv")   # the file name exactly as the Files panel shows it
df.head()'''),
 md('''## 3. From Google Drive

Put the file in a folder of your Drive, for example **MyDrive/JKU_T2**. The next cell connects your Drive to this session; Colab asks for your permission first. Your Drive then appears in the Files panel under `drive/MyDrive`, and **files there stay after the session ends**.'''),
 code('''import pandas as pd
from google.colab import drive
drive.mount("/content/drive")
df = pd.read_csv("/content/drive/MyDrive/JKU_T2/sample_weekly_sales.csv")
df.head()'''),
 md('''## 4. Explore the table

Whichever route you took, the table is now called `df`: one row per product and week, with units sold, price, discount and a promotion flag.'''),
 code('''print(df.shape)   # rows and columns
df.info()         # column names, types, missing values'''),
 code('''df.describe()     # the numbers at a glance: count, mean, smallest, largest'''),
 code('''df["week_start"] = pd.to_datetime(df["week_start"])
df.groupby("product_name")["units"].agg(["mean", "sum", "max"]).round(1)   # per product: average week, total, best week'''),
 code('''weekly = df.pivot(index="week_start", columns="product_name", values="units")
weekly.plot(figsize=(10, 4), title="Units sold per week, store");'''),
 md('''**Ask Gemini**, for example: "Summarise this table", "Which product sells most in December?" or "Plot the Plush Bear's weekly units and mark the promotion weeks." Read the code it writes before you run it, and check its answer against the table.

**Upload only data you are allowed to share.** Files in Colab and in your Drive sit on Google's servers; company data needs the company's permission.''')],
 lead='''*UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi, Logistikum*

Three ways to get a file into Colab, tried with one small sample, **sample_weekly_sales.csv**: weekly store sales of three products from 2023 to 2025 (synthetic, the same data as the course). Pick one route in sections 1 to 3, then explore the table in section 4. Run a cell with the play button next to it, or Shift + Enter.''')

notebook('1_see_and_forecast.ipynb','1 · Demand Patterns and Forecast Comparison',[
 md('''## 1. Read the demand

The retailer sells 40 products through two channels, store and online: 80 series of weekly sales. The data is synthetic, built to look like a real retailer's numbers.

Read the channel picture. Is one channel growing? Do you see a recurring busy season?'''),
 code('lesson.totals()'),
 md('''**What to see.** Online totals grow from 2023 to 2025. Store totals fluctuate rather than growing each year. A monthly total also depends on how many weekly observations fall in that month.

Four demand patterns are useful descriptions, not guarantees of forecast accuracy: **smooth** means regular sales and fairly regular amounts; **erratic** means frequent sales with variable amounts; **intermittent** means many weeks without sales; **lumpy** adds variable amounts to those gaps.'''),
 code('lesson.pattern_examples()'),
 md('''Two numbers place every product on one picture. **ADI** (average demand interval) is the average number of weeks between sales: 1 means a sale every week. **CV²** (squared coefficient of variation) says how irregular the amounts are from one sale to the next. The dashed lines are the usual limits from the literature, 1.32 and 0.49; they are a convention, not a law.'''),
 code('lesson.pattern_map()'),
 md('''## 2. Compare three approaches

- **Repeat last year**, also called seasonal naive: reuse sales from 52 weeks earlier.
- **Smooth recent demand**, or simple exponential smoothing: give more weight to recent sales; this version forecasts a constant level.
- **Global model**: one model learns from all products and channels, using past sales and planned prices, promotions and calendar information.

We forecast the same four weeks with each method and then reveal actual sales. An **average miss** is the average absolute difference in units. **Total miss as a percentage of demand** adds up those absolute misses and divides by total actual demand. **Bias** is the average forecast minus actual: positive means too high, negative means too low.

First look at the Christmas example. Do not choose a winner from the shape alone; read its errors too.'''),
 code("lesson.compare('Plush Bear, store', 'Christmas')"),
 md('''**What to see.** The printed sentence names the method with the smallest miss in this window. The bias column adds direction; the same method need not win in another season or for another product.

**Try it using the dropdowns.** Change Window to Summer. Then select Garden Hose, store. Compare that product in Summer and Christmas. Change Plush Bear from store to online to investigate a channel difference.'''),
 code('forecast_controls = lesson.forecast_widget()',True),
 md('''## 3. Your first recommendation

Discuss these questions with your partner:

1. Which product and window did you choose?
2. Which method has the smallest miss, and is its bias positive or negative?
3. Does your choice change with the season or channel?

Complete this sentence in your notes: **“For ___ in ___, I would start with ___ because ___. Before ordering, I would also check ___.”**

Stop here for the first exercise. The following sections support the evaluation discussion after the break.'''),
 md('''## 4. Check more than one window

A **backtest** replays past forecast decisions using only information available then. Here every method is tested at 13 dates, predicting the next four weeks each time; the test weeks run from 30 December 2024 to 22 December 2025.

**WAPE**, weighted absolute percentage error, is total absolute miss divided by total actual demand. **Percentage bias** is total signed error divided by total demand; earlier we expressed bias in units instead. Both definitions use forecast minus actual for the sign.'''),
 code('lesson.backtest()'),
 md('''**What to see.** The table compares all 80 series on the same dates; the four extra rows are the further local methods from the slides (Holt-Winters, Croston SBA, TSB, a moving average), scored the same way. Large sellers have more influence on these aggregate percentages, so the best assortment result does not promise the best result for your product. These results were computed in advance; the global model was retrained at each test date.'''),
 md('''## 5. A forecast can be right most weeks and still miss all sales

**MAPE**, mean absolute percentage error, computes a percentage miss for each week before averaging. A week with zero actual demand makes that division undefined.'''),
 code('lesson.zeros_demo()'),
 md('''**What to see.** The zero forecast matches three weeks exactly but misses the only sale. Counting correct weeks is not enough to judge an order policy; carry this question into notebook 2: how much stock would you need?

**Evidence note.** Four observations in the source data are flagged as censored sales: the shelf was empty, so demand is unknown. For training, these are replaced with the preceding four weeks’ mean using only past information; flagged test rows are excluded from forecast scoring. The README of the `data` folder in the course repository describes the data in full.''')])

notebook('2_forecast_to_stock.ipynb','2 · Safety Stock and Replenishment Policy',[
 md('''## 1. A forecast is not an order

**Lead time** is the wait between placing and receiving an order. We review stock every week; the **protection period** covers lead time plus that review week.

**Safety stock** is a cushion for uncertain demand. **Stock target**, also called the order up to level, is forecast demand over the protection period plus safety stock. **Order now** is the shortfall between that target and stock already on the shelf or on order, assuming no backorders.

This worked example uses round teaching numbers: five forecasts of 180, 190, 210, 200 and 220 units. The forecasts sum to 1,000; the 95 percent planning target adds 148 units of safety stock.'''),
 code('lesson.worked_order()'),
 md('''**What to see.** The stock target is 1,148 units, but the order is 348 because 800 units are already on the shelf or on order. Ordering the whole target would count that stock twice.

## 2. Choose a service target

A **cycle service level** is the probability of completing a replenishment cycle without a stockout. A target of 95 percent is a planning assumption, not a promise about the percentage of units served.

**Fill rate** is the share of demand actually served. The replay below reports fill rate, weeks with a stockout and average stock. We do not compare fill rate directly with the cycle service target.

The global model is trained before 2025. Error variability is measured on 25 weeks from January to June; the stock rule is then replayed on 26 weeks from 30 June to 22 December. Orders are rounded up to whole units.'''),
 code("lesson.stock_decision('Plush Bear, store', 95)"),
 md('''**Try it.** Move Target % from 50 to 80, then to 95 and 99. Does every increase serve more demand? Then choose Puzzle 1000 and Vacuum Filter; the same target need not deliver the same observed outcome.'''),
 code('stock_controls = lesson.stock_widget()',True),
 md('''**If the slider is unavailable**, compare these preset settings instead.'''),
 code("lesson.stock_table('Plush Bear, store')"),
 md('''**What to see.** Compare each increase in stock with the change in fill rate. If fill rate is unchanged, the extra stock served no additional demand in these test weeks; this does not prove that the extra cushion is useless in every future period.

## 3. Compare the products

All three products below use a 95 percent cycle service target. “Weeks of demand on shelf” divides average stock by average demand during the same replay period.'''),
 code('lesson.stock_compare()'),
 md('''**What to see.** A high target can coexist with shortages when the error assumptions are poor. A slow mover can hold many weeks of demand in only a few units; inspect unit cost before calling that stock excessive.

The safety stock calculation assumes independent weekly errors, a stable error distribution and an approximate bell curve. Those assumptions are especially weak for sparse demand. Treat the replay as a check on the rule, not a service guarantee.'''),
 md('''## 4. Your management recommendation

Choose one product and complete this in your own notes:

**“For ___, we would trial a cycle service target of ___ percent. The replay held about ___ units and served ___ percent of demand. We accept ___; before using this in a business, we would check ___.”**

Compare your answer with another pair. Different choices can be reasonable when shortage and holding costs differ. You have completed the core exercise; the next sections are optional.'''),
 md('''## 5. Optional: what is a better forecast worth?

Keep the same 95 percent target and change the forecasting approach. Each method uses its own error estimate. Read stock and service together; lower stock with substantially lower service is a different policy, not automatically a better one.'''),
 code('lesson.forecast_value()'),
 md('''**What to see.** The last column prices the average stock using this product’s unit cost and a 20 percent annual holding rate. It is an annualised scenario based on a half-year replay, not realised annual savings or a complete investment case; shortages and implementation costs are excluded.'''),
 md('''## 6. Optional: one Christmas order

The **newsvendor** problem is one order for one selling period. A unit short loses margin; a leftover unit costs its purchase price minus what we recover afterward.

The **critical ratio** is shortage cost divided by shortage cost plus leftover cost. It chooses a **quantile**: a demand level with a specified probability of not being exceeded. Here demand is approximated with a bell curve, so the 50th percentile is its mean.

This is a separate decision assuming the order can arrive for the selling week; it does not use the replenishment lead time. Use recovery below 100 percent: with no leftover cost, this unbounded model has no finite optimal order.'''),
 code('lesson.christmas_display(50)'),
 code('christmas_controls = lesson.christmas_widget()',True),
 md('''**What to see.** Higher recovery makes leftovers cheaper and increases the order, while the forecast stays fixed. This shows why a forecast alone cannot choose an order quantity.

**Take home.** Keep the forecast, its error and the business cost of a mistake together. Your recommendation should state a choice, evidence and a limitation.''')])
print('Built the three notebooks; the two course notebooks fetch',len(FILES),'files from',REPO_RAW)

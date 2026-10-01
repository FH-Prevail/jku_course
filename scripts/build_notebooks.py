"""Create the notebooks. Every code cell shows the code it runs: the functions are copied from src/classroom.py,
the same code that produced the numbers on the slides, so Gemini in Colab can explain or change any line.
The first cell loads the course data from the repository on GitHub. Nothing is trained in the notebooks; every
forecast was computed in advance by analysis/build_classroom.py."""
from pathlib import Path
import inspect
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import classroom as lesson
REPO_RAW='https://raw.githubusercontent.com/FH-Prevail/jku_course/main'
DATA_URL=REPO_RAW+'/data/classroom'
_mod=Path(lesson.__file__).read_text()
SETTINGS=_mod.split('# >>> settings (copied into the first cell of each notebook)\n')[1].split('# <<< settings')[0].rstrip()
# The worked examples in the task text (Plush Bear, store), computed from the course data so they match the outputs.
lesson.load(ROOT/'data/classroom')
EXP='Plush Bear, store'
def _best(product,window):
 t=lesson.score_table(lesson.comparison_data(product,window));m=t['Average miss, units'].idxmin()
 return m,float(t.loc[m,'Average miss, units']),float(t.loc[m,'Average bias, units']),t
_PAT=next(x for x in lesson.results['patterns']['series'] if x['series_id']==lesson.series_id(EXP))
_XM,_XMISS,_XBIAS,_XT=_best(EXP,'Christmas');_SM,_SMISS,_SBIAS,_=_best(EXP,'Summer')
_RLY=float(_XT.loc['Repeat last year','Average miss, units']);_SES=float(_XT.loc['Simple exponential smoothing','Average miss, units'])
_BT=lesson.results['backtest'];_YEAR=min(_BT,key=lambda m:_BT[m]['wape'])
_R90,_R95,_R99=(lesson.simulate(EXP,t) for t in (90,95,99))
SETUP_HEAD='''# Start here: run this cell first. It loads the course data from GitHub and prepares the pictures.
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display

SOURCE = "@@DATA_URL@@"   # the course data on GitHub
demand = pd.read_csv(f"{SOURCE}/demand.csv", parse_dates=["week_start"])          # weekly sales: one row per product, channel and week
products = pd.read_csv(f"{SOURCE}/products.csv").set_index("product_id")           # price, cost and supplier lead time of each product
comparisons = pd.read_csv(f"{SOURCE}/comparisons.csv", parse_dates=["week_start", "test_start"])  # the backtest forecasts
inventory = pd.read_csv(f"{SOURCE}/inventory.csv", parse_dates=["week_start"])     # forecasts 1 to 7 weeks ahead, for the stock replay
results = json.load(urllib.request.urlopen(f"{SOURCE}/results.json"))             # numbers computed in advance
'''
SETUP_TAIL='''
print("Ready. The course data is loaded.")
print("The forecasts were computed in advance, using only what was known at each forecast date.")
'''
setup=(SETUP_HEAD.replace('@@DATA_URL@@',DATA_URL)+'\n'+SETTINGS+'\n\n\n'+inspect.getsource(lesson.series_id)+'\n\n'+inspect.getsource(lesson.neat)+SETUP_TAIL)
intro='''*UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi, Logistikum*

Choose **Copy to Drive** (File, Save a copy in Drive), then **Runtime → Run all**. You do not have to write code: run the cells, read the outputs and use the controls.

Every cell shows its code above its result. The first cell loads the course data and the precomputed forecasts from the course repository, github.com/FH-Prevail/jku_course; it does not train a model, and it takes a few seconds.

**Ask Gemini about the code:** select a cell and ask, for example, “Explain this code line by line in plain English” or “Change this to show the Puzzle 1000 instead”. Read what it changed before you run it, and check its answer against the numbers. If Gemini is unavailable, ask a neighbour or the lecturer. Use only the synthetic course data.

**If a control is missing:** use the static example directly above it. If you edited a cell by accident, Undo, then run it again; otherwise reopen a fresh copy from the repository. Restarting does not undo edits.
'''
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s,hidden=False):
 c=nbf.v4.new_code_cell(s)
 if hidden:c.metadata={'cellView':'form','jupyter':{'source_hidden':True}}
 return c

def run(*functions,call):
 """A code cell: the functions it needs, copied from src/classroom.py, then the line that runs them."""
 return code('\n\n\n'.join(inspect.getsource(f).rstrip() for f in functions)+'\n\n\n'+call)

def colab_badge(name):
 url=f'https://colab.research.google.com/github/FH-Prevail/jku_course/blob/main/notebooks/{name}'
 # The official badge: target="_parent" lets the link leave GitHub's sandboxed notebook frame, where a plain link can be blocked.
 return (f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
         f'\n\nIf the button does not open, use this link, or copy it into the browser: <a href="{url}" target="_parent">{url}</a>\n'
         '\nOr in Colab: **File, Open notebook, GitHub**, type `FH-Prevail/jku_course` and choose the notebook.')

def notebook(name,title,cells,lead=None,assignment=None):
 """A course notebook: badge, intro, the assignment box and the setup cell, then the cells. With `lead`, a stand-alone
 notebook instead: badge and that text, no setup cell (the data-loading demo)."""
 head=([md('# '+title+'\n\n'+colab_badge(name)+'\n\n'+intro+('\n'+assignment if assignment else '')),code(setup)] if lead is None
       else [md('# '+title+'\n\n'+colab_badge(name)+'\n\n'+lead)])
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
df.groupby("product_name")["units"].agg(["mean", "sum", "max"]).round(1).reset_index()   # per product: average week, total, best week'''),
 code('''weekly = df.pivot(index="week_start", columns="product_name", values="units")
weekly.plot(figsize=(10, 4), title="Units sold per week, store");'''),
 md('''**Ask Gemini**, for example: "Summarise this table", "Which product sells most in December?" or "Plot the Plush Bear's weekly units and mark the promotion weeks." Read the code it writes before you run it, and check its answer against the table.

**Upload only data you are allowed to share.** Files in Colab and in your Drive sit on Google's servers; company data needs the company's permission.''')],
 lead='''*UE Digital Analytics im Handel · JKU Linz · 03.10.2026 · Sina Mirshahi, Logistikum*

Three ways to get a file into Colab, tried with one small sample, **sample_weekly_sales.csv**: weekly store sales of three products from 2023 to 2025 (synthetic, the same data as the course). Pick one route in sections 1 to 3, then explore the table in section 4. Run a cell with the play button next to it, or Shift + Enter.''')

DEADLINE='Saturday 17 October 2026, 23:59'
def assignment(number,other,tasks):
 return f'''## Your assignment (graded)

This notebook is **half of your T2 assignment**; notebook {other} is the other half. Together they count **25 % of the course grade**, plus 5 % for taking part in class. This notebook has **{tasks} tasks and 10 points**.

- **Every task says step by step what to do** and shows a worked example for the Plush Bear in store. Do the same for your own product, with your own numbers.
- **Write your answers** in the cells marked **Your answer**: double-click the cell, replace each ___ with your answer, then press Shift + Enter.
- **Individual work.** You may discuss with others, but the answers you submit are your own.
- **Gemini is allowed.** Say in your answer where you used it, and check what it says against the numbers.
- **Submit by {DEADLINE}** in the T2 assignment on MS Teams: in Colab choose **File, Download, Download .ipynb**, name the file `T2_Notebook{number}_Lastname_Firstname.ipynb` and upload it.
- Run all cells before you download, so your results are saved next to your answers.'''
NAME=md('''**Your name:** ___

**Student number:** ___''')
def submit(number):
 return md(f'''## Before you submit

1. **Runtime, Run all**, so every result is saved next to your answers.
2. Check that every **Your answer** cell is filled in: no ___ left.
3. **File, Download, Download .ipynb**. Name the file `T2_Notebook{number}_Lastname_Firstname.ipynb`.
4. Upload it to the T2 assignment on MS Teams by **{DEADLINE}**.''')

notebook('1_see_and_forecast.ipynb','1 · Demand Patterns and Forecast Comparison',[
 NAME,
 md('''## 1. Read the demand

The retailer sells 40 products through two channels, store and online: 80 series of weekly sales. The data is synthetic, built to look like a real retailer's numbers.

Read the channel picture. Is one channel growing? Do you see a recurring busy season?'''),
 run(lesson.totals,call='totals()'),
 md('''**What to see.** Online totals grow from 2023 to 2025. Store totals fluctuate rather than growing each year. A monthly total also depends on how many weekly observations fall in that month.

Four demand patterns are useful descriptions, not guarantees of forecast accuracy: **smooth** means regular sales and fairly regular amounts; **erratic** means frequent sales with variable amounts; **intermittent** means many weeks without sales; **lumpy** adds variable amounts to those gaps.'''),
 run(lesson.pattern_examples,call='pattern_examples()'),
 md('''Two numbers place every product on one picture. **ADI** (average demand interval) is the average number of weeks between sales: 1 means a sale every week. **CV²** (squared coefficient of variation) says how irregular the amounts are from one sale to the next. The dashed lines are the usual limits from the literature, 1.32 and 0.49; they are a convention, not a law.'''),
 run(lesson.pattern_map,call='pattern_map()'),
 md(f'''### Task 1.1 · Your product's demand pattern (2 points)

**Choose your product** for this notebook, one of these three: `Plush Bear, online` · `Puzzle 1000, store` · `Garden Hose, store`. You use the same product in all three tasks. The Plush Bear in store is the worked example, so it is not one of the choices.

**What to do**
1. In the cell below, replace `Plush Bear, store` with the name of your product. Keep the quotation marks.
2. Run the cell. It prints two numbers, ADI and CV², and the demand pattern.
3. Copy them into the answer cell and complete the two sentences. Rule of thumb: an ADI close to 1 means a sale every week, above 1.32 many weeks without a sale; a CV² below 0.49 means fairly regular amounts, above 0.49 very variable amounts.

**Worked example, Plush Bear in store:** ADI {_PAT['adi']:.2f}, so it sells every week; CV² {_PAT['cv2']:.2f}, below 0.49, so the amounts are fairly regular; demand pattern: {_PAT['pattern']}.'''),
 run(lesson.pattern_of,call='pattern_of("Plush Bear, store")   # put the name of your product here'),
 md('''**Your answer, Task 1.1**

- My product: ___
- ADI: ___, so the product sells ___ (every week, or with many weeks without a sale)
- CV²: ___, so the amounts are ___ (fairly regular, or very variable)
- Demand pattern: ___ (smooth, erratic, intermittent or lumpy)'''),
 md('''## 2. Compare four approaches

- **Repeat last year**, also called seasonal naive: reuse sales from 52 weeks earlier.
- **Simple exponential smoothing (SES)**: a weighted average that gives more weight to recent sales; it forecasts one constant level for all weeks ahead.
- **Prophet**, a local model from Meta: one model per product and channel that adds up a trend, a yearly season and the effect of the planned discount.
- **LightGBM in global mode**: one LightGBM model learns from all products and channels, using past sales and planned prices, promotions and calendar information. (LightGBM can also be trained one series at a time, in local mode; here it runs in global mode.)

We forecast the same four weeks with each method and then reveal actual sales. An **average miss** is the average absolute difference in units. **Total miss as a percentage of demand** adds up those absolute misses and divides by total actual demand. **Bias** is the average forecast minus actual: positive means too high, negative means too low.

First look at the Christmas example. Do not choose a winner from the shape alone; read its errors too.'''),
 run(lesson.comparison_data,lesson.score_table,lesson.compare,call="compare('Plush Bear, store', 'Christmas')"),
 md('''**What to see.** The printed sentence names the method with the smallest miss in this window. The bias column adds direction; the same method need not win in another season or for another product.

**Try it using the dropdowns.** Change Window to Summer. Then select Garden Hose, store. Compare that product in Summer and Christmas. Change Plush Bear from store to online to investigate a channel difference.'''),
 run(lesson.forecast_widget,call='forecast_controls = forecast_widget()'),
 md(f'''### Task 1.2 · Two windows compared (4 points)

**What to do**
1. In the dropdowns above, choose **your product** and the window **Christmas**. No dropdowns? Type `compare("Garden Hose, store", "Christmas")` with your product into the empty code cell under this task and run it.
2. Look at the table under the chart. Find the **smallest number in the column "Average miss, units"**; the sentence under the table names that method too.
3. Write down that method, its average miss and its **average bias with its sign**: + means the forecast was too high, − too low.
4. Change the window to **Summer** and do the same.
5. Answer the two questions under the table. For the second one, choose the reason that fits your table best:
   - **A.** Christmas has a peak. Methods that see the yearly season (repeat last year, Prophet, LightGBM in global mode) can follow it; simple exponential smoothing draws one flat line.
   - **B.** Only Prophet and LightGBM in global mode know the planned discounts.
   - **C.** Last year's season can mislead: when this year's peak is smaller or larger, the methods that copy last year miss.
   - **D.** LightGBM in global mode learns from all 80 series at once, so it does well for many products.

**Worked example, Plush Bear in store:** at Christmas **{_XM}** has the smallest average miss, **{_XMISS:.1f}** units, with a bias of **{_XBIAS:+.1f}**. In Summer it is **{_SM}**, **{_SMISS:.1f}** units, bias **{_SBIAS:+.1f}**. Not the same method. Reason A: simple exponential smoothing drew one flat line and missed the Christmas weeks by {_SES:.1f} units on average.'''),
 md('''**Your answer, Task 1.2**

| Window | Method with the smallest average miss | Its average miss, units | Its average bias, units, with + or − |
|---|---|---|---|
| Christmas | ___ | ___ | ___ |
| Summer | ___ | ___ | ___ |

- Is it the same method in both windows? ___ (yes or no)
- The reason that fits my table best: ___ (A, B, C or D), because ___ (one sentence with your numbers)'''),
 md('''**Optional, not graded: change the code with Gemini.** Ask Gemini to write the code that shows your product in Spring, for example: *"Use compare to show Garden Hose, store in Spring"*, and run it in the empty cell below. No Gemini on your account? Copy the last line of the cell in section 2 into the empty cell and change the product and the window.'''),
 code('# Optional: the code from Gemini, or your own, goes here\n'),
 md(f'''## 3. Your first recommendation

### Task 1.3 · Your forecast choice (4 points)

Imagine the store manager asks you: *"Which forecast should we use for this product?"* Answer in five short lines.

**What to do**
1. Choose **one window** from your Task 1.2 table: Christmas or Summer.
2. **Recommend a method**, usually the one with the smallest average miss. Another one is fine if you say why.
3. **Give your evidence:** its average miss from your table, and the average miss of repeat last year in the same window (the row "Repeat last year" in the table under the chart). If your method is repeat last year, compare it with the second best method instead.
4. **Name one risk** of trusting this evidence, and explain it in one sentence with your numbers:
   - **a.** It is only one window of four weeks.
   - **b.** The bias shows that the method runs too high or too low.
   - **c.** It is one product in one channel.
5. **Name one check** before ordering, and say why:
   - **a.** The whole-year table in section 4, after the break.
   - **b.** The same product in the other channel.
   - **c.** What running out costs compared with leftover stock.

**Worked example, Plush Bear in store:** (1) Christmas. (2) {_XM}. (3) Its average miss was {_XMISS:.1f} units a week, against {_RLY:.1f} for repeat last year. (4) Risk b: its bias was {_XBIAS:+.1f}, so it ran about {abs(_XBIAS):.0f} units a week too low and the shelf could run empty. (5) Check a: over the whole year, {_YEAR} has the lowest WAPE ({_BT[_YEAR]['wape']:.1f} % against {_BT[_XM]['wape']:.1f} % for {_XM}), so the Christmas result may not hold in other weeks.'''),
 md('''**Your answer, Task 1.3**

1. Window: ___
2. Method I recommend: ___
3. My evidence: its average miss was ___ units a week, against ___ units for ___.
4. Risk (a, b or c): ___, because ___
5. Check (a, b or c): ___, because ___'''),
 md('''Stop here for the first exercise. The following sections support the evaluation discussion after the break.'''),
 md('''## 4. Check more than one window

A **backtest** replays past forecast decisions using only information available then. Here every method is tested at 13 dates, predicting the next four weeks each time; the test weeks run from 30 December 2024 to 22 December 2025.

**WAPE**, weighted absolute percentage error, is total absolute miss divided by total actual demand. **Percentage bias** is total signed error divided by total demand; earlier we expressed bias in units instead. Both definitions use forecast minus actual for the sign.'''),
 run(lesson.backtest,call='backtest()'),
 md('''**What to see.** The table compares all 80 series on the same dates; the four extra rows are the further local methods from the slides (Holt-Winters, Croston SBA, TSB, a moving average), scored the same way. Large sellers have more influence on these aggregate percentages, so the best assortment result does not promise the best result for your product. These results were computed in advance; LightGBM in global mode was retrained at each test date.'''),
 md('''## 5. A forecast can be right most weeks and still miss all sales

**MAPE**, mean absolute percentage error, computes a percentage miss for each week before averaging. A week with zero actual demand makes that division undefined.'''),
 run(lesson.zeros_demo,call='zeros_demo()'),
 md('''**What to see.** The zero forecast matches three weeks exactly but misses the only sale. Counting correct weeks is not enough to judge an order policy; carry this question into notebook 2: how much stock would you need?

**Evidence note.** Four observations in the source data are flagged as censored sales: the shelf was empty, so demand is unknown. For training, these are replaced with the preceding four weeks’ mean using only past information; flagged test rows are excluded from forecast scoring. The README of the `data` folder in the course repository describes the data in full.'''),
 submit(1)],assignment=assignment(1,2,3))

notebook('2_forecast_to_stock.ipynb','2 · Safety Stock and Replenishment Policy',[
 NAME,
 md('''## 1. A forecast is not an order

**Lead time** is the wait between placing and receiving an order. We review stock every week; the **protection period** covers lead time plus that review week.

**Safety stock** is a cushion for uncertain demand. **Stock target**, also called the order up to level, is forecast demand over the protection period plus safety stock. **Order now** is the shortfall between that target and stock already on the shelf or on order, assuming no backorders.

This worked example uses round teaching numbers: five forecasts of 180, 190, 210, 200 and 220 units. The forecasts sum to 1,000; the 95 percent planning target adds 148 units of safety stock.'''),
 run(lesson.order_example,lesson.worked_order,call='worked_order()'),
 md('''**What to see.** The stock target is 1,148 units, but the order is 348 because 800 units are already on the shelf or on order. Ordering the whole target would count that stock twice.

### Task 2.1 · The worked order at 99 percent (2 points)

**What to do**
1. In the cell below, change **95** to **99** and run it. It shows the same table as above, for a 99 percent target.
2. Copy three numbers into the answer table: the safety stock, the stock target and the order now.
3. Complete the sentence under the table.'''),
 code('worked_order(95)   # change 95 to 99'),
 md('''**Your answer, Task 2.1**

| | At 95 % (the table above) | At 99 % (your result) |
|---|---|---|
| Safety stock, units | 148 | ___ |
| Stock target, units | 1,148 | ___ |
| Order now, units | 348 | ___ |

- The order is smaller than the stock target because ___ units are already on the shelf or on order; ordering the whole target would ___.'''),
 md('''## 2. Choose a service target

A **cycle service level** is the probability of completing a replenishment cycle without a stockout. A target of 95 percent is a planning assumption, not a promise about the percentage of units served.

**Fill rate** is the share of demand actually served. The replay below reports fill rate, weeks with a stockout and average stock. We do not compare fill rate directly with the cycle service target.

LightGBM in global mode is trained once, before 2025. Error variability is measured on 25 weeks from January to June. Orders are rounded up to whole units.

**The replay** is like the backtest, but for stock: we run the ordering rule week by week on the real demand of 30 June to 22 December 2025, as if we had used it then. Each week it orders up to the stock target, the delivery arrives after the lead time, customers buy what is on the shelf, and a sale we miss is lost. Then we count the average stock, the fill rate and the weeks with a stock-out.'''),
 run(lesson.profile,lesson.simulate,lesson.stock_decision,call="stock_decision('Plush Bear, store', 95)"),
 md('''**Try it.** Move Target % from 50 to 80, then to 95 and 99. Does every increase serve more demand? Then choose Puzzle 1000 and Vacuum Filter; the same target need not deliver the same observed outcome.'''),
 run(lesson.stock_widget,call='stock_controls = stock_widget()'),
 md('''**If the slider is unavailable**, compare these preset settings instead.'''),
 run(lesson.stock_table,call="stock_table('Plush Bear, store')"),
 md(f'''**What to see.** Compare each increase in stock with the change in fill rate. If fill rate is unchanged, the extra stock served no additional demand in these test weeks; this does not prove that the extra cushion is useless in every future period.

### Task 2.2 · What does each step buy? (4 points)

**What to do**
1. Choose **your product**: `Puzzle 1000, store` or `Vacuum Filter, store`. You use it in Task 2.3 too.
2. In the cell below, put its name (keep the quotation marks) and run it. The table shows the replay at five targets.
3. Copy the rows for **80, 90, 95 and 99 %** into the answer table; skip 50 %.
4. Compare the 95 % row with the 99 % row and complete the two sentences.

**Worked example, Plush Bear in store** (the table above): from 95 to 99 % the average stock rises from {_R95['average_stock']:.0f} to {_R99['average_stock']:.0f} units, and the fill rate goes from {_R95['fill_rate']:.1f} % to {_R99['fill_rate']:.1f} %. The extra {round(_R99['average_stock']) - round(_R95['average_stock'])} units served no more customers in these weeks, so the step to 99 % is not worth it for the Bear.'''),
 code('stock_table("Puzzle 1000, store")   # or "Vacuum Filter, store"'),
 md('''**Your answer, Task 2.2**

My product: ___

| Target | Average stock, units | Fill rate | Weeks with a stock-out |
|---|---|---|---|
| 80 % | ___ | ___ | ___ |
| 90 % | ___ | ___ | ___ |
| 95 % | ___ | ___ | ___ |
| 99 % | ___ | ___ | ___ |

- From 95 to 99 %, the average stock rises from ___ to ___ units (what it costs), and the fill rate goes from ___ % to ___ % (what it buys).
- Is the step to 99 % worth it for this product? ___ (yes or no), because ___'''),
 md('''## 3. Compare the products

All three products below use a 95 percent cycle service target. “Weeks of demand on shelf” divides average stock by average demand during the same replay period.'''),
 run(lesson.stock_compare,call='stock_compare()'),
 md('''**What to see.** A high target can coexist with shortages when the error assumptions are poor. A slow mover can hold many weeks of demand in only a few units; inspect unit cost before calling that stock excessive.

The safety stock calculation assumes independent weekly errors, a stable error distribution and an approximate bell curve. Those assumptions are especially weak for sparse demand. Treat the replay as a check on the rule, not a service guarantee.'''),
 md(f'''## 4. Your management recommendation

### Task 2.3 · Your stock recommendation (4 points)

Imagine the store manager asks you: *"Which service target should we use for this product?"* Answer in five short lines, for your product from Task 2.2.

**What to do**
1. Write your product.
2. **Choose one target:** 80, 90, 95 or 99 %.
3. **Give your evidence:** at that target, the average stock and the fill rate from your Task 2.2 table.
4. **Say what you give up**, and explain it in one sentence with your numbers:
   - **a.** More stock tied up on the shelf.
   - **b.** Some customers not served.
   - **c.** Less safety in unusual weeks.
5. **Name one check** before using it in the real store, and say why:
   - **a.** What holding one unit for a year costs.
   - **b.** What a lost sale costs.
   - **c.** Whether the forecast runs too high or too low (its bias).

**Worked example, Plush Bear in store:** (1) Plush Bear, store. (2) 90 %. (3) The shelf held on average {_R90['average_stock']:.0f} units and served {_R90['fill_rate']:.1f} % of demand. (4) Give up c: less safety in unusual weeks; but in these 26 weeks 95 and 99 % served no more customers and held {round(_R95['average_stock']) - round(_R90['average_stock'])} and {round(_R99['average_stock']) - round(_R90['average_stock'])} more units. (5) Check b: what a lost sale at Christmas costs, because one empty-shelf week could cost more than the stock we save.'''),
 md('''**Your answer, Task 2.3**

1. Product: ___
2. Target: ___ %
3. My evidence: at this target the shelf held on average ___ units and served ___ % of demand.
4. What I give up (a, b or c): ___, because ___
5. Check (a, b or c): ___, because ___'''),
 md('''You have completed the graded tasks; the next sections are optional.'''),
 md('''## 5. Optional: what is a better forecast worth?

Keep the same 95 percent target and change the forecasting approach. Each method uses its own error estimate. Read stock and service together; lower stock with substantially lower service is a different policy, not automatically a better one.'''),
 run(lesson.forecast_value,call='forecast_value()'),
 md('''**What to see.** The last column prices the average stock using this product’s unit cost and a 20 percent annual holding rate. It is an annualised scenario based on a half-year replay, not realised annual savings or a complete investment case; shortages and implementation costs are excluded.'''),
 md('''## 6. Optional: one Christmas order

The **newsvendor** problem is one order for one selling period. A unit short loses margin; a leftover unit costs its purchase price minus what we recover afterward.

The **critical ratio** is shortage cost divided by shortage cost plus leftover cost. It chooses a **quantile**: a demand level with a specified probability of not being exceeded. Here demand is approximated with a bell curve, so the 50th percentile is its mean.

This is a separate decision assuming the order can arrive for the selling week; it does not use the replenishment lead time. Use recovery below 100 percent: with no leftover cost, this unbounded model has no finite optimal order.'''),
 run(lesson.christmas_order,lesson.christmas_display,call='christmas_display(50)'),
 run(lesson.christmas_widget,call='christmas_controls = christmas_widget()'),
 md('''**What to see.** Higher recovery makes leftovers cheaper and increases the order, while the forecast stays fixed. This shows why a forecast alone cannot choose an order quantity.

**Take home.** Keep the forecast, its error and the business cost of a mistake together. Your recommendation should state a choice, evidence and a limitation.'''),
 submit(2)],assignment=assignment(2,1,3))

print('Built the three notebooks; notebooks 1 and 2 load the course data from',DATA_URL)

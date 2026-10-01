"""Student-facing T2 displays. Uses the frozen classroom data, never trains a model."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

BLUE = '#2F789F'
# The deck's line colours: one colour per method, the same on every slide and in every notebook chart.
LINE_COL = {'Actual sales': '#262626', 'Repeat last year': '#E66A1F', 'Simple exponential smoothing': '#1B9E5A',
            'Prophet': '#D04F95', 'LightGBM in global mode': '#0B5CAD'}
LINE_MARK = {'Repeat last year': 's', 'Simple exponential smoothing': '^', 'Prophet': 'X', 'LightGBM in global mode': 'D'}
INK = '#253746'
METHODS = ['Repeat last year', 'Simple exponential smoothing', 'LightGBM in global mode']            # the stock replay uses these three
COMPARE = ['Repeat last year', 'Simple exponential smoothing', 'Prophet', 'LightGBM in global mode']  # the forecast comparisons: two simple rules, Prophet (local), LightGBM trained on all series (global mode)
PRODUCTS = {'Plush Bear, store': 'P010_store', 'Plush Bear, online': 'P010_online',
            'Puzzle 1000, store': 'P011_store', 'Vacuum Filter, store': 'P035_store',
            'Garden Hose, store': 'P015_store'}
WINDOWS = {'Spring': '2025-03-24', 'Summer': '2025-06-16', 'Christmas': '2025-12-01'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .2, 'figure.dpi': 110})
D = P = B = I = R = None


def load(folder):
    global D, P, B, I, R
    folder = Path(folder)
    D = pd.read_csv(folder/'demand.csv', parse_dates=['week_start'])
    P = pd.read_csv(folder/'products.csv').set_index('product_id')
    B = pd.read_csv(folder/'comparisons.csv', parse_dates=['week_start', 'test_start'])
    I = pd.read_csv(folder/'inventory.csv', parse_dates=['week_start'])
    R = json.loads((folder/'results.json').read_text())


def show(obj):
    from IPython.display import display
    display(obj)


def _headless():
    """Headless test runs (nbclient) can stall on widget display; the T2_NO_WIDGETS flag skips it. Colab shows the controls."""
    if os.environ.get('T2_NO_WIDGETS'):
        print('The interactive controls appear when you run this notebook in Colab. Until then, use the static example above.');return True
    return False


def _sid(product):
    if product in PRODUCTS: return PRODUCTS[product]
    if product in PRODUCTS.values(): return product
    raise ValueError('Choose one of the product names in the dropdown.')


def totals():
    t=D.groupby(['week_start','channel']).units.sum().unstack().resample('MS').sum()
    fig,ax=plt.subplots(figsize=(9,3.8))
    for ch,col in [('store','#0B5CAD'),('online','#E66A1F')]: ax.plot(t.index,t[ch],color=col,lw=2,label=ch)
    ax.set(ylabel='Units sold per month',title='All products: store and online');ax.legend();fig.tight_layout();plt.show()
    y=D.assign(year=D.week_start.dt.year).groupby(['year','channel']).units.sum().unstack()
    return y.rename_axis('Year').rename(columns={'store':'Store units','online':'Online units'})


def pattern_examples():
    fig,axes=plt.subplots(2,2,figsize=(10,5.7),layout='constrained')
    examples=[('P010_store','Smooth: sales most weeks'),('P011_store','Erratic: frequent, variable amounts'),
              ('P035_store','Intermittent: many weeks without sales'),('P038_store','Lumpy: gaps and variable amounts')]
    for ax,(sid,title) in zip(axes.flat,examples):
        t=D[(D.series_id==sid)&(D.week_start.dt.year==2025)]
        ax.bar(t.week_start,t.units,width=6,color=BLUE);ax.set(title=title,ylabel='Units');ax.tick_params(axis='x',rotation=25,labelsize=9)
    plt.show()


def pattern_map():
    """Every series placed by its two numbers, ADI and CV²: the four demand patterns as one picture."""
    s=pd.DataFrame(R['patterns']['series']);cuts=R['patterns']['cuts'];counts=R['patterns']['counts']
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    ax.scatter(s.cv2.clip(upper=2),s.adi.clip(upper=10),color=BLUE,s=30,alpha=.85,edgecolor='white',lw=.5)
    ax.axvline(cuts['cv2'],color='#888888',ls='--',lw=1);ax.axhline(cuts['adi'],color='#888888',ls='--',lw=1)
    ax.set(xlabel='CV²: how irregular the sale sizes are',ylabel='ADI: average weeks between sales',xlim=(0,2.06),ylim=(0,10.6))
    for name,x,y in [('Smooth',.03,.25),('Erratic',1.62,.25),('Intermittent',.03,9.8),('Lumpy',1.62,9.8)]:
        ax.text(x,y,f"{name} ({counts[name.lower()]})",fontsize=11,color=INK,weight='bold')
    fig.tight_layout();plt.show()
    print('Each dot is one product in one channel, classified on 2023 and 2024. Values beyond the edge are drawn on the edge.')
    print('ADI: the average number of weeks between sales (1 = a sale every week). CV²: how irregular the amounts are from one sale to the next.')


def explore(product='Plush Bear, store'):
    sid=_sid(product);t=D[(D.series_id==sid)&(D.week_start.dt.year==2025)]
    fig,ax=plt.subplots(figsize=(9,3.4));ax.plot(t.week_start,t.units,color=BLUE,lw=1.8)
    promo=t[t.promo_flag==1];ax.scatter(promo.week_start,promo.units,color=BLUE,marker='o',s=55,label='Promotion week')
    ax.set(title=product,ylabel='Units sold');ax.legend();fig.tight_layout();plt.show()
    print('Look for a season, a changing level, and weeks without sales. Circles mark promotions, not their causal effect.')


def comparison_data(product='Plush Bear, store',window='Christmas'):
    sid=_sid(product)
    if window not in WINDOWS: raise ValueError('Choose Spring, Summer or Christmas.')
    return B[(B.series_id==sid)&(B.test_start==pd.Timestamp(WINDOWS[window]))].copy()


def score_table(g):
    rows=[]
    for m in COMPARE:
        q=g[(g.method==m)&(g.scoreable==1)];a=q.actual.to_numpy();f=q.forecast.to_numpy();total=a.sum()
        rows.append({'Method':m,'Average miss, units':np.abs(f-a).mean(),
                     'Total miss, % of demand':np.abs(f-a).sum()/total*100 if total else np.nan,
                     'Average bias, units':(f-a).mean()})
    return pd.DataFrame(rows).set_index('Method')


def compare(product='Plush Bear, store',window='Christmas'):
    g=comparison_data(product,window);a=g[g.method==METHODS[0]].sort_values('week_start')
    fig,ax=plt.subplots(figsize=(9,3.8))
    ax.bar(np.arange(4)-.12,a.actual,width=.24,color='#A8B4BE',label='Actual sales')
    for m in COMPARE:
        q=g[g.method==m].sort_values('week_start');ax.plot(range(4),q.forecast,color=LINE_COL[m],marker=LINE_MARK[m],lw=2,label=m)
    ax.set(xticks=range(4),xticklabels=a.week_start.dt.strftime('%d %b'),ylabel='Units',title=f'{product} | {window}')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=10);fig.tight_layout();plt.show()
    print(f'Forecast made after { (a.week_start.min()-pd.Timedelta(weeks=1)).date() }; all four weeks forecast together.')
    s=score_table(g)
    show(s.style.format({'Average miss, units':'{:.1f}','Total miss, % of demand':'{:.1f} %','Average bias, units':'{:+.1f}'},na_rep='Undefined: no demand'))
    winner=s['Average miss, units'].idxmin()
    print(f'In this window, {winner} has the smallest average miss. This is evidence for this product and window only.')
    print('Bias is forecast minus actual: positive means too high; negative means too low.')


def forecast_widget():
    if _headless(): return None
    from ipywidgets import interactive,Dropdown
    w=interactive(compare,product=Dropdown(options=list(PRODUCTS),value='Plush Bear, store',description='Product'),
                  window=Dropdown(options=list(WINDOWS),value='Christmas',description='Window'))
    show(w);return w


def backtest():
    rows=[]
    for m in COMPARE:
        q=B[(B.method==m)&(B.scoreable==1)];e=q.forecast-q.actual
        rows.append([m,f'{e.abs().sum()/q.actual.sum()*100:.1f} %',f'{e.sum()/q.actual.sum()*100:+.1f} %'])
    for m in ['Holt-Winters','Croston SBA','TSB','Moving average']:
        s=R['backtest_all'][m]['all'];rows.append([m,f"{s['wape']:.1f} %",f"{s['bias_pct']:+.1f} %"])
    print('13 test dates; four weeks ahead each; all 80 series. The same weeks and products for every method.')
    print('The four methods of the window comparison come first; four more local methods from the slides were scored the same way.')
    print('Flagged stockout observations are excluded from scoring because their demand is unknown.')
    return pd.DataFrame(rows,columns=['Method','WAPE: total miss / total demand','Bias: net error / total demand']).set_index('Method')


def zeros_demo():
    a=np.array([0,0,0,4]);f=np.zeros(4)
    print('Teaching example, not an extracted course product.')
    show(pd.DataFrame({'Week':[1,2,3,4],'Actual units':a,'Forecast units':f}))
    print('Three weeks exactly right, but all four units of demand missed.')
    print('MAPE is undefined because actual demand is zero in three weeks. WAPE is 100 %.')
    print('Do not choose a stock policy by counting the weeks a forecast gets right.')


def profile(product='Plush Bear, store',method='LightGBM in global mode'):
    sid=_sid(product)
    if method not in METHODS: raise ValueError('Choose a method shown in the table.')
    g=I[(I.series_id==sid)&(I.method==method)].sort_values('week_start')
    if g.empty: raise ValueError('This product has no stock exercise.')
    a=g.actual.to_numpy(float);f=g[[f'h{x}' for x in range(1,8)]].to_numpy(float)
    p=P.loc[sid.split('_')[0]];L=int(p.lead_time_weeks);f=f[:,:L+1]
    half=25;sig=float(np.std(a[:half]-f[:half,0],ddof=1))
    return g,a,f,sig,L,half


def simulate(product='Plush Bear, store',target=95,method='LightGBM in global mode',details=False):
    if not np.isfinite(target) or not 50<=target<=99: raise ValueError('Choose a target from 50 to 99 percent.')
    g,a,f,sig,L,half=profile(product,method)
    ss=float(norm.ppf(target/100)*sig*np.sqrt(L+1));on=float(np.ceil(f[half].sum()));pipeline=[];rows=[]
    for t in range(half,len(a)):
        on+=sum(q for arr,q in pipeline if arr==t);pipeline=[(arr,q) for arr,q in pipeline if arr!=t]
        level=float(np.ceil(f[t].sum()+ss));order=max(0.,level-on-sum(q for _,q in pipeline));pipeline.append((t+L,order))
        served=min(on,a[t]);lost=a[t]-served;on-=served
        rows.append({'week':str(g.week_start.iloc[t].date()),'actual':float(a[t]),'forecast_sum':float(f[t].sum()),'level':level,'order':order,'lost':lost,'stock':on})
    q=pd.DataFrame(rows);total=q.actual.sum()
    out={'target':float(target),'safety_stock':float(np.ceil(ss)),'average_stock':float(q.stock.mean()),
         'fill_rate':float(100*(1-q.lost.sum()/total)) if total else 100.,'stockout_weeks':int((q.lost>0).sum()),
         'lost_units':float(q.lost.sum()),'weeks_stock':float(q.stock.mean()/q.actual.mean()) if total else 0.,
         'weekly_sigma':sig,'protection_weeks':L+1,'test_weeks':len(q)}
    return (out,rows) if details else out


def stock_table(product='Plush Bear, store',targets=(50,80,90,95,99)):
    rows=[]
    for t in targets:
        r=simulate(product,t)
        rows.append([f'{t} %',f"{r['average_stock']:.0f}",f"{r['fill_rate']:.1f} %",r['stockout_weeks'],f"{r['lost_units']:.0f}"])
    return pd.DataFrame(rows,columns=['Cycle service target','Average stock, units','Fill rate','Weeks with a stockout','Units not served']).set_index('Cycle service target')


def stock_decision(product='Plush Bear, store',target=95):
    r=simulate(product,target);_,_,_,_,L,_=profile(product)
    print(f'{product}: cycle service target {target} %; supplier lead time {L} weeks.')
    show(pd.DataFrame({'Observed result':[f"{r['average_stock']:.0f} units",f"{r['fill_rate']:.1f} %",f"{r['stockout_weeks']} of {r['test_weeks']} weeks",f"{r['lost_units']:.0f} units"]},
                       index=['Average stock on the shelf','Fill rate: share of demand served','Weeks with a stockout','Demand not served']))
    print(f"Safety stock allowance: about {r['safety_stock']:.0f} units. This is part of the stock target, not the order quantity.")
    print('A cycle service target is a planning probability; fill rate measures units served. They are different measures.')


def stock_widget():
    if _headless(): return None
    from ipywidgets import interactive,Dropdown,IntSlider
    w=interactive(stock_decision,product=Dropdown(options=['Plush Bear, store','Puzzle 1000, store','Vacuum Filter, store'],description='Product'),
                  target=IntSlider(value=95,min=50,max=99,step=1,description='Target %',continuous_update=False))
    show(w);return w


def order_example(target=95):
    # A transparent teaching scenario, kept separate from the measured stock replay.
    f=np.array([180,190,210,200,220]);sigma_week=40.;L=4
    ss=int(np.ceil(norm.ppf(target/100)*sigma_week*np.sqrt(L+1)))
    S=int(f.sum()+ss);on=300;incoming=500;order=max(0,S-on-incoming)
    return {'weekly_forecasts':f.tolist(),'forecast_total':int(f.sum()),'weekly_sigma':sigma_week,
            'protection_weeks':L+1,'safety_stock':ss,'stock_target':S,'on_hand':on,'on_order':incoming,'order':order}


def worked_order():
    e=order_example();print('Teaching scenario: five weekly forecasts, one order decision. These are round illustrative numbers.')
    show(pd.DataFrame({'Quantity':['Forecast over five weeks','Safety stock','Stock target','Already on the shelf','Already on order','Order now'],
                       'Units':[e[k] for k in ['forecast_total','safety_stock','stock_target','on_hand','on_order','order']]}).set_index('Quantity'))
    print('We assume no backorders and arrivals as scheduled. The stock target is not the quantity to order.')


def stock_compare():
    rows=[]
    for p in ['Plush Bear, store','Puzzle 1000, store','Vacuum Filter, store']:
        r=simulate(p,95);rows.append([p,f"{r['fill_rate']:.1f} %",r['stockout_weeks'],f"{r['weeks_stock']:.1f}"])
    return pd.DataFrame(rows,columns=['Product','Fill rate at 95 % target','Weeks with a stockout','Weeks of demand on shelf']).set_index('Product')


def forecast_value(product='Plush Bear, store'):
    p=P.loc[_sid(product).split('_')[0]];rows=[]
    for m in METHODS:
        r=simulate(product,95,m);rows.append([m,f"{r['average_stock']:.0f}",f"{r['fill_rate']:.1f} %",f"EUR {r['average_stock']*p.unit_cost*p.holding_cost_rate:.0f}"])
    print(f'{product}. Same 95 % target, each method uses its own measured errors.')
    print(f'Unit cost EUR {p.unit_cost:.2f}; annual holding rate {p.holding_cost_rate:.0%}.')
    print('Annualised stock holding cost only: a scenario based on the replay average, excluding shortages and implementation.')
    return pd.DataFrame(rows,columns=['Method','Average stock, units','Fill rate','Annualised holding cost']).set_index('Method')


def christmas_order(salvage=50):
    if not np.isfinite(salvage) or not 0<=salvage<=95: raise ValueError('Choose recovery from 0 to 95 percent.')
    g,a,f,sig,L,half=profile('Puzzle 1000, store');i=np.flatnonzero(g.week_start==pd.Timestamp('2025-12-15'))[0]
    p=P.loc['P011'];under=float(p.unit_price-p.unit_cost);over=float(p.unit_cost*(1-salvage/100));ratio=under/(under+over)
    order=max(0,int(np.ceil(f[i,0]+norm.ppf(ratio)*sig)))
    return {'forecast':float(f[i,0]),'sigma':sig,'under':under,'over':over,'ratio':ratio,'order':order,'actual':float(a[i])}


def christmas_display(salvage=50):
    r=christmas_order(salvage)
    print(f"Puzzle 1000, store, 15 December 2025: forecast {r['forecast']:.0f} units; actual {r['actual']:.0f}.")
    print(f"Recover {salvage} % of purchase cost on leftovers: a unit short costs EUR {r['under']:.2f}, a leftover costs EUR {r['over']:.2f}.")
    print(f"Critical ratio {r['ratio']:.2f}: choose that quantile of demand. Suggested order {r['order']} units.")
    print('A quantile is a demand level with a chosen probability of not being exceeded. Here we approximate demand with a bell curve.')
    print('This separate exercise assumes one order can arrive for the selling week; it does not simulate the supplier lead time.')


def christmas_widget():
    if _headless(): return None
    from ipywidgets import interactive,IntSlider
    w=interactive(christmas_display,salvage=IntSlider(value=50,min=0,max=95,step=5,description='Recovery %',continuous_update=False));show(w);return w

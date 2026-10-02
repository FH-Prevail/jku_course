"""Build the frozen evidence used by the redesigned slides and both notebooks."""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import retail_fc as rf
warnings.filterwarnings('ignore')
OUT=ROOT/'data/classroom';OUT.mkdir(exist_ok=True)
d=rf.load_data(ROOT/'data');raw=d['demand'];products=d['products'];clean=raw.copy()
# Causal history repair: replace a flagged observation with the preceding four observed/repaired weeks.
for sid,g in clean.groupby('series_id'):
    hist=[]
    for idx,row in g.iterrows():
        v=float(np.mean(hist[-4:])) if row.stockout_flag and hist else float(row.units)
        clean.loc[idx,'units']=v;hist.append(v)
wide=rf.to_wide(clean);actual=rf.to_wide(raw);weeks=wide.index;pos={w:i for i,w in enumerate(weeks)}
disc=raw.pivot(index='week_start',columns='series_id',values='discount_pct').sort_index()   # planned, so known for the forecast weeks too
patterns=rf.pattern_table(wide.iloc[:104]) # training history only
cols=rf.FEATURE_COLS+['horizon'];features=pd.concat([rf.make_features(clean,horizon=h) for h in range(1,8)],ignore_index=True)
features['t']=features.week_start.map(pos)
FLAGGED=set(zip(raw.series_id[raw.stockout_flag==1],raw.week_start[raw.stockout_flag==1]))
rows=[];more=[]   # more: further local methods, scored the same way, shown on the slides and in notebook 1 section 4
for o in range(104,153,4):
    tr=features[(features.t<o)&(features.horizon<=4)];te=features[(features.t>=o)&(features.t<o+4)&(features.horizon==features.t-o+1)]
    model=lgb.LGBMRegressor(**rf.lgbm_params(n_jobs=2)).fit(tr[cols],tr.units)
    preds=model.predict(te[cols]);pred_lookup=dict(zip(zip(te.series_id,te.horizon),preds))
    for sid in wide.columns:
        train=wide[sid].iloc[:o].to_numpy(float)
        forecasts={'Repeat last year':rf.seasonal_naive(train,4),'Simple exponential smoothing':rf.ses(train,4),
                   'Prophet':rf.prophet_forecast(weeks[:o],train,disc[sid].iloc[:o],weeks[o:o+4],disc[sid].iloc[o:o+4]),
                   'LightGBM in global mode':np.array([pred_lookup[sid,h] for h in range(1,5)])}
        for m,fc in forecasts.items():
            for h in range(4):
                t=o+h;flag=int(raw[(raw.series_id==sid)&(raw.week_start==weeks[t])].stockout_flag.iloc[0])
                rows.append([sid,str(weeks[o].date()),str(weeks[t].date()),h+1,m,float(actual[sid].iloc[t]),max(0.,float(fc[h])),1-flag,patterns.loc[sid,'pattern']])
        extra={'Holt-Winters':rf.holt_winters(train,4),'Croston':rf.croston(train,4),'Croston SBA':rf.croston(train,4,variant='sba'),
               'TSB':rf.tsb(train,4),'Moving average':rf.moving_average(train,4)}
        for m,fc in extra.items():
            for h in range(4):
                t=o+h;more.append([sid,str(weeks[o].date()),str(weeks[t].date()),h+1,m,float(actual[sid].iloc[t]),max(0.,float(fc[h])),int((sid,weeks[t]) not in FLAGGED),patterns.loc[sid,'pattern']])
    print('Backtest',weeks[o].date(),flush=True)
bt=pd.DataFrame(rows,columns=['series_id','test_start','week_start','horizon','method','actual','forecast','scoreable','pattern']);bt.to_csv(OUT/'comparisons.csv',index=False)
more=pd.DataFrame(more,columns=bt.columns);allbt=pd.concat([bt,more],ignore_index=True)
# Freeze one model before 2025; weekly history inputs are updated causally, weights stay fixed.
cut=pd.Timestamp('2025-01-01');tr=features[features.week_start<cut];te=features[features.week_start>=cut].copy()
model=lgb.LGBMRegressor(**rf.lgbm_params(n_jobs=2)).fit(tr[cols],tr.units);te['forecast']=np.clip(model.predict(te[cols]),0,None)
te['review_t']=te.t-te.horizon+1;lookup=te.set_index(['series_id','review_t','horizon']).forecast
selected=['P010_store','P010_online','P011_store','P035_store','P015_store'];inv=[]
for sid in selected:
    y=wide[sid].to_numpy(float)
    for t in sorted(te.t.unique()):
        fs=[float(lookup.loc[(sid,t,h)]) for h in range(1,8) if t+h-1<len(weeks)]
        fs+= [fs[-1]]*(7-len(fs))
        fut=pd.date_range(weeks[t],periods=7,freq='7D')            # the next 7 weeks; past the data end the plan is unknown, so no discount
        pf=rf.prophet_forecast(weeks[:t],y[:t],disc[sid].iloc[:t],fut,disc[sid].reindex(fut).fillna(0))
        forecasts={'Repeat last year':rf.seasonal_naive(y[:t],7),'Simple exponential smoothing':rf.ses(y[:t],7),'Prophet':pf,'LightGBM in global mode':fs}
        for m,f in forecasts.items():inv.append([sid,str(weeks[t].date()),m,float(actual[sid].iloc[t]),*map(float,f)])
pd.DataFrame(inv,columns=['series_id','week_start','method','actual']+[f'h{h}' for h in range(1,8)]).to_csv(OUT/'inventory.csv',index=False)
raw.to_csv(OUT/'demand.csv',index=False);products.to_csv(OUT/'products.csv',index=False)
r={'release':'2026-09-30','scope':'Sina, T2: demand forecasting and inventory decisions',
   'n_rows':len(raw),'n_series':80,'n_weeks':156,'total_units':int(raw.units.sum()),
   'data_start':str(weeks[0].date()),'data_end':str(weeks[-1].date()),
   'backtest':{},'backtest_setup':{'origins':13,'horizon':4,'first_test_week':str(weeks[104].date()),'last_test_week':str(weeks[-1].date()),'flagged_test_rows_excluded':int(bt[bt.method=='LightGBM in global mode'].scoreable.eq(0).sum())},
   'model':rf.lgbm_params(n_jobs=2),'inventory_setup':{'training_end':'2024-12-30','calibration_start':'2025-01-06','calibration_end':'2025-06-23','test_start':'2025-06-30','test_end':'2025-12-22','calibration_weeks':25,'test_weeks':26,'integer_orders':True,'initial_stock':'ceiling of first protection forecast, no pipeline','tail':'extend the last available forecast from the same origin'},
   'history_repair':'4 flagged sales observations replaced by mean of preceding four weeks; flagged test observations excluded from forecast scores'}
for m,g in bt[bt.scoreable==1].groupby('method'):
    e=g.forecast-g.actual;r['backtest'][m]={'wape':float(e.abs().sum()/g.actual.sum()*100),'bias_pct':float(e.sum()/g.actual.sum()*100)}
def _score(g):
    e=g.forecast-g.actual;return {'wape':float(e.abs().sum()/g.actual.sum()*100),'bias_pct':float(e.sum()/g.actual.sum()*100)}
sc=allbt[allbt.scoreable==1]
r['backtest_all']={m:{'all':_score(g),**{p:_score(gp) for p,gp in g.groupby('pattern')}} for m,g in sc.groupby('method')}
r['by_horizon']={m:{str(h):_score(g[g.horizon==h])['wape'] for h in range(1,5)} for m,g in sc.groupby('method')}   # WAPE by weeks ahead
sm=sc[sc.pattern=='smooth'];origins=sorted(sm.test_start.unique())
r['per_origin']={'origins':origins,'smooth':{m:[_score(g[g.test_start==o])['wape'] for o in origins] for m,g in sm.groupby('method')}}
r['split_demo']={'window':origins[-1],'fixed_last':{m:_score(g[g.test_start==origins[-1]])['wape'] for m,g in sm.groupby('method')},
                 'rolling':{m:_score(g)['wape'] for m,g in sm.groupby('method')}}
# --- Added 29.09.2026 for the integrated deck: pattern map, results by pattern, model reasons ---
full_pat=rf.pattern_table(wide)   # classification on the full history, for the picture of all 80 series
r['patterns']={'counts':{k:int(v) for k,v in patterns.pattern.value_counts().items()},
               'counts_full_history':{k:int(v) for k,v in full_pat.pattern.value_counts().items()},
               'cuts':{'adi':float(rf.ADI_CUT),'cv2':float(rf.CV2_CUT)},
               'series':[{'series_id':sid,'adi':float(row.adi),'cv2':float(row.cv2),'pattern':row.pattern,
                          'units_2025':float(actual[sid].iloc[104:].sum())} for sid,row in patterns.iterrows()],
               'units_share_test_year':{k:float(actual.loc[:,patterns.index[patterns.pattern==k]].iloc[104:].sum().sum()/actual.iloc[104:].sum().sum()) for k in ['smooth','erratic','intermittent','lumpy']}}
r['backtest_by_pattern']={}
for (m,pat),g in bt[bt.scoreable==1].groupby(['method','pattern']):
    e=g.forecast-g.actual;r['backtest_by_pattern'].setdefault(m,{})[pat]={'wape':float(e.abs().sum()/g.actual.sum()*100),'bias_pct':float(e.sum()/g.actual.sum()*100),'n_series':int(g.series_id.nunique())}
r['examples']={}
for pat,sid in [('smooth','P010_store'),('erratic','P011_store'),('intermittent','P035_store'),('lumpy','P038_store')]:
    q=raw[(raw.series_id==sid)&(raw.week_start.dt.year==2025)].sort_values('week_start');a_,c_=rf.adi_cv2(q.units.to_numpy(float))
    r['examples'][pat]={'series_id':sid,'name':q.product_name.iloc[0],'channel':q.channel.iloc[0],'weeks':[str(w.date()) for w in q.week_start],'units':q.units.astype(float).tolist(),'adi_2025':float(a_),'cv2_2025':float(c_)}
mon=raw.groupby([raw.week_start.dt.to_period('M'),'channel']).units.sum().unstack()
r['monthly']={'months':[str(m) for m in mon.index],'store':mon.store.astype(float).tolist(),'online':mon.online.astype(float).tolist()}
x=raw.copy();x['previous_promo']=x.groupby('series_id').promo_flag.shift(1).fillna(0)
x=x[x.groupby('series_id').promo_flag.transform('sum')>0];base=x[(x.promo_flag==0)&(x.previous_promo==0)].groupby('series_id').units.mean()
x['relative']=x.units/x.series_id.map(base)
r['promo']={'lift_by_discount':{str(int(k)):float(v) for k,v in x[x.promo_flag==1].groupby('discount_pct').relative.mean().items()},
            'lift_by_channel':{k:float(v) for k,v in x[x.promo_flag==1].groupby('channel').relative.mean().items()},
            'week_after':float(x[(x.promo_flag==0)&(x.previous_promo==1)].relative.mean()),
            'n_promo_weeks':int(raw.promo_flag.sum()),'n_stockout_weeks':int(raw.stockout_flag.sum())}
# Model reasons, on the frozen model used for the stock replay: scramble one input on the 2025 rows (mean of 3 shuffles),
# and split one forecast into its inputs with LightGBM's own SHAP-style contributions (log scale, shown as factors).
rng=np.random.default_rng(0);te4=te[te.horizon<=4].copy();base_wape=float(np.abs(te4.forecast-te4.units).sum()/te4.units.sum()*100);perm={}
for col in cols:
    inc=[]
    for k in range(3):
        z=te4.reset_index(drop=True).copy();z[col]=z[col].sample(frac=1,random_state=int(rng.integers(1e9))).reset_index(drop=True);pr=np.clip(model.predict(z[cols]),0,None)
        inc.append(float(np.abs(pr-z.units).sum()/z.units.sum()*100)-base_wape)
    perm[col]=float(np.mean(inc))
perm=dict(sorted(perm.items(),key=lambda kv:-kv[1]))
one=te[(te.series_id=='P010_store')&(te.horizon==1)&(te.week_start==pd.Timestamp('2025-12-01'))]
contrib=model.predict(one[cols],pred_contrib=True)[0];expected=float(contrib[-1]);shap=dict(zip(cols,contrib[:-1]))
top=sorted(shap.items(),key=lambda kv:-abs(kv[1]))[:6];rest=float(sum(v for k,v in shap.items() if k not in dict(top)))
val=lambda k:(str(one[k].iloc[0]) if str(one[k].dtype) in ('category','object') else float(one[k].iloc[0]))
r['explain']={'base_wape_2025':base_wape,'permutation_wape_increase':perm,'n_inputs':len(cols),'n_train_rows':int(len(tr)),'n_series_trained':int(tr.series_id.nunique()),
              'one':{'series_id':'P010_store','name':'Plush Bear','week':'2025-12-01','horizon':1,'discount_pct':float(one.discount_pct.iloc[0]),
                     'prediction':float(one.forecast.iloc[0]),'actual':float(actual['P010_store'].loc[pd.Timestamp('2025-12-01')]),
                     'base_units':float(np.exp(expected)),'contributions':[{'feature':k,'value':val(k),'log':float(v),'factor':float(np.exp(v))} for k,v in top],
                     'other_factor':float(np.exp(rest))}}
# --- Added 29.09.2026, second pass: one series seen as components and as features; hyperparameter tuning ---
# Classical additive decomposition, written out so every step can be explained in class:
# trend = centred 2x52 moving average (ends extended by a straight line fitted to the nearest 26 trend values),
# season = for each week of the year, the MEDIAN of the three years' deviations from the trend (a promotion spike in
# one year does not become "season"), centred to sum to zero over a year; remainder = sales - trend - season.
# No smoothing across neighbouring weeks: it would wrap around the year and lend the Christmas peak to the first week of January.
def _decompose(y,period=52):
    t=y.rolling(period,center=True).mean().rolling(2,center=True).mean().shift(-1).to_numpy().copy()
    idx=np.arange(len(y));ok=np.flatnonzero(~np.isnan(t));f0,f1=ok[0],ok[-1]
    a=np.polyfit(idx[f0:f0+26],t[f0:f0+26],1);t[:f0]=np.polyval(a,idx[:f0])
    b=np.polyfit(idx[f1-25:f1+1],t[f1-25:f1+1],1);t[f1+1:]=np.polyval(b,idx[f1+1:])
    det=y.to_numpy()-t;wk=idx%period
    sw=np.array([np.median(det[wk==k]) for k in range(period)])
    sw=sw-sw.mean();se=sw[wk]
    return t,se,y.to_numpy()-t-se
pb=actual['P010_store'].astype(float);_t,_s,_r=_decompose(pb)
pbw=raw[raw.series_id=='P010_store'].sort_values('week_start');_pr=pbw.promo_flag.to_numpy()==1
r['decomposition']={'series_id':'P010_store','name':'Plush Bear','channel':'store',
                    'method':'classical additive: centred 52-week moving average; median weekly season over the three years',
                    'weeks':[str(w.date()) for w in pb.index],'observed':pb.tolist(),'trend':[float(v) for v in _t],
                    'seasonal':[float(v) for v in _s],'remainder':[float(v) for v in _r],'promo':pbw.promo_flag.astype(int).tolist(),
                    'season_peak_week':str(pb.index[int(np.argmax(_s[:52]))].date())[5:],
                    'remainder_promo_mean':float(_r[_pr].mean()),'remainder_normal_mean':float(_r[~_pr].mean())}
ti=pos[pd.Timestamp('2025-12-01')]
frow=features[(features.series_id=='P010_store')&(features.horizon==1)&(features.week_start==pd.Timestamp('2025-12-01'))].iloc[0]
win=actual['P010_store'].iloc[ti-55:ti+1]
r['feature_row']={'series_id':'P010_store','name':'Plush Bear','channel':'store','target_week':'2025-12-01','horizon':1,
                  'origin_week':str(weeks[ti-1].date()),'values':{k:(frow[k] if isinstance(frow[k],str) else float(frow[k])) for k in cols},
                  'actual':float(actual['P010_store'].iloc[ti]),
                  'window':{'weeks':[str(w.date()) for w in win.index],'units':win.astype(float).tolist()}}
# Tuning: train on target weeks t < V0, choose settings on the validation weeks V0 <= t < V1, never look at 2025.
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
V0,V1=78,104
ftr=features[(features.t<V0)&(features.horizon<=4)];fva=features[(features.t>=V0)&(features.t<V1)&(features.horizon<=4)]
DEFAULTS=rf.lgbm_params()
SPACE={'learning_rate':(0.01,0.2,'log'),'n_estimators':(100,1200,'int'),'num_leaves':(8,128,'intlog'),
       'min_child_samples':(5,100,'intlog'),'subsample':(0.5,1.0,'float'),'colsample_bytree':(0.4,1.0,'float'),
       'tweedie_variance_power':(1.05,1.8,'float')}
def _suggest(trial):
    out={}
    for k,(lo,hi,kind) in SPACE.items():
        if kind=='log': out[k]=trial.suggest_float(k,lo,hi,log=True)
        elif kind=='int': out[k]=trial.suggest_int(k,lo,hi,step=50)
        elif kind=='intlog': out[k]=trial.suggest_int(k,lo,hi,log=True)
        else: out[k]=trial.suggest_float(k,lo,hi)
    return out
def _fit_score(params,train,test):
    m=lgb.LGBMRegressor(**rf.lgbm_params(n_jobs=4,**params)).fit(train[cols],train.units)
    pr=np.clip(m.predict(test[cols]),0,None);return float(np.abs(pr-test.units).sum()/test.units.sum()*100)
study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
study.enqueue_trial({k:DEFAULTS[k] for k in SPACE})          # trial 0 is the setting the course uses
study.optimize(lambda trial:_fit_score(_suggest(trial),ftr,fva),n_trials=60)
ftr_all=features[(features.t<104)&(features.horizon<=4)];fte=features[(features.t>=104)&(features.horizon<=4)]
vals=[t.value for t in study.trials]
try:
    imp={k:float(v) for k,v in optuna.importance.get_param_importances(study).items()}
except Exception:
    imp={}
r['tuning']={'n_trials':len(vals),'sampler':'TPE (tree-structured Parzen estimator), seed 42','metric':'WAPE, horizons 1 to 4',
             'space':{k:[lo,hi,kind] for k,(lo,hi,kind) in SPACE.items()},
             'split':{'train':f"{weeks[52].date()} to {weeks[V0-1].date()}",'validation':f"{weeks[V0].date()} to {weeks[V1-1].date()}",
                      'test':f"{weeks[104].date()} to {weeks[-1].date()}",'n_train_rows':int(len(ftr)),'n_val_rows':int(len(fva))},
             'default':{'params':{k:DEFAULTS[k] for k in SPACE},'val_wape':vals[0],'test_wape':_fit_score({k:DEFAULTS[k] for k in SPACE},ftr_all,fte)},
             'best':{'params':study.best_params,'val_wape':float(study.best_value),'trial':int(study.best_trial.number),
                     'test_wape':_fit_score(study.best_params,ftr_all,fte)},
             'trial_wape':[float(v) for v in vals],'best_so_far':[float(v) for v in np.minimum.accumulate(vals)],'importance':imp}
print('Tuning done: default val',round(vals[0],2),'best val',round(study.best_value,2),'default test',round(r['tuning']['default']['test_wape'],2),'best test',round(r['tuning']['best']['test_wape'],2),flush=True)
# --- Facts quoted on the slides after the 29.09.2026 review, computed here so every sentence has a source ---
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
_al={}
for sid in wide.columns:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _al[sid]=float(SimpleExpSmoothing(wide[sid].iloc[:152].to_numpy(float),initialization_method='estimated').fit().params['smoothing_level'])
_wk=raw.drop_duplicates('week_start').week_start
_mon=raw.groupby([raw.week_start.dt.to_period('M'),'channel']).units.sum().unstack()
_nw=_wk.groupby(_wk.dt.to_period('M')).nunique()
_yr=raw.groupby([raw.week_start.dt.year,'channel']).units.sum().unstack();_nwy=_wk.groupby(_wk.dt.year).nunique()
r['facts']={'ses_alpha':{'near_zero_below_0_02':int(sum(v<0.02 for v in _al.values())),'n_series':len(_al),
                         'median':float(np.median(list(_al.values()))),'P010_store':_al['P010_store'],'P011_store':_al['P011_store'],
                         'fitted_on':'weeks up to the last backtest origin (t < 152)'},
            'promo_share':float(raw.promo_flag.mean()),'weeks_per_year':{str(k):int(v) for k,v in _nwy.items()},
            'weekly_avg_by_month':{'months':[str(m) for m in _mon.index],'store':[float(v) for v in (_mon.store/_nw.values)],
                                   'online':[float(v) for v in (_mon.online/_nw.values)],'weeks':[int(v) for v in _nw.values]},
            'weekly_avg_by_year':{str(y):{'store':float(_yr.loc[y,'store']/_nwy[y]),'online':float(_yr.loc[y,'online']/_nwy[y]),'weeks':int(_nwy[y])} for y in _yr.index}}
(OUT/'results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
import classroom as c
c.load(OUT)
r['order_example']=c.order_example();r['stock']={p:{str(t):c.simulate(p,t) for t in [50,80,90,95,99]} for p in ['Plush Bear, store','Puzzle 1000, store','Vacuum Filter, store']}
r['forecast_value']={m:c.simulate('Plush Bear, store',95,m) for m in c.METHODS}
_rb={}
for _p in ['Plush Bear, store','Puzzle 1000, store','Vacuum Filter, store']:
    _g,_a,_f,_sig,_L,_half=c.profile(_p,'LightGBM in global mode')
    _test=slice(_half,len(_a));_e=_f[_test,0]-_a[_test]
    _fp=_f.sum(1);_ap=np.array([_a[t:t+_L+1].sum() if t+_L+1<=len(_a) else np.nan for t in range(len(_a))])
    _ok=(~np.isnan(_ap))&(np.arange(len(_a))>=_half)
    _rb[_p]={'one_week_bias_pct_test':float(_e.sum()/_a[_test].sum()*100),
             'protection_bias_pct_test':float((_fp[_ok]-_ap[_ok]).sum()/_ap[_ok].sum()*100),
             'error_sd_calibration':float(_sig),'error_sd_test':float(np.std(_a[_test]-_f[_test,0],ddof=1))}
r['facts']['replay_bias']=_rb
r['christmas']={str(s):c.christmas_order(s) for s in [0,50,95]}
r['cases']={p:{w:c.score_table(c.comparison_data(p,w)).to_dict('index') for w in c.WINDOWS} for p in c.PRODUCTS}
x=raw.assign(month=raw.week_start.dt.month)
r['season']={cat:{str(int(m)):float(v) for m,v in (g.groupby('month').units.mean()/g.units.mean()).items()} for cat,g in x.groupby('category')}
r['yearly']=raw.assign(year=raw.week_start.dt.year).groupby(['year','channel']).units.sum().unstack().to_dict('index')
r['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/retail_fc.py',ROOT/'src/classroom.py',ROOT/'analysis/build_classroom.py',ROOT/'data/demand_weekly.csv',ROOT/'data/products.csv']}
def finite(obj):
    if isinstance(obj, dict): return {k: finite(v) for k,v in obj.items()}
    if isinstance(obj, list): return [finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj
(OUT/'results.json').write_text(json.dumps(finite(r),indent=2,allow_nan=False),encoding='utf-8')
print('Saved classroom evidence',OUT,flush=True)

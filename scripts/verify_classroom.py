"""Check evidence coherence, causal feature construction and student interactions."""
from pathlib import Path
import os,sys,json,argparse,tempfile,shutil,contextlib,io,concurrent.futures
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import classroom as c
import retail_fc as rf
parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
c.load(ROOT/'data/classroom');r=c.R
# Whole-unit order arithmetic and conservation, independently checked from the replay trace.
e=c.order_example();assert e['forecast_total']==1000 and e['stock_target']==1148 and e['order']==348
for p in ['Plush Bear, store','Puzzle 1000, store','Vacuum Filter, store']:
 for t in [50,80,90,95,99]:
  observed,trace=c.simulate(p,t,details=True);saved=r['stock'][p][str(t)]
  for k in saved:assert np.isclose(saved[k],observed[k]),(p,t,k)
  assert 0<=observed['fill_rate']<=100 and 0<=observed['stockout_weeks']<=26
  assert all(x['order']>=0 and x['order']==int(x['order']) and 0<=x['lost']<=x['actual'] and x['stock']>=0 for x in trace)
  assert np.isclose(observed['fill_rate'],100*(sum(x['actual']-x['lost'] for x in trace)/sum(x['actual'] for x in trace)))
for m,g in c.B[c.B.scoreable==1].groupby('method'):
 assert np.isclose(100*np.abs(g.forecast-g.actual).sum()/g.actual.sum(),r['backtest'][m]['wape'])
for p in c.PRODUCTS:
 for w in c.WINDOWS:
  q=c.comparison_data(p,w);assert len(q)==4*len(c.COMPARE) and set(q.method)==set(c.COMPARE)
  s=c.score_table(q);assert np.isfinite(s['Average miss, units']).all()
  # No-demand windows remain valid comparisons in units, not fake finite percentages.
  if q.actual.sum()==0:assert s['Total miss, % of demand'].isna().all()
for s in range(0,96,5):assert np.isfinite(c.christmas_order(s)['order'])
assert c.christmas_order(0)['order']<=c.christmas_order(50)['order']<=c.christmas_order(95)['order']
for invalid in [100,-1,float('nan')]:
 try:c.christmas_order(invalid)
 except ValueError:pass
 else:raise AssertionError('Invalid recovery accepted')
# Perturb unknown target/future sales: history features for the same origin must remain unchanged.
d=rf.load_data(ROOT/'data')['demand'];o=pd.Timestamp('2025-06-16');changed=d.copy();changed.loc[changed.week_start>=o,'units']+=10000
for h in range(1,8):
 original=rf.make_features(d,h);modified=rf.make_features(changed,h);target=o+pd.Timedelta(weeks=h-1)
 a=original[original.week_start==target][rf.FEATURE_COLS].reset_index(drop=True)
 b=modified[modified.week_start==target][rf.FEATURE_COLS].reset_index(drop=True)
 pd.testing.assert_frame_equal(a,b)
# Controls notify their callbacks when changed. Capture normal notebook display noise.
with contextlib.redirect_stdout(io.StringIO()):
 w=c.stock_widget();w.children[0].value='Puzzle 1000, store';w.children[1].value=99
 assert w.result is None
 z=c.christmas_widget();z.children[0].value=95
 f=c.forecast_widget();f.children[0].value='Vacuum Filter, store';f.children[1].value='Summer'
import matplotlib.pyplot as plt
plt.close('all')
print('PASS: evidence, order arithmetic, finite endpoints, zero windows, feature causality and control callbacks.')
if args.execute:
 os.environ['T2_NO_WIDGETS']='1'   # headless kernels can stall on widget display; the callbacks were exercised above
 import nbformat
 from nbclient import NotebookClient
 def execute(name):
  dest=Path(tempfile.mkdtemp(prefix='t2-student-'));path=ROOT/'notebooks'/name;shutil.copy2(path,dest/name)
  nb=nbformat.read(dest/name,4)
  NotebookClient(nb,timeout=400,kernel_name='python3',resources={'metadata':{'path':str(dest)}},
                 on_cell_start=lambda **kw: print(name, 'cell', kw['cell_index'], flush=True)).execute()
  errors=[o for cell in nb.cells for o in cell.get('outputs',[]) if o.output_type=='error']
  assert not errors
  warnings=[o for cell in nb.cells for o in cell.get('outputs',[]) if o.get('name')=='stderr']
  assert not warnings, warnings
  if name.startswith('1_'):
   assert sum('image/png' in o.get('data',{}) for cell in nb.cells for o in cell.get('outputs',[]))>=3
  nbformat.write(nb,path)
  return name,len([cell for cell in nb.cells if cell.cell_type=='code']),str(dest)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(execute,['1_see_and_forecast.ipynb','2_forecast_to_stock.ipynb']):print('PASS clean notebook:',item)

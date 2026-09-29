import argparse,json
from pathlib import Path
from .data import load_m5
from .forecast import train_evaluate,Forecaster

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default='artifacts');p.add_argument('--store',default='CA_1');p.add_argument('--max-products',type=int,default=120);p.add_argument('--history-days',type=int,default=730)
    a=p.parse_args();print('Loading M5...',flush=True)
    panel,calendar,audit=load_m5(a.data_dir,a.store,a.max_products,a.history_days)
    print(f'Training {audit["products"]} products, {len(panel)} daily records',flush=True)
    report=train_evaluate(panel,calendar,a.output,audit)
    out=Path(a.output);last=panel.date.max();product=sorted(panel.product_id.unique())[0]
    hist=panel[(panel.product_id==product)&(panel.date>=last-__import__('pandas').Timedelta(days=34))]
    future=calendar[calendar.date.between(last+__import__('pandas').Timedelta(days=1),last+__import__('pandas').Timedelta(days=2))]
    req={'product_id':product,'as_of':str(last.date()),'history':[{'date':str(r.date.date()),'sales':float(r.sales),'status':r.status} for r in hist.itertuples()],
         'calendar':[{'date':str(r.date.date()),'is_holiday':int(r.is_holiday),'has_event':int(r.has_event)} for r in future.itertuples()], 'country':'US'}
    (out/'example_forecast_request.json').write_text(json.dumps(req,indent=2))
    prediction,_=Forecaster(out).predict(hist,calendar,last)
    (out/'example_forecast_result.json').write_text(json.dumps(prediction,indent=2))
    print(json.dumps(report['horizons'],indent=2),flush=True)
if __name__=='__main__':main()

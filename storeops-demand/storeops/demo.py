"""SRS synthetic P001 fixture; never included in M5 training."""
from pathlib import Path
import json
import pandas as pd
from .orders import OrderInput,calculate_order

def write_demo(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame({'date':pd.date_range('2026-03-08',periods=8),'store_id':'S01','product_id':'P001',
                     'sales':[12,13,14,15,12,14,15,17],'status':'observed','data_label':'synthetic'})
    df.to_csv(output/'P001_synthetic.csv',index=False)
    example=calculate_order(OrderInput(product_id='P001',target_date='2026-03-15',d1=16,d2=12,on_hand=15,data_label='synthetic',forecast_source='mock'))
    example['fixture_notes']={'mean7_rounded':round(df.sales.iloc[:7].mean(),1),'same_weekday_last_week':12,'actual_2026_03_15':17,
                             'prediction_is_mock':True,'not_in_training':True,'korean_holiday_data':'not integrated; no KR model claim'}
    (output/'P001_order_example.json').write_text(json.dumps(example,ensure_ascii=False,indent=2))
if __name__=='__main__':write_demo('examples')

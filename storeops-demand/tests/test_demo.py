import json
import pandas as pd
from storeops.demo import write_demo

def test_srs_synthetic_fixture(tmp_path):
    write_demo(tmp_path);f=pd.read_csv(tmp_path/'P001_synthetic.csv');r=json.loads((tmp_path/'P001_order_example.json').read_text())
    assert set(f.data_label)=={'synthetic'}
    assert round(f.sales.iloc[:7].mean(),1)==13.6 and f.sales.iloc[0]==12 and f.sales.iloc[-1]==17
    assert r['forecast_source']=='mock' and r['recommended']=={'qty':24,'boxes':4}

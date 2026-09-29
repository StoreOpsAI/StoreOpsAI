from datetime import date
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from storeops.data import normalize_sales
from storeops.forecast import feature_table, FEATURES, train_evaluate, Forecaster
from storeops.orders import OrderInput,calculate_order

def order(**kwargs):
    values=dict(product_id='P001',target_date='2026-03-15',d1=16,d2=12,on_hand=15)
    values.update(kwargs);return OrderInput(**values)

def test_srs_example_and_rounding():
    r=calculate_order(order());assert r['need']==23 and r['recommended']=={'qty':24,'boxes':4}
    assert r['shortage_before_arrival']==13
    assert calculate_order(order(on_hand=100))['recommended']['qty']==0
    assert calculate_order(order(on_hand=14))['recommended']['qty']==24
    assert calculate_order(order(d1=.1,d2=.2,on_hand=0,safety_stock=0,pack_size=1))['recommended']['qty']==1

def test_shortage_receipt_timing():
    assert calculate_order(order(incoming=20,incoming_day=1))['shortage_before_arrival']==0
    assert calculate_order(order(incoming=20,incoming_day=2))['shortage_before_arrival']==1
    assert calculate_order(order(arrival_after_days=1))['shortage_before_arrival']==1
    assert calculate_order(order(arrival_after_days=0))['shortage_before_arrival']==0

@pytest.mark.parametrize('kwargs',[{'d1':-1},{'d1':float('nan')},{'d1':float('inf')},{'pack_size':0},{'on_hand':1.5},{'on_hand':True},{'incoming':-2},{'arrival_after_days':3}])
def test_bad_order_inputs(kwargs):
    with pytest.raises(ValidationError):order(**kwargs)

def test_missing_not_zero():
    frame=pd.DataFrame([dict(store_id='S',product_id='P',date='2026-01-01',sales=0,status='observed'),
                        dict(store_id='S',product_id='P',date='2026-01-03',sales=0,status='stockout'),
                        dict(store_id='S',product_id='P',date='2026-01-04',sales=0,status='closed')])
    panel,audit=normalize_sales(frame)
    assert audit['inserted_missing_days']==1 and panel.iloc[0].demand==0
    assert panel.iloc[1:].demand.isna().all()
    assert pd.isna(panel.iloc[1].sales)
    assert list(panel.status)==['observed','missing','stockout','closed']
    with pytest.raises(ValueError):normalize_sales(pd.concat([frame,frame]))

def sample():
    dates=pd.date_range('2025-01-01',periods=100)
    raw=pd.DataFrame({'date':dates,'store_id':'S','product_id':'P','sales':np.arange(100)%7+10})
    panel,audit=normalize_sales(raw)
    cal=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=102),'is_holiday':0,'has_event':0})
    return raw,panel,cal,audit

@pytest.mark.parametrize('h',[1,2])
def test_features_have_no_future_sales(h):
    raw,panel,cal,_=sample();origin=pd.Timestamp('2025-02-15')
    original=feature_table(panel,cal,h);raw.loc[raw.date>origin,'sales']=99999
    changed,_=normalize_sales(raw);new=feature_table(changed,cal,h)
    a=original[original.origin==origin];b=new[new.origin==origin]
    np.testing.assert_equal(a[FEATURES].to_numpy(),b[FEATURES].to_numpy())
    assert a.iloc[0].same_weekday_last_week==panel.loc[panel.date==origin+pd.Timedelta(days=h-7),'sales'].iloc[0]
    assert a.iloc[0].mean7==panel.loc[panel.date.between(origin-pd.Timedelta(days=6),origin),'sales'].mean()

def test_train_save_load_and_bounds(tmp_path):
    raw,panel,cal,audit=sample();report=train_evaluate(panel,cal,tmp_path,audit,validation_days=14)
    f=Forecaster(tmp_path);pred,_=f.predict(raw,cal,panel.date.max())
    assert pred['total']==pred['d1']+pred['d2'] and min(pred.values())>=0
    predictions=pd.read_csv(tmp_path/'validation_predictions.csv')
    assert (pd.to_datetime(predictions.target_date)>pd.Timestamp(report['train_target_through'])).all()
    assert predictions.groupby('horizon').size().to_dict()=={1:13,2:13}
    with pytest.raises(ValueError):f.predict(raw,cal.iloc[:1],panel.date.max())

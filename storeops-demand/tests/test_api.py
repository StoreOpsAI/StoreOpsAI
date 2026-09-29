import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from storeops.api import create_app

USERS={'a':{'store_id':'S01','actor':'owner-01','role':'owner'},'b':{'store_id':'S02','actor':'owner-02','role':'owner'},'agent':{'store_id':'S01','actor':'agent','role':'reader'}}
AUTH={'Authorization':'Bearer a'}
INPUT={'product_id':'P001','target_date':'2026-03-15','d1':16,'d2':12,'on_hand':15,'incoming':0,'data_label':'synthetic','forecast_source':'mock'}
@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path/'db.sqlite',principals=USERS))
def create(client,key='one'):
    r=client.post('/api/orders/drafts',headers={**AUTH,'Idempotency-Key':key},json=INPUT);assert r.status_code==201;return r.json()

def test_approve_contract_and_repeat(client):
    d=create(client);id=d['draft_id']
    r=client.post(f'/api/orders/drafts/{id}/approve',headers=AUTH,json={'qty':18});assert r.status_code==200
    a=r.json();assert a['status']=='draft_saved' and a['approved']['boxes']==3
    assert a['approved']['by']=='owner-01' and datetime.fromisoformat(a['approved']['at']).utcoffset().total_seconds()==9*3600
    for field in ['on_hand','incoming','sent_to_supplier','forecast','recommended']:assert a[field]==d[field]
    assert client.post(f'/api/orders/drafts/{id}/approve',headers=AUTH,json={'qty':18}).json()==a
    assert client.post(f'/api/orders/drafts/{id}/approve',headers=AUTH,json={'qty':24}).status_code==409
    with client.app.state.repo.connect() as con:assert con.execute('SELECT COUNT(*) FROM order_approvals').fetchone()[0]==1

def test_tenant_and_role_isolation(client):
    d=create(client)
    assert client.get('/api/orders/drafts').status_code==401
    assert client.get('/api/orders/drafts',headers={'Authorization':'Bearer b'}).json()==[]
    for method,suffix,data in [('get','',None),('post','/approve',{'qty':18})]:
        r=client.request(method,'/api/orders/drafts/'+d['draft_id']+suffix,headers={'Authorization':'Bearer b'},json=data);assert r.status_code==404
    assert client.post('/api/orders/drafts/'+d['draft_id']+'/approve',headers={'Authorization':'Bearer agent'},json={'qty':18}).status_code==403

@pytest.mark.parametrize('qty,status',[(17,409),(-6,422),(1.2,422),(True,422)])
def test_invalid_approval(client,qty,status):
    d=create(client);assert client.post('/api/orders/drafts/'+d['draft_id']+'/approve',headers=AUTH,json={'qty':qty}).status_code==status
    assert client.get('/api/orders/drafts/'+d['draft_id'],headers=AUTH).json()['status']=='recommended'

def test_idempotency_and_sql_persistence(client):
    first=create(client);assert create(client)['draft_id']==first['draft_id']
    changed={**INPUT,'on_hand':3}
    assert client.post('/api/orders/drafts',headers={**AUTH,'Idempotency-Key':'one'},json=changed).status_code==409
    new=TestClient(create_app(client.app.state.repo.path,principals=USERS))
    assert len(new.get('/api/orders/drafts',headers=AUTH).json())==1

def test_concurrent_approval_once(client):
    d=create(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(lambda _:client.post('/api/orders/drafts/'+d['draft_id']+'/approve',headers=AUTH,json={'qty':18}),range(2)))
    assert all(r.status_code==200 for r in responses)
    with client.app.state.repo.connect() as con:assert con.execute('SELECT COUNT(*) FROM order_approvals').fetchone()[0]==1

def test_real_model_api_pipeline(client):
    path=Path(__file__).parents[1]/'artifacts/example_forecast_request.json'
    if not path.exists():pytest.skip('Run M5 training first')
    req=json.loads(path.read_text())
    r=client.post('/api/forecasts',headers=AUTH,json=req);assert r.status_code==200,r.text
    body={'demand':req,'on_hand':15,'incoming':0}
    r=client.post('/api/orders/forecast-draft',headers={**AUTH,'Idempotency-Key':'real'},json=body);assert r.status_code==201,r.text
    assert r.json()['forecast_source']=='xgboost' and r.json()['sent_to_supplier'] is False
    id=r.json()['draft_id'];qty=r.json()['recommended']['qty']
    assert client.post('/api/orders/drafts/'+id+'/approve',headers=AUTH,json={'qty':qty}).status_code==200
    req['country']='KR';assert client.post('/api/forecasts',headers=AUTH,json=req).status_code==422

def test_model_absence_and_future_rejection(tmp_path):
    client=TestClient(create_app(tmp_path/'db',tmp_path/'no-models',principals=USERS))
    req={'product_id':'P','as_of':'2026-01-07','history':[{'date':f'2026-01-{i:02}','sales':1} for i in range(1,8)],'calendar':[{'date':'2026-01-08','is_holiday':0},{'date':'2026-01-09','is_holiday':0}]}
    assert client.post('/api/forecasts',headers=AUTH,json=req).status_code==503

def test_future_history_and_historical_asof_rejected(client):
    path=Path(__file__).parents[1]/'artifacts/example_forecast_request.json'
    if not path.exists():pytest.skip('Run M5 training first')
    req=json.loads(path.read_text());req['history'].append({'date':'2099-01-01','sales':1})
    assert client.post('/api/forecasts',headers=AUTH,json=req).status_code==422
    req=json.loads(path.read_text());req['as_of']='2016-05-21'
    assert client.post('/api/forecasts',headers=AUTH,json=req).status_code==422

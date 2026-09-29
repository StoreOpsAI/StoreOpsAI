import json,os
from pathlib import Path
from datetime import date,timedelta
from typing import Literal
import pandas as pd
from fastapi import FastAPI,Depends,Header,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import Field,StrictInt
from .orders import StrictModel,OrderInput,ApprovalInput,calculate_order
from .repository import Repository
from .forecast import Forecaster

ROOT=Path(__file__).resolve().parents[1]
class DailySale(StrictModel):
    date: date
    sales: float | None = Field(default=None,ge=0)
    status: Literal['observed','closed','stockout','missing']='observed'
class CalendarDay(StrictModel):
    date: date
    is_holiday: StrictInt = Field(ge=0,le=1)
    has_event: StrictInt = Field(default=0,ge=0,le=1)
class ForecastInput(StrictModel):
    product_id: str=Field(min_length=1,max_length=100)
    as_of: date
    history: list[DailySale]=Field(min_length=7,max_length=2000)
    calendar: list[CalendarDay]=Field(min_length=2,max_length=2000)
    country: Literal['US','KR']='US'
class PredictOrderInput(StrictModel):
    demand: ForecastInput
    on_hand: StrictInt=Field(ge=0)
    incoming: StrictInt=Field(default=0,ge=0)
    incoming_day: StrictInt=Field(default=1,ge=1,le=2)
    arrival_after_days: StrictInt=Field(default=2,ge=0,le=2)


def create_app(db_path=None,model_dir=None,principals=None):
    config=json.loads((ROOT/'config.json').read_text())
    repo=Repository(db_path or os.getenv('STOREOPS_DB',str(ROOT/'runtime/orders.sqlite3')))
    models=Path(model_dir or os.getenv('STOREOPS_MODELS',str(ROOT/'artifacts')))
    if principals is None:
        principals=json.loads(os.getenv('STOREOPS_TOKENS_JSON','{}'))
        if os.getenv('STOREOPS_DEMO')=='1':principals={'demo-owner-s01':{'store_id':'S01','actor':'owner-01','role':'owner'}}
    app=FastAPI(title='StoreOps 수요예측·발주 초안',version='0.1.0');app.state.repo=repo
    def owner(authorization: str | None=Header(default=None)):
        token=authorization.removeprefix('Bearer ') if authorization and authorization.startswith('Bearer ') else None
        if token not in principals:raise HTTPException(401,'Bearer token required')
        p=principals[token]
        if p.get('role')!='owner':raise HTTPException(403,'Owner access required')
        return p
    def forecast(req,principal):
        if not (models/'d1.json').is_file() or not (models/'d2.json').is_file():raise HTTPException(503,'Train models first')
        report_path=models/'evaluation.json'
        if report_path.exists() and req.as_of < date.fromisoformat(json.loads(report_path.read_text())['last_observed']):
            raise HTTPException(422,'as_of predates serving model training cutoff; use chronological backtest instead')
        if req.country!='US':raise HTTPException(422,'This model was trained on US M5. KR requires separate training and local holiday data.')
        if any(r.date>req.as_of for r in req.history):raise HTTPException(422,'Future observations are not accepted')
        frame=pd.DataFrame([r.model_dump() for r in req.history]);frame['store_id']=principal['store_id'];frame['product_id']=req.product_id
        cal=pd.DataFrame([r.model_dump() for r in req.calendar])
        try:
            result,audit=Forecaster(models).predict(frame,cal,req.as_of)
        except ValueError as e:raise HTTPException(422,str(e)) from e
        return {'product_id':req.product_id,'store_id':principal['store_id'],'as_of':req.as_of.isoformat(),
                'forecast':result,'model':'xgboost_m5_us','data_label':'real','quality':audit}
    @app.get('/',response_class=HTMLResponse)
    def screen():return (ROOT/'storeops/ui.html').read_text()
    @app.get('/health')
    def health():return {'status':'ok','models_ready':all((models/f'd{h}.json').exists() for h in [1,2])}
    @app.post('/api/forecasts')
    def predict(req:ForecastInput,p=Depends(owner)):return forecast(req,p)
    def store_draft(req,p,key):
        try:return repo.create(p['store_id'],p['actor'],calculate_order(req,config['safety_stock'],config['pack_size']),key)
        except ValueError as e:raise HTTPException(409,str(e)) from e
    @app.post('/api/orders/drafts',status_code=201)
    def draft(req:OrderInput,p=Depends(owner),idempotency_key:str=Header(min_length=1,max_length=120)):
        return store_draft(req,p,idempotency_key)
    @app.post('/api/orders/forecast-draft',status_code=201)
    def forecast_draft(req:PredictOrderInput,p=Depends(owner),idempotency_key:str=Header(min_length=1,max_length=120)):
        f=forecast(req.demand,p)['forecast']
        order=OrderInput(product_id=req.demand.product_id,target_date=req.demand.as_of+timedelta(days=1),d1=f['d1'],d2=f['d2'],on_hand=req.on_hand,incoming=req.incoming,incoming_day=req.incoming_day,arrival_after_days=req.arrival_after_days,forecast_source='xgboost')
        return store_draft(order,p,idempotency_key)
    @app.get('/api/orders/drafts')
    def drafts(p=Depends(owner)):return repo.list(p['store_id'])
    @app.get('/api/orders/drafts/{draft_id}')
    def get(draft_id:str,p=Depends(owner)):
        try:return repo.get(p['store_id'],draft_id)
        except KeyError as e:raise HTTPException(404,'Draft not found') from e
    @app.post('/api/orders/drafts/{draft_id}/approve')
    def approve(draft_id:str,req:ApprovalInput,p=Depends(owner)):
        try:return repo.approve(p['store_id'],p['actor'],draft_id,req.qty)
        except KeyError as e:raise HTTPException(404,'Draft not found') from e
        except ValueError as e:raise HTTPException(409,str(e)) from e
    @app.get('/api/demo/forecast-input')
    def example(p=Depends(owner)):
        file=models/'example_forecast_request.json'
        if not file.exists():raise HTTPException(503,'Train models first')
        return json.loads(file.read_text())
    return app

app=create_app()

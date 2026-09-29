from decimal import Decimal, ROUND_CEILING
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class OrderInput(StrictModel):
    product_id: str = Field(min_length=1, max_length=100)
    target_date: date
    d1: float = Field(ge=0)
    d2: float = Field(ge=0)
    on_hand: StrictInt = Field(ge=0)
    incoming: StrictInt = Field(default=0, ge=0)
    incoming_day: StrictInt = Field(default=1, ge=1, le=2)
    arrival_after_days: StrictInt = Field(default=2, ge=0, le=2)
    safety_stock: StrictInt | None = Field(default=None, ge=0)
    pack_size: StrictInt | None = Field(default=None, ge=1)
    data_label: Literal['real', 'synthetic'] = 'real'
    forecast_source: Literal['provided', 'mock', 'xgboost'] = 'provided'

class ApprovalInput(StrictModel):
    qty: StrictInt = Field(ge=0)

def calculate_order(request: OrderInput, default_safety=10, default_pack=6):
    safety = request.safety_stock if request.safety_stock is not None else default_safety
    pack = request.pack_size if request.pack_size is not None else default_pack
    if safety < 0 or pack < 1:
        raise ValueError('Invalid configured stock/pack size')
    d1,d2=Decimal(str(request.d1)),Decimal(str(request.d2))
    total=d1+d2
    need=max(Decimal(0),total+safety-request.on_hand-request.incoming)
    boxes=int((need/pack).to_integral_value(rounding=ROUND_CEILING))
    # Proposed policy: maximum cumulative deficit before new order arrival.
    # Existing incoming enters at START of incoming_day. New order enters after lead days.
    balance=Decimal(request.on_hand);shortage=Decimal(0)
    for day,demand in enumerate([d1,d2],1):
        if day>request.arrival_after_days:break
        if day==request.incoming_day:balance+=request.incoming
        balance-=demand
        shortage=max(shortage,-balance)
    return {'product_id':request.product_id,'target_date':request.target_date.isoformat(),
            'forecast':{'d1':float(d1),'d2':float(d2),'total':float(total)},
            'safety_stock':safety,'on_hand':request.on_hand,'incoming':request.incoming,
            'incoming_day':request.incoming_day,'arrival_after_days':request.arrival_after_days,
            'need':float(need),'pack_size':pack,'recommended':{'qty':boxes*pack,'boxes':boxes},
            'shortage_before_arrival':float(max(0,shortage)),
            'shortage_policy':'Maximum cumulative deficit before arrival; existing receipts at start of day',
            'data_label':request.data_label,'forecast_source':request.forecast_source,
            'approved':None,'status':'recommended','sent_to_supplier':False}

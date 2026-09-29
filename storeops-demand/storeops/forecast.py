from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .data import normalize_sales

FEATURES = ['mean7', 'same_weekday_last_week', 'weekday', 'is_holiday', 'mean28', 'month', 'has_event']

def feature_table(panel, calendar, horizon):
    if horizon not in (1, 2):
        raise ValueError('Only d1/d2 supported')
    cal = calendar.copy()
    cal['date'] = pd.to_datetime(cal.date).dt.normalize()
    if cal.date.duplicated().any() or not {'is_holiday', 'has_event'} <= set(cal):
        raise ValueError('Calendar must have unique dates and explicit holiday/event flags')
    parts = []
    for (store, product), g in panel.groupby(['store_id', 'product_id'], sort=True):
        g = g.sort_values('date').copy()
        y = g.demand
        f = pd.DataFrame({'origin': g.date, 'target_date': g.date + pd.Timedelta(days=horizon),
                          'mean7': y.rolling(7, min_periods=7).mean(),
                          'mean28': y.rolling(28, min_periods=7).mean(),
                          'same_weekday_last_week': y.shift(7 - horizon), 'target': y.shift(-horizon)})
        f['store_id'], f['product_id'] = store, product
        parts.append(f)
    out = pd.concat(parts, ignore_index=True).merge(cal.rename(columns={'date': 'target_date'}), on='target_date', how='left', validate='many_to_one')
    out['weekday'] = out.target_date.dt.dayofweek
    out['month'] = out.target_date.dt.month
    return out

def new_model():
    return XGBRegressor(n_estimators=180, max_depth=4, learning_rate=.05, subsample=.9,
                        colsample_bytree=.9, objective='reg:squarederror', tree_method='hist',
                        n_jobs=4, random_state=42)

def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'mae': float(np.mean(np.abs(y-pred))), 'rmse': float(np.sqrt(np.mean((y-pred)**2))),
            'wape': float(np.abs(y-pred).sum()/np.abs(y).sum()) if np.abs(y).sum() else None}

def train_evaluate(panel, calendar, output, audit, validation_days=28):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    last = panel.date.max(); cutoff = last - pd.Timedelta(days=validation_days)
    if validation_days < 3:
        raise ValueError('Need at least 3 holdout days')
    report = {'data': audit, 'last_observed': str(last.date()), 'train_target_through': str(cutoff.date()),
              'validation_origin_from': str(cutoff.date()), 'validation_origin_through': str((last-pd.Timedelta(days=2)).date()),
              'protocol': 'Fixed model rolling-origin evaluation. Both horizons predicted at each origin using only observations through that origin. No tuning on holdout; final serving models refit after evaluation.',
              'features': FEATURES, 'horizons': {}}
    predictions = []
    for h in (1, 2):
        f = feature_table(panel, calendar, h)
        usable = f[FEATURES + ['target']].notna().all(axis=1)
        train = f[usable & (f.target_date <= cutoff)]
        test = f[usable & (f.origin >= cutoff) & (f.origin <= last-pd.Timedelta(days=2))]
        if train.empty or test.empty:
            raise ValueError('Insufficient complete history/targets for chronological evaluation')
        model = new_model(); model.fit(train[FEATURES], train.target)
        pred = np.maximum(0, model.predict(test[FEATURES]))
        report['horizons'][f'd{h}'] = {'train_samples': len(train), 'validation_samples': len(test),
                                       'xgboost': metrics(test.target, pred), 'mean7': metrics(test.target, test.mean7)}
        p = test[['store_id', 'product_id', 'origin', 'target_date', 'target', 'mean7']].copy()
        p['horizon'], p['xgboost'] = h, pred; predictions.append(p)
        final = f[usable & (f.target_date <= last)]
        model = new_model(); model.fit(final[FEATURES], final.target)
        model.save_model(output / f'd{h}.json')
    pd.concat(predictions).to_csv(output/'validation_predictions.csv', index=False)
    report['model_type'] = 'Two direct XGBoost regressors; no future sales used for d2'
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

class Forecaster:
    def __init__(self, directory):
        self.models = {}
        for h in (1,2):
            model = XGBRegressor(); model.load_model(Path(directory)/f'd{h}.json'); self.models[h] = model
    def predict(self, history, calendar, as_of):
        date = pd.Timestamp(as_of).normalize()
        panel, audit = normalize_sales(history)
        panel = panel[panel.date <= date]
        if panel.empty or panel[['store_id','product_id']].drop_duplicates().shape[0] != 1:
            raise ValueError('Supply history for exactly one store/product')
        result = {}
        for h in (1,2):
            f = feature_table(panel, calendar, h)
            row = f[f.origin == date]
            if len(row) != 1 or row[FEATURES].isna().any().any():
                raise ValueError('Need recent 7 complete observed days, weekday lag, and future calendar flags')
            result[f'd{h}'] = float(max(0, self.models[h].predict(row[FEATURES])[0]))
        result['total'] = result['d1'] + result['d2']
        return result, audit

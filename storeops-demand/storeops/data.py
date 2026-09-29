from pathlib import Path
import numpy as np
import pandas as pd

STATUSES = {'observed', 'closed', 'stockout', 'missing'}

def normalize_sales(frame):
    """Reindex daily records without inventing demand for missing/closed/stockout days."""
    f = frame.copy()
    required = {'store_id', 'product_id', 'date', 'sales'}
    if not required <= set(f):
        raise ValueError(f'Missing columns: {sorted(required - set(f))}')
    if f.empty or f[['store_id', 'product_id', 'date']].isna().any().any():
        raise ValueError('Empty data or missing identity/date')
    f['date'] = pd.to_datetime(f['date'], errors='raise').dt.normalize()
    if f.duplicated(['store_id', 'product_id', 'date']).any():
        raise ValueError('Duplicate store/product/date')
    f['sales'] = pd.to_numeric(f['sales'], errors='raise')
    present = f.sales.dropna()
    if (~np.isfinite(present)).any() or (present < 0).any():
        raise ValueError('Sales must be finite and nonnegative')
    if 'status' not in f:
        f['status'] = np.where(f.sales.isna(), 'missing', 'observed')
    if not set(f.status) <= STATUSES:
        raise ValueError('status must be observed/closed/stockout/missing')
    if ((f.status == 'observed') & f.sales.isna()).any():
        raise ValueError('Observed sales require a value')
    parts = []
    inserted = 0
    for (store, product), g in f.groupby(['store_id', 'product_id'], sort=True):
        idx = pd.date_range(g.date.min(), g.date.max(), freq='D')
        inserted += len(idx) - len(g)
        g = g.set_index('date').reindex(idx).rename_axis('date').reset_index()
        g['store_id'], g['product_id'] = store, product
        g['status'] = g.status.fillna('missing')
        # Observed zero stays zero. Other statuses are retained, but not demand targets.
        g['demand'] = g.sales.where(g.status == 'observed')
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    audit = {'inserted_missing_days': inserted, 'status_counts': out.status.value_counts().to_dict(),
             'observed_zero_days': int(((out.status == 'observed') & (out.sales == 0)).sum()),
             'policy': 'No zero imputation. Closed/stockout/missing excluded from demand targets and history.'}
    return out, audit

def load_m5(directory, store='CA_1', max_products=120, history_days=730, seed=42):
    directory = Path(directory)
    sales_file = directory / 'sales_train_evaluation.csv'
    if not sales_file.exists():
        sales_file = directory / 'sales_train_validation.csv'
    meta = ['id', 'item_id', 'store_id', 'state_id', 'cat_id']
    header = pd.read_csv(sales_file, nrows=0).columns
    days = sorted((x for x in header if x.startswith('d_')), key=lambda x: int(x[2:]))
    days = days[-(history_days + 28):]
    catalog = pd.read_csv(sales_file, usecols=meta)
    catalog = catalog[catalog.store_id == store]
    if catalog.empty:
        raise ValueError('Unknown M5 store')
    if max_products > 0 and len(catalog) > max_products:
        catalog = catalog.sample(n=max_products, random_state=seed)
    chosen = set(catalog.id)
    chunks = []
    for c in pd.read_csv(sales_file, usecols=meta + days, chunksize=2000):
        sub = c[c.id.isin(chosen)]
        if not sub.empty:
            chunks.append(sub)
    wide = pd.concat(chunks, ignore_index=True)
    cal = pd.read_csv(directory / 'calendar.csv', parse_dates=['date'])
    # M5 National event flag; not Korean holidays, and not all religious/cultural events.
    cal['is_holiday'] = ((cal.event_type_1 == 'National') | (cal.event_type_2 == 'National')).astype(int)
    cal['has_event'] = (cal.event_name_1.notna() | cal.event_name_2.notna()).astype(int)
    long = wide.melt(id_vars=meta, value_vars=days, var_name='d', value_name='sales')
    long = long.merge(cal[['d', 'date']], on='d', validate='many_to_one')
    long = long.rename(columns={'item_id': 'product_id'})
    long['status'] = 'observed'
    panel, audit = normalize_sales(long[['store_id', 'product_id', 'date', 'sales', 'status']])
    audit.update({'source': sales_file.name, 'store': store, 'products': len(wide),
                  'sample_seed': seed, 'country': 'US', 'weather_used': False,
                  'limitation': 'M5 has no daily closure/stockout reason. Observed zeros are not reclassified.',
                  'prices_used': False})
    return panel, cal[['date', 'is_holiday', 'has_event']], audit

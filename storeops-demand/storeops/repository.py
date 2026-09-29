import sqlite3,json,uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

class Repository:
    def __init__(self,path):
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS order_drafts (
              draft_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, request_key TEXT NOT NULL,
              request_json TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(store_id,request_key));
            CREATE TABLE IF NOT EXISTS order_approvals (
              draft_id TEXT PRIMARY KEY REFERENCES order_drafts(draft_id), approved_by TEXT NOT NULL,
              approved_at TEXT NOT NULL, qty INTEGER NOT NULL, boxes INTEGER NOT NULL);
            ''')
    def connect(self):
        con=sqlite3.connect(self.path,timeout=15);con.row_factory=sqlite3.Row
        con.execute('PRAGMA foreign_keys=ON');return con
    def create(self,store,actor,body,request_key):
        canonical=json.dumps(body,sort_keys=True)
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            old=con.execute('SELECT * FROM order_drafts WHERE store_id=? AND request_key=?',(store,request_key)).fetchone()
            if old:
                if old['request_json']!=canonical:raise ValueError('Idempotency key reused with different data')
                return json.loads(old['body'])
            result={**body,'draft_id':str(uuid.uuid4()),'store_id':store,'created_by':actor,'created_at':now()}
            con.execute('INSERT INTO order_drafts VALUES (?,?,?,?,?)',(result['draft_id'],store,request_key,canonical,json.dumps(result)))
            return result
    def get(self,store,draft_id):
        with self.connect() as con:
            r=con.execute('SELECT body FROM order_drafts WHERE store_id=? AND draft_id=?',(store,draft_id)).fetchone()
        if not r:raise KeyError('Draft not found')
        return json.loads(r['body'])
    def list(self,store):
        with self.connect() as con:
            return [json.loads(r['body']) for r in con.execute('SELECT body FROM order_drafts WHERE store_id=? ORDER BY rowid DESC',(store,))]
    def approve(self,store,actor,draft_id,qty):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            r=con.execute('SELECT body FROM order_drafts WHERE store_id=? AND draft_id=?',(store,draft_id)).fetchone()
            if not r:raise KeyError('Draft not found')
            body=json.loads(r['body'])
            if not isinstance(qty,int) or isinstance(qty,bool) or qty<0 or qty%body['pack_size']:
                raise ValueError('Approval quantity must be a nonnegative pack multiple')
            if body['approved']:
                if body['approved']['qty']==qty and body['approved']['by']==actor:return body
                raise ValueError('Draft already approved; create a new recommendation to change quantity')
            approval={'qty':qty,'boxes':qty//body['pack_size'],'by':actor,'at':now()}
            body.update(approved=approval,status='draft_saved')
            con.execute('INSERT INTO order_approvals VALUES (?,?,?,?,?)',(draft_id,actor,approval['at'],qty,approval['boxes']))
            con.execute('UPDATE order_drafts SET body=? WHERE draft_id=?',(json.dumps(body),draft_id))
            return body

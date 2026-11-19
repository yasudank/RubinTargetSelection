import os
import glob
import json
import argparse
import gc
import threading
import io
from math import floor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed, FIRST_COMPLETED

import numpy as np
import pandas as pd
import fastavro
import psycopg2
from sqlalchemy import create_engine, text
from tqdm import tqdm

# 設定 (デフォルト)
DB_PARAMS = {
    "host": "localhost",
    "database": "cosmos_alert",
    "user": "lsst",
    "password": "rubin"
}
FITS_ROOT = 'cutouts'
BATCH_SIZE = 50
WORKERS = 12
THREADS_PER_WORKER = 8

dir_lock = threading.Lock()

# alert データベースから取得した全インデックスの静的定義リスト
ALERT_DB_INDEXES = [
    # alerts
    ('alerts', 'idx_alerts_diaSourceId', 'CREATE INDEX IF NOT EXISTS "idx_alerts_diaSourceId" ON public.alerts USING btree ("diaSourceId")'),
    
    # clf
    ('clf', 'idx_clf_parent', 'CREATE INDEX IF NOT EXISTS idx_clf_parent ON public.clf USING btree ("parent_diaSourceId")'),
    
    # diaObject
    ('diaObject', 'idx_diaObject_diaObjectId', 'CREATE INDEX IF NOT EXISTS "idx_diaObject_diaObjectId" ON public."diaObject" USING btree ("diaObjectId")'),
    ('diaObject', 'idx_diaObject_parent', 'CREATE INDEX IF NOT EXISTS "idx_diaObject_parent" ON public."diaObject" USING btree ("parent_diaSourceId")'),
    ('diaObject', 'idx_diaObject_q3c', 'CREATE INDEX IF NOT EXISTS "idx_diaObject_q3c" ON public."diaObject" USING btree (q3c_ang2ipix(ra, "dec"))'),
    
    # diaSource
    ('diaSource', 'idx_diaSource_diaObjectId', 'CREATE INDEX IF NOT EXISTS "idx_diaSource_diaObjectId" ON public."diaSource" USING btree ("diaObjectId")'),
    ('diaSource', 'idx_diaSource_diaSourceId', 'CREATE INDEX IF NOT EXISTS "idx_diaSource_diaSourceId" ON public."diaSource" USING btree ("diaSourceId")'),
    ('diaSource', 'idx_diaSource_parent', 'CREATE INDEX IF NOT EXISTS "idx_diaSource_parent" ON public."diaSource" USING btree ("parent_diaSourceId")'),
    ('diaSource', 'idx_diaSource_q3c', 'CREATE INDEX IF NOT EXISTS "idx_diaSource_q3c" ON public."diaSource" USING btree (q3c_ang2ipix(ra, "dec"))'),
    
    # lc_features
    ('lc_features', 'idx_lc_features_parent', 'CREATE INDEX IF NOT EXISTS idx_lc_features_parent ON public.lc_features USING btree ("parent_diaSourceId")'),
    
    # misc
    ('misc', 'idx_misc_parent', 'CREATE INDEX IF NOT EXISTS idx_misc_parent ON public.misc USING btree ("parent_diaSourceId")'),
    
    # mpc_orbits
    ('mpc_orbits', 'idx_mpc_orbits_parent', 'CREATE INDEX IF NOT EXISTS idx_mpc_orbits_parent ON public.mpc_orbits USING btree ("parent_diaSourceId")'),
    
    # pred
    ('pred', 'idx_pred_classifier_parent', 'CREATE INDEX IF NOT EXISTS idx_pred_classifier_parent ON public.pred USING btree (main_label_classifier, "parent_diaSourceId")'),
    ('pred', 'idx_pred_classifier_sourceid', 'CREATE INDEX IF NOT EXISTS idx_pred_classifier_sourceid ON public.pred USING btree (main_label_classifier, "parent_diaSourceId")'),
    ('pred', 'idx_pred_main_label_classifier', 'CREATE INDEX IF NOT EXISTS idx_pred_main_label_classifier ON public.pred USING btree (main_label_classifier)'),
    ('pred', 'idx_pred_parent', 'CREATE INDEX IF NOT EXISTS idx_pred_parent ON public.pred USING btree ("parent_diaSourceId")'),
    
    # prvDiaForcedSources
    ('prvDiaForcedSources', 'idx_prvDiaForcedSources_diaForcedSourceId', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaForcedSources_diaForcedSourceId" ON public."prvDiaForcedSources" USING btree ("diaForcedSourceId")'),
    ('prvDiaForcedSources', 'idx_prvDiaForcedSources_diaObjectId', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaForcedSources_diaObjectId" ON public."prvDiaForcedSources" USING btree ("diaObjectId")'),
    ('prvDiaForcedSources', 'idx_prvDiaForcedSources_parent', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaForcedSources_parent" ON public."prvDiaForcedSources" USING btree ("parent_diaSourceId")'),
    ('prvDiaForcedSources', 'idx_prvDiaForcedSources_q3c', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaForcedSources_q3c" ON public."prvDiaForcedSources" USING btree (q3c_ang2ipix(ra, "dec"))'),
    
    # prvDiaSources
    ('prvDiaSources', 'idx_prvDiaSources_diaObjectId', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaSources_diaObjectId" ON public."prvDiaSources" USING btree ("diaObjectId")'),
    ('prvDiaSources', 'idx_prvDiaSources_diaSourceId', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaSources_diaSourceId" ON public."prvDiaSources" USING btree ("diaSourceId")'),
    ('prvDiaSources', 'idx_prvDiaSources_parent', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaSources_parent" ON public."prvDiaSources" USING btree ("parent_diaSourceId")'),
    ('prvDiaSources', 'idx_prvDiaSources_q3c', 'CREATE INDEX IF NOT EXISTS "idx_prvDiaSources_q3c" ON public."prvDiaSources" USING btree (q3c_ang2ipix(ra, "dec"))'),
    ('prvDiaSources', 'idx_prvdiasources_mjd_floor', 'CREATE INDEX IF NOT EXISTS idx_prvdiasources_mjd_floor ON public."prvDiaSources" USING btree (floor(("midpointMjdTai" - (0.5)::double precision)))'),
    ('prvDiaSources', 'idx_prvdiasources_parent', 'CREATE INDEX IF NOT EXISTS idx_prvdiasources_parent ON public."prvDiaSources" USING btree ("parent_diaSourceId")'),
    ('prvDiaSources', 'idx_prvdiasources_parentsourceid_mjd', 'CREATE INDEX IF NOT EXISTS idx_prvdiasources_parentsourceid_mjd ON public."prvDiaSources" USING btree ("parent_diaSourceId", "midpointMjdTai")'),
    
    # ssSource
    ('ssSource', 'idx_ssSource_diaSourceId', 'CREATE INDEX IF NOT EXISTS "idx_ssSource_diaSourceId" ON public."ssSource" USING btree ("diaSourceId")'),
    ('ssSource', 'idx_ssSource_parent', 'CREATE INDEX IF NOT EXISTS "idx_ssSource_parent" ON public."ssSource" USING btree ("parent_diaSourceId")'),
    
    # xm
    ('xm', 'idx_xm_parent', 'CREATE INDEX IF NOT EXISTS idx_xm_parent ON public.xm USING btree ("parent_diaSourceId")'),
]

# alert テーブルから除外する構造体・リスト・バイナリ等のキー
EXCLUDED_ALERT_KEYS = {
    'diaSourceId', 'diaSource', 'prvDiaSources', 'prvDiaForcedSources',
    'diaObject', 'ssSource', 'mpc_orbits', 'cutoutDifference',
    'cutoutScience', 'cutoutTemplate', 'lc_features', 'clf', 'xm', 'pred', 'misc'
}

def get_db_url(db_params=None):
    params = db_params or DB_PARAMS
    return f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['database']}"

def check_disk_space(check_paths=None, min_free_gb=5.0):
    """
    指定されたパス（ルート / や fits_root など）の空きディスク容量をチェックする。
    空き容量が min_free_gb 未満の場合は False と詳細情報を返す。
    """
    import shutil
    if check_paths is None:
        check_paths = ['/', FITS_ROOT]
    for p in check_paths:
        try:
            abs_p = os.path.abspath(p)
            while not os.path.exists(abs_p) and abs_p != os.path.dirname(abs_p):
                abs_p = os.path.dirname(abs_p)
            usage = shutil.disk_usage(abs_p)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < min_free_gb:
                return False, f"Path '{p}' free space: {free_gb:.2f} GB (minimum required: {min_free_gb} GB)"
        except Exception:
            pass
    return True, ""

def ensure_database_exists(db_params, tablespace='ts_alert'):
    """データベースが存在しない場合は自動作成する (指定テーブルスペースがあれば優先使用)"""
    target_db = db_params['database']
    check_params = db_params.copy()
    check_params['database'] = 'postgres'
    try:
        conn = psycopg2.connect(**check_params)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            db_exists = cur.fetchone()

            # テーブルスペースの存在確認
            ts_to_use = None
            if tablespace:
                cur.execute("SELECT 1 FROM pg_tablespace WHERE spcname = %s", (tablespace,))
                if cur.fetchone():
                    ts_to_use = tablespace

            if not db_exists:
                print(f"Database '{target_db}' does not exist. Creating...")
                ts_clause = f' TABLESPACE "{ts_to_use}"' if ts_to_use else ""
                cur.execute(f'CREATE DATABASE "{target_db}"{ts_clause}')
                print(f"Database '{target_db}' created successfully{ts_clause}.")
            
            if ts_to_use:
                try:
                    cur.execute(f'ALTER DATABASE "{target_db}" SET default_tablespace = "{ts_to_use}"')
                except Exception:
                    pass
        conn.close()
    except Exception as e:
        print(f"Notice during database existence check: {e}")

def drop_ingestion_indexes(engine):
    """
    大量データ取り込み(COPY)を高速化し、WAL消費とディスク枯渇を防ぐため、
    prvDiaSources および prvDiaForcedSources のセカンダリインデックスを一時削除する。
    インデックスは取り込み完了後の apply_alert_indexes で再作成される。
    """
    indexes_to_drop = [
        "idx_prvDiaForcedSources_diaForcedSourceId",
        "idx_prvDiaForcedSources_diaObjectId",
        "idx_prvDiaForcedSources_parent",
        "idx_prvDiaForcedSources_q3c",
        "idx_prvDiaSources_diaObjectId",
        "idx_prvDiaSources_diaSourceId",
        "idx_prvDiaSources_parent",
        "idx_prvDiaSources_q3c",
        "idx_prvdiasources_mjd_floor",
        "idx_prvdiasources_parent",
        "idx_prvdiasources_parentsourceid_mjd"
    ]
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        res = conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))
        existing_indexes = set(r[0] for r in res.fetchall())
        dropped = 0
        for idx in indexes_to_drop:
            if idx in existing_indexes:
                try:
                    conn.execute(text(f'DROP INDEX IF EXISTS "{idx}"'))
                    dropped += 1
                except Exception as e:
                    print(f"Notice while dropping index {idx}: {e}")
        if dropped > 0:
            print(f"Temporarily dropped {dropped} secondary indexes on prvDia tables for high-speed COPY. (Will be rebuilt at finalization)")

def save_fits_task(dia_object_id, mjd, band, dia_source_id, science, template, difference, created_dirs, fits_root=FITS_ROOT):
    if dia_object_id is None or mjd is None or band is None or dia_source_id is None:
        return
    day_obs = int(floor(mjd - 0.5))
    try:
        sub_dir = f"{int(dia_object_id) % 1000:03d}"
    except (ValueError, TypeError):
        sub_dir = str(dia_object_id)[-3:].zfill(3)
    path = os.path.join(fits_root, sub_dir, str(dia_object_id), str(day_obs), str(band), str(dia_source_id))
    
    with dir_lock:
        if path not in created_dirs:
            os.makedirs(path, exist_ok=True)
            created_dirs.add(path)
    
    for name, data in [('science', science), ('template', template), ('difference', difference)]:
        if data is not None:
            file_path = os.path.join(path, f"{name}.fits")
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(data)

class NpEncoder(json.JSONEncoder):
    """Numpyの型をJSON化するためのエンコーダ"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

INT_COL_NAMES = {
    'diaSourceId', 'diaObjectId', 'diaForcedSourceId', 'parentDiaSourceId',
    'parent_diaSourceId', 'visit', 'detector', 'year', 'month', 'day',
    'nDiaSources', 'ssObjectId'
}

def copy_insert(conn, table_name, df, db_url=None):
    if df.empty:
        return
    
    # 整数型列の欠損値・指数表記化を防止し、オブジェクト型を文字列（JSON）に変換
    for col in df.columns:
        if col in INT_COL_NAMES:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: json.dumps(x, cls=NpEncoder) if isinstance(x, (dict, list, np.ndarray)) else x)

    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
    output.seek(0)
    
    cur = conn.cursor()
    try:
        cur.copy_from(output, table_name, sep='\t', null='\\N', columns=list(df.columns))
        conn.commit()
    except Exception:
        conn.rollback()
        # テーブル未作成などの場合は、PostgreSQLのアドバイザリロックで保護して並列作成の競合を防ぐ
        engine = create_engine(db_url or get_db_url())
        with engine.connect() as eng_conn:
            with eng_conn.begin():
                # テーブル名ごとのアドバイザリ排他ロックを取得
                eng_conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:tname))"), {"tname": table_name})
                tbl_exists = eng_conn.execute(
                    text("SELECT to_regclass(:tname)"), {"tname": f'public."{table_name}"'}
                ).scalar()
                if not tbl_exists:
                    # 0行のDataFrameでテーブル構造を作成
                    df.head(0).to_sql(table_name, eng_conn, if_exists='append', index=False)
        
        # テーブル作成後に再度 COPY で挿入を試みる
        output.seek(0)
        with conn.cursor() as retry_cur:
            retry_cur.copy_from(output, table_name, sep='\t', null='\\N', columns=list(df.columns))
        conn.commit()
    finally:
        cur.close()

def load_alert_records(file_path):
    """Avro または Parquet ファイルからアラートレコードのリストを読み込む"""
    if file_path.endswith('.avro'):
        with open(file_path, 'rb') as f:
            return list(fastavro.reader(f))
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    else:
        try:
            with open(file_path, 'rb') as f:
                return list(fastavro.reader(f))
        except Exception:
            df = pd.read_parquet(file_path)
            return df.to_dict('records')

def initialize_tables(all_files, engine, current_db_params):
    """
    並列ワーカー起動前に、親プロセスで全テーブルを作成しておく。
    これにより複数ワーカーによる同時 CREATE TABLE の競合
    (pg_type_typname_nsp_index や relation already exists) を防止する。
    """
    print("Checking and initializing database tables...")
    tables_needed = [
        'alerts', 'diaSource', 'diaObject', 'ssSource', 'mpc_orbits',
        'clf', 'xm', 'pred', 'misc', 'prvDiaSources', 'prvDiaForcedSources', 'lc_features'
    ]
    
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        existing_tables = set()
        for t in tables_needed:
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": f'public."{t}"'}).scalar():
                existing_tables.add(t)
    
    missing = [t for t in tables_needed if t not in existing_tables]
    if not missing:
        print("All database tables already exist.")
        return

    print(f"Missing tables to initialize: {missing}")
    all_data = {k: [] for k in tables_needed}
    
    # サンプルファイルからスキーマ構築用データを抽出
    for fpath in all_files[:10]:
        rows = load_alert_records(fpath)
        if not rows:
            continue
        for row in rows:
            ds_id = row.get('diaSourceId', 0)
            if 'alerts' in missing:
                alert_meta = {
                    k: v for k, v in row.items() 
                    if k not in EXCLUDED_ALERT_KEYS and not isinstance(v, (dict, list, np.ndarray, bytes))
                }
                alert_meta['diaSourceId'] = ds_id
                all_data['alerts'].append(alert_meta)

            for s in ['diaSource', 'diaObject', 'ssSource', 'mpc_orbits', 'clf', 'xm', 'pred', 'misc']:
                if s in missing:
                    data = row.get(s)
                    if isinstance(data, dict):
                        if s == 'diaObject' and data.get('diaObjectId') is None:
                            continue
                        if s == 'ssSource' and data.get('diaSourceId') is None:
                            continue
                        if s == 'mpc_orbits' and not any(v is not None for v in data.values()):
                            continue
                        d_copy = data.copy()
                        d_copy['parent_diaSourceId'] = ds_id
                        all_data[s].append(d_copy)

            for l in ['prvDiaSources', 'prvDiaForcedSources']:
                if l in missing:
                    items = row.get(l)
                    if isinstance(items, (list, np.ndarray)):
                        for item in items:
                            d = item.copy() if isinstance(item, dict) else {'value': item}
                            d['parent_diaSourceId'] = ds_id
                            all_data[l].append(d)

            if 'lc_features' in missing:
                feat = row.get('lc_features')
                if feat and isinstance(feat, dict):
                    for b, f in feat.items():
                        if isinstance(f, dict):
                            f_copy = f.copy()
                            f_copy['band_id'], f_copy['parent_diaSourceId'] = b, ds_id
                            all_data['lc_features'].append(f_copy)
        
        # 必要なテーブルのデータが揃ったか確認
        if all(len(all_data[t]) > 0 for t in missing if t != 'prvDiaForcedSources'):
            break

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for t in missing:
            records = all_data[t]
            if records:
                df = pd.DataFrame(records)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(lambda x: json.dumps(x, cls=NpEncoder) if isinstance(x, (dict, list, np.ndarray)) else x)
                exists = conn.execute(text("SELECT to_regclass(:t)"), {"t": f'public."{t}"'}).scalar()
                if not exists:
                    df.head(0).to_sql(t, conn, if_exists='append', index=False)
                    print(f"Table '{t}' created.")
    print("Database table initialization complete.")

def process_batch(file_list, db_params=None, fits_root=FITS_ROOT):
    current_db_params = db_params or DB_PARAMS
    db_url = get_db_url(current_db_params)
    conn = psycopg2.connect(**current_db_params)
    created_dirs = set()
    
    try:
        with ThreadPoolExecutor(max_workers=THREADS_PER_WORKER) as thread_pool:
            for file_path in file_list:
                try:
                    rows = load_alert_records(file_path)
                    if not rows:
                        file_key = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"
                        copy_insert(conn, 'processed_files', pd.DataFrame([{'file_path': file_key}]), db_url=db_url)
                        continue
                    
                    # DBに既に存在するdiaSourceIdを取得してスキップ
                    existing_ids = set()
                    ds_ids = tuple(r['diaSourceId'] for r in rows if 'diaSourceId' in r and r['diaSourceId'] is not None)
                    if ds_ids:
                        try:
                            with conn.cursor() as cur:
                                if len(ds_ids) == 1:
                                    cur.execute('SELECT "diaSourceId" FROM alerts WHERE "diaSourceId" = %s', (ds_ids[0],))
                                else:
                                    cur.execute('SELECT "diaSourceId" FROM alerts WHERE "diaSourceId" IN %s', (ds_ids,))
                                existing_ids = set(r[0] for r in cur.fetchall())
                        except Exception:
                            conn.rollback()
                                
                    if existing_ids:
                        rows = [r for r in rows if r.get('diaSourceId') not in existing_ids]
                    
                    if not rows:
                        file_key = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"
                        copy_insert(conn, 'processed_files', pd.DataFrame([{'file_path': file_key}]), db_url=db_url)
                        continue
                    
                    # 1. FITS保存 (マルチスレッド)
                    fits_futures = []
                    for row in rows:
                        ds = row.get('diaSource')
                        if isinstance(ds, dict):
                            fits_futures.append(thread_pool.submit(
                                save_fits_task,
                                ds.get('diaObjectId'),
                                ds.get('midpointMjdTai'),
                                ds.get('band') or ds.get('filterName'),
                                row.get('diaSourceId'),
                                row.get('cutoutScience'),
                                row.get('cutoutTemplate'),
                                row.get('cutoutDifference'),
                                created_dirs,
                                fits_root
                            ))
                    
                    # 2. DB用データの準備
                    all_data = {
                        k: [] for k in [
                            'alerts', 'diaSource', 'diaObject', 'ssSource', 'mpc_orbits',
                            'clf', 'xm', 'pred', 'misc', 'prvDiaSources', 'prvDiaForcedSources', 'lc_features'
                        ]
                    }
                    
                    for row in rows:
                        ds_id = row['diaSourceId']
                        
                        # alerts: 基本型以外（構造体、リスト、バイナリ）を除外
                        alert_meta = {
                            k: v for k, v in row.items() 
                            if k not in EXCLUDED_ALERT_KEYS and not isinstance(v, (dict, list, np.ndarray, bytes))
                        }
                        alert_meta['diaSourceId'] = ds_id
                        all_data['alerts'].append(alert_meta)
                        
                        # structs (1:1)
                        for s in ['diaSource', 'diaObject', 'ssSource', 'mpc_orbits', 'clf', 'xm', 'pred', 'misc']:
                            data = row.get(s)
                            if isinstance(data, dict):
                                if s == 'diaObject' and data.get('diaObjectId') is None:
                                    continue
                                if s == 'ssSource' and data.get('diaSourceId') is None:
                                    continue
                                if s == 'mpc_orbits' and not any(v is not None for v in data.values()):
                                    continue
                                data_copy = data.copy()
                                data_copy['parent_diaSourceId'] = ds_id
                                all_data[s].append(data_copy)
                        
                        # lists (1:N)
                        for l in ['prvDiaSources', 'prvDiaForcedSources']:
                            items = row.get(l)
                            if isinstance(items, (list, np.ndarray)):
                                for item in items:
                                    d = item.copy() if isinstance(item, dict) else {'value': item}
                                    d['parent_diaSourceId'] = ds_id
                                    all_data[l].append(d)
                        
                        # lc_features
                        feat = row.get('lc_features')
                        if feat:
                            if isinstance(feat, dict):
                                for b, f in feat.items():
                                    if isinstance(f, dict):
                                        f_copy = f.copy()
                                        f_copy['band_id'], f_copy['parent_diaSourceId'] = b, ds_id
                                        all_data['lc_features'].append(f_copy)
                            elif isinstance(feat, (list, np.ndarray)):
                                for item in feat:
                                    if isinstance(item, (list, tuple)) and len(item) == 2:
                                        b, f = item
                                        if isinstance(f, dict):
                                            f_copy = f.copy()
                                            f_copy['band_id'], f_copy['parent_diaSourceId'] = b, ds_id
                                            all_data['lc_features'].append(f_copy)

                    # 3. バッチ内重複排除 (メモリ上での事前重複排除)
                    # (a) prvDiaSources: 同一バッチ内の重複diaSourceIdを排除
                    if all_data['prvDiaSources']:
                        df_prv = pd.DataFrame(all_data['prvDiaSources'])
                        if 'diaSourceId' in df_prv.columns:
                            df_prv = df_prv.drop_duplicates(subset=['diaSourceId'], keep='last')
                        all_data['prvDiaSources'] = df_prv

                    # (b) prvDiaForcedSources: 同一バッチ内の重複diaForcedSourceIdを排除
                    if all_data['prvDiaForcedSources']:
                        df_prv_fs = pd.DataFrame(all_data['prvDiaForcedSources'])
                        if 'diaForcedSourceId' in df_prv_fs.columns:
                            df_prv_fs = df_prv_fs.drop_duplicates(subset=['diaForcedSourceId'], keep='last')
                        all_data['prvDiaForcedSources'] = df_prv_fs

                    # (c) diaObject: 同一バッチ内でdiaObjectId毎に最新レコード(validityStartMjdTai / parent_diaSourceId)のみを保持
                    if all_data['diaObject']:
                        df_obj = pd.DataFrame(all_data['diaObject'])
                        if 'diaObjectId' in df_obj.columns:
                            sort_cols = [c for c in ['validityStartMjdTai', 'parent_diaSourceId'] if c in df_obj.columns]
                            if sort_cols:
                                df_obj = df_obj.sort_values(by=sort_cols)
                            df_obj = df_obj.drop_duplicates(subset=['diaObjectId'], keep='last')
                        all_data['diaObject'] = df_obj

                    # 4. DBへCOPY挿入
                    for table, records in all_data.items():
                        if isinstance(records, pd.DataFrame):
                            if not records.empty:
                                copy_insert(conn, table, records, db_url=db_url)
                        elif records:
                            copy_insert(conn, table, pd.DataFrame(records), db_url=db_url)
                    
                    file_key = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"
                    copy_insert(conn, 'processed_files', pd.DataFrame([{'file_path': file_key}]), db_url=db_url)
                    
                    wait(fits_futures)
                    del rows, all_data
                            
                except Exception as e:
                    print(f"Error in {file_path}: {e}")
                    conn.rollback()
    finally:
        conn.close()
    return len(file_list)

def post_ingestion_deduplication(engine):
    """
    全バッチ取り込み完了後、DB全体で以下を実行する (高速テーブルスワップ方式):
    1. prvDiaSources の重複レコード削除 (同一 diaSourceId のうち最新の 1 レコードのみを残す)
    2. prvDiaForcedSources の重複レコード削除 (同一 diaForcedSourceId のうち最新の 1 レコードのみを残す)
    3. diaObject の古いレコード削除 (同一 diaObjectId のうち最新の validityStartMjdTai / parent_diaSourceId のみを残す)
    4. diaSource および alerts の重複レコード削除 (ファイル間重複の解消)
    """
    print("Deduplicating tables using high-speed table swap (DISTINCT ON)...")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text("SET work_mem = '4GB';"))
            conn.execute(text("SET maintenance_work_mem = '4GB';"))
        except Exception:
            pass
        # 1. prvDiaSources の重複削除
        try:
            if conn.execute(text("SELECT to_regclass('public.\"prvDiaSources\"')")).scalar():
                print("Deduplicating prvDiaSources (table swap method)...")
                conn.execute(text("""
                    DROP TABLE IF EXISTS "prvDiaSources_dedup";
                    CREATE TABLE "prvDiaSources_dedup" AS
                    SELECT DISTINCT ON ("diaSourceId") *
                    FROM "prvDiaSources"
                    WHERE "diaSourceId" IS NOT NULL
                    ORDER BY "diaSourceId", "parent_diaSourceId" DESC NULLS LAST;
                    
                    DROP TABLE "prvDiaSources";
                    ALTER TABLE "prvDiaSources_dedup" RENAME TO "prvDiaSources";
                """))
                print("prvDiaSources deduplication complete.")
        except Exception as e:
            print(f"Warning during prvDiaSources deduplication: {e}")

        # 2. prvDiaForcedSources の重複削除
        try:
            if conn.execute(text("SELECT to_regclass('public.\"prvDiaForcedSources\"')")).scalar():
                print("Deduplicating prvDiaForcedSources (table swap method)...")
                conn.execute(text("""
                    DROP TABLE IF EXISTS "prvDiaForcedSources_dedup";
                    CREATE TABLE "prvDiaForcedSources_dedup" AS
                    SELECT DISTINCT ON ("diaForcedSourceId") *
                    FROM "prvDiaForcedSources"
                    WHERE "diaForcedSourceId" IS NOT NULL
                    ORDER BY "diaForcedSourceId", "parent_diaSourceId" DESC NULLS LAST;
                    
                    DROP TABLE "prvDiaForcedSources";
                    ALTER TABLE "prvDiaForcedSources_dedup" RENAME TO "prvDiaForcedSources";
                """))
                print("prvDiaForcedSources deduplication complete.")
        except Exception as e:
            print(f"Warning during prvDiaForcedSources deduplication: {e}")

        # 3. diaObject の重複削除 (最新のみ保持)
        try:
            if conn.execute(text("SELECT to_regclass('public.\"diaObject\"')")).scalar():
                print("Keeping only the latest record for each diaObject (table swap method)...")
                conn.execute(text("""
                    DROP TABLE IF EXISTS "diaObject_dedup";
                    CREATE TABLE "diaObject_dedup" AS
                    SELECT DISTINCT ON ("diaObjectId") *
                    FROM "diaObject"
                    WHERE "diaObjectId" IS NOT NULL
                    ORDER BY "diaObjectId", "validityStartMjdTai" DESC NULLS LAST, "parent_diaSourceId" DESC NULLS LAST;
                    
                    DROP TABLE "diaObject";
                    ALTER TABLE "diaObject_dedup" RENAME TO "diaObject";
                """))
                print("diaObject deduplication complete.")
        except Exception as e:
            print(f"Warning during diaObject deduplication: {e}")

        # 4. diaSource の重複削除
        try:
            if conn.execute(text("SELECT to_regclass('public.\"diaSource\"')")).scalar():
                print("Deduplicating diaSource (table swap method)...")
                conn.execute(text("""
                    DROP TABLE IF EXISTS "diaSource_dedup";
                    CREATE TABLE "diaSource_dedup" AS
                    SELECT DISTINCT ON ("diaSourceId") *
                    FROM "diaSource"
                    WHERE "diaSourceId" IS NOT NULL
                    ORDER BY "diaSourceId", "parent_diaSourceId" DESC NULLS LAST;
                    
                    DROP TABLE "diaSource";
                    ALTER TABLE "diaSource_dedup" RENAME TO "diaSource";
                """))
                print("diaSource deduplication complete.")
        except Exception as e:
            print(f"Warning during diaSource deduplication: {e}")

        # 5. alerts の重複削除
        try:
            if conn.execute(text("SELECT to_regclass('public.\"alerts\"')")).scalar():
                print("Deduplicating alerts (table swap method)...")
                conn.execute(text("""
                    DROP TABLE IF EXISTS "alerts_dedup";
                    CREATE TABLE "alerts_dedup" AS
                    SELECT DISTINCT ON ("diaSourceId") *
                    FROM "alerts"
                    WHERE "diaSourceId" IS NOT NULL
                    ORDER BY "diaSourceId";
                    
                    DROP TABLE "alerts";
                    ALTER TABLE "alerts_dedup" RENAME TO "alerts";
                """))
                print("alerts deduplication complete.")
        except Exception as e:
            print(f"Warning during alerts deduplication: {e}")

def apply_alert_indexes(engine, cluster_enabled=False):
    """
    alert データベースと同一のインデックス定義を新しいデータベースに静的設定・適用する。
    """
    print("Applying exact index definitions matching 'alert' database...")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text("SET maintenance_work_mem = '4GB';"))
        except Exception:
            pass
        # Q3C 拡張のチェック
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS q3c"))
            print("Q3C extension checked.")
        except Exception as e:
            print(f"Notice (Q3C): {e}")

        # 既存インデックス名一覧を取得
        res = conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))
        existing_indexes = set(r[0] for r in res.fetchall())

        applied_count = 0
        for tbl, idx_name, idx_sql in ALERT_DB_INDEXES:
            # テーブルが存在するか確認
            if not conn.execute(text("SELECT to_regclass(:t)"), {"t": f'public."{tbl}"'}).scalar():
                continue

            if idx_name in existing_indexes:
                continue

            try:
                conn.execute(text(idx_sql))
                print(f"Created index: {idx_name} on {tbl}")
                existing_indexes.add(idx_name)
                applied_count += 1
            except Exception as e:
                print(f"Notice while creating index {idx_name}: {e}")

        print(f"Applied {applied_count} index(es) matching alert database.")

        if cluster_enabled:
            for tbl in ["diaSource", "diaObject", "prvDiaSources", "prvDiaForcedSources"]:
                idx = f"idx_{tbl}_q3c"
                if idx in existing_indexes:
                    try:
                        conn.execute(text(f'CLUSTER "{tbl}" USING "{idx}"'))
                        print(f"Clustered {tbl} using {idx}")
                    except Exception as e:
                        print(f"Cluster notice ({tbl}): {e}")

        # 各テーブルの ANALYZE
        res = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        for (tbl,) in res.fetchall():
            try:
                conn.execute(text(f'ANALYZE "{tbl}"'))
            except Exception:
                pass
        print("Index creation and table optimization complete.")

def main():
    parser = argparse.ArgumentParser(description="Ingest LSST Avro/Parquet alerts into PostgreSQL and save FITS cutouts.")
    parser.add_argument('--source_dir', type=str, default='ftransfer_lsst_cosmos_20251025_20251231', help='Directory containing avro/parquet files')
    parser.add_argument('--dbname', type=str, default=DB_PARAMS['database'], help='PostgreSQL target database name')
    parser.add_argument('--dbhost', type=str, default=DB_PARAMS['host'], help='PostgreSQL host')
    parser.add_argument('--dbuser', type=str, default=DB_PARAMS['user'], help='PostgreSQL user')
    parser.add_argument('--dbpassword', type=str, default=DB_PARAMS['password'], help='PostgreSQL password')
    parser.add_argument('--fits_root', type=str, default=FITS_ROOT, help='Root directory to save FITS cutouts')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Number of worker processes')
    parser.add_argument('--threads', type=int, default=THREADS_PER_WORKER, help='Threads per worker for FITS writing')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Files per batch chunk assigned to each worker (default: 50)')
    parser.add_argument('--max_files', type=int, default=0, help='Maximum remaining files to process in this run (0 = all remaining)')
    parser.add_argument('--tablespace', type=str, default='ts_alert', help='PostgreSQL tablespace for storage (default: ts_alert)')
    parser.add_argument('--min_disk_free_gb', type=float, default=5.0, help='Minimum free disk space (GB) on root/storage to proceed (default: 5.0)')
    parser.add_argument('--finalize', action='store_true', help='Force post-ingestion deduplication and index rebuild even if files remain')
    parser.add_argument('--no_finalize', action='store_true', help='Skip deduplication and index rebuild even if all files are processed')
    parser.add_argument('--cluster', action='store_true', help='Run CLUSTER command after ingestion (can be very slow)')
    parser.add_argument('--test', type=int, default=0, help='Process only N batches for testing')
    args = parser.parse_args()

    # ディスク容量の事前チェック
    disk_ok, disk_msg = check_disk_space(['/', args.fits_root], min_free_gb=args.min_disk_free_gb)
    if not disk_ok:
        print(f"Error: Low disk space detected before starting: {disk_msg}")
        return

    # DB接続パラメータの更新
    current_db_params = {
        "host": args.dbhost,
        "database": args.dbname,
        "user": args.dbuser,
        "password": args.dbpassword
    }
    db_url = get_db_url(current_db_params)
    ensure_database_exists(current_db_params, tablespace=args.tablespace)

    # 対象ファイルの探索 (avro優先、なければparquet)
    all_files = sorted(glob.glob(os.path.join(args.source_dir, '*.avro')))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(args.source_dir, '*.parquet')))

    if not all_files:
        print(f"No avro or parquet files found in {args.source_dir}")
        return

    print(f"Found {len(all_files)} total files in {args.source_dir}. Target DB: {args.dbname}")

    engine = create_engine(db_url)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS processed_files (file_path TEXT PRIMARY KEY)"))
        res = conn.execute(text("SELECT file_path FROM processed_files"))
        processed_files = set(r[0] for r in res.fetchall())

    # 親プロセスで全テーブルを事前初期化 (マルチワーカーによる CREATE TABLE 競合を防止)
    initialize_tables(all_files, engine, current_db_params)
    
    source_prefix = f"{os.path.basename(args.source_dir)}/"
    processed_in_dir = sum(1 for f in processed_files if f.startswith(source_prefix))
    
    files_to_process = [
        f for f in all_files 
        if f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}" not in processed_files
    ]
    remaining_total = len(files_to_process)

    if args.test > 0:
        files_to_process = files_to_process[:args.batch_size * args.test]
    elif args.max_files > 0:
        files_to_process = files_to_process[:args.max_files]

    print(f"Directory: {args.source_dir}")
    print(f"  - Total files: {len(all_files)}")
    print(f"  - Already processed: {processed_in_dir}")
    print(f"  - Remaining before run: {remaining_total}")
    print(f"  - Files to process in this run: {len(files_to_process)} (batch_size={args.batch_size}, workers={args.workers})")

    if not files_to_process:
        print("No files to process in this run.")
    else:
        # 大量COPY時のWAL肥大化・ディスク枯渇を防ぐためprvテーブルのインデックスを一時削除
        drop_ingestion_indexes(engine)

        chunks = [files_to_process[i:i + args.batch_size] for i in range(0, len(files_to_process), args.batch_size)]
        print(f"Starting ingestion with {args.workers} workers (COPY method, {len(chunks)} chunks)...")

        processed_count = 0
        checkpoint_interval = 1000  # 1000ファイル毎にCHECKPOINT実行してWALを解放
        last_checkpoint = 0

        # ワーカー数に基づいたスライディングウィンドウでタスク投入
        max_active_tasks = args.workers * 2
        chunk_iter = iter(chunks)
        active_futures = {}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(total=len(files_to_process)) as pbar:
                # 最初に max_active_tasks 分を投入
                for chunk in chunk_iter:
                    f = executor.submit(process_batch, chunk, current_db_params, args.fits_root)
                    active_futures[f] = len(chunk)
                    if len(active_futures) >= max_active_tasks:
                        break

                disk_aborted = False
                while active_futures:
                    done_futures, _ = wait(list(active_futures.keys()), return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        count = active_futures.pop(future)
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Batch processing error: {e}")
                        pbar.update(count)
                        processed_count += count

                        # 定期的CHECKPOINT実行でWALを解放
                        if processed_count - last_checkpoint >= checkpoint_interval:
                            last_checkpoint = processed_count
                            try:
                                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                                    conn.execute(text("CHECKPOINT"))
                            except Exception:
                                pass

                    # ディスク容量の定期チェック
                    d_ok, d_msg = check_disk_space(['/', args.fits_root], min_free_gb=args.min_disk_free_gb)
                    if not d_ok:
                        print(f"\n[WARNING] Disk space safeguard triggered: {d_msg}")
                        print("Gracefully stopping queueing new batches to prevent disk exhaustion.")
                        disk_aborted = True
                        break

                    # 空いた枠に次のチャンクを順次投入
                    if not disk_aborted:
                        for chunk in chunk_iter:
                            f = executor.submit(process_batch, chunk, current_db_params, args.fits_root)
                            active_futures[f] = len(chunk)
                            if len(active_futures) >= max_active_tasks:
                                break

        # 完了後のCHECKPOINT
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("CHECKPOINT"))
        except Exception:
            pass

    # 4. 全ファイル完了時の重複削除と最新 diaObject の保持、インデックス適用
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        res = conn.execute(text("SELECT file_path FROM processed_files WHERE file_path LIKE :p"), {"p": f"{source_prefix}%"})
        done_now = set(r[0] for r in res.fetchall())
    
    remaining_now = len(all_files) - len(done_now)
    print(f"\nProgress for {args.source_dir}: {len(done_now)}/{len(all_files)} files completed ({remaining_now} remaining).")

    should_finalize = False
    if args.finalize:
        should_finalize = True
    elif remaining_now == 0 and not args.no_finalize:
        should_finalize = True

    if should_finalize:
        print("Running deduplication and index creation...")
        post_ingestion_deduplication(engine)
        apply_alert_indexes(engine, cluster_enabled=args.cluster)
        print("Finalization complete.")
    else:
        print(f"Ingestion run finished. {remaining_now} files still remaining.")
        print("Run again to continue ingesting remaining files (e.g., with --max_files 2000).")
        print("Deduplication and index rebuild will be automatically performed once all files are ingested (or pass --finalize).")

    print(f"Finished. DB: {db_url}")

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

# Préparation migration PostgreSQL
try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

class DBManager:
    def __init__(self, db_path='sessions.db', use_postgres=False, pg_config=None):
        self.use_postgres = use_postgres
        self.pg_config = pg_config or {}
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        if self.use_postgres:
            if not HAS_PG:
                raise ImportError("psycopg2 requis pour PostgreSQL")
            return psycopg2.connect(**self.pg_config)
        else:
            return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        # SQL compatible pour SQLite et PostgreSQL
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                duration REAL,
                sentiment_score REAL,
                pause_ratio REAL,
                transcription TEXT,
                full_metrics_json TEXT
            )
        '''
        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()

    def save_session(self, metrics: dict):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            insert_sql = '''
                INSERT INTO sessions (timestamp, duration, sentiment_score, pause_ratio, transcription, full_metrics_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''' if self.use_postgres else '''
                INSERT INTO sessions (timestamp, duration, sentiment_score, pause_ratio, transcription, full_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                metrics.get('timestamp', datetime.now().isoformat()),
                metrics.get('duration'),
                metrics.get('sentiment_score'),
                metrics.get('pause_ratio'),
                metrics.get('transcription'),
                json.dumps(metrics)
            )
            cursor.execute(insert_sql, params)
            conn.commit()
        except Exception as e:
            print(f"[DBManager] Error saving session: {e}")
        finally:
            conn.close()

    def get_all_sessions(self):
        try:
            conn = self._get_conn()
            df = pd.read_sql_query('SELECT * FROM sessions ORDER BY timestamp DESC', conn)
            return df
        except Exception as e:
            print(f"[DBManager] Error loading sessions: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

# Pour utiliser PostgreSQL :
# db = DBManager(use_postgres=True, pg_config={
#     'host': 'localhost', 'port': 5432, 'user': 'user', 'password': 'pass', 'dbname': 'db'
# })

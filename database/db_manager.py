import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

class DBManager:
    def __init__(self, db_path='sessions.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                duration REAL,
                sentiment_score REAL,
                pause_ratio REAL,
                transcription TEXT,
                full_metrics_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_session(self, metrics: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (timestamp, duration, sentiment_score, pause_ratio, transcription, full_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('timestamp', datetime.now().isoformat()),
                metrics.get('duration'),
                metrics.get('sentiment_score'),
                metrics.get('pause_ratio'),
                metrics.get('transcription'),
                json.dumps(metrics)
            ))
            conn.commit()
        except Exception as e:
            print(f"[DBManager] Error saving session: {e}")
        finally:
            conn.close()

    def get_all_sessions(self):
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM sessions ORDER BY timestamp DESC', conn)
            return df
        except Exception as e:
            print(f"[DBManager] Error loading sessions: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

import psycopg2
import yaml
import os

def load_pg_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('postgres', {})

def init_postgres():
    pg_cfg = load_pg_config()
    conn = psycopg2.connect(**pg_cfg)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            timestamp TEXT NOT NULL,
            duration REAL,
            sentiment_score REAL,
            pause_ratio REAL,
            transcription TEXT,
            full_metrics_json TEXT
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Table 'sessions' créée ou déjà existante.")

if __name__ == '__main__':
    init_postgres()
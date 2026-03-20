import sys
import os
import yaml

# Ajouter le dossier racine au path pour importer database.db_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database.db_manager import DBManager

def load_pg_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('postgres', {})

def init_postgres():
    pg_cfg = load_pg_config()
    db = DBManager(use_postgres=True, pg_config=pg_cfg)
    print("✅ Table 'sessions' créée ou déjà existante.")

if __name__ == '__main__':
    init_postgres()
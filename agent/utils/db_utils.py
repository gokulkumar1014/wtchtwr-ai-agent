import os

def get_db_path():
    '''
    Return the absolute path to the local DuckDB database.
    This ensures all components (backend, LangGraph, Slackbot) reference the same DB.
    '''
    base_dir = os.path.abspath(os.getcwd())
    db_path = os.path.join(base_dir, 'db', 'airbnb.duckdb')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'''Expected DuckDB file not found at: {db_path}. Ensure \'db/airbnb.duckdb\' exists or run the data setup script first.''')
    return db_path


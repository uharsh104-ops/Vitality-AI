import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "health_assistant.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create the users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    
    # Create the health_logs table linked to users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            sleep_hours REAL,
            exercise_minutes INTEGER,
            meal_type TEXT,
            mood TEXT,
            water_intake REAL,
            diabetes INTEGER,
            obesity INTEGER,
            hypertension INTEGER,
            health_score REAL,
            UNIQUE(user_id, log_date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Ensure guest user ALWAYS exists with ID 0 (or whatever auto-increments first if we just use username)
    try:
        cursor.execute("INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)", ('Guest', ''))
    except sqlite3.Error:
        pass
        
    conn.commit()
    conn.close()

# --- User Auth Functions ---
def add_user(username, password_hash):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username exists
    finally:
        conn.close()

def get_user(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# --- Log Functions ---
def save_log(user_id, date, sleep, exercise, meal, mood, water, diab, obesi, hyper, score):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO health_logs 
            (user_id, log_date, sleep_hours, exercise_minutes, meal_type, mood, water_intake, diabetes, obesity, hypertension, health_score) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, date, sleep, exercise, meal, mood, water, int(diab), int(obesi), int(hyper), score))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving log: {e}")
        return False
    finally:
        conn.close()

def get_recent_logs(user_id, days=30):
    conn = get_connection()
    query = f"""
        SELECT log_date, sleep_hours, exercise_minutes, meal_type, mood, water_intake, health_score 
        FROM health_logs 
        WHERE user_id = {user_id}
        ORDER BY log_date DESC 
        LIMIT {days}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['log_date'] = pd.to_datetime(df['log_date']).dt.date
        df = df.sort_values(by="log_date")
    
    return df

def get_all_logs(user_id):
    conn = get_connection()
    query = f"SELECT * FROM health_logs WHERE user_id = {user_id} ORDER BY log_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_user_conditions(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT diabetes, obesity, hypertension FROM health_logs WHERE user_id = ? ORDER BY log_date DESC LIMIT 1", (user_id,))
    res = cursor.fetchone()
    conn.close()
    if res:
        return {'diabetes': bool(res[0]), 'obesity': bool(res[1]), 'hypertension': bool(res[2])}
    return {'diabetes': False, 'obesity': False, 'hypertension': False}

def get_streak(user_id):
    """Calculates consecutive days logged."""
    df = get_recent_logs(user_id, 30)
    if df.empty:
        return 0
    
    df = df.sort_values(by="log_date", ascending=False)
    dates = df['log_date'].tolist()
    
    current_date = datetime.now().date()
    streak = 0
    
    if current_date not in dates and (current_date - timedelta(days=1)) not in dates:
        return 0
        
    check_date = dates[0]
    for d in dates:
        if d == check_date:
            streak += 1
            check_date -= timedelta(days=1)
        else:
            break
            
    return streak

def clear_user_logs(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM health_logs WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

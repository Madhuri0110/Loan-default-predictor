import sqlite3

def create_connection():
    conn = sqlite3.connect("loan.db")
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        income REAL,
        loan_amount REAL,
        prediction INTEGER
    )
    """)
    
    conn.commit()
    conn.close()

def insert_data(age, income, loan_amount, prediction):
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO predictions (age, income, loan_amount, prediction)
    VALUES (?, ?, ?, ?)
    """, (age, income, loan_amount, prediction))
    
    conn.commit()
    conn.close()
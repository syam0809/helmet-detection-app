import sqlite3

def create_table():
    conn = sqlite3.connect('fines.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS fines 
                (plate TEXT, violation TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def add_fine(plate, violation, timestamp):
    conn = sqlite3.connect('fines.db')
    c = conn.cursor()
    c.execute("INSERT INTO fines VALUES (?, ?, ?)", (plate, violation, timestamp))
    conn.commit()
    conn.close()

def get_fines():
    conn = sqlite3.connect('fines.db')
    c = conn.cursor()
    c.execute("SELECT * FROM fines")
    data = c.fetchall()
    conn.close()
    return data

import sqlite3
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Connect to SQLite (this creates the DB file if it doesn't exist)
conn = sqlite3.connect("data/palm_auth.db")
cursor = conn.cursor()

# Create Users Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    features BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("✅ SQLite Database setup complete! Database file: data/palm_auth.db")

import sqlite3
from pathlib import Path

Path("data").mkdir(exist_ok=True)

conn = sqlite3.connect("data/gresham_demo.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS portfolio_companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    sector TEXT,
    investment_date TEXT,
    investment_value_gbp INTEGER,
    status TEXT
)
""")

portfolio_data = [
    (1, "OnSecurity", "Cybersecurity", "2024-03-15", 5500000, "Active"),
    (2, "Accredit Solutions", "Security Automation", "2024-06-20", 10000000, "Active"),
    (3, "GreenTech Energy", "Renewable Energy", "2023-11-10", 8000000, "Active"),
]

cursor.executemany("INSERT OR REPLACE INTO portfolio_companies VALUES (?, ?, ?, ?, ?, ?)", portfolio_data)

conn.commit()
conn.close()
print("Database created!")

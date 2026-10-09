import sqlite3
import random
from datetime import date, timedelta

random.seed(42)

conn = sqlite3.connect("./data/documents/database.sqlite")
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    industry TEXT,
    churn_date TEXT,
    satisfaction_score REAL
);

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    region TEXT,
    product TEXT,
    revenue REAL,
    date TEXT,
    units_sold INTEGER
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    department TEXT,
    satisfaction_score REAL,
    tenure_years REAL
);
""")

industries = ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing"]
regions = ["North", "South", "East", "West"]
products = ["Product A", "Product B", "Product C", "Product D"]
departments = ["Engineering", "Sales", "HR", "Marketing", "Support"]

base = date(2023, 1, 1)

customers = []
for i in range(1, 101):
    churn = (base + timedelta(days=random.randint(0, 365))).isoformat() if random.random() < 0.2 else None
    customers.append((i, f"Customer {i}", random.choice(industries), churn, round(random.uniform(1, 10), 1)))
cur.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?)", customers)

sales_rows = []
for i in range(1, 501):
    d = (base + timedelta(days=random.randint(0, 364))).isoformat()
    sales_rows.append((i, random.choice(regions), random.choice(products),
                       round(random.uniform(500, 50000), 2), d, random.randint(1, 200)))
cur.executemany("INSERT OR REPLACE INTO sales VALUES (?,?,?,?,?,?)", sales_rows)

employees = []
for i in range(1, 51):
    employees.append((i, random.choice(departments), round(random.uniform(1, 10), 1), round(random.uniform(0.5, 15), 1)))
cur.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?,?)", employees)

conn.commit()
conn.close()
print("Database seeded: 100 customers, 500 sales, 50 employees")

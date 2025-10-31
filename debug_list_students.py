import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "dental_project_DB.db")
print('Using DB:', DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT student_id, name, batch_id, department_id, round_id, phone, email FROM student ORDER BY rowid DESC LIMIT 20;")
rows = c.fetchall()
print(f"Found {len(rows)} students (most recent first):")
for r in rows:
    print(dict(r))
conn.close()

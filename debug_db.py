import sqlite3
import os

DB_PATH = r"c:\Users\pc\Desktop\last version DentAI-Hospital-System-ahmed\DentAI-Hospital-System-ahmed\dental_project_DB.db"

def check_db():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print("--- Departments ---")
        cursor.execute("SELECT * FROM department ORDER BY department_id")
        depts = cursor.fetchall()
        for d in depts:
            print(dict(d))
        print(f"Total Departments: {len(depts)}")

        print("\n--- Rounds ---")
        cursor.execute("SELECT * FROM rounds")
        rounds = cursor.fetchall()
        for r in rounds:
            print(dict(r))
        print(f"Total Rounds: {len(rounds)}")

        print("\n--- Students ---")
        cursor.execute("SELECT student_id, name, department_id, round_id, batch_id FROM student LIMIT 5")
        students = cursor.fetchall()
        for s in students:
            print(dict(s))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_db()

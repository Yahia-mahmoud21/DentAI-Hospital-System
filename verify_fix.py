import sqlite3
import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from doctor_query import approve_student_round
from db import connect as db_connect

def verify():
    conn = db_connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Find a doctor in d003
        print("Finding doctor in d003...")
        cursor.execute("SELECT faculty_members_id FROM faculty_members WHERE department_id = 'd003' LIMIT 1")
        doc = cursor.fetchone()
        if not doc:
            # Create a dummy doctor if needed, or pick another dept
            print("No doctor in d003, creating temp one")
            cursor.execute("INSERT INTO faculty_members (faculty_members_id, name, department_id, password) VALUES ('TEMP_DOC', 'Temp Doc', 'd003', '123')")
            doc_id = 'TEMP_DOC'
        else:
            doc_id = doc['faculty_members_id']
        print(f"Using Doctor: {doc_id}")

        # 2. Find or Create a student in d003, R003
        print("Setting up student s001 in d003, R003...")
        cursor.execute("UPDATE student SET department_id = 'd003', round_id = 'R003' WHERE student_id = 's001'")
        conn.commit()

        # 3. Attempt Approval
        print("Attempting approval...")
        result = approve_student_round('s001', doc_id, True)
        print(f"Result: {result}")

        # 4. Verify New State
        cursor.execute("SELECT department_id, round_id FROM student WHERE student_id = 's001'")
        student = cursor.fetchone()
        print(f"New State: Dept={student['department_id']}, Round={student['round_id']}")

        if result['success'] and student['department_id'] == 'd004' and student['round_id'] == 'R003':
            print("SUCCESS: Student moved to d004 and stayed in R003 (as rounds are exhausted)")
        elif result['success'] and student['department_id'] == 'd004':
             print(f"PARTIAL SUCCESS: Student moved to d004, Round is {student['round_id']}")
        else:
            print("FAILURE: Student did not progress as expected")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup temp doc if created
        if 'doc_id' in locals() and doc_id == 'TEMP_DOC':
            cursor.execute("DELETE FROM faculty_members WHERE faculty_members_id = 'TEMP_DOC'")
            conn.commit()
        conn.close()

if __name__ == "__main__":
    verify()

import sqlite3
import time
import uuid
from db import connect as db_connect
# Removed duplicate 'import uuid'
# Removed unused 'import time'

# 1. Modify get_cases_data to accept student_id and use parameterization
def get_cases_data(student_id: str):
    """Fetches case data for a specific student."""
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use '?' as a placeholder for the student_id
        cursor.execute("""
            SELECT s.case_id, c.name, s.description, s.treatment, s.before_photo,
                   s.after_photo, s.appointment_date, s.checked,
                   s.department_reffere
            FROM student_department_cases s
            JOIN cases c ON c.case_id = s.case_id
            WHERE student_id = ?
            ORDER BY s.case_id DESC;
        """, (student_id,)) # Pass the student_id as a tuple

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error in get_cases_data: {e}")
        return []


def get_student_data():
    """Fetches basic student data and case count for department 'd001'."""
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # NOTE: Your query currently filters on a specific department ID ('d001')
        # in the HAVING clause, but the GROUP BY applies to all rows first.
        # It assumes student_department_cases contains the department_id.
        # A more robust query might be:
        # WHERE sc.department_id = 'd001'
        # GROUP BY s.student_id, s.name;
        # For now, keeping your original query structure:
        cursor.execute("""SELECT s.student_id, s.name, count(sc.case_id) as 'Cases count'
            FROM student s
            JOIN student_department_cases sc
            ON s.student_id = sc.student_id
            GROUP by s.student_id
            HAVING sc.department_id = 'd001';
        """)

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error in get_student_data: {e}")
        return []

# 2. Create a new function to combine the data
def get_all_student_data_with_cases():
    """Fetches student data and embeds the list of cases for each student."""
    # 1. Get the list of students
    students = get_student_data()

    # 2. Iterate through the student list and fetch cases for each
    for student in students:
        student_id = student.get('student_id')
        if student_id:
            # Call the modified get_cases_data with the student's ID
            cases = get_cases_data(student_id)
            # 3. Add the cases list to the student dictionary
            student['cases'] = cases
        else:
            student['cases'] = [] # Handle students with no valid ID

    return students

# 3. Call the new function
final_data = get_all_student_data_with_cases()
print(final_data)
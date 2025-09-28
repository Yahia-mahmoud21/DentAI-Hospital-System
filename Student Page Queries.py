import sqlite3

def get_student(student_id: str):
    """Fetch student details by student_id."""
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, batch_id, department_id, name, email FROM student WHERE student_id = ?;",
            (student_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            data = dict(row)
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Student not found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

#----------------------------------------------------------------#

import sqlite3

def get_case_of_student_(student_id: str):
    """Fetch  """
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "select s.case_id, c.name, description , checked from student_department_cases as s join cases as c on s.case_id = c.case_id where s.student_id =?;",
            (student_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            data = dict(row)
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Student not found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

#----------------------------------------------------------------#


def get_patient_Queue():
    """Fetch all the patients without a student."""
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "select c.case_id, c.name, c.phone, c.age, c.gender, s.appointment_date, s.description   from student_department_cases as s join cases as c on s.case_id = c.case_id where s.student_id is null;",
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        data = [dict(row) for row in rows]
        if data:
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Student not found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


print(get_patient_Queue())
#----------------------------------------------------------------#


def get_patient_basic_info(case_id : int):
    """Fetch basic information of a patient by case_id."""
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "select case_id, batch_id, name, age, gender, phone from cases where case_id=1;",
            (case_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            data = dict(row)
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Student not found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}
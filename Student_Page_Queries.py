import sqlite3
from db import connect as db_connect

def get_student(student_id: str):
    """Fetch student details by student_id."""
    try:
        conn = db_connect()
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

def get_case_of_student_(student_id: str):
    """Fetch  """
    try:
        conn = db_connect()
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
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "select c.case_id, c.name, c.phone, c.age, c.gender, s.appointment_date, s.description from student_department_cases as s join cases as c on s.case_id = c.case_id where s.student_id is null;",
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        data = [dict(row) for row in rows]
        if data:
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "No patients found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}
#----------------------------------------------------------------#


def get_patient_basic_info(case_id : int):
    """Fetch basic information of a patient by case_id."""
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "select case_id, batch_id, name, age, gender, phone from cases where case_id = ?;",
            (case_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            data = dict(row)
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Patient not found"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}
    

def update_edit_case(description, treatment, before_photo, after_photo, department_reffere , appointment_date, student_id, case_id, department_id):
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
                       update student_department_cases
                        set description = ?, treatment = ?, before_photo = ?, after_photo = ?, department_reffere = ?, 
                       appointment_date = ?, appointment_time = null
                        where student_id = ? and case_id = ? and department_id = ?;
                       """, (description, treatment, before_photo, after_photo, department_reffere , appointment_date, student_id, case_id, department_id))
        conn.commit()
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        print(f"DataBase error : {e}")
        return {"success": False, "message": "Server error, please try again"}
    
def update_edit_case_all(description, treatment, after_photo, student_id, case_id, department_id):
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
                       update student_department_cases
                        set description = ?, treatment = ?, after_photo = ?
                        where student_id = ? and case_id = ? and department_id = ?;
                       """, (description, treatment, after_photo, student_id, case_id, department_id))
        conn.commit()
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        print(f"DataBase error : {e}")
        return {"success": False, "message": "Server error, please try again"}

#----------------------------------------------------------------#

def get_patient_report(case_id: str):
    """
    Fetch patient report data by case_id from cases and student_department_cases tables.
    Returns combined data including patient info, case details, student info, and department info.
    """
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.case_id,
                c.name AS patient_name,
                c.age,
                c.gender,
                c.phone,
                c.batch_id,
                sdc.appointment_date,
                sdc.appointment_time,
                sdc.treatment,
                sdc.description,
                sdc.before_photo,
                sdc.after_photo,
                sdc.department_reffere,
                sdc.checked,
                sdc.notes,
                s.name AS student_name,
                s.email AS student_email,
                d.name AS department_name,
                d.department_id,
                (SELECT fm.name FROM faculty_members fm 
                 WHERE fm.department_id = sdc.department_id 
                 LIMIT 1) AS doctor_name
            FROM cases c
            LEFT JOIN student_department_cases sdc ON c.case_id = sdc.case_id
            LEFT JOIN student s ON sdc.student_id = s.student_id
            LEFT JOIN department d ON sdc.department_id = d.department_id
            WHERE c.case_id = ?
            ORDER BY sdc.appointment_date DESC, sdc.appointment_time DESC
            LIMIT 1;
        """, (case_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            row_dict = dict(row)
            return {"success": True, "data": row_dict}
        else:
            return {"success": False, "message": "Case not found"}
            
    except Exception as e:
        print(f"Database error in get_patient_report: {e}")
        return {"success": False, "message": "Server error, please try again"}
    

# def show_student_cases():
#     try:
#         conn = sqlite3.connect("dental_project_DB.db")
#         conn.row_factory = sqlite3.Row

#         cursor = conn.cursor()

#         cursor.execute("""
#                         select 

# """)
        

#     except sqlite3.Error as e:
#         print(f"DataBase error : {e}")
#         return {"success": False, "message": "Server error, please try again"}
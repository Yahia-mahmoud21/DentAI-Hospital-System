import sqlite3
from .db import connect as db_connect

def get_doctor_data(doctorID):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
select f.name, f.faculty_members_id, f.title, f.department_id, d.name as 'department_name', f.email
from faculty_members f
join department d
on f.department_id = d.department_id
where f.faculty_members_id = ?;
""", ((doctorID,)))
        row = cursor.fetchone()

        if row:
            data = dict(row)
            
        cursor.execute("""
SELECT f.name as 'manager' from department d
join faculty_members f
on d.manager = f.faculty_members_id
where d.department_id = ?;
""", ((data['department_id'],)))
        row = cursor.fetchone()

        if row:
            manager = dict(row)
            data.update(manager)
       
       
        cursor.close()
        conn.close()

        
        
        return data

    except sqlite3.Error as e:
        print(f"DataBase error : {e}")
        return {"success": False, "message": "Server error, please try again"}
    


def get_doctor_student_cases(doctorID):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
select f.department_id 
from faculty_members f
join department d
on f.department_id = d.department_id
where f.faculty_members_id = ?;
""", ((doctorID,)))
        row = cursor.fetchone()

        if row:
            department = dict(row)
            
        cursor.execute("""
SELECT sd.case_id as 'id', 
	   s.name,
	   sd.before_photo as 'before',
	   sd.description,
	   sd.after_photo as 'after',
	   sd.treatment,
	   sd.department_reffere as 'department'
from student_department_cases sd
join student s
on sd.student_id = s.student_id
where sd.department_id = ? and sd.before_photo is not null and checked = '-1';
""", ((department['department_id'],)))
        rows = cursor.fetchall()

        if row:
            data = [dict(row) for row in rows]
        
       
       
        cursor.close()
        conn.close()

        
        
        return data

    except sqlite3.Error as e:
        print(f"DataBase error : {e}")
        return {"success": False, "message": "Server error, please try again"}
    

def update_doctor_student_case(description, treatment, department_reffere, checked, notes, case_id, department_id):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # تحديث بيانات الحالة الحالية
        cursor.execute("""
            UPDATE student_department_cases
            SET description = ?, treatment = ?, department_reffere = ?, checked = ?, notes = ?
            WHERE case_id = ? AND department_id = ?;
        """, (description, treatment, department_reffere, checked, notes, case_id, department_id))
        conn.commit()

        # جلب ID القسم الجديد
        cursor.execute("SELECT department_id FROM department WHERE name = ?;", (department_reffere,))
        new_department = cursor.fetchone()

        if not new_department:
            raise ValueError(f"Department '{department_reffere}' not found")

        new_department_id = new_department["department_id"]

        # جلب البيانات القديمة للحالة
        cursor.execute("""
            SELECT description, before_photo 
            FROM student_department_cases 
            WHERE case_id = ? AND department_id = ?;
        """, (case_id, department_id))
        data = cursor.fetchone()

        if not data:
            raise ValueError("Case data not found")

        new_description = data["description"]
        new_before = data["before_photo"]

        # إدخال نسخة جديدة للحالة في القسم الجديد
        cursor.execute("""
            INSERT INTO student_department_cases (department_id, case_id, description, before_photo)
            VALUES (?, ?, ?, ?);
        """, (new_department_id, case_id, new_description, new_before))

        conn.commit()
        cursor.close()
        conn.close()

        return {"success": True, "message": "Case updated successfully"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "message": str(e)}

import sqlite3
from db import connect as db_connect

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
SELECT 
    sd.case_id            AS id,
    c.name                AS name,
    s.name                AS student_name,
    s.student_id          AS student_id,
    sd.before_photo       AS before,
    sd.description        AS description,
    sd.after_photo        AS after,
    sd.treatment          AS treatment,
    sd.department_reffere AS department,
    sd.checked            AS checked,
    sd.notes              AS notes,
    sd.appointment_date   AS appointment_date
FROM student_department_cases sd
JOIN student s
    ON sd.student_id = s.student_id
JOIN cases c
    ON c.case_id = sd.case_id
WHERE sd.department_id = ?
  AND s.department_id = ?
  AND sd.before_photo IS NOT NULL
ORDER BY sd.case_id DESC;
""", ((department['department_id'], department['department_id'])))
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

        should_advance = str(checked) == "1" and department_reffere

        if should_advance:
            cursor.execute("SELECT department_id FROM department WHERE name = ?;", (department_reffere,))
            new_department = cursor.fetchone()

            if not new_department:
                raise ValueError(f"Department '{department_reffere}' not found")

            new_department_id = new_department["department_id"]

            cursor.execute("""
                SELECT 1 FROM student_department_cases 
                WHERE case_id = ? AND department_id = ?;
            """, (case_id, new_department_id))
            already_exists = cursor.fetchone()

            if not already_exists:
                cursor.execute("""
                    SELECT description, before_photo 
                    FROM student_department_cases 
                    WHERE case_id = ? AND department_id = ?;
                """, (case_id, department_id))
                data = cursor.fetchone()

                if not data:
                    raise ValueError("Case data not found")

                cursor.execute("""
                    INSERT INTO student_department_cases (department_id, case_id, description, before_photo)
                    VALUES (?, ?, ?, ?);
                """, (new_department_id, case_id, data["description"], data["before_photo"]))

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


def approve_student_round(student_id, doctor_id, approved):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Get Doctor's Department
        cursor.execute("SELECT department_id FROM faculty_members WHERE faculty_members_id = ?", (doctor_id,))
        doc_row = cursor.fetchone()
        if not doc_row:
            return {"success": False, "message": "Doctor not found"}
        doc_dept_id = doc_row['department_id']

        # 2. Get Student's Current State
        cursor.execute("SELECT department_id, round_id, batch_id FROM student WHERE student_id = ?", (student_id,))
        student_row = cursor.fetchone()
        if not student_row:
            return {"success": False, "message": "Student not found"}
        
        stu_dept_id = student_row['department_id']
        stu_round_id = student_row['round_id']
        batch_id = student_row['batch_id']

        # 3. Verify Department Match
        if doc_dept_id != stu_dept_id:
             return {"success": False, "message": "Student is not in your department"}

        if approved:
            # 4. Find Next Department
            # We exclude d001 if it's considered a special 'Diagnosis' department not part of rotation, 
            # BUT if the student is currently IN d001, we must include it to find the index.
            # Let's fetch all and filter if needed. For now, assuming all departments are part of rotation.
            cursor.execute("SELECT department_id FROM department WHERE department_id != 'd001' ORDER BY department_id")
            depts = [row['department_id'] for row in cursor.fetchall()]
            
            # If student is in d001, and it's not in list, we might have an issue. 
            # But usually rotation starts after diagnosis or diagnosis is one of them.
            # If student is in a department not in the list (e.g. d001), we can't advance them based on this list.
            # Let's check if stu_dept_id is in depts.
            
            if stu_dept_id not in depts:
                 # Try fetching ALL departments to see if we can find it
                 cursor.execute("SELECT department_id FROM department ORDER BY department_id")
                 all_depts = [row['department_id'] for row in cursor.fetchall()]
                 if stu_dept_id in all_depts:
                     depts = all_depts # Use all departments if current is not in the filtered list
                 else:
                     return {"success": False, "message": f"Current department {stu_dept_id} not found in department list"}

            current_dept_index = depts.index(stu_dept_id)
            next_dept_index = current_dept_index + 1

            # 5. Find Next Round
            cursor.execute("SELECT round_id FROM rounds WHERE batch_id = ? ORDER BY round_id", (batch_id,))
            rounds = [row['round_id'] for row in cursor.fetchall()]
            
            if stu_round_id in rounds:
                current_round_index = rounds.index(stu_round_id)
                next_round_index = current_round_index + 1
            else:
                # If current round is not in the list (maybe null?), start from 0?
                # Or return error.
                if not stu_round_id and rounds:
                    next_round_index = 0 # Start first round
                else:
                    return {"success": False, "message": f"Current round {stu_round_id} not found in rounds list for batch {batch_id}"}

            # Check if we can advance
            if next_dept_index < len(depts):
                 new_dept_id = depts[next_dept_index]
                 
                 # Determine new round
                 if next_round_index < len(rounds):
                     new_round_id = rounds[next_round_index]
                 else:
                     # If rounds are exhausted, keep the current round (or the last one)
                     # This allows department progression even if rounds run out
                     new_round_id = stu_round_id 
                 
                 cursor.execute("UPDATE student SET department_id = ?, round_id = ? WHERE student_id = ?", 
                                (new_dept_id, new_round_id, student_id))
                 conn.commit()
                 return {"success": True, "message": f"Student promoted to {new_dept_id} (Round: {new_round_id})"}
            
            elif next_dept_index >= len(depts):
                 return {"success": True, "message": "Student has completed all departments!"}

        else:
            return {"success": True, "message": "Student not approved. Stays in current department."}

    except Exception as e:
        print(f"Error in approve_student_round: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()

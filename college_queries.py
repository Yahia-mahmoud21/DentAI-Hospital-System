import sqlite3


def get_batchs_data():
    try :
        conn = sqlite3.connect('dental_project_DB.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
       

        cursor.execute("select batch_id as 'id', from_date as 'from', to_date as 'to' from batch;")

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    

def get_college_doctor():
    try :
        conn = sqlite3.connect('dental_project_DB.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
       

        cursor.execute("""select f.faculty_members_id as 'id', f.name as 'name', d.name as 'department', f.title as 'title', f.phone as 'phone', f.email as 'email' from faculty_members f
                            join department d
                            on d.department_id = f.department_id;
                       """)

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def get_college_student():
    try :
        conn = sqlite3.connect('dental_project_DB.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
       

        cursor.execute("""SELECT student_id as 'id', s.name, batch_id as 'batch', d.name as 'department', s.round_id as 'round', phone, email from student s 
                        join department d
                        on d.department_id = s.department_id;
                                            """)

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def get_college_departments():
    try :
        conn = sqlite3.connect('dental_project_DB.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
       

        cursor.execute("""SELECT 
                            d.department_id as 'id', 
                            d.name AS 'name',
                            m.name AS 'manager',
                            d.capacity as 'capacity',
                            COUNT(DISTINCT f.faculty_members_id) AS 'totalDoctors',
                            COUNT(DISTINCT s.student_id) AS 'totalStudents'
                        FROM department d
                        LEFT JOIN faculty_members m
                            ON d.manager = m.faculty_members_id
                        LEFT JOIN faculty_members f
                            ON f.department_id = d.department_id
                        LEFT JOIN student s
                            ON d.department_id = s.department_id
                        GROUP BY 
                            d.department_id, d.name, m.name, d.capacity;
                                                                    """)

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    

def get_college_rounds():
    try :
        conn = sqlite3.connect('dental_project_DB.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
       

        cursor.execute("""SELECT round_id as 'id', batch_id as 'batch', month_year as 'month' FROM rounds;""")

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []



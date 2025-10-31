
import sqlite3
import time
import uuid
from db import connect as db_connect
import uuid


def get_batchs_data():
    try:
        conn = db_connect()
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


def create_batch(batch_id: str, from_year: str, to_year: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO batch (batch_id, from_date, to_date)
            VALUES (?, ?, ?);
            """,
            (batch_id, from_year, to_year)
        )
        conn.commit()
        # Ensure WAL contents are checkpointed so external viewers see the new rows immediately
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "UNIQUE" in msg or "PRIMARY KEY" in msg:
            return {"success": False, "message": f"Batch ID '{batch_id}' already exists"}
        return {"success": False, "message": f"Database constraint error: {msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


# ---------------------- Department ID utilities ---------------------- #
def generate_next_department_id() -> str:
    """Return next department id in the format d001, d002, ... based on max existing id."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT department_id FROM department WHERE department_id LIKE 'd%' OR department_id LIKE 'D%';")
        rows = cursor.fetchall()
        max_num = 0
        for (dep_id,) in rows:
            if not dep_id:
                continue
            s = str(dep_id).strip()
            # allow patterns like d001 or D12 etc.
            if s and (s[0] in ('d','D')):
                num_part = ''.join(ch for ch in s[1:] if ch.isdigit())
                if num_part.isdigit():
                    max_num = max(max_num, int(num_part))
        next_num = max_num + 1
        new_id = f"d{next_num:03d}"
        cursor.close()
        conn.close()
        return new_id
    except sqlite3.Error:
        # Fallback if any DB error occurs
        return "d001"


# ---------------------- Faculty (Doctor) ID utilities ---------------------- #
def generate_next_faculty_id() -> str:
    """Return next faculty_members_id in the format FM001, FM002, ..."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT faculty_members_id FROM faculty_members WHERE faculty_members_id LIKE 'FM%' OR faculty_members_id LIKE 'fm%';")
        rows = cursor.fetchall()
        max_num = 0
        for (fm_id,) in rows:
            if not fm_id:
                continue
            s = str(fm_id).strip()
            if len(s) >= 3 and (s[0].upper() == 'F' and s[1].upper() == 'M'):
                num_part = ''.join(ch for ch in s[2:] if ch.isdigit())
                if num_part.isdigit():
                    max_num = max(max_num, int(num_part))
        next_num = max_num + 1
        new_id = f"FM{next_num:03d}"
        cursor.close()
        conn.close()
        return new_id
    except sqlite3.Error:
        return "FM001"


# ---------------------- Student ID utilities ---------------------- #
def generate_next_student_id() -> str:
    """Return next student_id in the format stu001, stu002, ..."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id FROM student WHERE student_id LIKE 'stu%' OR student_id LIKE 'STU%';")
        rows = cursor.fetchall()
        max_num = 0
        for (st_id,) in rows:
            if not st_id:
                continue
            s = str(st_id).strip()
            s_up = s.upper()
            if s_up.startswith('STU'):
                num_part = ''.join(ch for ch in s[3:] if ch.isdigit())
                if num_part.isdigit():
                    max_num = max(max_num, int(num_part))
        next_num = max_num + 1
        new_id = f"stu{next_num:03d}"
        cursor.close()
        conn.close()
        return new_id
    except sqlite3.Error:
        return "stu001"


# ---------------------- Rounds CRUD ---------------------- #
def generate_next_round_id() -> str:
    """Return next round_id in the format R001, R002, ..."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT round_id FROM rounds WHERE round_id LIKE 'R%' OR round_id LIKE 'r%';")
        rows = cursor.fetchall()
        max_num = 0
        for (rid,) in rows:
            if not rid:
                continue
            s = str(rid).strip()
            if s and (s[0] in ('R','r')):
                num_part = ''.join(ch for ch in s[1:] if ch.isdigit())
                if num_part.isdigit():
                    max_num = max(max_num, int(num_part))
        next_num = max_num + 1
        new_id = f"R{next_num:03d}"
        cursor.close()
        conn.close()
        return new_id
    except sqlite3.Error:
        return "R001"

def create_round(batch_id: str, month: str, name: str):
    try:
        if not batch_id:
            return {"success": False, "message": "Batch is required"}
        if not month:
            return {"success": False, "message": "Month is required"}
        
        # Helper: convert given month to YYYY-MM using the batch start year when needed
        def to_year_month(month_value: str, batch_start_year: str) -> str | None:
            m = (month_value or "").strip()
            if not m:
                return None
            # Already in YYYY-MM
            if len(m) == 7 and m[:4].isdigit() and m[4] == '-' and m[5:7].isdigit():
                return m
            months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            name_to_num = {name.lower(): idx+1 for idx, name in enumerate(months)}
            name_to_num.update({name[:3].lower(): idx+1 for idx, name in enumerate(months)})
            # Numeric month
            if m.isdigit() and 1 <= int(m) <= 12:
                return f"{batch_start_year}-{int(m):02d}"
            # Name/abbr
            if m.lower() in name_to_num:
                return f"{batch_start_year}-{name_to_num[m.lower()]:02d}"
            # Formats like MM-YYYY or YYYY/MM
            if '-' in m or '/' in m:
                sep = '-' if '-' in m else '/'
                parts = [p.strip() for p in m.split(sep) if p.strip()]
                if len(parts) == 2:
                    a, b = parts
                    if a.isdigit() and len(a) == 4 and b.isdigit():
                        # YYYY-MM or YYYY-M
                        if 1 <= int(b) <= 12:
                            return f"{a}-{int(b):02d}"
                    if b.isdigit() and len(b) == 4 and a.isdigit():
                        # MM-YYYY or M-YYYY
                        if 1 <= int(a) <= 12:
                            return f"{b}-{int(a):02d}"
            return None
        # 'name' column may not exist; ignore it and use month_year only
        conn = db_connect()
        cursor = conn.cursor()
        # Resolve/validate batch id (accept BYYYY or YYYY); auto-create if inferrable
        resolved_batch = None
        candidates = []
        s = str(batch_id).strip()
        if s:
            if s.upper().startswith('B') and s[1:].isdigit():
                candidates = [s]
            elif s.isdigit():
                candidates = [f"B{s}"]
            else:
                # Accept display strings like "2013-2014" → use starting year as batch id
                # Also tolerate spaces around dash
                first_part = s.split('-', 1)[0].strip()
                if first_part.isdigit():
                    candidates = [f"B{first_part}"]
            # Note: additional display-string handling is covered above; avoid duplicate else
        for candidate in candidates:
            cursor.execute("SELECT 1 FROM batch WHERE batch_id = ?;", (candidate,))
            if cursor.fetchone():
                resolved_batch = candidate
                break
        if not resolved_batch and candidates:
            # Auto-create batch using inferred years: from=YYYY, to=YYYY+1
            try:
                year = int(candidates[0][1:]) if candidates[0].upper().startswith('B') else int(candidates[0])
                from_year, to_year = str(year), str(year + 1)
                cursor.execute(
                    """
                    INSERT INTO batch (batch_id, from_date, to_date)
                    VALUES (?, ?, ?);
                    """,
                    (f"B{from_year}", from_year, to_year)
                )
                conn.commit()
                resolved_batch = f"B{from_year}"
            except Exception:
                pass
        if not resolved_batch:
            conn.close()
            return {"success": False, "message": f"Batch '{batch_id}' not found"}
        # Compute YYYY-MM value for DB constraint
        start_year = (resolved_batch[1:5] if resolved_batch and len(resolved_batch) >= 5 else None)
        ym_value = to_year_month(month, start_year or "0000")
        if not ym_value or not (len(ym_value) == 7 and ym_value[:4].isdigit() and ym_value[4] == '-' and ym_value[5:7].isdigit()):
            return {"success": False, "message": f"Invalid month format. Expected YYYY-MM; got '{month}'."}

        # Debug logging for diagnosis
        try:
            print({"create_round": {"input_batch": batch_id, "resolved_batch": resolved_batch, "input_month": month}})
        except Exception:
            pass
        rid = generate_next_round_id()
        # Try multiple schema variants for compatibility
        inserted = False
        try:
            cursor.execute(
                """
                INSERT INTO rounds (round_id, batch_id, month_year, name)
                VALUES (?, ?, ?, ?);
                """,
                (rid, resolved_batch, ym_value, name)
            )
            inserted = True
        except sqlite3.OperationalError:
            try:
                cursor.execute(
                    """
                    INSERT INTO rounds (round_id, batch_id, month_year)
                    VALUES (?, ?, ?);
                    """,
                    (rid, resolved_batch, ym_value)
                )
                inserted = True
            except sqlite3.OperationalError:
                try:
                    cursor.execute(
                        """
                        INSERT INTO rounds (round_id, batch_id, month, name)
                        VALUES (?, ?, ?, ?);
                        """,
                        (rid, resolved_batch, ym_value, name)
                    )
                    inserted = True
                except sqlite3.OperationalError:
                    cursor.execute(
                        """
                        INSERT INTO rounds (round_id, batch_id, month)
                        VALUES (?, ?, ?);
                        """,
                        (rid, resolved_batch, ym_value)
                    )
                    inserted = True
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "UNIQUE" in msg or "PRIMARY KEY" in msg:
            return {"success": False, "message": "Round already exists"}
        return {"success": False, "message": f"Database constraint error: {msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": f"DB error while creating round: {str(e)}"}

def update_round(round_id: str, batch_id: str, month: str, name: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        # Convert to YYYY-MM using batch start year if necessary
        def to_year_month(month_value: str, batch_start_year: str) -> str | None:
            m = (month_value or "").strip()
            if not m:
                return None
            if len(m) == 7 and m[:4].isdigit() and m[4] == '-' and m[5:7].isdigit():
                return m
            months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            name_to_num = {name.lower(): idx+1 for idx, name in enumerate(months)}
            name_to_num.update({name[:3].lower(): idx+1 for idx, name in enumerate(months)})
            if m.isdigit() and 1 <= int(m) <= 12:
                return f"{batch_start_year}-{int(m):02d}"
            if m.lower() in name_to_num:
                return f"{batch_start_year}-{name_to_num[m.lower()]:02d}"
            if '-' in m or '/' in m:
                sep = '-' if '-' in m else '/'
                parts = [p.strip() for p in m.split(sep) if p.strip()]
                if len(parts) == 2:
                    a, b = parts
                    if a.isdigit() and len(a) == 4 and b.isdigit() and 1 <= int(b) <= 12:
                        return f"{a}-{int(b):02d}"
                    if b.isdigit() and len(b) == 4 and a.isdigit() and 1 <= int(a) <= 12:
                        return f"{b}-{int(a):02d}"
            return None
        # Resolve batch id similar to create; auto-create if inferrable
        resolved_batch = None
        candidates = []
        s = str(batch_id).strip()
        if s:
            if s.upper().startswith('B') and s[1:].isdigit():
                candidates = [s]
            elif s.isdigit():
                candidates = [f"B{s}"]
        for candidate in candidates:
            cursor.execute("SELECT 1 FROM batch WHERE batch_id = ?;", (candidate,))
            if cursor.fetchone():
                resolved_batch = candidate
                break
        if not resolved_batch and candidates:
            try:
                year = int(candidates[0][1:]) if candidates[0].upper().startswith('B') else int(candidates[0])
                from_year, to_year = str(year), str(year + 1)
                cursor.execute(
                    """
                    INSERT INTO batch (batch_id, from_date, to_date)
                    VALUES (?, ?, ?);
                    """,
                    (f"B{from_year}", from_year, to_year)
                )
                conn.commit()
                resolved_batch = f"B{from_year}"
            except Exception:
                pass
        if not resolved_batch:
            cursor.close()
            conn.close()
            return {"success": False, "message": f"Batch '{batch_id}' not found"}
        # Compute YYYY-MM using batch start year
        start_year = (resolved_batch[1:5] if resolved_batch and len(resolved_batch) >= 5 else None)
        ym_value = to_year_month(month, start_year or "0000") or month

        # Try multiple schema variants for compatibility
        try:
            cursor.execute(
                """
                UPDATE rounds
                SET batch_id = ?, month_year = ?, name = ?
                WHERE round_id = ?;
                """,
                (resolved_batch, ym_value, name, round_id)
            )
        except sqlite3.OperationalError:
            try:
                cursor.execute(
                    """
                    UPDATE rounds
                    SET batch_id = ?, month_year = ?
                    WHERE round_id = ?;
                    """,
                    (resolved_batch, ym_value, round_id)
                )
            except sqlite3.OperationalError:
                try:
                    cursor.execute(
                        """
                        UPDATE rounds
                        SET batch_id = ?, month = ?, name = ?
                        WHERE round_id = ?;
                        """,
                        (resolved_batch, ym_value, name, round_id)
                    )
                except sqlite3.OperationalError:
                    cursor.execute(
                        """
                        UPDATE rounds
                        SET batch_id = ?, month = ?
                        WHERE round_id = ?;
                        """,
                        (resolved_batch, ym_value, round_id)
                    )
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        msg = str(e)
        return {"success": False, "message": f"Database constraint error: {msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": f"DB error while updating round: {str(e)}"}

def delete_round(round_id: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rounds WHERE round_id = ?;", (round_id,))
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

def update_batch(batch_id: str, from_year: str, to_year: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE batch
            SET from_date = ?, to_date = ?
            WHERE batch_id = ?;
            """,
            (from_year, to_year, batch_id)
        )
        conn.commit()
        # Ensure WAL contents are checkpointed so external viewers see the updates immediately
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def delete_batch(batch_id: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM batch WHERE batch_id = ?;", (batch_id,))
        conn.commit()
        # Ensure WAL contents are checkpointed so external viewers see the deletion immediately
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def get_college_doctor():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                f.faculty_members_id AS 'id', 
                f.name AS 'name', 
                COALESCE(d.name, '') AS 'department', 
                COALESCE(f.title, '') AS 'title', 
                COALESCE(f.phone, '') AS 'phone', 
                COALESCE(f.email, '') AS 'email'
            FROM faculty_members f
            LEFT JOIN department d
                ON d.department_id = f.department_id
            ORDER BY f.faculty_members_id;
            """
        )

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []


def create_college_doctor(doctor_id: str, name: str, department_name: str, title: str, phone: str, email: str):
    try:
        department_id = _get_department_id_by_name(department_name)
        if not department_id:
            # Auto-create department if not found
            try:
                new_dep_id = generate_next_department_id()
                conn_tmp = db_connect()
                cur_tmp = conn_tmp.cursor()
                cur_tmp.execute(
                    """
                    INSERT INTO department (department_id, name, manager, capacity, description)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (new_dep_id, department_name, None, 0, None)
                )
                conn_tmp.commit()
                cur_tmp.close()
                conn_tmp.close()
                department_id = new_dep_id
            except sqlite3.Error as e:
                print(f"Auto-create department failed: {e}")
                return {"success": False, "message": "Department not found and could not be created"}

        conn = db_connect()
        cursor = conn.cursor()
        # Provide a default random password if the schema requires it
        default_password = uuid.uuid4().hex
        try:
            cursor.execute(
                """
                INSERT INTO faculty_members (faculty_members_id, name, department_id, title, phone, email, password)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (doctor_id, name, department_id, title, phone or None, email or None, default_password)
            )
        except sqlite3.OperationalError:
            # Fallback for schemas without password column
            cursor.execute(
                """
                INSERT INTO faculty_members (faculty_members_id, name, department_id, title, phone, email)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (doctor_id, name, department_id, title, phone or None, email or None)
            )
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        # If the doctor ID already exists, update the existing doctor instead of failing
        msg = str(e)
        print(f"Integrity error on insert doctor: {msg}")
        try:
            if "UNIQUE" in msg or "PRIMARY KEY" in msg:
                conn = db_connect()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE faculty_members
                    SET name = ?, department_id = ?, title = ?, phone = ?, email = ?
                    WHERE faculty_members_id = ?;
                    """,
                    (name, department_id, title, phone or None, email or None, doctor_id)
                )
                conn.commit()
                cursor.close()
                conn.close()
                return {"success": True}
        except Exception as upsert_err:
            print(f"Upsert doctor failed: {upsert_err}")
            return {"success": False, "message": f"Database error while updating existing doctor '{doctor_id}': {upsert_err}"}
        return {"success": False, "message": f"Database constraint error: {msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def update_college_doctor(doctor_id: str, name: str, department_name: str, title: str, phone: str, email: str):
    try:
        department_id = _get_department_id_by_name(department_name)
        if not department_id:
            return {"success": False, "message": "Department not found"}

        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE faculty_members
            SET name = ?, department_id = ?, title = ?, phone = ?, email = ?
            WHERE faculty_members_id = ?;
            """,
            (name, department_id, title, phone, email, doctor_id)
        )
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def delete_college_doctor(doctor_id: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faculty_members WHERE faculty_members_id = ?;", (doctor_id,))
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

def get_college_student():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use LEFT JOIN so students without a matching department record (e.g. inserted manually)
        # still appear in the listing. Department name will be NULL in that case.
        cursor.execute(
            """SELECT student_id as 'id', s.name, batch_id as 'batch', d.name as 'department', s.round_id as 'round', phone, email
                        FROM student s
                        LEFT JOIN department d
                        ON d.department_id = s.department_id;"""
        )

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []


def get_college_departments():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """SELECT 
                            d.department_id as 'id', 
                            d.name AS 'name',
                            m.faculty_members_id AS 'managerId',
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
                            d.department_id, d.name, m.faculty_members_id, m.name, d.capacity;"""
        )

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []


def get_college_rounds():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Be compatible with schemas that have either month_year or month
        try:
            cursor.execute("SELECT round_id as 'id', batch_id as 'batch', month_year as 'month' FROM rounds;")
        except sqlite3.OperationalError:
            cursor.execute("SELECT round_id as 'id', batch_id as 'batch', month as 'month', COALESCE(name, month) as 'name' FROM rounds;")

        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        cursor.close()
        conn.close()

        return data

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []


# ---------------------- Students CRUD (College) ---------------------- #


def _get_department_id_by_name(department_name: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # If caller already passed an ID-like value, try it first
        if isinstance(department_name, str) and department_name.upper().startswith('D'):
            cursor.execute("SELECT department_id FROM department WHERE department_id = ?;", (department_name,))
            row = cursor.fetchone()
            if row:
                cursor.close()
                conn.close()
                return row["department_id"]

        # Otherwise, try to resolve by name
        cursor.execute("SELECT department_id FROM department WHERE name = ?;", (department_name,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None
        return row["department_id"]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None


def update_college_student(student_id: str, name: str, batch_id: str, department_name: str, round_id: str, phone: str, email: str):
    try:
        department_id = _get_department_id_by_name(department_name)
        if not department_id:
            return {"success": False, "message": "Department not found"}

        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE student
            SET name = ?, batch_id = ?, department_id = ?, round_id = ?, phone = ?, email = ?
            WHERE student_id = ?;
            """,
            (name, batch_id, department_id, round_id, phone, email, student_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def delete_college_student(student_id: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student WHERE student_id = ?;", (student_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def delete_college_department(department_id: str):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM department WHERE department_id = ?;", (department_id,))
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}

def create_college_student(student_id: str, name: str, batch_id: str, department_name: str, round_id: str, phone: str, email: str):
    """
    Create a new student.
    Note: Requires an explicit student_id provided by the caller to avoid guessing ID scheme.
    """
    try:
        department_id = _get_department_id_by_name(department_name)
        if not department_id:
            return {"success": False, "message": f"Department '{department_name}' not found. Please check the department name."}

        # Verify batch exists
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT batch_id FROM batch WHERE batch_id = ?;", (batch_id,))
        if not cursor.fetchone():
            conn.close()
            return {"success": False, "message": f"Batch '{batch_id}' not found. Please check the batch ID."}
        
        # Resolve/verify round id (accept exact id, "ID - Month", or month value)
        provided_round = (round_id or "").strip()
        db_round_id = None

        # 1) Try exact id
        cursor.execute("SELECT round_id FROM rounds WHERE round_id = ?;", (provided_round,))
        row = cursor.fetchone()
        if row:
            db_round_id = row[0]
        else:
            # 2) Try split format: "ID - Month"
            if "-" in provided_round:
                candidate_id = provided_round.split("-", 1)[0].strip()
                cursor.execute("SELECT round_id FROM rounds WHERE round_id = ?;", (candidate_id,))
                row = cursor.fetchone()
                if row:
                    db_round_id = row[0]
            # 3) Try month_year match
            if not db_round_id:
                cursor.execute("SELECT round_id FROM rounds WHERE month_year = ?;", (provided_round,))
                row = cursor.fetchone()
                if row:
                    db_round_id = row[0]

        if not db_round_id:
            conn.close()
            return {"success": False, "message": f"Round '{round_id}' not found. Please select a valid round."}

        # Normalize optional fields and satisfy NOT NULL if required by schema
        phone = phone if (phone and str(phone).strip()) else None
        email = email if (email and str(email).strip()) else None

        # Check NOT NULL constraints for phone/email
        try:
            cursor.execute("PRAGMA table_info(student);")
            cols = cursor.fetchall()
            col_notnull = {row[1]: row[3] for row in cols}  # name -> notnull(0/1)
            if phone is None and col_notnull.get('phone') == 1:
                phone = ""
            if email is None and col_notnull.get('email') == 1:
                email = ""
        except Exception as _:
            # Fallback: leave values as-is
            pass

        # Create a random default password to avoid NOT NULL/UNIQUE constraints on password
        default_password = uuid.uuid4().hex
        # Retry on database lock
        attempts = 0
        while True:
            try:
                cursor.execute(
                    """
                    INSERT INTO student (student_id, name, batch_id, department_id, round_id, phone, email, password)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (student_id, name, batch_id, department_id, db_round_id, phone, email, default_password)
                )
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempts < 5:
                    attempts += 1
                    time.sleep(0.3 * attempts)
                    continue
                raise
        cursor.close()
        conn.close()

        return {"success": True}
    except sqlite3.IntegrityError as e:
        # If the student_id already exists, update the existing student instead of failing
        error_msg = str(e)
        print(f"Integrity error on insert student: {e}")
        try:
            if "UNIQUE" in error_msg or "PRIMARY KEY" in error_msg:
                # Perform UPDATE as an upsert behavior
                conn = db_connect()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE student
                    SET name = ?, batch_id = ?, department_id = ?, round_id = ?, phone = ?, email = ?
                    WHERE student_id = ?;
                    """,
                    (name, batch_id, department_id, db_round_id, phone, email, student_id)
                )
                conn.commit()
                cursor.close()
                conn.close()
                return {"success": True}
        except Exception as upsert_err:
            print(f"Upsert failed: {upsert_err}")
            return {"success": False, "message": f"Database error while updating existing student '{student_id}': {upsert_err}"}
        # Fallback generic error
        return {"success": False, "message": f"Database constraint error: {error_msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        import traceback
        print(traceback.format_exc())
        return {"success": False, "message": f"Database error: {str(e)}"}


def create_college_department(department_id, name, manager_id, capacity, description=None):
    """Create a department (college management). Includes debug prints and detailed error reporting."""
    try:
        print(f"create_college_department args: id={department_id!r}, name={name!r}, manager_id={manager_id!r}, capacity={capacity!r}, description={description!r}")
        conn = db_connect()
        cursor = conn.cursor()
        # Validate manager exists; if not, set NULL to avoid FK violation
        mgr = manager_id if manager_id else None
        try:
            if mgr:
                cursor.execute("SELECT 1 FROM faculty_members WHERE faculty_members_id = ?;", (mgr,))
                if not cursor.fetchone():
                    mgr = None
        except Exception:
            mgr = None
        cursor.execute(
            """
            INSERT INTO department (department_id, name, manager, capacity, description)
            VALUES (?, ?, ?, ?, ?);
            """,
            (department_id, name, mgr, int(capacity) if capacity is not None else None, description or None)
        )
        rowcount = cursor.rowcount
        print(f"create_college_department rowcount after insert: {rowcount}")
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        if rowcount > 0:
            return {"success": True}
        else:
            return {"success": False, "message": f"DB insert failed; rowcount={rowcount}; id={department_id}, name={name}, manager_id={manager_id}, capacity={capacity}"}
    except sqlite3.IntegrityError as e:
        msg = str(e)
        print(f"IntegrityError: {msg}")
        return {"success": False, "message": f"DB Integrity Error: {msg}"}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": f"DB Error: {str(e)}"}

def update_college_department(department_id, name, manager_id, capacity, description=None):
    """Update a department (college management)."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE department SET name=?, manager=?, capacity=?, description=? WHERE department_id=?;
            """,
            (name, manager_id, int(capacity) if capacity is not None else None, description or None, department_id)
        )
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        cursor.close()
        conn.close()
        return {"success": True}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}




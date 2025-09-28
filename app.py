from fastapi import FastAPI, Request,Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import uvicorn
import sqlite3
from queries import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 1. GET endpoint بسيط
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/student", response_class=HTMLResponse)
def student(request: Request):

    return templates.TemplateResponse("student.html", {"request": request})

@app.get("/doctor", response_class=HTMLResponse)
def doctor(request: Request):
    return templates.TemplateResponse("doctor_all.html", {"request": request})

@app.get("/college", response_class=HTMLResponse)
def collge(request: Request):
    return templates.TemplateResponse("college.html", {"request": request})

@app.get("/patient", response_class=HTMLResponse)
def patient(request: Request):
    return templates.TemplateResponse("patient.html", {"request": request})









# Login pages
@app.get("/student/login", response_class=HTMLResponse)
def login_student(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request})
















@app.get("/doctor/login", response_class=HTMLResponse)
def login_doctor(request: Request):
    return templates.TemplateResponse("login_doctor.html", {"request": request})

@app.get("/college/login", response_class=HTMLResponse)
def login_collge(request: Request):
    return templates.TemplateResponse("login_college.html", {"request": request})

@app.get("/patient/login", response_class=HTMLResponse)
def login_patient(request: Request):
    return templates.TemplateResponse("login_patient.html", {"request": request})




# login for student
class LoginStudentModel(BaseModel):
    email: str
    password: str

@app.post("/api/v1/student/login")
def api_login_student(data: LoginStudentModel):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        st_email = data.email
        st_password = data.password
        cursor.execute("SELECT student_id, email, password  FROM student WHERE email = ? AND password = ?", (st_email, st_password))
        result = cursor.fetchone()
        conn.close()

        if result:  
            return {
                "success": True,
                "message": "Login successful",
                "ID" : result[0]
            }
        else:
            return {"success": False, "message": "Invalid credentials"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}







class LoginDoctorModel(BaseModel):
    email: str
    password: str

@app.post("/api/v1/doctor/login")
def api_login_doctor(data: LoginDoctorModel):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        doc_email = data.email
        doc_password = data.password
        cursor.execute("SELECT faculty_members_id, email, password  FROM faculty_members WHERE email = ? AND password = ?", (doc_email, doc_password))
        result = cursor.fetchone()
        conn.close()

        if result:  
            return {
                "success": True,
                "message": "Login successful"
            }
        else:
            return {"success": False, "message": "Invalid credentials"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}



class LoginCollegeModel(BaseModel):
    email: str
    password: str

@app.post("/api/v1/college/login")
def api_login_collge(data: LoginCollegeModel):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        col_email = data.email
        col_password = data.password
        cursor.execute("SELECT collage_id, email, password  FROM collage WHERE email = ? AND password = ?", (col_email, col_password))
        result = cursor.fetchone()
        conn.close()

        if result:  
            return {
                "success": True,
                "message": "Login successful"
            }
        else:
            return {"success": False, "message": "Invalid credentials"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}



class LoginPatientModel(BaseModel):
    Id: str
    Name: str

@app.post("/api/v1/patient/login")
def api_login_patient(data: LoginPatientModel):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        case_Id = int(data.Id)
        case_Name = data.Name
        cursor.execute("SELECT case_id, name FROM cases WHERE case_id = ? AND name = ?", (case_Id, case_Name))
        result = cursor.fetchone()
        conn.close()

        if result:  
            return {
                "success": True,
                "message": "Login successful"
            }
        else:
            return {"success": False, "message": "Invalid credentials"}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}



# get case data from home page 
class GetCaseData(BaseModel):
    patientName: str
    patientAge: int
    gender: str
    phoneNumber: str
    sick : str
    

@app.post("/api/v1/home/case")
def get_cases_data_regestration(data: GetCaseData):
    print(data.patientName, data.patientAge, data.gender,data.phoneNumber, data.sick)

    try:
        insert_cases(data.patientName, int(data.patientAge), data.gender, data.phoneNumber, data.sick)

        return {
                "success": True,
                "message": "Data Added successful"
            }
    except sqlite3.Error as e:
         print(f"Database error: {e}")
         return {"success": False, "message": "Server error, please try again"}


   




# get Student basic info
@app.get("/api/v1/student/{student_id}")
def get_student(student_id: str):
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
    
@app.get("/api/v1/student/{student_id}")
def get_student(student_id: str):
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


@app.get("/api/patients/{department_id}")
def get_patients(department_id : str):
    conn = sqlite3.connect("dental_project_DB.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        select c.case_id, c.name, c.phone, c.age, c.gender, c.sick, 
               s.appointment_date, s.description
        from student_department_cases as s 
        join cases as c on s.case_id = c.case_id 
        where s.student_id is null and s.department_id = ? order by  s.appointment_date DESC ;
    """, (department_id,))
    
    rows = cursor.fetchall()
    conn.close()

    patients = [dict(row) for row in rows]
    return {"patients": patients}



@app.get("/api/patients/all")
def get_patients_all():
    conn = sqlite3.connect("dental_project_DB.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        select c.case_id, c.name, c.phone, c.age, c.gender, 
               s.appointment_date, s.description
        from student_department_cases as s 
        join cases as c on s.case_id = c.case_id 
        where s.student_id is null and s.department_id != 'D001';
    """)
    
    rows = cursor.fetchall()
    conn.close()

    patients = [dict(row) for row in rows]
    return {"patients": patients}



@app.get("/api/student/cases/table/{student_id}")
def get_student_cases_table(student_id : str):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""select s.case_id , c.name , s.description, s.treatment, s.before_photo, 
                            s.after_photo, s.department_id, s.appointment_date, s.checked 
                            from student_department_cases s
                            join cases c
                            on c.case_id = s.case_id
                            where student_id = ?
                            order by s.case_id DESC;
                            """, (student_id,))
        
        rows = cursor.fetchall()
        conn.close()

        table = [dict(row) for row in rows]
        return {"success": True, "table" : table}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}



class PostPatientToPatient(BaseModel):
    case_id: int
    department_id: str
    student_id: str


@app.post("/api/v1/student/patient")
def post_patient_to_patient(data: PostPatientToPatient):
    print(data.case_id, data.department_id, data.student_id)

    update_studentID_for_case( data.student_id, data.case_id, data.department_id)
    
    return {
            "success": True,
            "message": "Data Added successful"
        }


@app.post("/api/v1/edit/case")
async def edit_case(
    student_id: str = Form(...),
    case_id: str = Form(...),
    description: str = Form(...),
    treatment: str = Form(...),
    department: str = Form(...),
    date: str = Form(...),
    before: UploadFile = File(...),
    after: UploadFile = File(...)
):
    print("Student:", student_id)
    print("Case:", case_id)
    print("Desc:", description)
    print("Dept:", department)
    print("Date:", date)
    print("Before file:", before.filename if before else "No file")
    print("After file:", after.filename if after else "No file")

    return {
        "success": True,
        "message": "Data Added successful"
    }


model_path = "yasserrmd/SciReason-LFM2-2.6B"  

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

class AiDiagnosis(BaseModel):
    age: int
    sick: str

def build_prompt(age, sick):
    return f"""
You are an experienced dental doctor.
A patient is {age} years old and reports the following issue: "{sick}".
Please provide:
1. The most likely dental diagnosis.
2. Possible related conditions.
3. Brief explanation.
4. Suggested examinations or next steps.
"""

@app.post("/api/ai/diagnosis")
def Ai_diagnosis(data: AiDiagnosis):
    print("Age:", data.age, "Sick:", data.sick)

    messages = [
        {"role": "user", "content": build_prompt(data.age, data.sick)}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print("AI Response:", response_text)

    return {
        "AI_response": response_text,
        "success": True,
        "message": "Data Added successful"
    }





if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)



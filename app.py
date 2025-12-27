from fastapi import FastAPI, Request,Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import uvicorn
import sqlite3
from DataBase.db import connect as db_connect
from DataBase.queries import *
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import torchvision.transforms as transforms
from Image_classification_model.model import RegularizedDentalClassifier
from PIL import Image
import os
import io
import uuid
from threading import Thread
from DataBase.Student_Page_Queries import *
from DataBase.doctor_query import *
from DataBase.college_queries import *

app = FastAPI()

# Enable CORS for frontend requests and preflight handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. GET endpoint بسيط
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Avoid browser 404 spam for favicon when none is provided
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(status_code=204)

# Render the home page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Get a list of department names except with ID 'D001'
def get_departments():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM department where department_id != 'd001';")
    departments = [row[0] for row in cursor.fetchall()]
    conn.close()
    return departments


# Render the student page with a list of available departments
@app.get("/student", response_class=HTMLResponse)
def student(request: Request):
    departments = get_departments()
    return templates.TemplateResponse(
        "student.html", {"request": request, "departments": departments}
    )


# Render the doctor page with a list of departments
@app.get("/doctor", response_class=HTMLResponse)
def doctor(request: Request):
    departments = get_departments()
    return templates.TemplateResponse(
        "doctor_all.html", {"request": request, "departments": departments}
    )

# Render the AI page
@app.get("/AI", response_class=HTMLResponse)
def AI(request: Request):
    return templates.TemplateResponse("Ai.html", {"request": request})

# Render the college page
@app.get("/college", response_class=HTMLResponse)
def collge(request: Request):
    return templates.TemplateResponse("college.html", {"request": request})

# Render the patient page
@app.get("/patient", response_class=HTMLResponse)
def patient(request: Request):
    return templates.TemplateResponse("patient copy.html", {"request": request})


# Render the book page
@app.get("/book", response_class=HTMLResponse)
def book(request: Request):
    return templates.TemplateResponse("book.html", {"request": request})









# Login pages
# Render the login page for students
@app.get("/student/login", response_class=HTMLResponse)
def login_student(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request})


# Render the login page for secretary
@app.get("/secretary/login", response_class=HTMLResponse)
def login_student(request: Request):
    return templates.TemplateResponse("login_secretary.html", {"request": request})


# Render the login page for doctors
@app.get("/doctor/login", response_class=HTMLResponse)
def login_doctor(request: Request):
    return templates.TemplateResponse("login_doctor.html", {"request": request})

# Render the login page for college staff
@app.get("/college/login", response_class=HTMLResponse)
def login_collge(request: Request):
    return templates.TemplateResponse("login_college.html", {"request": request})

# Render the login page for patients
@app.get("/patient/login", response_class=HTMLResponse)
def login_patient(request: Request):
    return templates.TemplateResponse("login_patient.html", {"request": request})



# login for student
class LoginStudentModel(BaseModel):
    email: str
    password: str
    
# Student login API endpoint
@app.post("/api/v1/student/login")
def api_login_student(data: LoginStudentModel):
    try:
        conn = db_connect()
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
    

class RegisterStudentModel(BaseModel):
    id : str
    email: str
    password : str

# Student registration API endpoint
@app.post("/api/v1/student/register")
def api_register_student(data: RegisterStudentModel):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        st_email = data.email
        st_password = data.password
        st_id = data.id
        
        # Check if student exists and email is not already registered
        cursor.execute("SELECT student_id FROM student WHERE student_id = ? AND (email IS NULL OR email = '')", (st_id,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            return {
                "success": False,
                "message": "Student ID not found or already registered. Please contact the administration"
            }
        
        # Update student with email and password
        cursor.execute("UPDATE student SET email = ?, password = ? WHERE student_id = ?", (st_email, st_password, st_id))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Account created successfully",
            "student_id": st_id
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        return {
            "success": False,
            "message": f"Error creating account: {str(e)}"
        }





# login for doctor
class LoginDoctorModel(BaseModel):
    email: str
    password: str

# Doctor login API endpoint
@app.post("/api/v1/doctor/login")
def api_login_doctor(data: LoginDoctorModel):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        dr_email = data.email
        dr_password = data.password
        cursor.execute("SELECT faculty_members_id, email, password  FROM faculty_members WHERE email = ? AND password = ?", (dr_email, dr_password))
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
    

class RegisterDoctorModel(BaseModel):
    id : str
    email: str
    password : str

# Doctor registration API endpoint
@app.post("/api/v1/doctor/register")
def api_register_student(data: RegisterDoctorModel):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        dr_email = data.email
        dr_password = data.password
        dr_id = data.id
        
        # Check if student exists and email is not already registered
        cursor.execute("SELECT faculty_members_id FROM faculty_members WHERE faculty_members_id = ? AND (email IS NULL OR email = '')", (dr_id,))
        doctor = cursor.fetchone()
        
        if not doctor:
            conn.close()
            return {
                "success": False,
                "message": "faculty_member ID not found or already registered. Please contact the administration"
            }
        
        # Update student with email and password
        cursor.execute("UPDATE faculty_members SET email = ?, password = ? WHERE faculty_members_id = ?", (dr_email, dr_password, dr_id))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Account created successfully",
            "student_id": dr_id
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        return {
            "success": False,
            "message": f"Error creating account: {str(e)}"
        }
###############################################################################################3



class LoginCollegeModel(BaseModel):
    email: str
    password: str

# College login API endpoint
@app.post("/api/v1/college/login")
def api_login_collge(data: LoginCollegeModel):
    try:
        conn = db_connect()
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

##############################################################################################

#login

class LoginAdminSecretarymodel(BaseModel):
    email: str
    password: str

# Secretary login API endpoint (checks against hardcoded admin credentials)
@app.post("/api/v1/Secretary/login")
def api_login_AdminSecretary(data: LoginAdminSecretarymodel):
       
        col_email = data.email
        col_password = data.password
       

        if col_email == "admin@gmail.com" and col_password=="1234":  
            return {
                "success": True,
                "message": "Login successful"
            }
        else:
            return {"success": False, "message": "Invalid credentials"}

###############################################################################################

class LoginPatientModel(BaseModel):
    Id: str
    Name: str

# Patient login API endpoint
@app.post("/api/v1/patient/login")
def api_login_patient(data: LoginPatientModel):
    try:
        conn = db_connect()
        cursor = conn.cursor()
        case_Id = int(data.Id)
        case_Name = data.Name
        cursor.execute("SELECT case_id, phone FROM cases WHERE case_id = ? AND phone = ?", (case_Id, case_Name))
        result = cursor.fetchone()
        conn.close()

        if result:  
            return {
                "success": True,
                "message": "Login successful",
                "case_id": result[0]
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
    

# API for registering a new case from homepage
@app.post("/api/v1/home/case")
def get_cases_data_regestration(data: GetCaseData):
    print(data.patientName, data.patientAge, data.gender,data.phoneNumber)

    try:
        insert_cases(data.patientName, int(data.patientAge), data.gender, data.phoneNumber)

        return {
                "success": True,
                "message": "Data Added successful"
            }
    except sqlite3.Error as e:
         print(f"Database error: {e}")
         return {"success": False, "message": "Server error, please try again"}



# get case data from home page 
class updateCaseData(BaseModel):
    editpatientID: str
    editpatientName: str
    editpatientAge: int
    editgender: str
    editphoneNumber: str


# API for editing case information
@app.post("/api/v1/home/edit/case")
def update_book_case(data: updateCaseData):
    print(data.editpatientID, data.editpatientName, data.editpatientAge, data.editgender,data.editphoneNumber)

    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""                                      
                    UPDATE cases 
                    set name = ?,
                    age = ?, 
                    phone = ?, 
                    gender = ? 
                    where case_id = ?; 
  """, (data.editpatientName, int(data.editpatientAge), data.editphoneNumber,data.editgender, data.editpatientID ))
        conn.commit()

        cursor.close()
        conn.close()

        return {
                "success": True,
                "message": "Data Added successful"
            }
    except sqlite3.Error as e:
         print(f"Database error: {e}")
         return {"success": False, "message": "Server error, please try again"}


# API for listing all unassigned cases
@app.get("/api/home/show/cases")
def show_patients():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""  
                  SELECT 
                    c.case_id, 
                    c.name, 
                    c.age, 
                    c.phone, 
                    c.gender, 
                    sc.appointment_date, 
                    sc.appointment_time
                FROM cases c
                JOIN student_department_cases sc
                    ON sc.case_id = c.case_id
                WHERE sc.department_id = 'd001' 
                AND sc.appointment_date = DATE('now')
                ORDER BY c.case_id DESC;


  """)
        
        rows = cursor.fetchall()
        conn.close()

        patients = [dict(row) for row in rows]
        return {"success": True, "patients": patients}
    except sqlite3.Error as e:
        print('database error : ', e )
        return {"success": False}

   


# app.mount replaced below with BASE_DIR-aware STATIC_DIR


# Get student basic information by student_id
@app.get("/api/v1/student/{student_id}")
def get_student(student_id: str):
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
    
@app.get("/api/v1/student/{student_id}")
def get_student(student_id: str):
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


# Get patient list for a department
@app.get("/api/patients/{department_id}")
def get_patients(department_id : str):
    conn = db_connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        select c.case_id, c.name, c.phone, c.age, c.gender, c.sick, 
               s.appointment_date, s.description
        from student_department_cases as s 
        join cases as c on s.case_id = c.case_id 
        where s.student_id is null and s.department_id = ? and s.checked = '-1' order by  s.appointment_date DESC ;
    """, (department_id,))
    
    rows = cursor.fetchall()
    conn.close()

    patients = [dict(row) for row in rows]
    return {"patients": patients}



# Get all patients (excluding D001)
@app.get("/api/patients/all")
def get_patients_all():
    conn = db_connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        select c.case_id, c.name, c.phone, c.age, c.gender, 
               s.appointment_date, s.description
        from student_department_cases as s 
        join cases as c on s.case_id = c.case_id 
        where s.student_id is null and s.department_id != 'd001';
    """)
    
    rows = cursor.fetchall()
    conn.close()

    patients = [dict(row) for row in rows]
    return {"patients": patients}


# Get all cases associated with a specific student
@app.get("/api/student/cases/table/{student_id}")
def get_student_cases_table(student_id: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.case_id, c.name, s.description, s.treatment, s.before_photo,
                   s.after_photo, s.department_id, s.appointment_date, s.checked,
                   s.department_reffere, s.notes
            FROM student_department_cases s
            JOIN cases c ON c.case_id = s.case_id
            WHERE student_id = ?
            ORDER BY s.case_id DESC;
        """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        table = []
        for row in rows:
            data = {
                "case_id": row["case_id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "treatment": row["treatment"] or "",
                "before_photo": row["before_photo"] or "",
                "after_photo": row["after_photo"] or "",
                "department_id": row["department_id"],
                "appointment_date": row["appointment_date"] or "",
                "checked": row["checked"],
                "department_reffere": row["department_reffere"] or "",
                "notes": row["notes"] or ""  # 👈 هنا التحويل الصريح
            }
            table.append(data)

        print(table)
        return {"success": True, "table": table}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


class PostPatientToPatient(BaseModel):
    case_id: int
    department_id: str
    student_id: str


# Assign a case to a student
@app.post("/api/v1/student/patient")
def post_patient_to_patient(data: PostPatientToPatient):
    print(data.case_id, data.department_id, data.student_id)

    update_studentID_for_case( data.student_id, data.case_id, data.department_id)
    
    return {
            "success": True,
            "message": "Data Added successful"
        }


# UPLOAD_FOLDER is defined later using BASE_DIR to ensure absolute path
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Edit details of a case, handling before and after photos
@app.post("/api/v1/edit/case")
async def edit_case(
    
        student_id: str = Form(None),
        case_id: str = Form(None),
        department_id: str = Form(None),
        description: str = Form(None),
        treatment: str = Form(None),
        department: str = Form(None),
        date: str = Form(None),
        before: UploadFile = File(None),
        after: UploadFile = File(None)
        ):
    try:
        conn = db_connect()
        print("Student ID:", student_id)
        print("Case ID:", case_id)
        print("department ID:", department_id)
        print("Description:", description)
        print("Treatment:", treatment)
        print("Department:", department)
        print("Date:", date)

        if before:
            ext = os.path.splitext(before.filename)[1]  # الامتداد
            filename_before = f"{uuid.uuid4()}{ext}"         # اسم فريد
            filepath_before = os.path.join(UPLOAD_FOLDER, filename_before)

            with open(filepath_before, 'wb') as f:
                f.write(await before.read())
            
            print("before path :", filepath_before)
        else:
            print("Before file: No file")

        if after:
            ext = os.path.splitext(after.filename)[1]  # الامتداد
            filename_after = f"{uuid.uuid4()}{ext}"         # اسم فريد
            filepath_after = os.path.join(UPLOAD_FOLDER, filename_after)

            with open(filepath_after, 'wb') as f:
                f.write(await after.read())
            
            print("after path :", filepath_after)
        else:
            filename_after = "unkown"
            print("After file: No file")

        if department_id == 'd001':
            update_edit_case(description, treatment, filename_before, filename_after, department, date, student_id, case_id, department_id)
        else:
            update_edit_case_all(description, treatment, filename_after, student_id, case_id, department_id)
        print("="*50)
        

        return {"success": True, "message": "Data received"}
    except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {"success": False, "message": "Server error, please try again"} 





# model_name = "SciReason-LFM2-2.6B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model.eval()

# class AiDiagnosis(BaseModel):
#     prompt: str

# def build_prompt(prompt):
#     return prompt

# @app.post("/api/ai/diagnosis/stream")
# def ai_diagnosis_stream(data: AiDiagnosis):
#     messages = [
#         {"role": "user", "content": f"{build_prompt(data.prompt)}, with short answer"}
#     ]

#     # تجهيز الإدخال
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_tensors="pt",
#         return_dict=True
#     ).to(model.device)

#     # Streamer لإخراج التوكنات أثناء التوليد
#     streamer = TextIteratorStreamer(
#         tokenizer,
#         skip_prompt=True,
#         skip_special_tokens=True
#     )

#     # تشغيل التوليد في Thread منفصل
#     generation_thread = Thread(
#         target=model.generate,
#         kwargs=dict(
#             **inputs,
#             max_new_tokens=800,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             streamer=streamer
#         )
#     )
#     generation_thread.start()

#     def token_stream():
#         for new_text in streamer:
#             yield new_text  

#     return StreamingResponse(token_stream(), media_type="text/plain")


_MODEL_CACHE = {
    "loaded": False,
    "model": None,
    "checkpoint": None,
    "transform": None,
    "class_names": None,
}


def model_classification(image_path):
    """Lazy-load the model and required libraries, then run inference.

    Caches loaded model and transforms in _MODEL_CACHE so subsequent calls
    are fast.
    """
    # Lazy import here to avoid importing heavy libraries during app startup
    import torch
    import torchvision.transforms as transforms
    from Image_classification_model.model import RegularizedDentalClassifier

    if not _MODEL_CACHE["loaded"]:
        checkpoint = torch.load('Image_classification_model/dental_classifier_balanced.pth', weights_only=False)
        model = RegularizedDentalClassifier(num_classes=6)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        _MODEL_CACHE.update({
            "loaded": True,
            "model": model,
            "checkpoint": checkpoint,
            "transform": transform,
            "class_names": checkpoint.get('class_names', [])
        })

    model = _MODEL_CACHE["model"]
    transform = _MODEL_CACHE["transform"]
    checkpoint = _MODEL_CACHE["checkpoint"]

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = _MODEL_CACHE.get('class_names', [])
    name = class_names[predicted.item()] if class_names and len(class_names) > predicted.item() else str(predicted.item())
    print(f"Predicted: {name}")
    return name




# ML: Classify dental images using a pre-trained model
@app.post("/api/v1/AI/classification")
async def classify(
      image: UploadFile = File(None),
        ):
    try :

        if image:
            ext = os.path.splitext(image.filename)[1]  # الامتداد
            filename_image = f"{uuid.uuid4()}{ext}"         # اسم فريد
            filepath_image = os.path.join(UPLOAD_FOLDER, filename_image)

            with open(filepath_image, 'wb') as f:
                f.write(await image.read())
            
            prediction = model_classification(filepath_image)
        else:
            print("image file: No file")

       
        

        return {"success": True, "predict": prediction}
    except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {"success": False, "message": "Server error, please try again"} 




















##############################################################################################################

################# Doctor page #####################

# Get all doctor data by doctorID
@app.get("/api/v1/doctor/data/{doctorID}")
def get_doctor_data_api(doctorID : str):
    try:

        data = get_doctor_data(doctorID)

        return {"success": True, "data": data}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}
    


# Get cases for a doctor by doctorID
@app.get("/api/v1/doctor/student/cases/{doctorID}")
def get_doctor_data_api(doctorID : str):
    try:

        data = get_doctor_student_cases(doctorID)
        return {"success": True, "data": data}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}




# Approve a doctor's case changes
@app.post("/api/v1/approve/case")
async def edit_case(
    
        case_id: str = Form(None),
        department_id: str = Form(None),
        description: str = Form(None),
        treatment: str = Form(None),
        department: str = Form(None),
        approval: str = Form(None),
        notes: str = Form(None),
      
        ):
    try :
        print("="*50)
        print("Case ID:", case_id)
        print("department ID:", department_id)
        print("Description:", description)
        print("Treatment:", treatment)
        print("Department:", department)
        print("approval:", approval)
        print("notes:", notes)

       
        print("="*50)
        update_doctor_student_case(description, treatment, department, approval,  notes, case_id, department_id)

        return {"success": True, "message": "Data received"}
    except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {"success": False, "message": "Server error, please try again"} 




@app.get("/api/v1/college/batchs")
async def get_batchs():
    try:
        data = get_batchs_data()
        print(data)

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  
    

class CollegeBatchCreateModel(BaseModel):
    id: str
    from_: str | None = None
    from_: str = None  # placeholder to satisfy type checkers if needed

class CollegeBatchPayload(BaseModel):
    from_: str
    to: str

    class Config:
        fields = {"from_": "from"}


# Create a college batch
@app.post("/api/v1/college/batchs")
async def college_batch_create(data: dict):
    try:
        # Accept plain dict to map 'from' key
        batch_id = data.get("id", "").strip()
        from_year = str(data.get("from", "")).strip()
        to_year = str(data.get("to", "")).strip()
        # Make batch_id equal to 'B' + from value (e.g., From=2025 => B2025)
        if from_year:
            batch_id = f"B{from_year}"

        if not batch_id:
            return {"success": False, "message": "Batch ID is required"}
        if not from_year or not to_year:
            return {"success": False, "message": "From and To are required"}
        return create_batch(batch_id, from_year, to_year)
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}


# Update college batch info by batch_id
@app.put("/api/v1/college/batchs/{batch_id}")
async def college_batch_update(batch_id: str, data: dict):
    try:
        from_year = str(data.get("from", "")).strip()
        to_year = str(data.get("to", "")).strip()
        if not from_year or not to_year:
            return {"success": False, "message": "From and To are required"}
        return update_batch(batch_id, from_year, to_year)
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}


# Delete a specific batch
@app.delete("/api/v1/college/batchs/{batch_id}")
async def college_batch_delete(batch_id: str):
    try:
        return delete_batch(batch_id)
    except Exception as e:
        print(f"Error {e}")


# Get feedback table data for a department
@app.get("/api/v1/feedback/table/{department_id}")
def get_feedback_table(department_id: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.case_id, c.name, s.description, s.treatment, s.before_photo,
                   s.after_photo, s.department_id, s.appointment_date, 
                   s.department_reffere, s.notes
            FROM student_department_cases s
            JOIN cases c ON c.case_id = s.case_id
            WHERE s.department_id = ?
            ORDER BY s.case_id DESC;
        """, (department_id,))

        rows = cursor.fetchall()
        conn.close()

        table = []
        for row in rows:
            data = {
                "case_id": row["case_id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "treatment": row["treatment"] or "",
                "before_photo": row["before_photo"] or "",
                "after_photo": row["after_photo"] or "",
                "department_id": row["department_id"],
                "appointment_date": row["appointment_date"] or "",
                "department_reffere": row["department_reffere"] or "",
                "notes": row["notes"] or ""
            }
            table.append(data)

        return {"success": True, "table": table}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


# Get all students with round_check status for college page
@app.get("/api/v1/college/students")
def get_college_students():
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT 
                s.student_id,
                s.name,
                s.batch_id,
                s.department_id,
                d.name as department_name,
                s.email,
                s.phone,
                MAX(sdc.round_check) as round_check,
                GROUP_CONCAT(DISTINCT r.round_id) as round_ids
            FROM student s
            LEFT JOIN department d ON s.department_id = d.department_id
            LEFT JOIN student_department_cases sdc ON s.student_id = sdc.student_id
            LEFT JOIN rounds r ON s.batch_id = r.batch_id
            GROUP BY s.student_id, s.name, s.batch_id, s.department_id
            ORDER BY s.student_id;
        """)

        rows = cursor.fetchall()
        conn.close()

        students = []
        for row in rows:
            # Determine approval status based on round_check value
            round_check_value = row["round_check"] if row["round_check"] is not None else 0
            approval_status = "Approved" if round_check_value == 1 else "Pending"
            
            # Get the first round ID if available
            round_ids = row["round_ids"]
            round_id = round_ids.split(',')[0] if round_ids else ""
            
            print(f"DEBUG: Student {row['student_id']} - round_check: {round_check_value}, status: {approval_status}, round_id: {round_id}")
            
            data = {
                "id": row["student_id"],
                "name": row["name"],
                "batch": row["batch_id"] or "",
                "department": row["department_name"] or "",
                "round": round_id,
                "round_check": round_check_value,
                "approval_status": approval_status,
                "phone": row["phone"] or "",
                "email": row["email"] or ""
            }
            students.append(data)

        return {"success": True, "data": students}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


# Get student approval data for a department
@app.get("/api/v1/doctor/students/approval/{department_id}")
def get_students_approval(department_id: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT 
                s.student_id,
                s.name,
                COUNT(sdc.case_id) as cases_count,
                MAX(sdc.round_check) as round_check
            FROM student s
            JOIN student_department_cases sdc ON s.student_id = sdc.student_id
            WHERE sdc.department_id = ?
            GROUP BY s.student_id, s.name
            ORDER BY s.student_id;
        """, (department_id,))

        rows = cursor.fetchall()
        conn.close()

        students = []
        for row in rows:
            # Determine approval status based on round_check value
            round_check_value = row["round_check"] if row["round_check"] is not None else 0
            approval_status = "Approved" if round_check_value == 1 else "Pending"
            
            print(f"DEBUG: Doctor approval - Student {row['student_id']} - round_check: {round_check_value}, status: {approval_status}")
            
            data = {
                "id": row["student_id"],
                "name": row["name"],
                "casesCount": row["cases_count"],
                "round_check": round_check_value,
                "approval_status": approval_status
            }
            students.append(data)

        return {"success": True, "students": students}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


# Get cases for a specific student in a department
@app.get("/api/v1/doctor/student/cases/{student_id}/{department_id}")
def get_student_cases(student_id: str, department_id: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                sdc.case_id,
                c.name,
                sdc.description,
                sdc.treatment,
                sdc.before_photo,
                sdc.after_photo,
                sdc.appointment_date,
                sdc.checked,
                sdc.notes
            FROM student_department_cases sdc
            JOIN cases c ON sdc.case_id = c.case_id
            WHERE sdc.student_id = ? AND sdc.department_id = ?
            ORDER BY sdc.case_id DESC;
        """, (student_id, department_id))

        rows = cursor.fetchall()
        conn.close()

        cases = []
        for row in rows:
            # Determine approval status based on checked value
            checked_value = row["checked"]
            print(f"DEBUG: Case {row['case_id']} - checked value: {checked_value} (type: {type(checked_value)})")
            
            if checked_value == 1:
                approval = "Approved"
            elif checked_value == -1:
                approval = "Rejected"
            else:
                approval = "Pending"

            data = {
                "id": row["case_id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "treatment": row["treatment"] or "",
                "before": row["before_photo"] or "",
                "after": row["after_photo"] or "",
                "date": row["appointment_date"] or "",
                "approval": approval,
                "checked": checked_value,
                "notes": row["notes"] or ""
            }
            cases.append(data)

        return {"success": True, "cases": cases}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


class StudentApprovalModel(BaseModel):
    student_id: str
    approval: str
    department_id: str


# Submit student approval and set round_check = 1
@app.post("/api/v1/doctor/student/approval")
def submit_student_approval(data: StudentApprovalModel):
    try:
        student_id = data.student_id
        approval = data.approval  # "approve" or "not-approve"
        department_id = data.department_id

        if not student_id or not approval or not department_id:
            return {"success": False, "message": "Missing required fields"}

        conn = db_connect()
        cursor = conn.cursor()

        print(f"Updating approval for student {student_id} in department {department_id}: {approval}")

        # If approval is "approve", set round_check = 1
        if approval == "approve":
            cursor.execute("""
                UPDATE student_department_cases
                SET round_check = 1
                WHERE student_id = ? AND department_id = ?;
            """, (student_id, department_id))
            print(f"✅ Set round_check = 1 for student {student_id}")
        else:
            # If not approved, set round_check = 0
            cursor.execute("""
                UPDATE student_department_cases
                SET round_check = 0
                WHERE student_id = ? AND department_id = ?;
            """, (student_id, department_id))
            print(f"❌ Set round_check = 0 for student {student_id}")

        conn.commit()
        
        # Verify the update
        cursor.execute("""
            SELECT COUNT(*) as count FROM student_department_cases
            WHERE student_id = ? AND department_id = ?;
        """, (student_id, department_id))
        result = cursor.fetchone()
        rows_updated = result[0] if result else 0
        
        conn.close()

        return {"success": True, "message": f"Student approval submitted: {approval}", "rows_updated": rows_updated}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": f"Server error: {str(e)}"}
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

# Get all college faculty members (doctors)
@app.get("/api/v1/college/doctor")
async def get_college_faculty_member():
    try:
        data = get_college_doctor()
        print(data)

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  

class CollegeDoctorCreateModel(BaseModel):
    id: str
    name: str
    department: str
    title: str
    phone: str = ""
    email: str = ""

class CollegeDoctorUpdateModel(BaseModel):
    name: str
    department: str
    title: str
    phone: str = ""
    email: str = ""

# Create doctor record for college
@app.post("/api/v1/college/doctor")
async def college_doctor_create(data: CollegeDoctorCreateModel):
    try:
        # Normalize or auto-generate FM-style ID (FM001, FM002, ...)
        doctor_id = (data.id or "").strip() if hasattr(data, 'id') else ""
        try:
            if not doctor_id:
                doctor_id = generate_next_faculty_id()
            else:
                s = doctor_id.strip()
                if len(s) >= 2 and s[0].upper() == 'F' and s[1].upper() == 'M':
                    num_part = ''.join(ch for ch in s[2:] if ch.isdigit())
                    doctor_id = f"FM{int(num_part):03d}" if num_part.isdigit() else generate_next_faculty_id()
                else:
                    digits = ''.join(ch for ch in s if ch.isdigit())
                    doctor_id = f"FM{int(digits):03d}" if digits.isdigit() else generate_next_faculty_id()
        except Exception:
            doctor_id = generate_next_faculty_id()
        if not data.name or not data.name.strip():
            return {"success": False, "message": "Name is required"}
        if not data.department or not data.department.strip():
            return {"success": False, "message": "Department is required"}
        if not data.title or not data.title.strip():
            return {"success": False, "message": "Title is required"}

        result = create_college_doctor(
            doctor_id,
            data.name.strip(),
            data.department.strip(),
            data.title.strip(),
            (data.phone.strip() if data.phone else None),
            (data.email.strip() if data.email else None),
        )
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}

# Update college doctor record by doctor_id
@app.put("/api/v1/college/doctor/{doctor_id}")
async def college_doctor_update(doctor_id: str, data: CollegeDoctorUpdateModel):
    try:
        result = update_college_doctor(
            doctor_id,
            data.name,
            data.department,
            data.title,
            data.phone,
            data.email,
        )
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}

# Delete a college doctor by doctor_id
@app.delete("/api/v1/college/doctor/{doctor_id}")
async def college_doctor_delete(doctor_id: str):
    try:
        result = delete_college_doctor(doctor_id)
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}

# Get all students (college side)
@app.get("/api/v1/college/student")
async def get_college_faculty_member():
    try:
        data = get_college_student()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  
    

# ---------------------- College Students Management ---------------------- #

class CollegeStudentCreateModel(BaseModel):
    id: str
    name: str
    batch: str
    department: str
    round: str
    phone: str = ""
    email: str = ""


class CollegeStudentUpdateModel(BaseModel):
    name: str
    batch: str
    department: str
    round: str
    phone: str = ""
    email: str = ""


# Create a college student record
@app.post("/api/v1/college/student")
async def college_student_create(data: CollegeStudentCreateModel):
    try:
        # Validate required fields and preserve the provided student ID exactly as in DB
        student_id = (data.id or "").strip()
        if not student_id:
            return {"success": False, "message": "Student ID is required"}
        if not data.name or not data.name.strip():
            return {"success": False, "message": "Name is required"}
        if not data.batch or not data.batch.strip():
            return {"success": False, "message": "Batch is required"}
        if not data.department or not data.department.strip():
            return {"success": False, "message": "Department is required"}
        if not data.round or not data.round.strip():
            return {"success": False, "message": "Round is required"}
        
        # Debug: log received payload
        print({
            "id": data.id,
            "name": data.name,
            "batch": data.batch,
            "department": data.department,
            "round": data.round,
            "phone": data.phone,
            "email": data.email,
        })

        # If student already exists, perform an update instead of insert to keep the same ID
        try:
            _conn2 = db_connect()
            _cur2 = _conn2.cursor()
            _cur2.execute("SELECT 1 FROM student WHERE student_id = ?;", (student_id,))
            exists = _cur2.fetchone() is not None
            _cur2.close()
            _conn2.close()
        except Exception:
            exists = False

        if exists:
            result = update_college_student(
                student_id,
                data.name.strip(),
                data.batch.strip(),
                data.department.strip(),
                data.round.strip(),
                (data.phone.strip() if data.phone else None),
                (data.email.strip() if data.email else None),
            )
            return result
        else:
            result = create_college_student(
                student_id,
                data.name.strip(),
                data.batch.strip(),
                data.department.strip(),
                data.round.strip(),
                (data.phone.strip() if data.phone else None),
                (data.email.strip() if data.email else None),
            )
            return result
    except Exception as e:
        import traceback
        print(f"Error creating student: {e}")
        print(traceback.format_exc())
        return {"success": False, "message": f"Server error: {str(e)}"}


# Update a college student record by student_id
@app.put("/api/v1/college/student/{student_id}")
async def college_student_update(student_id: str, data: CollegeStudentUpdateModel):
    try:
        result = update_college_student(
            student_id, data.name, data.batch, data.department, data.round, data.phone, data.email
        )
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}


# Delete a college student by student_id
@app.delete("/api/v1/college/student/{student_id}")
async def college_student_delete(student_id: str):
    try:
        result = delete_college_student(student_id)
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}


# Get all college departments
@app.get("/api/v1/college/departments")
async def get_college_departments_data():
    try:
        data = get_college_departments()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  


@app.delete("/api/v1/college/departments/{department_id}")
async def college_department_delete(department_id: str):
    try:
        result = delete_college_department(department_id)
        return result
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}

# Create department for college management
@app.post("/api/v1/college/departments")
async def college_department_create(data: dict):
    dep_id = data.get("id", "").strip()
    name = data.get("name", "").strip()
    manager = data.get("manager", "").strip()
    capacity = data.get("capacity", None)
    description = data.get("description", None)
    # Auto-generate department id if not provided or invalid
    try:
        if not dep_id or not (len(dep_id) >= 2 and dep_id[0] in ('d','D') and any(ch.isdigit() for ch in dep_id[1:])):
            dep_id = generate_next_department_id()
        else:
            # Normalize to dNNN format if a number can be extracted
            num_part = ''.join(ch for ch in dep_id[1:] if ch.isdigit())
            if num_part.isdigit():
                dep_id = f"d{int(num_part):03d}"
            else:
                dep_id = generate_next_department_id()
    except Exception:
        # Fallback to server-generated id if anything goes wrong
        dep_id = generate_next_department_id()
    if not name:
        return {"success": False, "message": "Name is required"}
  
    if not capacity:
        return {"success": False, "message": "Capacity is required"}
    return create_college_department(dep_id, name, manager, capacity, description)

# Update department for college management
@app.put("/api/v1/college/departments/{department_id}")
async def college_department_update(department_id: str, data: dict):
    name = data.get("name", "").strip()
    manager = data.get("manager", "").strip()
    capacity = data.get("capacity", None)
    description = data.get("description", None)
    if not name:
        return {"success": False, "message": "Name is required"}
    
    if not capacity:
        return {"success": False, "message": "Capacity is required"}
    return update_college_department(department_id, name, manager, capacity, description)


# Get all rounds (unique time periods/batch rounds) for college
@app.get("/api/v1/college/rounds")
async def get_college_rounds_data():
    try:
        data = get_college_rounds()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  


# Create a round
@app.post("/api/v1/college/rounds")
async def college_round_create(data: dict):
    try:
        batch = str(data.get("batch", "")).strip()
        month = str(data.get("month", "")).strip()
        name = str(data.get("name", "")).strip()
        result = create_round(batch, month, name)
        return result
    except Exception as e:
        print(f"Error creating round: {e}")
        return {"success": False, "message": f"Server error: {str(e)}"}


# Update a round
@app.put("/api/v1/college/rounds/{round_id}")
async def college_round_update(round_id: str, data: dict):
    try:
        batch = str(data.get("batch", "")).strip()
        month = str(data.get("month", "")).strip()
        name = str(data.get("name", "")).strip()
        result = update_round(round_id, batch, month, name)
        return result
    except Exception as e:
        print(f"Error updating round: {e}")
        return {"success": False, "message": f"Server error: {str(e)}"}


# Delete a round
@app.delete("/api/v1/college/rounds/{round_id}")
async def college_round_delete(round_id: str):
    try:
        result = delete_round(round_id)
        return result
    except Exception as e:
        print(f"Error deleting round: {e}")
        return {"success": False, "message": f"Server error: {str(e)}"}





## Analatical for student

@app.get("/api/home/analyize/cases/{student_id}")
def num_of_gender(student_id: str):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                c.gender,
                COUNT(c.gender) AS total_gender
            FROM cases c
            JOIN student_department_cases sdc
                ON c.case_id = sdc.case_id
            JOIN student s
                ON s.student_id = sdc.student_id
            WHERE s.student_id = ?
            GROUP BY c.gender;
        """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        num_male = 0
        num_female = 0

        for gender, count in rows:
            if gender and gender.upper().startswith("M"):
                num_male = count
            elif gender and gender.upper().startswith("F"):
                num_female = count

        return {
            "success": True,
            "num of Males": num_male,
            "num of Females": num_female
        }

    except sqlite3.Error as e:
        print("Database error:", e)
        return {"success": False, "error": str(e)}








@app.get("/api/home/analyize/checked/{student_id}")
def num_check_of_case(student_id: str):
    
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
             SELECT 
                sdc.checked,
                COUNT(sdc.checked) 
     
            FROM student_department_cases sdc
            JOIN student s
                ON s.student_id = sdc.student_id
            WHERE s.student_id = ?
            GROUP BY sdc.checked
			ORDER by sdc.checked;
        """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            num_of_pending= rows[0][1]
            num_of_rejectied= rows[1][1]
            num_of_approved= rows[2][1]

            return {
                "success": True,
                "num_of_pending": num_of_pending,
                "num_of_rejectied": num_of_rejectied,
                "num_of_approved": num_of_approved,
                 }
        else:

            return {
             "success": True,
                 "num_of_pending": 0,
                 "num_of_rejectied": 0,
                "num_of_approved": 0,
                }










@app.get("/api/home/analyize/case_by_department/{student_id}")
def num_check_of_case(student_id: str):
        list_Departments=[]
        list_number_cases=[]
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
              SELECT 
                d.name,
                COUNT(sdc.case_id) 
     
            FROM student_department_cases sdc
            JOIN student s
                ON s.student_id = sdc.student_id
			join department d
			on d.department_id=sdc.department_id
            WHERE s.student_id = ?
            GROUP BY sdc.department_id
			ORDER by sdc.department_id;
        """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            for row in rows:
                list_Departments.append(row[0])
                list_number_cases.append(row[1])

            return {
                "success": True,
                "list_Departments": list_Departments,
                "list_number_cases": list_number_cases,
                 }
        else:

            return {
             "success": False,
                }



@app.get("/api/home/analyize/Treatment/{student_id}")
def num_check_of_case(student_id: str):
        list_Treatment=[]
        list_number_cases=[]
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
        LOWER(sdc.treatment) AS treatment,
        COUNT(*) AS total
    FROM student_department_cases sdc
    JOIN student s
        ON s.student_id = sdc.student_id
    WHERE s.student_id = ? 
      AND sdc.treatment IS NOT NULL
      AND TRIM(sdc.treatment) != ""
      AND LOWER(sdc.treatment) != "unkown"
    GROUP BY LOWER(sdc.treatment)
    ORDER BY total DESC;
        """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            for row in rows:
                list_Treatment.append(row[0])
                list_number_cases.append(row[1])

            return {
                "success": True,
                "list_Treatment": list_Treatment,
                "list_number_cases": list_number_cases,
                 }
        else:

            return {
             "success": False,
                }



## Analatical for doctor

@app.get("/api/home/analyize_doctor/cases/{doctorID}")
def num_of_gender(doctorID: str):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
           SELECT 
    c.gender,
    COUNT(c.gender) AS total_gender
FROM cases c
JOIN student_department_cases sdc
    ON c.case_id = sdc.case_id
JOIN faculty_members fm
    ON fm.department_id = sdc.department_id
WHERE fm.faculty_members_id = ?
GROUP BY c.gender;

        """, (doctorID,))

        rows = cursor.fetchall()
        conn.close()

        num_male = 0
        num_female = 0

        for gender, count in rows:
            if gender and gender.upper().startswith("M"):
                num_male = count
            elif gender and gender.upper().startswith("F"):
                num_female = count

        return {
            "success": True,
            "num of Males": num_male,
            "num of Females": num_female
        }

    except sqlite3.Error as e:
        print("Database error:", e)
        return {"success": False, "error": str(e)}






@app.get("/api/home/analyize/checked_doctor/{doctorID}")
def num_check_of_case(doctorID: str):
    
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
             SELECT 
                sdc.checked,
                COUNT(sdc.checked) 
     
            FROM student_department_cases sdc
            JOIN faculty_members fm
                ON fm.department_id = sdc.department_id
            WHERE fm.faculty_members_id = ?
            GROUP BY sdc.checked
			ORDER by sdc.checked;
        """, (doctorID,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            num_of_pending= rows[0][1]
            num_of_rejectied= rows[1][1]
            num_of_approved= rows[2][1]

            return {
                "success": True,
                "num_of_pending": num_of_pending,
                "num_of_rejectied": num_of_rejectied,
                "num_of_approved": num_of_approved,
                 }
        else:

            return {
             "success": True,
                 "num_of_pending": 0,
                 "num_of_rejectied": 0,
                "num_of_approved": 0,
                }



@app.get("/api/home/analyize/case_by_department_doctor/{doctorID}")
def num_check_of_case(doctorID: str):
        list_Departments=[]
        list_number_cases=[]
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
             SELECT 
    s.name AS student_name,
    COUNT(sdc.case_id) AS total_cases
FROM student_department_cases sdc
JOIN faculty_members fm ON fm.department_id = sdc.department_id
JOIN student s ON s.student_id = sdc.student_id
WHERE fm.faculty_members_id = ?
GROUP BY s.name, s.student_id
ORDER BY s.student_id;
        """, (doctorID,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            for row in rows:
                list_Departments.append(row[0])
                list_number_cases.append(row[1])

            return {
                "success": True,
                "list_Departments": list_Departments,
                "list_number_cases": list_number_cases,
                 }
        else:

            return {
             "success": False,
                }
        

@app.get("/api/home/analyize/Treatment_doctor/{doctorID}")
def num_check_of_case(doctorID: str):
        list_Treatment=[]
        list_number_cases=[]
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
        LOWER(sdc.treatment) AS treatment,
        COUNT(*) AS total
    FROM student_department_cases sdc
    JOIN faculty_members fm
        ON fm.department_id = sdc.department_id
    WHERE fm.faculty_members_id = ? 
      AND sdc.treatment IS NOT NULL
      AND TRIM(sdc.treatment) != ""
      AND LOWER(sdc.treatment) != "unkown"
    GROUP BY LOWER(sdc.treatment)
    ORDER BY total DESC;
        """, (doctorID,))

        rows = cursor.fetchall()
        conn.close()

        
        if rows :
            for row in rows:
                list_Treatment.append(row[0])
                list_number_cases.append(row[1])

            return {
                "success": True,
                "list_Treatment": list_Treatment,
                "list_number_cases": list_number_cases,
                 }
        else:

            return {
             "success": False,
                }



# Get patient dashboard data by case_id
@app.get("/api/v1/patient/dashboard/{case_id}")
def get_patient_dashboard(case_id: str):
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.case_id,
                c.name,
                d.name AS department_name,
                sdc.appointment_date,
                sdc.appointment_time,
                sdc.checked,
                s.name AS student_name
            FROM cases c
            LEFT JOIN student_department_cases sdc ON c.case_id = sdc.case_id
            LEFT JOIN department d ON sdc.department_id = d.department_id
            LEFT JOIN student s ON sdc.student_id = s.student_id
            WHERE c.case_id = ?
            ORDER BY sdc.appointment_date DESC, sdc.appointment_time DESC
            LIMIT 1;
        """, (case_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Determine status based on checked value
            checked_value = row["checked"]
            if checked_value == 1:
                status = "Approved"
            elif checked_value == -1:
                status = "Rejected"
            else:
                status = "Pending"
            
            # Format date
            appointment_date = row["appointment_date"] or ""
            if appointment_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(appointment_date, "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%d/%m/%Y")
                except:
                    formatted_date = appointment_date
            else:
                formatted_date = "N/A"
            
            # Format time
            appointment_time = row["appointment_time"] or "N/A"
            
            data = {
                "case_id": row["case_id"],
                "name": row["name"] or "",
                "department": row["department_name"] or "N/A",
                "date": formatted_date,
                "time": appointment_time,
                "status": status,
                "student_name": row["student_name"] or "Not Assigned"
            }
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "Patient case not found"}
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


# Get patient report data by case_id
@app.get("/api/v1/patient/report/{case_id}")
def api_get_patient_report(case_id: str):
    """
    Endpoint to get patient report data.
    Pulls from cases and student_department_cases tables.
    """
    try:
        # Call the dedicated function from Student_Page_Queries
        result = get_patient_report(case_id)
        
        if result["success"]:
            row = result["data"]
            
            # Format date
            appointment_date = row.get("appointment_date") or ""
            if appointment_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(appointment_date, "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%B %d, %Y")
                except:
                    formatted_date = appointment_date
            else:
                formatted_date = "N/A"
            
            # Format patient ID
            patient_id = f"DENT-{row['case_id']:05d}"
            
            # Get treatment or default
            treatment = row.get("treatment") or "No treatment recorded"
            
            data = {
                "patient_name": row.get("patient_name") or "",
                "patient_id": patient_id,
                "doctor_name": row.get("doctor_name") or "Not Assigned",
                "department": row.get("department_name") or "N/A",
                "student_name": row.get("student_name") or "Not Assigned",
                "date": formatted_date,
                "treatment": treatment,
                "description": row.get("description") or "",
                "age": row.get("age") or "",
                "gender": row.get("gender") or "",
                "phone": row.get("phone") or "",
                "batch_id": row.get("batch_id") or "",
                "department_id": row.get("department_id") or "",
                "before_photo": row.get("before_photo") or "",
                "after_photo": row.get("after_photo") or "",
                "appointment_time": row.get("appointment_time") or "",
                "checked": row.get("checked") or "",
                "notes": row.get("notes") or "",
                "department_reffere": row.get("department_reffere") or "",
                "student_email": row.get("student_email") or ""
            }
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": result.get("message", "Patient case not found")}
            
    except Exception as e:
        print(f"Error in get_patient_report: {e}")
        return {"success": False, "message": "Server error, please try again"}



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)



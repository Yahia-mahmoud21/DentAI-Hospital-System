from fastapi import FastAPI, Request,Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import uvicorn
import sqlite3
from queries import *
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import torchvision.transforms as transforms
from model import RegularizedDentalClassifier
import os
import io
import uuid
from threading import Thread
from Student_Page_Queries import *
from doctor_query import *
from college_queries import *

app = FastAPI()

# 1. GET endpoint بسيط
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


def get_departments():
    conn = sqlite3.connect("dental_project_DB.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM department where department_id != 'D001';")
    departments = [row[0] for row in cursor.fetchall()]
    conn.close()
    return departments


@app.get("/student", response_class=HTMLResponse)
def student(request: Request):
    departments = get_departments()
    return templates.TemplateResponse(
        "student.html", {"request": request, "departments": departments}
    )


@app.get("/doctor", response_class=HTMLResponse)
def doctor(request: Request):
    departments = get_departments()
    return templates.TemplateResponse(
        "doctor_all.html", {"request": request, "departments": departments}
    )

@app.get("/AI", response_class=HTMLResponse)
def AI(request: Request):
    return templates.TemplateResponse("Ai.html", {"request": request})

@app.get("/college", response_class=HTMLResponse)
def collge(request: Request):
    return templates.TemplateResponse("college.html", {"request": request})

@app.get("/patient", response_class=HTMLResponse)
def patient(request: Request):
    return templates.TemplateResponse("patient.html", {"request": request})


@app.get("/book", response_class=HTMLResponse)
def book(request: Request):
    return templates.TemplateResponse("book.html", {"request": request})









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
    

class RegisterStudentModel(BaseModel):
    id : str
    email: str
    password : str

@app.post("/api/v1/student/register")
def api_register_student(data: RegisterStudentModel):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
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
                "message": "Login successful",
                "ID" : result[0]
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


@app.post("/api/v1/home/edit/case")
def update_book_case(data: updateCaseData):
    print(data.editpatientID, data.editpatientName, data.editpatientAge, data.editgender,data.editphoneNumber)

    try:
        conn = sqlite3.connect("dental_project_DB.db")
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


@app.get("/api/home/show/cases")
def show_patients():
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""  
                  SELECT c.case_id, c.name, c.age, c.phone, c.gender, sc.appointment_date, sc.appointment_time
                from cases c 
                join student_department_cases sc
                on sc.case_id = c.case_id
                where sc.student_id is NULL
                ORDER by c.case_id DESC;

  """)
        
        rows = cursor.fetchall()
        conn.close()

        patients = [dict(row) for row in rows]
        return {"success": True, "patients": patients}
    except sqlite3.Error as e:
        print('database error : ', e )
        return {"success": False}

   


app.mount("/static", StaticFiles(directory="static"), name="static")


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
def get_student_cases_table(student_id: str):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
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


@app.post("/api/v1/student/patient")
def post_patient_to_patient(data: PostPatientToPatient):
    print(data.case_id, data.department_id, data.student_id)

    update_studentID_for_case( data.student_id, data.case_id, data.department_id)
    
    return {
            "success": True,
            "message": "Data Added successful"
        }


UPLOAD_FOLDER = os.path.join("static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    try :
        print("="*50)
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

        if department_id == 'D001':
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
#         {"role": "user", "content": build_prompt(data.prompt)}
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

#     # دالة generator تبعت التوكنات لحظة بلحظة
#     def token_stream():
#         for new_text in streamer:
#             yield new_text  # 👈 كل توكن يتبعت أول بأول

#     # إرجاع StreamResponse للـ frontend
#     return StreamingResponse(token_stream(), media_type="text/plain")


def model_classification(image_path):
    checkpoint = torch.load('dental_classifier_balanced.pth', weights_only=False)
    model = RegularizedDentalClassifier(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    class_names = checkpoint['class_names']
    print(f"Predicted: {class_names[predicted.item()]}")
    return class_names[predicted.item()]




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

@app.get("/api/v1/doctor/data/{doctorID}")
def get_doctor_data_api(doctorID : str):
    try:

        data = get_doctor_data(doctorID)

        return {"success": True, "data": data}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}
    


@app.get("/api/v1/doctor/student/cases/{doctorID}")
def get_doctor_data_api(doctorID : str):
    try:

        data = get_doctor_student_cases(doctorID)
        return {"success": True, "data": data}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}




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
    

@app.get("/api/v1/college/doctor")
async def get_college_faculty_member():
    try:
        data = get_college_doctor()
        print(data)

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  

@app.get("/api/v1/college/student")
async def get_college_faculty_member():
    try:
        data = get_college_student()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  
    

@app.get("/api/v1/college/departments")
async def get_college_departments_data():
    try:
        data = get_college_departments()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  



@app.get("/api/v1/college/rounds")
async def get_college_rounds_data():
    try:
        data = get_college_rounds()

        return {"success": True, "data" : data}
    except Exception as e:
        print(f"Error {e}")
        return {"success": False, "message": "Server error, please try again"}  











if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)



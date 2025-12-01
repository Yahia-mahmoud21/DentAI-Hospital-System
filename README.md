# 🦷 DentAI Hospital System Features Document

## 1. System Overview
A fully integrated web system for managing the workflow in a  **Assuit University Dental Hospital**, built using **FastAPI**.  
The system includes a **Deep Learning** model to classify dental diseases from images and provides multiple user interfaces according to each user role in the hospital.

---

## 2. Artificial Intelligence (AI) Module

### A. Dental Disease Classification

- **Model Used:**  
  A **ResNet-50** deep learning model trained to classify **6 types** of dental diseases.

- **Supported Diseases:**
  1. Calculus  
  2. Dental Caries  
  3. Gingivitis  
  4. Hypodontia (Congenital missing teeth)  
  5. Mouth Ulcer  
  6. Tooth Discoloration

- **User Interface:**  
  The `AI` page provides a simple drag-and-drop interface to upload tooth images and get an instant diagnosis.

- **API Endpoint:**  
  Allows developers to integrate the classification service with other systems.

---

### B. Intelligent Medical Assistant (AI Chat)

An integrated chat interface inside the AI page that allows users to interact with a virtual assistant.  
The assistant is trained on medical data to provide accurate dental-related responses.

---

## 3. Features by User Role

### A. Secretary

**Dashboard `book`:**
- Add new patients.  
- View a waiting list of new unassigned cases.  
- Edit patient data (name, age, contact info, etc.).

---

### B. Student

- **Login & Registration Page:**  
  Students can create an account linked to their university ID.

- **Dashboard `student`:**
  - View personal and academic information.  
  - View available cases in their department.  
  - **Case Management:**
    - Assign a case to themselves.  
    - Edit case details (description, treatment plan, before/after photos).  
    - Refer the case to another department if needed.  
    - View supervisor feedback on approved or rejected cases.  
  - **Analytics & Dashboard:**
    - Case distribution by patient gender.  
    - Case status (Approved / Rejected / Under Review).  
    - Number of cases per department.  
    - Most common treatment types.

---

### C. Doctor

- **Login Page:** Secure login portal for doctors.  
- **Dashboard `/doctor`:**
  - View personal and department information.  
  - Review cases submitted by students.  
  - **Decision Options:**
    - ✅ **Approve:** Accept the student’s treatment plan.  
    - ❌ **Refuse:** Reject the case and add comments for guidance.  
    - 🔄 **Referral:** Refer the case to another department.
- **Analytics Dashboard**

---

### D. College Administration

**Dashboard `/college`:**
- View overall statistics (batches, doctors, students, departments, rotations).  
- Manage tables (view, filter, and sort data).  
- **Import & Export Data** from or to Excel files.  
- Manage departments and assign department heads.

---

### E. Patient

- **Patient Portal `patient`:**  
  Login using the case number and phone number.  
- **Case Details:**  
  View their treatment progress and **download the medical report as a PDF**.

---


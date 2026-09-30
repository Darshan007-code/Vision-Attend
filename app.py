"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Main Flask Web Application & REST API
"""

import os
import json
import functools
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for, 
    session, flash, jsonify, Response, send_file
)
import cv2

import config
from core.database import (
    initialize_db, authenticate_user, add_student, get_all_students,
    get_student_by_id, delete_student, update_student_samples_count,
    add_teacher, get_all_teachers, add_subject, get_all_subjects,
    get_subject_by_id, record_attendance, get_attendance_records,
    get_student_summary, get_student_subject_breakdown,
    get_setting, set_setting, get_all_settings
)
from core.attendance_manager import attendance_manager
from services.export_service import export_service
from services.stats_service import stats_service

# Initialize Flask App
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Ensure Database and folders are ready
initialize_db()

# --- Decorators for Role-Based Access Control ---

def login_required(roles=None):
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("login_page"))
            if roles and session.get("role") not in roles:
                flash("You do not have permission to access this resource.", "danger")
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- Authentication Routes ---

@app.route("/")
def index():
    """Portal Home: Role Selection / Landing Page."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role_selected = request.form.get("role", "admin")

        user = authenticate_user(username, password)
        if user and user['role'] == role_selected:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            session["reference_id"] = user["reference_id"]
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials or role mismatch. Please try again.", "danger")
            return redirect(url_for("login_page", role=role_selected))

    selected_role = request.args.get("role", "admin")
    return render_template("login.html", role=selected_role)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for("login_page"))

# --- Dashboard Redirection by Role ---

@app.route("/dashboard")
@login_required()
def dashboard():
    role = session.get("role")
    if role == "admin":
        return redirect(url_for("kiosk"))
    elif role == "teacher":
        return redirect(url_for("kiosk"))
    elif role == "student":
        return redirect(url_for("student_portal"))
    return redirect(url_for("login_page"))

# --- Live Attendance Kiosk (Teacher / Admin) ---

@app.route("/kiosk")
@login_required(roles=["admin", "teacher"])
def kiosk():
    subjects = get_all_subjects()
    kpis = stats_service.get_dashboard_kpis()
    active_subject_id = attendance_manager.active_subject_id
    active_session_type = attendance_manager.active_session_type
    all_students = get_all_students()
    liveness_enabled = get_setting("liveness_enabled", "1") == "1"
    return render_template(
        "kiosk.html",
        subjects=subjects,
        kpis=kpis,
        active_subject_id=active_subject_id,
        active_session_type=active_session_type,
        students=all_students,
        liveness_enabled=liveness_enabled
    )

@app.route("/video_feed")
@login_required(roles=["admin", "teacher"])
def video_feed():
    """MJPEG Live Webcam Stream."""
    return Response(
        attendance_manager.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/live_events")
@login_required(roles=["admin", "teacher"])
def api_live_events():
    """Returns recent recognized student events for real-time live feed."""
    return jsonify(list(attendance_manager.recent_events))

@app.route("/api/set_session", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_set_session():
    data = request.get_json() or {}
    subject_id = data.get("subject_id")
    session_type = data.get("session_type", "Regular Class")
    if subject_id:
        attendance_manager.set_session(int(subject_id), session_type)
        return jsonify({"success": True, "subject_id": subject_id, "session_type": session_type})
    return jsonify({"success": False, "error": "Invalid subject ID"}), 400

@app.route("/api/toggle_liveness", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_toggle_liveness():
    current = get_setting("liveness_enabled", "1") == "1"
    new_val = "0" if current else "1"
    set_setting("liveness_enabled", new_val)
    return jsonify({"success": True, "liveness_enabled": new_val == "1"})

@app.route("/api/process_browser_frame", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_process_browser_frame():
    """
    Processes video frames sent directly from client-side browser webcam (HTML5 getUserMedia).
    Enables live interactive face recognition & Haar eye-blink anti-spoofing in cloud deployments
    where the server does not have direct hardware USB camera access.
    """
    import base64
    import numpy as np

    data = request.get_json() or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "No image data provided"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Image decode failed"}), 400

        # Execute full computer vision pipeline: detection, Haar eye action, liveness, recognition, HUD
        annotated_frame, frame_events = attendance_manager.process_frame(frame)

        # Encode back to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({"success": False, "error": "Frame encode failed"}), 500

        resp_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{resp_b64}",
            "events": frame_events,
            "fps": attendance_manager.current_fps
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/manual_mark", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_manual_mark():
    """Allows teacher/admin to manually override and mark attendance."""
    student_id = request.form.get("student_id")
    subject_id = request.form.get("subject_id")
    status = request.form.get("status", "Present")
    session_type = request.form.get("session_type", "Manual Entry")

    if not student_id or not subject_id:
        flash("Student and Subject must be specified.", "warning")
        return redirect(url_for("kiosk"))

    today_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    record_attendance(
        student_id=int(student_id),
        subject_id=int(subject_id),
        date=today_str,
        time_str=time_str,
        status=status,
        confidence=100.0,
        liveness=1,
        session_type=session_type,
        method="Manual Override"
    )
    flash("Manual attendance record created successfully.", "success")
    return redirect(url_for("kiosk"))

# --- Attendance History & Records ---

@app.route("/attendance")
@login_required(roles=["admin", "teacher"])
def attendance_records_page():
    date_filter = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    subject_filter = request.args.get("subject_id")
    dept_filter = request.args.get("department")

    records = get_attendance_records(
        date=date_filter if date_filter else None,
        subject_id=int(subject_filter) if subject_filter else None,
        department=dept_filter if dept_filter else None
    )
    subjects = get_all_subjects()
    
    return render_template(
        "attendance.html",
        records=records,
        subjects=subjects,
        selected_date=date_filter,
        selected_subject=subject_filter,
        selected_dept=dept_filter
    )

# --- Export Endpoints ---

@app.route("/export/csv")
@login_required(roles=["admin", "teacher"])
def export_csv():
    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject_id")
    dept_filter = request.args.get("department")

    records = get_attendance_records(
        date=date_filter,
        subject_id=int(subject_filter) if subject_filter else None,
        department=dept_filter
    )
    csv_data = export_service.export_csv(records)
    filename = f"Attendance_{date_filter or 'All'}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@app.route("/export/excel")
@login_required(roles=["admin", "teacher"])
def export_excel():
    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject_id")
    dept_filter = request.args.get("department")

    records = get_attendance_records(
        date=date_filter,
        subject_id=int(subject_filter) if subject_filter else None,
        department=dept_filter
    )
    sub_name = "All Courses"
    if subject_filter:
        s = get_subject_by_id(int(subject_filter))
        if s: sub_name = f"{s['code']} - {s['name']}"

    file_path = export_service.export_excel(records, subject_name=sub_name, date_str=date_filter or "")
    return send_file(file_path, as_attachment=True)

@app.route("/export/pdf")
@login_required(roles=["admin", "teacher"])
def export_pdf():
    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject_id")
    dept_filter = request.args.get("department")

    records = get_attendance_records(
        date=date_filter,
        subject_id=int(subject_filter) if subject_filter else None,
        department=dept_filter
    )
    sub_name = "All Courses"
    if subject_filter:
        s = get_subject_by_id(int(subject_filter))
        if s: sub_name = f"{s['code']} - {s['name']}"

    file_path = export_service.export_pdf(records, subject_name=sub_name, date_str=date_filter or "")
    return send_file(file_path, as_attachment=True)

# --- Student Management ---

@app.route("/students")
@login_required(roles=["admin", "teacher"])
def students_page():
    students = get_all_students()
    return render_template("students.html", students=students)

@app.route("/students/register", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def register_student():
    roll_number = request.form.get("roll_number", "").strip()
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    semester = request.form.get("semester", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()

    if not roll_number or not name or not department:
        flash("Roll Number, Name, and Department are required.", "danger")
        return redirect(url_for("students_page"))

    try:
        student_id = add_student(roll_number, name, department, semester, email, phone)
        flash(f"Student '{name}' registered successfully with ID #{student_id}.", "success")
        return redirect(url_for("student_profile", student_id=student_id))
    except Exception as e:
        flash(f"Error registering student: {str(e)}", "danger")
        return redirect(url_for("students_page"))

@app.route("/students/<int:student_id>")
@login_required(roles=["admin", "teacher"])
def student_profile(student_id):
    student = get_student_by_id(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("students_page"))

    summary = get_student_summary(student_id)
    breakdown = get_student_subject_breakdown(student_id)
    history = get_attendance_records(student_id=student_id)
    
    # List sample files
    sample_dir = config.DATASET_DIR / str(student_id)
    samples = []
    if sample_dir.exists():
        samples = [f.name for f in sample_dir.glob("*.jpg")]

    return render_template(
        "student_profile.html",
        student=student,
        summary=summary,
        breakdown=breakdown,
        history=history,
        samples=samples
    )

@app.route("/students/<int:student_id>/delete", methods=["POST"])
@login_required(roles=["admin"])
def delete_student_route(student_id):
    delete_student(student_id)
    # Remove dataset folder
    sample_dir = config.DATASET_DIR / str(student_id)
    if sample_dir.exists():
        import shutil
        shutil.rmtree(str(sample_dir), ignore_errors=True)
    flash("Student and training data deleted.", "info")
    return redirect(url_for("students_page"))

@app.route("/api/capture_sample", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_capture_sample():
    """Captures a single face crop from camera for student enrollment wizard."""
    data = request.get_json() or {}
    student_id = data.get("student_id")
    sample_idx = data.get("sample_idx", 1)

    if not student_id:
        return jsonify({"success": False, "error": "Missing student_id"}), 400

    result = attendance_manager.capture_training_sample(int(student_id), int(sample_idx))
    if result.get("success"):
        # Update count in DB
        sample_dir = config.DATASET_DIR / str(student_id)
        count = len(list(sample_dir.glob("*.jpg")))
        update_student_samples_count(int(student_id), count)
        result["total_samples"] = count

    return jsonify(result)

@app.route("/api/upload_samples/<int:student_id>", methods=["POST"])
@login_required(roles=["admin", "teacher"])
def api_upload_samples(student_id):
    """Uploads pre-existing photo files for student."""
    files = request.files.getlist("photos")
    if not files:
        flash("No files selected.", "warning")
        return redirect(url_for("student_profile", student_id=student_id))

    student_dir = config.DATASET_DIR / str(student_id)
    student_dir.mkdir(parents=True, exist_ok=True)
    
    current_count = len(list(student_dir.glob("*.jpg")))
    saved = 0

    for f in files:
        if f.filename:
            # Read image and detect face
            import numpy as np
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                faces = attendance_manager.face_engine.detect_faces(img)
                if faces:
                    crop_gray = faces[0]['crop_gray']
                    current_count += 1
                    attendance_manager.face_engine.save_face_sample(student_id, crop_gray, current_count)
                    saved += 1

    update_student_samples_count(student_id, current_count)
    flash(f"Successfully processed and saved {saved} face samples!", "success")
    return redirect(url_for("student_profile", student_id=student_id))

@app.route("/dataset_sample/<int:student_id>/<filename>")
@login_required()
def dataset_sample_image(student_id, filename):
    """Serves high-resolution face sample crops for student profile biometric studio."""
    safe_dir = config.DATASET_DIR / str(student_id)
    safe_path = safe_dir / filename
    if safe_path.exists() and safe_path.is_file():
        return send_file(safe_path, mimetype="image/jpeg")
    return "Sample image not found", 404

@app.route("/student_avatar/<int:student_id>")
def student_avatar(student_id):
    """Serves primary biometric face crop avatar for ID card and live detection feed."""
    student_dir = config.DATASET_DIR / str(student_id)
    if student_dir.exists():
        samples = sorted(list(student_dir.glob("*.jpg")))
        if samples:
            return send_file(samples[0], mimetype="image/jpeg")
    return redirect(url_for('static', filename='img/logo.png'))

# --- Faculty & Subject Management (Admin) ---

@app.route("/teachers", methods=["GET", "POST"])
@login_required(roles=["admin"])
def teachers_page():
    if request.method == "POST":
        code = request.form.get("teacher_code", "").strip()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        if code and name and department:
            try:
                add_teacher(code, name, department, email, phone)
                flash(f"Faculty '{name}' added successfully.", "success")
            except Exception as e:
                flash(f"Error adding teacher: {str(e)}", "danger")
        else:
            flash("All required fields must be filled.", "warning")
        return redirect(url_for("teachers_page"))

    teachers = get_all_teachers()
    return render_template("teachers.html", teachers=teachers)

@app.route("/subjects", methods=["GET", "POST"])
@login_required(roles=["admin"])
def subjects_page():
    if request.method == "POST":
        code = request.form.get("code", "").strip()
        name = request.form.get("name", "").strip()
        dept = request.form.get("department", "").strip()
        semester = request.form.get("semester", "").strip()
        teacher_id = request.form.get("teacher_id")

        if code and name:
            try:
                add_subject(code, name, dept, semester, int(teacher_id) if teacher_id else None)
                flash(f"Subject '{name}' registered.", "success")
            except Exception as e:
                flash(f"Error registering subject: {str(e)}", "danger")
        return redirect(url_for("subjects_page"))

    subjects = get_all_subjects()
    teachers = get_all_teachers()
    return render_template("subjects.html", subjects=subjects, teachers=teachers)

# --- AI Model Training Center ---

@app.route("/train")
@login_required(roles=["admin"])
def train_page():
    # Load evaluation metrics if exists
    eval_file = config.MODELS_DIR / "evaluation.json"
    eval_data = None
    if eval_file.exists():
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        except Exception:
            pass

    # Count total samples
    total_samples = 0
    enrolled_students = 0
    for s_dir in config.DATASET_DIR.iterdir():
        if s_dir.is_dir():
            s_count = len(list(s_dir.glob("*.jpg")))
            if s_count > 0:
                enrolled_students += 1
                total_samples += s_count

    return render_template(
        "train.html",
        is_trained=attendance_manager.face_engine.is_trained,
        eval_data=eval_data,
        total_samples=total_samples,
        enrolled_students=enrolled_students
    )

@app.route("/api/train_model", methods=["POST"])
@login_required(roles=["admin"])
def api_train_model():
    """Triggers the computer vision model training pipeline."""
    students = get_all_students()
    metadata = {s['id']: {'student_id': s['id'], 'name': s['name'], 'roll_number': s['roll_number']} for s in students}
    
    result = attendance_manager.face_engine.train_model(metadata)
    return jsonify(result)

# --- Analytics & Insights ---

@app.route("/analytics")
@login_required(roles=["admin", "teacher"])
def analytics_page():
    kpis = stats_service.get_dashboard_kpis()
    trend = stats_service.get_7day_trend()
    dept_dist = stats_service.get_department_distribution()
    defaulters = stats_service.get_defaulters_list()

    return render_template(
        "analytics.html",
        kpis=kpis,
        trend=trend,
        dept_dist=dept_dist,
        defaulters=defaulters
    )

# --- Student Portal (Individual View) ---

@app.route("/student_portal")
@login_required(roles=["student"])
def student_portal():
    roll_no = session.get("reference_id")
    student = None
    all_students = get_all_students()
    for s in all_students:
        if s['roll_number'] == roll_no:
            student = s
            break

    if not student:
        flash("Student profile record not found.", "warning")
        return redirect(url_for("logout"))

    student_id = student['id']
    summary = get_student_summary(student_id)
    breakdown = get_student_subject_breakdown(student_id)
    history = get_attendance_records(student_id=student_id)

    return render_template(
        "student_portal.html",
        student=student,
        summary=summary,
        breakdown=breakdown,
        history=history
    )

# --- System Settings ---

@app.route("/settings", methods=["GET", "POST"])
@login_required(roles=["admin"])
def settings_page():
    if request.method == "POST":
        set_setting("camera_index", request.form.get("camera_index", "0"))
        set_setting("confidence_threshold", request.form.get("confidence_threshold", "75"))
        set_setting("cooldown_minutes", request.form.get("cooldown_minutes", "60"))
        set_setting("class_start_time", request.form.get("class_start_time", "09:00"))
        set_setting("late_grace_minutes", request.form.get("late_grace_minutes", "15"))
        set_setting("liveness_enabled", "1" if request.form.get("liveness_enabled") else "0")
        set_setting("voice_enabled", "1" if request.form.get("voice_enabled") else "0")
        set_setting("min_attendance_pct", request.form.get("min_attendance_pct", "75"))

        flash("Settings saved successfully.", "success")
        return redirect(url_for("settings_page"))

    current_settings = get_all_settings()
    return render_template("settings.html", settings=current_settings)

if __name__ == "__main__":
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)

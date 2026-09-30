"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Core Database Module (SQLite)
"""

import sqlite3
import hashlib
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Computes SHA-256 hash for secure credential storage."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def initialize_db():
    """Initializes tables, indices, and default seed records."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Users table (Admin, Teachers, Students login)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
                full_name TEXT NOT NULL,
                email TEXT,
                reference_id TEXT, -- student_id or teacher_id
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Teachers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Subjects / Courses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                semester TEXT NOT NULL,
                teacher_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE SET NULL
            )
        """)

        # Students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                semester TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                face_samples_count INTEGER DEFAULT 0,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Attendance Logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                date TEXT NOT NULL, -- YYYY-MM-DD
                time TEXT NOT NULL, -- HH:MM:SS
                status TEXT NOT NULL CHECK(status IN ('Present', 'Late', 'Absent')),
                confidence REAL DEFAULT 0.0,
                liveness_verified INTEGER DEFAULT 1,
                session_type TEXT DEFAULT 'Regular Class',
                method TEXT DEFAULT 'Face Recognition',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            )
        """)

        # System Settings key-value table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_subject ON attendance(subject_id)")

        # Seed Default Admin if not exists
        admin = cursor.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        if not admin:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, full_name, email)
                VALUES (?, ?, ?, ?, ?)
            """, ('admin', hash_password('admin123'), 'admin', 'System Administrator', 'admin@visionattend.edu'))

        # Seed Default Settings if not exist
        default_settings = {
            'camera_index': str(config.DEFAULT_CAMERA_INDEX),
            'confidence_threshold': str(config.CONFIDENCE_DISTANCE_THRESHOLD),
            'cooldown_minutes': str(config.COOLDOWN_MINUTES),
            'class_start_time': config.CLASS_START_TIME,
            'late_grace_minutes': str(config.LATE_GRACE_MINUTES),
            'liveness_enabled': '1' if config.LIVENESS_ENABLED else '0',
            'voice_enabled': '1' if config.VOICE_ENABLED else '0',
            'min_attendance_pct': str(config.ATTENDANCE_MIN_PERCENTAGE)
        }

        for k, v in default_settings.items():
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))

        # Seed Sample Data if tables are empty
        teacher_count = cursor.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
        if teacher_count == 0:
            cursor.execute("""
                INSERT INTO teachers (teacher_code, name, department, email, phone)
                VALUES 
                ('T101', 'Dr. Sarah Mitchell', 'Computer Science', 's.mitchell@visionattend.edu', '+1 555-0101'),
                ('T102', 'Prof. David Vance', 'Information Technology', 'd.vance@visionattend.edu', '+1 555-0102')
            """)
            # Also create logins for them
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, full_name, email, reference_id)
                VALUES 
                ('T101', ?, 'teacher', 'Dr. Sarah Mitchell', 's.mitchell@visionattend.edu', 'T101'),
                ('T102', ?, 'teacher', 'Prof. David Vance', 'd.vance@visionattend.edu', 'T102')
            """, (hash_password('teacher123'), hash_password('teacher123')))

        subject_count = cursor.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        if subject_count == 0:
            cursor.execute("""
                INSERT INTO subjects (code, name, department, semester, teacher_id)
                VALUES 
                ('CS801', 'Computer Vision & AI', 'Computer Science', 'Semester 8', 1),
                ('CS802', 'Deep Learning & Neural Networks', 'Computer Science', 'Semester 8', 1),
                ('IT801', 'Cloud Infrastructure & DevOps', 'Information Technology', 'Semester 8', 2)
            """)

        conn.commit()

# --- User & Auth Queries ---

def authenticate_user(username, password):
    """Verifies credentials and returns user record if valid."""
    p_hash = hash_password(password)
    with get_connection() as conn:
        return conn.execute("""
            SELECT * FROM users WHERE username = ? AND password_hash = ?
        """, (username, p_hash)).fetchone()

# --- Student Queries ---

def add_student(roll_number, name, department, semester, email="", phone="", photo_path=None):
    """Adds a new student and generates student user credentials."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students (roll_number, name, department, semester, email, phone, photo_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (roll_number.strip(), name.strip(), department.strip(), semester.strip(), email.strip(), phone.strip(), photo_path))
        student_id = cursor.lastrowid

        # Also create a login account for the student (username=roll_number, password=roll_number)
        cursor.execute("""
            INSERT OR REPLACE INTO users (username, password_hash, role, full_name, email, reference_id)
            VALUES (?, ?, 'student', ?, ?, ?)
        """, (roll_number.strip(), hash_password(roll_number.strip()), name.strip(), email.strip(), roll_number.strip()))

        conn.commit()
        return student_id

def update_student_samples_count(student_id, count):
    """Updates the count of captured face images for a student."""
    with get_connection() as conn:
        conn.execute("UPDATE students SET face_samples_count = ? WHERE id = ?", (count, student_id))
        conn.commit()

def get_all_students():
    """Fetches all registered students."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM students ORDER BY roll_number ASC").fetchall()

def get_student_by_id(student_id):
    """Fetches a single student by numeric ID."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()

def get_student_by_roll(roll_number):
    """Fetches student by roll number."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM students WHERE roll_number = ?", (roll_number,)).fetchone()

def delete_student(student_id):
    """Deletes student and associated records."""
    with get_connection() as conn:
        student = conn.execute("SELECT roll_number FROM students WHERE id = ?", (student_id,)).fetchone()
        if student:
            conn.execute("DELETE FROM users WHERE username = ?", (student['roll_number'],))
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()

# --- Teacher Queries ---

def add_teacher(teacher_code, name, department, email="", phone=""):
    """Adds teacher and sets up login."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO teachers (teacher_code, name, department, email, phone)
            VALUES (?, ?, ?, ?, ?)
        """, (teacher_code.strip(), name.strip(), department.strip(), email.strip(), phone.strip()))
        teacher_id = cursor.lastrowid

        cursor.execute("""
            INSERT OR REPLACE INTO users (username, password_hash, role, full_name, email, reference_id)
            VALUES (?, ?, 'teacher', ?, ?, ?)
        """, (teacher_code.strip(), hash_password(teacher_code.strip()), name.strip(), email.strip(), teacher_code.strip()))

        conn.commit()
        return teacher_id

def get_all_teachers():
    """Fetches all teachers."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM teachers ORDER BY name ASC").fetchall()

def get_teacher_by_code(teacher_code):
    """Fetches teacher by teacher code."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM teachers WHERE teacher_code = ?", (teacher_code,)).fetchone()

# --- Subject Queries ---

def add_subject(code, name, department, semester, teacher_id=None):
    """Registers a new subject course."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO subjects (code, name, department, semester, teacher_id)
            VALUES (?, ?, ?, ?, ?)
        """, (code.strip().upper(), name.strip(), department.strip(), semester.strip(), teacher_id))
        conn.commit()
        return cursor.lastrowid

def get_all_subjects():
    """Fetches all subjects with assigned teacher name."""
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, t.name as teacher_name 
            FROM subjects s
            LEFT JOIN teachers t ON s.teacher_id = t.id
            ORDER BY s.code ASC
        """).fetchall()

def get_subject_by_id(subject_id):
    """Fetches subject details."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()

# --- Attendance Logging & Retrieval ---

def record_attendance(student_id, subject_id, date, time_str, status="Present", confidence=0.0, liveness=1, session_type="Regular Class", method="Face Recognition"):
    """Inserts a new attendance log entry."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance (student_id, subject_id, date, time, status, confidence, liveness_verified, session_type, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_id, subject_id, date, time_str, status, confidence, liveness, session_type, method))
        conn.commit()
        return cursor.lastrowid

def get_last_attendance(student_id, subject_id, date):
    """Returns the latest attendance record for a student on a specific date and subject."""
    with get_connection() as conn:
        return conn.execute("""
            SELECT * FROM attendance 
            WHERE student_id = ? AND subject_id = ? AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (student_id, subject_id, date)).fetchone()

def get_attendance_records(date=None, subject_id=None, department=None, student_id=None):
    """Filtered attendance query joining student and subject information."""
    query = """
        SELECT a.id, a.date, a.time, a.status, a.confidence, a.liveness_verified, a.session_type, a.method,
               s.id as student_id, s.roll_number, s.name as student_name, s.department as student_dept, s.semester,
               sub.id as subject_id, sub.code as subject_code, sub.name as subject_name
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN subjects sub ON a.subject_id = sub.id
        WHERE 1=1
    """
    params = []
    if date:
        query += " AND a.date = ?"
        params.append(date)
    if subject_id:
        query += " AND a.subject_id = ?"
        params.append(subject_id)
    if student_id:
        query += " AND a.student_id = ?"
        params.append(student_id)
    if department:
        query += " AND s.department = ?"
        params.append(department)

    query += " ORDER BY a.date DESC, a.time DESC"

    with get_connection() as conn:
        return conn.execute(query, params).fetchall()

def get_student_summary(student_id):
    """Computes total subjects, total sessions, classes attended, and attendance percentage."""
    with get_connection() as conn:
        # Total distinct dates with classes held in subjects the student attends
        total_sessions = conn.execute("""
            SELECT COUNT(DISTINCT date || '_' || subject_id) FROM attendance
        """).fetchone()[0]

        attended_sessions = conn.execute("""
            SELECT COUNT(DISTINCT date || '_' || subject_id) FROM attendance
            WHERE student_id = ?
        """, (student_id,)).fetchone()[0]

        rate = (attended_sessions / total_sessions * 100.0) if total_sessions > 0 else 0.0
        return {
            "total_sessions": total_sessions,
            "attended_sessions": attended_sessions,
            "missed_sessions": max(0, total_sessions - attended_sessions),
            "attendance_rate": round(rate, 1)
        }

def get_student_subject_breakdown(student_id):
    """Computes subject-wise attendance breakdown for an individual student."""
    with get_connection() as conn:
        subjects = conn.execute("SELECT * FROM subjects").fetchall()
        breakdown = []
        for sub in subjects:
            total_subject_sessions = conn.execute("""
                SELECT COUNT(DISTINCT date) FROM attendance WHERE subject_id = ?
            """, (sub['id'],)).fetchone()[0]

            attended = conn.execute("""
                SELECT COUNT(DISTINCT date) FROM attendance WHERE subject_id = ? AND student_id = ?
            """, (sub['id'], student_id)).fetchone()[0]

            rate = (attended / total_subject_sessions * 100.0) if total_subject_sessions > 0 else 0.0
            breakdown.append({
                "subject_code": sub['code'],
                "subject_name": sub['name'],
                "total": total_subject_sessions,
                "attended": attended,
                "percentage": round(rate, 1)
            })
        return breakdown

# --- Settings Queries ---

def get_setting(key, default=None):
    """Fetches setting value by key."""
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row['value'] if row else default

def set_setting(key, value):
    """Sets or updates setting key."""
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()

def get_all_settings():
    """Fetches dictionary of all system settings."""
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r['key']: r['value'] for r in rows}

if __name__ == "__main__":
    initialize_db()
    print("Database initialized successfully with default records.")

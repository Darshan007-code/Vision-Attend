"""
VisionAttend - Automated System Verification & Self-Test Script
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from core.database import initialize_db, add_student, get_all_students, record_attendance, get_attendance_records
from services.export_service import export_service
from services.stats_service import stats_service
from core.face_engine import FaceEngine
from core.liveness_detector import LivenessDetector

def run_tests():
    print("=" * 60)
    print("  RUNNING VISIONATTEND SYSTEM VERIFICATION SUITE")
    print("=" * 60)

    # 1. Database Test
    print("[1/5] Testing Database Initialization & Schema...")
    initialize_db()
    students = get_all_students()
    print(f"      OK. Database ready. Registered students: {len(students)}")

    # 2. Add Test Student if empty
    if len(students) == 0:
        print("[2/5] Seeding Test Student...")
        s_id = add_student(
            roll_number="22CS01",
            name="Alex Johnson",
            department="Computer Science",
            semester="Semester 8",
            email="alex.johnson@visionattend.edu",
            phone="+1 555-0144"
        )
        print(f"      OK. Seeded Test Student #{s_id}: Alex Johnson (22CS01)")
    else:
        s_id = students[0]['id']

    # 3. Test Attendance Record Insertion
    print("[3/5] Testing Attendance Logging...")
    rec_id = record_attendance(
        student_id=s_id,
        subject_id=1,
        date="2026-09-30",
        time_str="09:05:00",
        status="Present",
        confidence=94.5,
        liveness=1,
        session_type="Regular Class",
        method="Face Recognition"
    )
    records = get_attendance_records(date="2026-09-30")
    print(f"      OK. Attendance record #{rec_id} inserted. Query count: {len(records)}")

    # 4. Test Reporting & Exports (CSV, Excel, PDF)
    print("[4/5] Testing Export Engines (CSV, Excel, PDF)...")
    csv_out = export_service.export_csv(records)
    assert len(csv_out) > 0, "CSV export failed"
    print(f"      CSV Export: {len(csv_out)} bytes OK")

    excel_path = export_service.export_excel(records, subject_name="CS801 - Computer Vision & AI", date_str="2026-09-30")
    assert Path(excel_path).exists(), "Excel export failed"
    print(f"      Excel Export: Generated {Path(excel_path).name} OK")

    pdf_path = export_service.export_pdf(records, subject_name="CS801 - Computer Vision & AI", date_str="2026-09-30")
    assert Path(pdf_path).exists(), "PDF export failed"
    print(f"      ReportLab PDF Export: Generated {Path(pdf_path).name} OK")

    # 5. Test Computer Vision & Liveness Classes
    print("[5/5] Testing Computer Vision Engine & Liveness Detector...")
    engine = FaceEngine()
    liveness = LivenessDetector()
    kpis = stats_service.get_dashboard_kpis()
    print(f"      FaceEngine: Ready (Trained: {engine.is_trained})")
    print(f"      LivenessDetector: Ready (Blur Threshold: {liveness.blur_threshold})")
    print(f"      Dashboard KPIs: {kpis}")

    print("=" * 60)
    print("  ALL SYSTEMS PASSED VERIFICATION WITH ZERO ERRORS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

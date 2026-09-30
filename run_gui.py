"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Standalone Desktop OpenCV GUI Runner (Offline Viva & Presentation Mode)
"""

import cv2
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from core.database import (
    initialize_db, get_all_subjects, get_all_students, 
    add_student, update_student_samples_count
)
from core.attendance_manager import attendance_manager

def run_desktop_kiosk():
    print("=" * 60)
    print("  VISIONATTEND - STANDALONE DESKTOP BIOMETRIC KIOSK")
    print("=" * 60)
    
    initialize_db()
    subjects = get_all_subjects()
    if not subjects:
        print("[!] No subjects found in database. Initializing default...")
    
    active_subject_idx = 0
    subject_id = subjects[active_subject_idx]['id'] if subjects else 1
    subject_name = subjects[active_subject_idx]['name'] if subjects else "General"
    attendance_manager.set_session(subject_id, "Regular Class")

    print(f"[*] Active Subject: {subject_name} (ID: {subject_id})")
    print("[*] Controls:")
    print("    [Q] / [ESC] : Quit Desktop Kiosk")
    print("    [S]         : Switch Course / Subject")
    print("    [E]         : Enroll New Student (Webcam Face Capture)")
    print("    [T]         : Retrain LBPH Face Recognizer")
    print("    [L]         : Toggle Liveness / Anti-Spoofing Check")
    print("-" * 60)

    cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam index", config.DEFAULT_CAMERA_INDEX)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    window_name = "VisionAttend AI - Biometric Attendance Kiosk"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 680)

    liveness_enabled = config.LIVENESS_ENABLED

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        # Process frame
        annotated_frame, events = attendance_manager.process_frame(frame)

        # Bottom control instruction banner
        h, w = annotated_frame.shape[:2]
        banner_h = 32
        cv2.rectangle(annotated_frame, (0, h - banner_h), (w, h), (15, 23, 42), cv2.FILLED)
        ctrl_text = f"Subject: {subject_name} | [S] Switch Sub | [E] Enroll | [T] Train | [L] Liveness:{'ON' if liveness_enabled else 'OFF'} | [Q] Quit"
        cv2.putText(annotated_frame, ctrl_text, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q'), 27]: # Q or ESC
            print("[*] Exiting Desktop Kiosk...")
            break

        elif key in [ord('s'), ord('S')]: # Switch Subject
            if subjects:
                active_subject_idx = (active_subject_idx + 1) % len(subjects)
                subject_id = subjects[active_subject_idx]['id']
                subject_name = subjects[active_subject_idx]['name']
                attendance_manager.set_session(subject_id, "Regular Class")
                print(f"[+] Switched to Subject: {subject_name} (ID: {subject_id})")

        elif key in [ord('l'), ord('L')]: # Toggle Liveness
            liveness_enabled = not liveness_enabled
            from core.database import set_setting
            set_setting("liveness_enabled", "1" if liveness_enabled else "0")
            print(f"[+] Liveness Anti-Spoofing set to: {'ENABLED' if liveness_enabled else 'DISABLED'}")

        elif key in [ord('t'), ord('T')]: # Train Model
            print("[*] Initiating model training...")
            students = get_all_students()
            metadata = {s['id']: {'student_id': s['id'], 'name': s['name'], 'roll_number': s['roll_number']} for s in students}
            result = attendance_manager.face_engine.train_model(metadata)
            if result.get("success"):
                print(f"[SUCCESS] Model trained in {result['training_time_sec']}s on {result['total_faces']} samples across {result['total_students']} students!")
                print(f"          Test Accuracy: {result['metrics']['accuracy']}%")
            else:
                print(f"[!] Training failed: {result.get('error')}")

        elif key in [ord('e'), ord('E')]: # Enroll Student
            cv2.destroyWindow(window_name)
            cap.release()
            
            print("\n--- STUDENT BIOMETRIC ENROLLMENT WIZARD ---")
            roll_no = input("Enter Student Roll Number: ").strip()
            name = input("Enter Student Full Name: ").strip()
            dept = input("Enter Department (default: CSE): ").strip() or "Computer Science"
            sem = input("Enter Semester (default: Semester 8): ").strip() or "Semester 8"

            if roll_no and name:
                try:
                    student_id = add_student(roll_no, name, dept, sem)
                    print(f"[+] Student registered with ID #{student_id}. Starting face sample capture...")
                    print("    Please look directly at camera. Capturing 30 samples...")

                    enroll_cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
                    if not enroll_cap.isOpened():
                        enroll_cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)

                    sample_count = 0
                    while sample_count < 30:
                        s_ret, s_frame = enroll_cap.read()
                        if not s_ret: break

                        faces = attendance_manager.face_engine.detect_faces(s_frame)
                        h_f, w_f = s_frame.shape[:2]

                        if len(faces) == 1:
                            x, y, fw, fh = faces[0]['box']
                            crop_gray = faces[0]['crop_gray']
                            sample_count += 1
                            attendance_manager.face_engine.save_face_sample(student_id, crop_gray, sample_count)
                            
                            # Visual feedback
                            cv2.rectangle(s_frame, (x, y), (x + fw, y + fh), (34, 197, 94), 2)
                            cv2.putText(s_frame, f"Capturing: {sample_count}/30", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)
                        elif len(faces) > 1:
                            cv2.putText(s_frame, "Multiple faces detected!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(s_frame, "Face not found. Look at camera.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        # Progress Bar
                        bar_w = int((w_f - 60) * (sample_count / 30.0))
                        cv2.rectangle(s_frame, (30, h_f - 40), (w_f - 30, h_f - 20), (50, 50, 50), cv2.FILLED)
                        cv2.rectangle(s_frame, (30, h_f - 40), (30 + bar_w, h_f - 20), (34, 197, 94), cv2.FILLED)

                        cv2.imshow("Enrollment Capture - Look at Camera", s_frame)
                        cv2.waitKey(80) # Slight delay between frame captures for varied expressions

                    enroll_cap.release()
                    cv2.destroyAllWindows()
                    update_student_samples_count(student_id, sample_count)
                    print(f"[SUCCESS] Captured {sample_count} samples for {name}!")

                    # Ask to train now
                    train_now = input("Train AI model immediately? (y/n): ").strip().lower()
                    if train_now == 'y':
                        all_st = get_all_students()
                        meta = {s['id']: {'student_id': s['id'], 'name': s['name'], 'roll_number': s['roll_number']} for s in all_st}
                        res = attendance_manager.face_engine.train_model(meta)
                        print(f"[SUCCESS] Model trained: Accuracy {res['metrics']['accuracy']}%")

                except Exception as err:
                    print(f"[ERROR] Enrollment failed: {err}")

            # Re-open camera for desktop kiosk
            cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_desktop_kiosk()

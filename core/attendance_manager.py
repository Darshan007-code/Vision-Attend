"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Core Attendance Manager & Computer Vision Pipeline Orchestrator
"""

import cv2
import time
from datetime import datetime, timedelta
from collections import deque
import threading
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from core.database import (
    record_attendance, get_last_attendance, get_student_by_id, 
    get_setting, get_all_students
)
from core.face_engine import FaceEngine
from core.liveness_detector import LivenessDetector
from services.voice_service import voice_service

class AttendanceManager:
    def __init__(self):
        self.face_engine = FaceEngine()
        self.liveness_detector = LivenessDetector()

        # Session State
        self.active_subject_id = 1
        self.active_session_type = "Regular Class"
        self.camera_index = int(get_setting("camera_index", config.DEFAULT_CAMERA_INDEX))
        
        # In-memory cooldown tracking: (student_id, subject_id) -> timestamp
        self.cooldown_tracker = {}
        
        # Event buffer for real-time live attendance feed on the UI (last 20 events)
        self.recent_events = deque(maxlen=20)

        # Video capture handle & thread lock
        self.cap = None
        self.lock = threading.Lock()
        self.is_streaming = False

        # Performance metrics
        self.prev_frame_time = 0
        self.current_fps = 0.0

    def start_camera(self, camera_idx=None) -> bool:
        """Starts video capture device if not already running."""
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                return True

            idx = camera_idx if camera_idx is not None else self.camera_index
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default API
                self.cap = cv2.VideoCapture(idx)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self.is_streaming = True
                return True
            return False

    def stop_camera(self):
        """Releases the camera."""
        with self.lock:
            self.is_streaming = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def set_session(self, subject_id: int, session_type: str = "Regular Class"):
        """Configures current course subject and class session."""
        self.active_subject_id = int(subject_id)
        self.active_session_type = session_type

    def _determine_status(self) -> str:
        """
        Determines 'Present' or 'Late' status based on configured class schedule.
        """
        start_time_str = get_setting("class_start_time", config.CLASS_START_TIME)
        grace_mins = int(get_setting("late_grace_minutes", config.LATE_GRACE_MINUTES))

        now = datetime.now()
        try:
            start_hour, start_min = map(int, start_time_str.split(":"))
            scheduled_start = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
            cutoff_time = scheduled_start + timedelta(minutes=grace_mins)

            if now > cutoff_time:
                return "Late"
            return "Present"
        except Exception:
            return "Present"

    def _process_attendance_event(self, student_id: int, confidence: float, liveness: int) -> dict:
        """
        Checks cooldown and commits attendance to DB if valid.
        """
        cooldown_mins = int(get_setting("cooldown_minutes", config.COOLDOWN_MINUTES))
        now = time.time()
        cooldown_key = (student_id, self.active_subject_id)
        
        last_logged = self.cooldown_tracker.get(cooldown_key, 0)
        cooldown_seconds = cooldown_mins * 60

        student = get_student_by_id(student_id)
        student_name = student['name'] if student else f"Student #{student_id}"
        department = student['department'] if student else ""

        # Check in-memory cooldown
        if (now - last_logged) < cooldown_seconds:
            remaining = int((cooldown_seconds - (now - last_logged)) / 60)
            return {
                'status': 'cooldown',
                'message': f"Already marked recently ({remaining}m remaining)",
                'name': student_name,
                'student_id': student_id
            }

        # Check DB to see if already marked for this subject today
        today_str = datetime.now().strftime("%Y-%m-%d")
        last_db_record = get_last_attendance(student_id, self.active_subject_id, today_str)
        if last_db_record:
            self.cooldown_tracker[cooldown_key] = now
            return {
                'status': 'already_recorded',
                'message': f"Marked today at {last_db_record['time']}",
                'name': student_name,
                'student_id': student_id
            }

        # Determine Present vs Late
        attendance_status = self._determine_status()
        time_str = datetime.now().strftime("%H:%M:%S")

        # Save record
        record_id = record_attendance(
            student_id=student_id,
            subject_id=self.active_subject_id,
            date=today_str,
            time_str=time_str,
            status=attendance_status,
            confidence=confidence,
            liveness=liveness,
            session_type=self.active_session_type,
            method="Face Recognition"
        )

        self.cooldown_tracker[cooldown_key] = now

        # Add to recent events buffer for live UI feed
        event_data = {
            'id': record_id,
            'student_id': student_id,
            'name': student_name,
            'roll_number': student['roll_number'] if student else str(student_id),
            'department': department,
            'time': time_str,
            'status': attendance_status,
            'confidence': confidence,
            'photo': f"/student_avatar/{student_id}"
        }
        self.recent_events.appendleft(event_data)

        # Trigger voice feedback if enabled
        voice_on = get_setting("voice_enabled", "1") == "1"
        if voice_on:
            voice_service.announce_attendance(student_name, department, attendance_status)

        return {
            'status': 'recorded',
            'message': f"Marked {attendance_status} ({confidence}%)",
            'name': student_name,
            'student_id': student_id,
            'attendance_status': attendance_status
        }

    def _draw_hud(self, frame, x, y, w, h, name, roll_no, confidence, liveness_passed, status_text, is_recognized, eyes=None, blink_count=0, just_blinked=False):
        """Renders modern, sleek HUD graphics on detected face with ocular reticles and strict anti-spoofing alert."""
        # Color theme
        if not liveness_passed:
            color = (30, 30, 235)  # Crimson Red for Spoof / Phone / Photo (BGR)
            badge_color = (15, 15, 140)
        elif is_recognized and liveness_passed:
            color = (34, 197, 94)  # Vibrant Emerald Green for Live Recognized (BGR)
            badge_color = (22, 101, 52)
        else:
            color = (59, 130, 246)  # Blue for searching / unregistered live face
            badge_color = (30, 58, 138)

        # 1. Sleek Corner Brackets
        corner_len = int(min(w, h) * 0.25)
        thickness = 2
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # Subtle semi-transparent bounding box outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        # 2. Ocular Reticles & Eye Action Tracking Visuals
        if eyes:
            for (ex, ey, ew, eh) in eyes:
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                r = max(7, int(min(ew, eh) * 0.36))

                if just_blinked:
                    ocular_col = (0, 255, 255) # Bright Yellow Blink Indicator
                elif liveness_passed:
                    ocular_col = (255, 240, 0) # Cyan/Aqua Reticle
                else:
                    ocular_col = (60, 60, 220) # Reddish-Orange

                # Circular ocular tracking reticle
                cv2.circle(frame, (cx, cy), r, ocular_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
                # Crosshair notch ticks
                cv2.line(frame, (cx - r - 4, cy), (cx - r + 2, cy), ocular_col, 1)
                cv2.line(frame, (cx + r - 2, cy), (cx + r + 4, cy), ocular_col, 1)
                cv2.line(frame, (cx, cy - r - 4), (cx, cy - r + 2), ocular_col, 1)
                cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 4), ocular_col, 1)
                cv2.putText(frame, "EYE", (cx - 7, cy - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, ocular_col, 1, cv2.LINE_AA)

        # 3. Blink Confirmation Tag
        if just_blinked:
            cv2.rectangle(frame, (x, max(0, y - 48)), (x + 195, max(20, y - 24)), (0, 215, 255), cv2.FILLED)
            cv2.putText(frame, ">> BLINK VERIFIED <<", (x + 8, max(15, y - 31)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 23, 42), 1, cv2.LINE_AA)

        # 4. Header Name / Spoof Pill
        if not liveness_passed:
            header_text = "SPOOF DETECTED [REJECTED]"
        else:
            header_text = f"{name}"
            if roll_no and roll_no != "N/A":
                header_text += f" | {roll_no}"
            if blink_count > 0:
                header_text += f" (Blinks:{blink_count})"

        (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        pill_w = max(tw + 16, 120)
        pill_h = 24
        pill_x = x
        pill_y = max(26, y - 6)

        # Draw header banner
        cv2.rectangle(frame, (pill_x, pill_y - pill_h), (pill_x + pill_w, pill_y), badge_color, cv2.FILLED)
        cv2.rectangle(frame, (pill_x, pill_y - pill_h), (pill_x + pill_w, pill_y), color, 1)
        cv2.putText(frame, header_text, (pill_x + 8, pill_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Footer Status & Anti-Spoofing Diagnosis Badge
        conf_text = f"{confidence}%" if (confidence > 0 and liveness_passed) else ""
        sub_text = f"{status_text} [{conf_text}]" if conf_text else status_text

        (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        foot_y = y + h + 22
        if foot_y + 10 < frame.shape[0]:
            cv2.rectangle(frame, (x, y + h + 2), (x + max(stw + 16, 120), foot_y), (15, 23, 42), cv2.FILLED)
            cv2.rectangle(frame, (x, y + h + 2), (x + max(stw + 16, 120), foot_y), color, 1)
            cv2.putText(frame, sub_text, (x + 6, foot_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        """
        Runs full pipeline on a single frame:
        Detection -> Liveness -> Recognition -> Cooldown Check -> HUD.
        Returns: (annotated_frame, list of recognized events in frame)
        """
        if frame is None:
            return None, []

        # Calculate FPS
        curr_time = time.time()
        if self.prev_frame_time > 0:
            self.current_fps = round(1.0 / (curr_time - self.prev_frame_time), 1)
        self.prev_frame_time = curr_time

        liveness_on = get_setting("liveness_enabled", "1") == "1"
        faces = self.face_engine.detect_faces(frame)
        frame_events = []

        for face in faces:
            x, y, w, h = face['box']
            crop_bgr = face['crop_bgr']
            crop_gray = face['crop_gray']

            # Strict Multi-Layer Liveness & Anti-Spoofing Verification
            liveness_res = {'passed': True, 'score': 100.0, 'reason': 'Real Face'}
            if liveness_on:
                liveness_res = self.liveness_detector.verify_liveness(crop_bgr, crop_gray, box=(x, y, w, h))

            # LBPH Face Recognition Prediction
            pred = self.face_engine.predict(crop_gray)
            status_text = "Searching..."

            if not liveness_res['passed']:
                # STRICT SPOOF REJECTION: Mobile screens, digital displays, and paper photos are rejected
                status_text = f"REJECTED: {liveness_res['reason']}"
                # Audio alert with 5s cooldown
                now = time.time()
                if not hasattr(self, '_last_spoof_warn'): self._last_spoof_warn = 0
                if now - self._last_spoof_warn > 5.0:
                    self._last_spoof_warn = now
                    voice_service.announce_warning("Fake image or display screen detected. Real face required.")
            elif pred['recognized']:
                # Verified Live Face + Enrolled Student
                event_res = self._process_attendance_event(
                    student_id=pred['student_id'],
                    confidence=pred['confidence'],
                    liveness=1
                )
                status_text = event_res['message']
                frame_events.append(event_res)
            else:
                status_text = "Unregistered Live Face"

            # Render HUD
            self._draw_hud(
                frame=frame,
                x=x, y=y, w=w, h=h,
                name=pred['name'],
                roll_no=pred['roll_number'],
                confidence=pred['confidence'],
                liveness_passed=liveness_res['passed'],
                status_text=status_text,
                is_recognized=pred['recognized'],
                eyes=liveness_res.get('eyes', []),
                blink_count=liveness_res.get('blink_count', 0),
                just_blinked=liveness_res.get('just_blinked', False)
            )

        # Global Top-Bar Overlay
        h_frame, w_frame = frame.shape[:2]
        # Dark HUD top header strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 34), (15, 23, 42), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Camera info & FPS
        hud_info = f"VISIONATTEND AI  |  FPS: {self.current_fps}  |  Faces: {len(faces)}"
        cv2.putText(frame, hud_info, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (241, 245, 249), 1, cv2.LINE_AA)

        # Pulse indicator circle
        cv2.circle(frame, (w_frame - 20, 17), 6, (34, 197, 94), cv2.FILLED)

        return frame, frame_events

    def generate_frames(self):
        """Yields MJPEG stream frame bytes for Flask streaming response."""
        self.start_camera()
        while self.is_streaming:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            # Process frame through computer vision pipeline
            processed_frame, _ = self.process_frame(frame)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def capture_training_sample(self, student_id: int, sample_idx: int) -> dict:
        """
        Captures a single high-quality face sample from the camera for student enrollment.
        """
        self.start_camera()
        if self.cap is None or not self.cap.isOpened():
            return {'success': False, 'error': 'Camera not available'}

        success, frame = self.cap.read()
        if not success:
            return {'success': False, 'error': 'Failed to read camera frame'}

        faces = self.face_engine.detect_faces(frame)
        if len(faces) == 0:
            return {'success': False, 'error': 'No face detected in frame. Please face the camera.'}
        if len(faces) > 1:
            return {'success': False, 'error': 'Multiple faces detected. Ensure only one person is in view.'}

        crop_gray = faces[0]['crop_gray']
        filepath = self.face_engine.save_face_sample(student_id, crop_gray, sample_idx)

        return {
            'success': True,
            'filepath': filepath,
            'sample_idx': sample_idx
        }

# Global Singleton Manager
attendance_manager = AttendanceManager()

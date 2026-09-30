"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Haar Cascade Eye Action & Blink Liveness Detection Engine
Tracks Ocular Action, Pupil Centers, and Blinks to Authenticate Real Faces
"""

import cv2
import numpy as np
import time
from collections import deque
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

class LivenessDetector:
    def __init__(self):
        # 1. Haar Eye Cascades (Standard + Eyeglasses fallback)
        eye_path = cv2.data.haarcascades + config.EYE_CASCADE_FILENAME
        glass_eye_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"

        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        self.eye_glass_cascade = cv2.CascadeClassifier(glass_eye_path)
        self.blur_threshold = getattr(config, 'BLUR_THRESHOLD', 80.0)
        
        # 2. Per-Face Temporal Ocular & Blink Tracking State Machine
        # track_key -> {
        #   'eye_history': deque of eye_counts (last 20 frames),
        #   'eye_positions': deque of (cx, cy) relative to face crop,
        #   'blink_count': int,
        #   'last_state': 'OPEN' | 'BLINKING',
        #   'blink_start': float,
        #   'verified_live': bool,
        #   'live_reason': str,
        #   'frames_seen': int,
        #   'last_seen': float
        # }
        self.trackers = {}

    def _get_track_key(self, x, y, w, h) -> str:
        """Quantizes face coordinates to associate temporal frames with the same face."""
        qx = int(x / 45) * 45
        qy = int(y / 45) * 45
        return f"{qx}_{qy}"

    def detect_eyes(self, face_gray) -> list:
        """
        Detects eyes within the upper anatomical ocular region of the face.
        Applies CLAHE contrast enhancement for robust detection under all lighting.
        Returns list of (ex, ey, ew, eh) relative to the face box.
        """
        if face_gray is None or face_gray.size == 0:
            return []

        h, w = face_gray.shape
        # Ocular Region of Interest (ROI): upper 15% to 60% height of face
        roi_y1 = int(h * 0.15)
        roi_y2 = int(h * 0.60)
        roi_x1 = int(w * 0.05)
        roi_x2 = int(w * 0.95)

        roi_gray = face_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_gray.size == 0:
            return []

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced_roi = clahe.apply(roi_gray)

        # Detect with primary eye cascade
        eyes = self.eye_cascade.detectMultiScale(
            enhanced_roi,
            scaleFactor=1.10,
            minNeighbors=3,
            minSize=(int(w * 0.08), int(h * 0.08)),
            maxSize=(int(w * 0.45), int(h * 0.40))
        )

        # Fallback to eyeglasses cascade if primary detected nothing
        if len(eyes) == 0 and not self.eye_glass_cascade.empty():
            eyes = self.eye_glass_cascade.detectMultiScale(
                enhanced_roi,
                scaleFactor=1.10,
                minNeighbors=2,
                minSize=(int(w * 0.08), int(h * 0.08)),
                maxSize=(int(w * 0.45), int(h * 0.40))
            )

        # Fallback to natural grayscale if CLAHE enhanced was too saturated
        if len(eyes) == 0:
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.10,
                minNeighbors=2,
                minSize=(int(w * 0.08), int(h * 0.08)),
                maxSize=(int(w * 0.45), int(h * 0.40))
            )

        # Convert coordinates back relative to face_gray
        adjusted_eyes = []
        for (ex, ey, ew, eh) in eyes:
            adjusted_eyes.append((ex + roi_x1, ey + roi_y1, ew, eh))

        return adjusted_eyes

    def update_ocular_action(self, track_key: str, eyes: list) -> dict:
        """
        State Machine for Real-Time Human Eye Action & Blink Tracking:
        - When eyes are open: eye_count >= 1
        - When eyelid blinks: eye_count drops to 0 for 1-4 frames (0.05 - 0.75s)
        - When eyes reopen: valid human blink event registered!
        - Also tracks ocular micro-movement to defeat static photos.
        """
        now = time.time()
        eye_count = len(eyes)

        # Clean stale trackers older than 3 seconds
        stale_keys = [k for k, v in self.trackers.items() if now - v['last_seen'] > 3.0]
        for k in stale_keys:
            self.trackers.pop(k, None)

        if track_key not in self.trackers:
            self.trackers[track_key] = {
                'eye_history': deque(maxlen=25),
                'eye_positions': deque(maxlen=20),
                'blink_count': 0,
                'last_state': 'OPEN' if eye_count > 0 else 'BLINKING',
                'blink_start': now,
                'verified_live': False,
                'live_reason': 'Tracking Eyes',
                'frames_seen': 0,
                'last_seen': now
            }

        tracker = self.trackers[track_key]
        tracker['last_seen'] = now
        tracker['frames_seen'] += 1
        tracker['eye_history'].append(eye_count)

        if eye_count > 0:
            # Track primary eye center relative to face crop
            ex, ey, ew, eh = eyes[0]
            tracker['eye_positions'].append((ex + ew / 2.0, ey + eh / 2.0))

        just_blinked = False

        # --- Biological Blink Transition State Machine ---
        if eye_count == 0:
            if tracker['last_state'] == 'OPEN':
                tracker['last_state'] = 'BLINKING'
                tracker['blink_start'] = now
        else: # eye_count >= 1
            if tracker['last_state'] == 'BLINKING':
                blink_duration = now - tracker.get('blink_start', now)
                # Valid biological human blink: 50ms to 850ms
                if 0.04 <= blink_duration <= 0.85:
                    tracker['blink_count'] += 1
                    tracker['verified_live'] = True
                    tracker['live_reason'] = f"Blink Verified ({tracker['blink_count']} blinks)"
                    just_blinked = True
                tracker['last_state'] = 'OPEN'
            else:
                tracker['last_state'] = 'OPEN'

        # --- Ocular Micro-Movement Analysis (Rejects 2D Rigid Screen / Photo) ---
        if not tracker['verified_live'] and tracker['frames_seen'] >= 14:
            if len(tracker['eye_positions']) >= 8:
                xs = [p[0] for p in tracker['eye_positions']]
                ys = [p[1] for p in tracker['eye_positions']]
                var_pos = float(np.var(xs) + np.var(ys))
                # Natural human eye saccades and tremors have variance >= 0.8
                # Static photos / phone screens held still have variance < 0.2
                if var_pos >= 0.8 and eye_count >= 1:
                    tracker['verified_live'] = True
                    tracker['live_reason'] = "Live Ocular Dynamics Verified"

        return {
            'eyes_detected': eye_count,
            'blink_count': tracker['blink_count'],
            'just_blinked': just_blinked,
            'is_live_ocular': tracker['verified_live'],
            'live_reason': tracker['live_reason'],
            'frames_tracked': tracker['frames_seen']
        }

    def check_skin_tone(self, face_bgr) -> bool:
        """Biological skin chrominance check (YCrCb) with realistic lighting tolerance."""
        if face_bgr is None or face_bgr.size == 0:
            return False
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        # Broad biological skin envelope (Fair to Deep skin tones under diverse lighting)
        skin_mask = (cr >= 110) & (cr <= 195) & (cb >= 60) & (cb <= 150)
        skin_ratio = np.sum(skin_mask) / (face_bgr.shape[0] * face_bgr.shape[1])
        return skin_ratio >= 0.12

    def verify_liveness(self, face_bgr, face_gray, box=(0, 0, 100, 100)) -> dict:
        """
        Haar Cascade Eye Action Liveness Verification Pipeline:
        1. Detects ocular regions (Haar Eye Cascade + Glasses fallback + CLAHE)
        2. Tracks eye action & blink transitions over time
        3. Analyzes ocular micro-movement to reject static photos & phone displays
        4. Confirms biological skin chrominance
        """
        x, y, w, h = box
        track_key = self._get_track_key(x, y, w, h)

        # 1. Detect Eyes
        eyes = self.detect_eyes(face_gray)
        eye_count = len(eyes)

        # 2. Update Ocular & Blink Tracking
        ocular = self.update_ocular_action(track_key, eyes)

        # 3. Check Skin Tone
        has_skin = self.check_skin_tone(face_bgr)

        # 4. Decision: Real Face vs Spoof (Mobile / Photo)
        frames_tracked = ocular['frames_tracked']

        if frames_tracked < 12:
            # Initial tracking window (first ~0.5s):
            if ocular['is_live_ocular']:
                is_live = True
                status_desc = ocular['live_reason']
            elif eye_count >= 1:
                is_live = True  # Display cyan tracking reticles
                status_desc = "Eyes Active - Blink to Authenticate"
            else:
                is_live = False
                status_desc = "Awaiting Eye Alignment"
        else:
            # Established tracking window (after ~0.5s):
            if ocular['is_live_ocular'] and has_skin:
                is_live = True
                status_desc = ocular['live_reason']
            elif not has_skin:
                is_live = False
                status_desc = "REJECTED: No Biological Skin"
            else:
                # Static image or phone screen without biological eye action / blinks
                is_live = False
                status_desc = "REJECTED: Mobile Screen / Static Photo"

        return {
            'passed': is_live,
            'eyes': eyes, # List of (ex, ey, ew, eh) relative to face crop
            'eye_count': eye_count,
            'blink_count': ocular['blink_count'],
            'just_blinked': ocular['just_blinked'],
            'reason': status_desc,
            'score': 98.0 if is_live else 25.0
        }

if __name__ == "__main__":
    detector = LivenessDetector()
    print("Eye Action & Blink Liveness Detector initialized successfully.")

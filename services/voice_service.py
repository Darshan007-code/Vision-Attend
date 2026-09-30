"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Voice Feedback Service (Background Thread TTS)
"""

import threading
import queue
import pyttsx3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

class VoiceService:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Worker thread so speech generation never stalls camera frame rate."""
        # Initialize pyttsx3 inside thread for COM thread safety in Windows
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165) # comfortable speaking rate
            engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"[VoiceService] Error initializing pyttsx3: {e}")
            engine = None

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text is None:
                    break
                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        print(f"[VoiceService] Speech execution note: {err}")
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def announce_attendance(self, student_name: str, department: str = "", status: str = "Present"):
        """Queues a greeting announcement."""
        first_name = student_name.split()[0] if student_name else "Student"
        if status == "Late":
            msg = f"Welcome {first_name}. Attendance marked late."
        else:
            msg = f"Welcome {first_name}. Attendance verified."
        self.speech_queue.put(msg)

    def announce_warning(self, message: str):
        """Queues a warning notification."""
        self.speech_queue.put(message)

# Global singleton
voice_service = VoiceService()

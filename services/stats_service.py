"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Analytics & Statistics Service
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from core.database import get_connection, get_setting

class StatsService:
    def get_dashboard_kpis(self) -> dict:
        """Returns overall system counts and today's attendance KPIs."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        with get_connection() as conn:
            total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            total_subjects = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
            total_teachers = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
            
            # Today's attendance numbers
            today_records = conn.execute("""
                SELECT 
                    COUNT(DISTINCT student_id) as total_present,
                    SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as on_time,
                    SUM(CASE WHEN status = 'Late' THEN 1 ELSE 0 END) as late_count
                FROM attendance
                WHERE date = ?
            """, (today_str,)).fetchone()

            today_unique_present = today_records['total_present'] or 0
            on_time = today_records['on_time'] or 0
            late = today_records['late_count'] or 0

            absent = max(0, total_students - today_unique_present)
            rate = round((today_unique_present / total_students * 100.0), 1) if total_students > 0 else 0.0

            return {
                'total_students': total_students,
                'total_subjects': total_subjects,
                'total_teachers': total_teachers,
                'today_present': today_unique_present,
                'today_on_time': on_time,
                'today_late': late,
                'today_absent': absent,
                'today_rate': rate,
                'today_date': today_str
            }

    def get_7day_trend(self) -> dict:
        """Returns last 7 days date labels and unique student attendance counts."""
        dates = []
        counts = []
        now = datetime.now()

        with get_connection() as conn:
            for i in range(6, -1, -1):
                d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                d_label = (now - timedelta(days=i)).strftime("%d %b")
                dates.append(d_label)

                c = conn.execute("""
                    SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?
                """, (d,)).fetchone()[0]
                counts.append(c)

        return {'labels': dates, 'data': counts}

    def get_department_distribution(self) -> dict:
        """Returns attendance counts grouped by student department."""
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT s.department, COUNT(a.id) as log_count
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                GROUP BY s.department
            """).fetchall()

            labels = [r['department'] for r in rows] or ["Computer Science", "Information Technology"]
            counts = [r['log_count'] for r in rows] or [0, 0]
            return {'labels': labels, 'data': counts}

    def get_defaulters_list(self) -> list:
        """
        Identifies students whose overall attendance rate is below min_attendance_pct (default 75%).
        """
        min_pct = float(get_setting("min_attendance_pct", config.ATTENDANCE_MIN_PERCENTAGE))
        with get_connection() as conn:
            total_sessions = conn.execute("""
                SELECT COUNT(DISTINCT date || '_' || subject_id) FROM attendance
            """).fetchone()[0]

            if total_sessions == 0:
                return []

            students = conn.execute("SELECT * FROM students").fetchall()
            defaulters = []

            for s in students:
                attended = conn.execute("""
                    SELECT COUNT(DISTINCT date || '_' || subject_id) FROM attendance
                    WHERE student_id = ?
                """, (s['id'],)).fetchone()[0]

                pct = round((attended / total_sessions * 100.0), 1)
                if pct < min_pct:
                    defaulters.append({
                        'student_id': s['id'],
                        'roll_number': s['roll_number'],
                        'name': s['name'],
                        'department': s['department'],
                        'semester': s['semester'],
                        'attended': attended,
                        'total_sessions': total_sessions,
                        'percentage': pct,
                        'shortfall': round(min_pct - pct, 1)
                    })

            # Sort by lowest percentage first
            defaulters.sort(key=lambda x: x['percentage'])
            return defaulters

stats_service = StatsService()

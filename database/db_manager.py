import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path='database/system.db'):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Table for registered persons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    encoding BLOB
                )
            ''')
            # Table for detections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    FOREIGN KEY (person_id) REFERENCES registered_persons (id)
                )
            ''')
            # Table for video recordings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    file_path TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS occupancy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    count INTEGER
                )
            ''')
            conn.commit()

    def register_person(self, name, image_path, encoding):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registered_persons (name, image_path, encoding) VALUES (?, ?, ?)',
                (name, image_path, encoding)
            )
            conn.commit()
            return cursor.lastrowid

    def log_detection(self, person_id, camera_id, image_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO detections (person_id, camera_id, image_path) VALUES (?, ?, ?)',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def update_detection_person(self, camera_id, image_path, person_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE detections SET person_id = ? WHERE camera_id = ? AND image_path = ?',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def start_recording(self, camera_id, file_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO video_recordings (camera_id, file_path, start_time) VALUES (?, ?, CURRENT_TIMESTAMP)',
                (camera_id, file_path)
            )
            conn.commit()
            return cursor.lastrowid

    def end_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE video_recordings SET end_time = CURRENT_TIMESTAMP WHERE id = ?',
                (record_id,)
            )
            conn.commit()

    def search_recordings(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM video_recordings WHERE 1=1'
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND start_time >= ?"
                params.append(start_time)
            if end_time:
                query += " AND start_time <= ?"
                params.append(end_time)
            
            query += " ORDER BY start_time DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM video_recordings WHERE id = ?', (record_id,))
            return cursor.fetchone()

    def delete_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM video_recordings WHERE id = ?', (record_id,))
            conn.commit()


    def get_registered_persons(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM registered_persons')
            return cursor.fetchall()

    def search_detections(self, name=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT d.*, rp.name 
                FROM detections d 
                LEFT JOIN registered_persons rp ON d.person_id = rp.id
                WHERE 1=1
            '''
            params = []
            if name:
                query += " AND rp.name LIKE ?"
                params.append(f"%{name}%")
            if start_time:
                query += " AND d.timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND d.timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY d.timestamp DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def log_occupancy(self, camera_id, count):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO occupancy_log (camera_id, count) VALUES (?, ?)',
                (camera_id, int(count))
            )
            conn.commit()

    def search_occupancy(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT id, camera_id, timestamp, count FROM occupancy_log WHERE 1=1'
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def delete_occupancy_log(self, log_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM occupancy_log WHERE id = ?", (log_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_occupancy_logs_filtered(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "DELETE FROM occupancy_log WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def analytics_summary(self, camera_id=None, hours=24):
        """
        Summary numbers for the five stat cards at the top of the dashboard.
        Returns a dict with keys:
            total_detections, today_detections, unique_persons,
            registered_persons, unknown_detections,
            peak_occupancy, peak_time, busiest_hour, busiest_avg
        """
        h = int(hours)
        cam_d = "AND camera_id = ?" if camera_id else ""
        cam_p = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()

            # Total detections in window
            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            total = cur.fetchone()[0]

            # Today's detections
            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE date(timestamp) = date('now') {cam_d}
            """, cam_p)
            today = cur.fetchone()[0]

            # Unique known persons
            cur.execute(f"""
                SELECT COUNT(DISTINCT person_id) FROM detections
                WHERE person_id IS NOT NULL
                AND timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            unique_persons = cur.fetchone()[0]

            # Total registered persons
            cur.execute("SELECT COUNT(*) FROM registered_persons")
            registered = cur.fetchone()[0]

            # Unknown detections (no person_id)
            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE person_id IS NULL
                AND timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            unknown = cur.fetchone()[0]

            # Peak occupancy row
            occ_cam = "AND camera_id = ?" if camera_id else ""
            cur.execute(f"""
                SELECT MAX(count), timestamp FROM occupancy_log
                WHERE timestamp >= datetime('now', '-{h} hours') {occ_cam}
            """, cam_p)
            peak_row  = cur.fetchone()
            peak_cnt  = peak_row[0] or 0
            peak_time = peak_row[1] if peak_row else None

            # Busiest hour over last 7 days
            cur.execute(f"""
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hr,
                       AVG(count) AS avg_c
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-7 days') {occ_cam}
                GROUP BY hr
                ORDER BY avg_c DESC
                LIMIT 1
            """, cam_p)
            busy = cur.fetchone()

        return {
            "total_detections":   total,
            "today_detections":   today,
            "unique_persons":     unique_persons,
            "registered_persons": registered,
            "unknown_detections": unknown,
            "peak_occupancy":     peak_cnt,
            "peak_time":          peak_time,
            "busiest_hour":       busy[0] if busy else None,
            "busiest_avg":        round(busy[1], 2) if busy else 0,
        }

    def analytics_occupancy_trend(self, camera_id=None, hours=24):
        """
        Hourly average occupancy for the line chart.
        Returns list of {"hour": "2024-01-15T14:00:00", "avg_count": 2.5}
        """
        h   = int(hours)
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                    ROUND(AVG(count), 2)                     AS avg_count
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-{h} hours') {cam}
                GROUP BY hour
                ORDER BY hour ASC
            """, prm)
            return [{"hour": r[0], "avg_count": r[1]} for r in cur.fetchall()]

    def analytics_heatmap(self, camera_id=None):
        """
        7-day × 24-hour occupancy heatmap.
        Returns list of {"dow": 0-6, "hour": 0-23, "avg_count": float}
        where dow 0 = Sunday, 6 = Saturday (SQLite strftime %w convention).
        """
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    CAST(strftime('%w', timestamp) AS INTEGER) AS dow,
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    ROUND(AVG(count), 2)                       AS avg_count
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-7 days') {cam}
                GROUP BY dow, hour
                ORDER BY dow, hour
            """, prm)
            return [{"dow": r[0], "hour": r[1], "avg_count": r[2]} for r in cur.fetchall()]

    def analytics_top_persons(self, camera_id=None, hours=24, limit=10):
        """
        Top N persons by detection count in the given window.
        Returns list of {"name": str, "count": int, "top_camera": str}
        """
        h   = int(hours)
        lim = int(limit)
        cam = "AND d.camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    rp.name,
                    COUNT(*) AS cnt,
                    (
                        SELECT d2.camera_id
                        FROM   detections d2
                        WHERE  d2.person_id = d.person_id
                        AND    d2.timestamp >= datetime('now', '-{h} hours')
                        GROUP  BY d2.camera_id
                        ORDER  BY COUNT(*) DESC
                        LIMIT  1
                    ) AS top_cam
                FROM detections d
                JOIN registered_persons rp ON rp.id = d.person_id
                WHERE d.person_id IS NOT NULL
                AND   d.timestamp >= datetime('now', '-{h} hours')
                {cam}
                GROUP BY d.person_id
                ORDER BY cnt DESC
                LIMIT {lim}
            """, prm)
            return [{"name": r[0], "count": r[1], "top_camera": r[2]}
                    for r in cur.fetchall()]

    def analytics_per_camera(self, hours=24):
        """
        Detection count per camera (for bar chart).
        Returns list of {"camera_id": str, "count": int}
        """
        h = int(hours)
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT camera_id, COUNT(*) AS cnt
                FROM   detections
                WHERE  timestamp >= datetime('now', '-{h} hours')
                GROUP  BY camera_id
                ORDER  BY cnt DESC
            """)
            return [{"camera_id": r[0], "count": r[1]} for r in cur.fetchall()]

    def analytics_identity_breakdown(self, camera_id=None, hours=24):
        """
        Known vs unknown detection totals for the donut chart.
        Returns {"known": int, "unknown": int}
        """
        h   = int(hours)
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    SUM(CASE WHEN person_id IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN person_id IS NULL     THEN 1 ELSE 0 END)
                FROM detections
                WHERE timestamp >= datetime('now', '-{h} hours') {cam}
            """, prm)
            row = cur.fetchone()
            return {"known": row[0] or 0, "unknown": row[1] or 0}

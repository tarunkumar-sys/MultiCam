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
            # Table for unknown face clusters (DBSCAN groups)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_clusters (
                    id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL
                )
            ''')
            # Table for individual unknown sightings per cluster
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    snap_path TEXT,
                    FOREIGN KEY (cluster_id) REFERENCES unknown_clusters(id)
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

    # ------------------------------------------------------------------
    # Unknown cluster methods
    # ------------------------------------------------------------------

    def create_unknown_cluster(self, cluster_id: int, label: str):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO unknown_clusters (id, label) VALUES (?, ?)",
                (cluster_id, label)
            )
            conn.commit()

    def add_unknown_sighting(self, cluster_id: int, camera_id: str, snap_path: str):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO unknown_sightings (cluster_id, camera_id, snap_path) VALUES (?, ?, ?)",
                (cluster_id, camera_id, snap_path)
            )
            conn.commit()

    def get_all_unknown_clusters(self):
        with self.get_connection() as conn:
            return conn.execute("SELECT id, label FROM unknown_clusters ORDER BY id").fetchall()

    def get_unknown_clusters_with_sightings(self):
        """Returns (cluster_id, label, first_snap, camera_id, timestamp)."""
        with self.get_connection() as conn:
            return conn.execute('''
                SELECT uc.id, uc.label,
                       (SELECT snap_path FROM unknown_sightings WHERE cluster_id=uc.id ORDER BY id LIMIT 1),
                       us.camera_id, us.timestamp
                FROM unknown_clusters uc
                LEFT JOIN unknown_sightings us ON us.cluster_id = uc.id
                ORDER BY uc.id, us.timestamp
            ''').fetchall()

    def rename_unknown_cluster(self, cluster_id: int, new_label: str):
        with self.get_connection() as conn:
            conn.execute("UPDATE unknown_clusters SET label=? WHERE id=?", (new_label, cluster_id))
            conn.commit()

    def delete_unknown_cluster(self, cluster_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM unknown_sightings WHERE cluster_id=?", (cluster_id,))
            conn.execute("DELETE FROM unknown_clusters WHERE id=?", (cluster_id,))
            conn.commit()

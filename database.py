import sqlite3
import os
import pickle
from datetime import datetime

class DriverDatabase:
    def __init__(self, db_path='drivers.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with drivers table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                license_no TEXT NOT NULL UNIQUE,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully!")
    
    def add_driver(self, name, age, license_no, face_encoding):
        """Add a new driver to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # face_encoding is already pickled before passing!
            cursor.execute('''
                INSERT INTO drivers (name, age, license_no, face_encoding)
                VALUES (?, ?, ?, ?)
            ''', (name, age, license_no, face_encoding))
            
            driver_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return driver_id
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def get_driver_by_license(self, license_no):
        """Get driver information by license number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM drivers WHERE license_no = ?
        ''', (license_no,))
        
        driver = cursor.fetchone()
        conn.close()
        return driver
    
    def get_all_drivers(self):
        """Get all drivers from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM drivers')
        drivers = cursor.fetchall()
        conn.close()
        return drivers
    
    def get_driver_encodings(self):
        """Get all face encodings from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, face_encoding FROM drivers')
        drivers = cursor.fetchall()
        conn.close()
        
        result = []
        for driver in drivers:
            try:
                encoding = pickle.loads(driver[2])  # driver[2] is face_encoding blob
                result.append((driver[0], driver[1], encoding))  # (id, name, encoding)
            except:
                continue
        
        return result
    
    def driver_exists(self, face_encoding):
        """Check if a driver with the given face encoding exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_drivers = cursor.execute('SELECT face_encoding FROM drivers').fetchall()
        conn.close()
        
        import face_recognition
        for driver in all_drivers:
            stored_encoding = face_recognition.load_image_file(driver[0])
            stored_encoding = face_recognition.face_encodings(stored_encoding)[0]
            
            # Compare faces
            matches = face_recognition.compare_faces([stored_encoding], face_encoding)
            if matches[0]:
                return True
        
        return False

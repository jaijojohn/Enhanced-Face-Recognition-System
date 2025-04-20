import os
import sqlite3
import numpy as np
import pickle
import sys
from datetime import datetime

# Add root directory to path to allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import BASE_DIR

# Database path
DB_PATH = os.path.join(BASE_DIR, 'data', 'face_recognition.db')

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        name TEXT NOT NULL,
        details TEXT,
        date_created TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        image_data BLOB NOT NULL,
        face_encoding BLOB,
        date_captured TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def add_person(person_id, name, details=""):
    """Add a new person to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "INSERT INTO persons (person_id, name, details, date_created) VALUES (?, ?, ?, ?)",
        (person_id, name, details, date_created)
    )
    
    person_db_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return person_db_id

def add_face_sample(person_db_id, image_data, face_encoding=None):
    """Add a face sample for a person"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert numpy array to binary
    if face_encoding is not None:
        face_encoding_binary = pickle.dumps(face_encoding)
    else:
        face_encoding_binary = None
    
    date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "INSERT INTO face_samples (person_id, image_data, face_encoding, date_captured) VALUES (?, ?, ?, ?)",
        (person_db_id, image_data, face_encoding_binary, date_captured)
    )
    
    conn.commit()
    conn.close()

def get_all_persons():
    """Get all persons from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, person_id, name, details FROM persons")
    persons = cursor.fetchall()
    
    conn.close()
    return persons

def get_person_by_id(person_db_id):
    """Get person details by database ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, person_id, name, details FROM persons WHERE id = ?", (person_db_id,))
    person = cursor.fetchone()
    
    conn.close()
    return person

def get_face_samples(person_db_id=None):
    """Get face samples, optionally filtered by person ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if person_db_id:
        cursor.execute("""
            SELECT fs.id, fs.person_id, p.name, fs.face_encoding
            FROM face_samples fs
            JOIN persons p ON fs.person_id = p.id
            WHERE fs.person_id = ?
        """, (person_db_id,))
    else:
        cursor.execute("""
            SELECT fs.id, fs.person_id, p.name, fs.face_encoding
            FROM face_samples fs
            JOIN persons p ON fs.person_id = p.id
        """)
    
    samples = []
    for row in cursor.fetchall():
        sample_id, person_id, name, face_encoding_binary = row
        
        # Convert binary back to numpy array
        if face_encoding_binary:
            face_encoding = pickle.loads(face_encoding_binary)
        else:
            face_encoding = None
            
        samples.append({
            'id': sample_id,
            'person_id': person_id,
            'name': name,
            'face_encoding': face_encoding
        })
    
    conn.close()
    return samples

def get_all_encodings():
    """Get all face encodings with person information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT fs.face_encoding, p.name, p.details
        FROM face_samples fs
        JOIN persons p ON fs.person_id = p.id
        WHERE fs.face_encoding IS NOT NULL
    """)
    
    results = []
    for row in cursor.fetchall():
        face_encoding_binary, name, details = row
        
        if face_encoding_binary:
            face_encoding = pickle.loads(face_encoding_binary)
            results.append((face_encoding, name, details))
    
    conn.close()
    
    # Organize results
    encodings = []
    names = []
    details_list = []
    
    for encoding, name, details in results:
        encodings.append(encoding)
        names.append(name)
        details_list.append(details)
    
    return np.array(encodings), np.array(names), np.array(details_list)

def get_sample_image(sample_id):
    """Get a sample image by its ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_data FROM face_samples WHERE id = ?", (sample_id,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None

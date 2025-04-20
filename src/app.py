import os
import sys
import cv2
import numpy as np
import base64
import sqlite3
from io import BytesIO
from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime

# Add root directory to path to allow imports from project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Define paths directly instead of importing from config
BASE_DIR = root_dir
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'face_recognition.db')

# Initialize database
def init_db():
    """Initialize the database with required tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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
        date_captured TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Initialize database on startup
init_db()

# Test database to ensure it has data
def test_database():
    """Test function to verify database content"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check persons table
    cursor.execute("SELECT COUNT(*) FROM persons")
    person_count = cursor.fetchone()[0]
    
    print(f"Database test: Found {person_count} persons in the database")
    
    if person_count == 0:
        # Add a test person if none exist
        print("Adding a test person to the database...")
        add_person("test001", "Test Person", "Added for testing")
        print("Test person added successfully")
    
    # Verify again
    cursor.execute("SELECT id, person_id, name, details FROM persons")
    persons = cursor.fetchall()
    print(f"Persons in database: {persons}")
    
    conn.close()

# Run database test on startup
test_database()

# Database functions
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

def add_face_sample(person_db_id, image_binary, face_data=None):
    """Add a face sample for a person
    
    Args:
        person_db_id: Database ID of the person
        image_binary: Binary image data
        face_data: Dictionary with face coordinates and other metadata
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert face data to JSON string if provided
    face_data_json = None
    if face_data:
        import json
        face_data_json = json.dumps(face_data)
    
    # Check if the table has the face_data column
    cursor.execute("PRAGMA table_info(face_samples)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'face_data' not in columns:
        # Add the face_data column if it doesn't exist
        cursor.execute("ALTER TABLE face_samples ADD COLUMN face_data TEXT")
    
    cursor.execute(
        "INSERT INTO face_samples (person_id, image_data, face_data, date_captured) VALUES (?, ?, ?, ?)",
        (person_db_id, image_binary, face_data_json, date_captured)
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

def get_face_samples(person_db_id=None):
    """Get face samples, optionally filtered by person ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if person_db_id:
        cursor.execute("""
            SELECT fs.id, fs.person_id, p.name
            FROM face_samples fs
            JOIN persons p ON fs.person_id = p.id
            WHERE fs.person_id = ?
        """, (person_db_id,))
    else:
        cursor.execute("""
            SELECT fs.id, fs.person_id, p.name
            FROM face_samples fs
            JOIN persons p ON fs.person_id = p.id
        """)
    
    samples = []
    for row in cursor.fetchall():
        sample_id, person_id, name = row
        samples.append({
            'id': sample_id,
            'person_id': person_id,
            'name': name
        })
    
    conn.close()
    return samples

@app.route('/')
def index():
    """Home page with links to capture and recognition pages"""
    return render_template('index.html')

@app.route('/capture')
def capture():
    """Page for capturing face samples"""
    persons = get_all_persons()
    return render_template('capture.html', persons=persons)

@app.route('/recognize')
def recognize():
    """Page for live face recognition"""
    return render_template('recognize.html')

@app.route('/api/persons', methods=['GET', 'POST'])
def handle_persons():
    """API endpoint to get all persons or add a new person"""
    if request.method == 'GET':
        persons = get_all_persons()
        result = {'persons': [{'id': p[0], 'person_id': p[1], 'name': p[2], 'details': p[3]} for p in persons]}
        print(f"API /api/persons returning {len(result['persons'])} persons: {result}")
        return jsonify(result)
    
    elif request.method == 'POST':
        data = request.json
        person_id = data.get('person_id')
        name = data.get('name')
        details = data.get('details', '')
        
        if not person_id or not name:
            return jsonify({'error': 'Person ID and name are required'}), 400
        
        try:
            person_db_id = add_person(person_id, name, details)
            return jsonify({'success': True, 'id': person_db_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/person/<int:person_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_person(person_id):
    """API endpoint to get, update or delete a specific person"""
    print(f"API handle_person called with ID: {person_id}, method: {request.method}")
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if person exists
        cursor.execute("SELECT id, person_id, name, details FROM persons WHERE id = ?", (person_id,))
        person = cursor.fetchone()
        
        print(f"Person data from database: {person}")
        
        if not person:
            conn.close()
            print(f"Person with ID {person_id} not found")
            return jsonify({'error': 'Person not found'}), 404
        
        if request.method == 'GET':
            # Return person details
            conn.close()
            return jsonify({
                'id': person[0],
                'person_id': person[1],
                'name': person[2],
                'details': person[3]
            })
            
        elif request.method == 'PUT':
            # Update person details
            data = request.json
            print(f"PUT request data: {data}")
            
            new_person_id = data.get('person_id')
            new_name = data.get('name')
            new_details = data.get('details', '')
            
            print(f"Updating person {person_id} with new values: person_id={new_person_id}, name={new_name}, details={new_details}")
            
            if not new_person_id or not new_name:
                conn.close()
                print(f"Error: Person ID and name are required. Received: person_id={new_person_id}, name={new_name}")
                return jsonify({'error': 'Person ID and name are required'}), 400
            
            try:
                cursor.execute("""
                    UPDATE persons 
                    SET person_id = ?, name = ?, details = ? 
                    WHERE id = ?
                """, (new_person_id, new_name, new_details, person_id))
                
                # Verify the update was successful
                cursor.execute("SELECT id, person_id, name, details FROM persons WHERE id = ?", (person_id,))
                updated_person = cursor.fetchone()
                print(f"Updated person data: {updated_person}")
                
                conn.commit()
                conn.close()
                
                result = {
                    'success': True,
                    'message': 'Person updated successfully',
                    'person': {
                        'id': person_id,
                        'person_id': new_person_id,
                        'name': new_name,
                        'details': new_details
                    }
                }
                print(f"Returning success response: {result}")
                return jsonify(result)
            except Exception as e:
                print(f"Error during database update: {str(e)}")
                conn.rollback()
                conn.close()
                return jsonify({'error': f'Database error: {str(e)}'}), 500
            
        elif request.method == 'DELETE':
            # First check if there are any face samples for this person
            cursor.execute("SELECT COUNT(*) FROM face_samples WHERE person_id = ?", (person_id,))
            sample_count = cursor.fetchone()[0]
            
            # Delete the person
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            
            # If there are face samples, delete them too
            if sample_count > 0:
                cursor.execute("DELETE FROM face_samples WHERE person_id = ?", (person_id,))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'Person and all associated face samples deleted successfully',
                'samples_deleted': sample_count
            })
    
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/person_samples/<int:person_id>', methods=['GET'])
def get_person_samples(person_id):
    """API endpoint to get all face samples for a specific person"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get person details
        cursor.execute("SELECT id, person_id, name, details FROM persons WHERE id = ?", (person_id,))
        person = cursor.fetchone()
        
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        
        # Get all face samples for this person
        cursor.execute("""
            SELECT id, face_data, date_captured, image_data 
            FROM face_samples 
            WHERE person_id = ? 
            ORDER BY date_captured DESC
        """, (person_id,))
        
        samples = []
        for row in cursor.fetchall():
            sample_id, face_data_json, date_captured, image_data = row
            
            # Convert image data to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Parse face data if available
            face_data = None
            if face_data_json:
                import json
                face_data = json.loads(face_data_json)
            
            samples.append({
                'id': sample_id,
                'face_data': face_data,
                'date_captured': date_captured,
                'image': f'data:image/jpeg;base64,{image_base64}'
            })
        
        conn.close()
        
        return jsonify({
            'person': {
                'id': person[0],
                'person_id': person[1],
                'name': person[2],
                'details': person[3]
            },
            'samples': samples
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/face_sample/<int:sample_id>', methods=['DELETE'])
def delete_face_sample(sample_id):
    """API endpoint to delete a specific face sample"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the person_id before deleting (for returning)
        cursor.execute("SELECT person_id FROM face_samples WHERE id = ?", (sample_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'Face sample not found'}), 404
            
        person_id = result[0]
        
        # Delete the face sample
        cursor.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
        conn.commit()
        
        # Get count of remaining samples
        cursor.execute("SELECT COUNT(*) FROM face_samples WHERE person_id = ?", (person_id,))
        remaining_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Face sample deleted successfully',
            'person_id': person_id,
            'remaining_count': remaining_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture', methods=['POST'])
def handle_capture():
    """API endpoint to capture and save a face sample"""
    person_id = request.form.get('person_id')
    if not person_id:
        return jsonify({'error': 'Person ID is required'}), 400
    
    # Get the image data from the request
    image_data = request.form.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_binary = base64.b64decode(image_data)
        
        # Convert to OpenCV format for face detection
        nparr = np.frombuffer(image_binary, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Simple face detection using Haar Cascade (built into OpenCV)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Use the first detected face
        x, y, w, h = faces[0]
        
        # Create face data dictionary
        face_data = {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'capture_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': 1.0,  # Placeholder for confidence score
            'source': 'webcam'
        }
        
        # Save to database with face data
        add_face_sample(int(person_id), image_binary, face_data)
        
        return jsonify({'success': True, 'face': face_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """API endpoint to upload and save a face sample from a file"""
    person_id = request.form.get('person_id')
    if not person_id:
        return jsonify({'error': 'Person ID is required'}), 400
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read file into memory
        file_bytes = file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Simple face detection using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the uploaded image'}), 400
        
        # Use the first detected face
        x, y, w, h = faces[0]
        
        # Create face data dictionary
        face_data = {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'capture_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': 1.0,  # Placeholder for confidence score
            'source': 'upload',
            'filename': file.filename
        }
        
        # Save to database with face data
        add_face_sample(int(person_id), file_bytes, face_data)
        
        # Convert the image with face highlight to base64 for response
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True, 
            'face': face_data,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_person_faces():
    """Get all faces with person information from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT fs.id, p.name, p.details, fs.face_data, fs.image_data
        FROM face_samples fs
        JOIN persons p ON fs.person_id = p.id
        WHERE fs.face_data IS NOT NULL
    """)
    
    result = []
    for row in cursor.fetchall():
        sample_id, name, details, face_data_json, image_data = row
        
        if face_data_json:
            import json
            face_data = json.loads(face_data_json)
            
            # Create a thumbnail from the stored image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract face region
            x, y, w, h = face_data['x'], face_data['y'], face_data['width'], face_data['height']
            face_img = img[y:y+h, x:x+w]
            
            # Resize to standard size for comparison
            face_img = cv2.resize(face_img, (100, 100))
            
            result.append({
                'id': sample_id,
                'name': name,
                'details': details,
                'face_data': face_data,
                'face_img': face_img
            })
    
    conn.close()
    return result

def compare_faces(face1, face2):
    """Simple face comparison using histogram comparison"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def generate_frames():
    """Generator function for streaming video frames with face recognition"""
    # Create a fallback image for when camera is not available
    def create_fallback_image(message):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a dark gray background
        cv2.rectangle(img, (0, 0), (640, 480), (50, 50, 50), -1)
        
        # Add main error message
        cv2.putText(img, message, (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add helpful instructions
        cv2.putText(img, "Please check your camera permissions", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(img, "or try using the 'Upload Photo' option instead", (50, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Add a border
        cv2.rectangle(img, (20, 20), (620, 460), (0, 0, 255), 2)
        
        return img
    
    # Try different camera backends and indices
    backends = [
        cv2.CAP_ANY,      # Auto-detect
        cv2.CAP_DSHOW,    # DirectShow (Windows)
        cv2.CAP_MSMF,     # Microsoft Media Foundation
        cv2.CAP_V4L2      # Video for Linux
    ]
    
    camera_indices = [0, 1, -1]  # Try default (0), external (1), and system default (-1)
    cap = None
    
    # Try each backend with each camera index
    for backend in backends:
        for index in camera_indices:
            try:
                print(f"Trying to open camera with backend {backend} and index {index}...")
                cap = cv2.VideoCapture(index, backend)
                
                # Check if camera opened successfully
                if cap is not None and cap.isOpened():
                    # Read a test frame to confirm it's working
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Successfully opened camera with backend {backend} and index {index}")
                        
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        break
                    else:
                        print(f"Camera opened but couldn't read frame with backend {backend} and index {index}")
                        cap.release()
                        cap = None
                else:
                    print(f"Failed to open camera with backend {backend} and index {index}")
            except Exception as e:
                print(f"Error trying backend {backend} and index {index}: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
        
        if cap is not None and cap.isOpened():
            break
    
    # If no camera could be opened, return a static error image
    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera")
        while True:
            img = create_fallback_image("Camera not available")
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Load Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get known faces from database
    known_faces = get_person_faces()
    print(f"Loaded {len(known_faces)} known faces from database")
    
    # Frame counter for periodic update of known faces
    frame_counter = 0
    
    while True:
        # Read frame
        success, frame = cap.read()
        if not success:
            break
        
        # Update known faces every 100 frames
        frame_counter += 1
        if frame_counter >= 100:
            known_faces = get_person_faces()
            frame_counter = 0
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Skip if face is too small
            if w < 50 or h < 50:
                continue
                
            # Resize to standard size for comparison
            face_resized = cv2.resize(face_img, (100, 100))
            
            # Compare with known faces
            best_match = None
            best_similarity = 0.5  # Threshold for recognition
            
            for known_face in known_faces:
                similarity = compare_faces(face_resized, known_face['face_img'])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = known_face
            
            # Draw rectangle around face
            if best_match:
                # Recognized face - green
                color = (0, 255, 0)
                name = best_match['name']
                confidence = f"{best_similarity:.2f}"
                label = f"{name} ({confidence})"
            else:
                # Unknown face - red
                color = (0, 0, 255)
                label = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add text
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for video streaming"""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {str(e)}")
        # Create a fallback image with error message
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (640, 480), (50, 50, 50), -1)
        cv2.putText(img, "Camera Error", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(e), (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        ret, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Return a static image instead
        return f"<img src='data:image/jpeg;base64,{img_str}' width='640' height='480'>"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

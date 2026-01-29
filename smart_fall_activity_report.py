import cv2
import time
import numpy as np
import requests
import sys
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
import pickle
import face_recognition
import torchreid
from torchreid.utils import FeatureExtractor
from datetime import datetime, date
from flask import Flask, jsonify, request
import threading

data_lock = threading.Lock()

# ==================== Person Re-Identification (ReID) ====================
class ReIDManager:
    def __init__(self, threshold=0.75):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using OSNet (osnet_x1_0) - specialized for person re-identification
        # FeatureExtractor handles model loading and preprocessing (resize/norm)
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device=self.device,
            verbose=False
        )
        
        self.threshold = threshold
        # Identity Store: persistent_id -> {'embedding': tensor, 'last_seen': timestamp}
        self.identity_bank = {}
        self.next_persistent_id = 1
        self.bank_file = "reid_bank.pickle"
        self.load_bank()

    def load_bank(self):
        if os.path.exists(self.bank_file):
            try:
                with open(self.bank_file, "rb") as f:
                    data = pickle.load(f)
                    
                if isinstance(data, dict):
                    if 'bank' in data:
                        self.identity_bank = data['bank']
                        self.next_persistent_id = data.get('next_id', 1)
                    else:
                        # Legacy format: the whole pickle was the bank
                        self.identity_bank = data
                        # Estimate next_id from keys like "Person_N"
                        pids = [int(k.split('_')[1]) for k in data.keys() if isinstance(k, str) and k.startswith("Person_")]
                        self.next_persistent_id = max(pids) + 1 if pids else 1
                
                # Validation: Remove invalid entries that would cause KeyError
                valid_bank = {}
                for pid, entry in self.identity_bank.items():
                    if isinstance(entry, dict) and 'embedding' in entry:
                        valid_bank[pid] = entry
                    elif isinstance(entry, np.ndarray):
                        # Very old format where entry WAS the embedding
                        valid_bank[pid] = {'embedding': entry, 'last_seen': time.time()}
                
                self.identity_bank = valid_bank
                print(f"Loaded {len(self.identity_bank)} valid identities from ReID bank.")
            except Exception as e:
                print(f"Error loading ReID bank: {e}")

    def save_bank(self):
        try:
            with open(self.bank_file, "wb") as f:
                pickle.dump({'bank': self.identity_bank, 'next_id': self.next_persistent_id}, f)
        except Exception as e:
            print(f"Error saving ReID bank: {e}")

    @torch.no_grad()
    def get_embedding(self, person_crop):
        if person_crop is None or person_crop.size == 0: return None
        # Convert BGR (OpenCV) to RGB (expected by torchreid/PIL)
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        # extractor returns a torch tensor
        features = self.extractor([rgb_crop])
        # L2 Normalize for cosine similarity matching
        features = nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()

    def match_identity(self, current_embedding):
        if current_embedding is None: return None
        
        best_id = None
        max_sim = -1
        
        for pid, data in self.identity_bank.items():
            if 'embeddings' not in data: 
                # Backward compatibility for old format
                if 'embedding' in data:
                    data['embeddings'] = [data['embedding']]
                else:
                    continue
            
            # Compare current embedding against the BEST match in this person's gallery
            # This allows matching front view to front view, side to side, etc.
            person_sim = -1
            for gallery_emb in data['embeddings']:
                # Dimension check: prevents error if switching from ResNet50 (2048) to OSNet (512)
                if gallery_emb.shape != current_embedding.shape:
                    continue
                sim = np.dot(current_embedding, gallery_emb)
                if sim > person_sim:
                    person_sim = sim
            
            if person_sim > max_sim:
                max_sim = person_sim
                best_id = pid
        
        if max_sim > self.threshold:
            # Update last seen
            self.identity_bank[best_id]['last_seen'] = time.time()
            return best_id
        
        # New identity
        new_id = f"Person_{self.next_persistent_id}"
        self.next_persistent_id += 1
        self.identity_bank[new_id] = {
            'embeddings': [current_embedding], # Start gallery
            'last_seen': time.time(),
            'first_seen': time.time()
        }
        return new_id

    def add_to_gallery(self, pid, embedding):
        """Add a new angle to a person's signature if it's sufficiently different"""
        if pid not in self.identity_bank or embedding is None: return
        
        gallery = self.identity_bank[pid].setdefault('embeddings', [])
        
        # Only add if it's a 'new' angle (sim < 0.90 compared to existing ones)
        # and gallery is not too large (max 10 angles)
        is_new_angle = True
        for gallery_emb in gallery:
            if np.dot(embedding, gallery_emb) > 0.90:
                is_new_angle = False
                break
        
        if is_new_angle and len(gallery) < 10:
            gallery.append(embedding)
            print(f"📸 Captured new body angle for {pid} (Total: {len(gallery)})")

    def prune_bank(self, max_idle=86400, min_duration=5, protected_ids=None):
        """Remove short-lived 'ghost' IDs to prevent bank bloat. Default idle increased to 24h."""
        now = time.time()
        to_delete = []
        protected_ids = protected_ids or []
        for pid, data in self.identity_bank.items():
            if pid in protected_ids:
                continue # Never prune named individuals
                
            idle_time = now - data['last_seen']
            duration = data['last_seen'] - data.get('first_seen', data['last_seen'])
            
            # If seen once and never again for 24 hours, or tracked for < 5s then gone
            if idle_time > max_idle or (idle_time > 300 and duration < min_duration):
                to_delete.append(pid)
        
        for pid in to_delete:
            del self.identity_bank[pid]
        if to_delete:
            print(f"🧹 Pruned {len(to_delete)} ghost identities from ReID bank.")
        return to_delete

reid_manager = ReIDManager(threshold=0.65)
tracker_to_persistent = {} # Maps YOLO tracker_id -> persistent_id

# ==================== Face Recognition Setup ====================
FACES_DIR = "registered_faces"
ENCODINGS_FILE = "encodings.pickle"
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)

known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} registered faces.")

load_encodings()

def register_face(image_or_list, name, yolo_model=None):
    global known_face_encodings, known_face_names
    if image_or_list is None: return False
    
    images = image_or_list if isinstance(image_or_list, list) else [image_or_list]
    success_count = 0
    
    for image in images:
        try:
            if isinstance(image, np.ndarray):
                # If we have a YOLO model, try to crop people out first for better accuracy
                if yolo_model:
                    results = yolo_model(image, verbose=False)
                    if results[0].boxes:
                        for box in results[0].boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            crop = image[y1:y2, x1:x2]
                            if process_single_image(crop, name):
                                success_count += 1
                        continue # Already processed crops for this frame
                
                if process_single_image(image, name):
                    success_count += 1
            else: # Flask FileStorage
                file_bytes = np.frombuffer(image.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None: continue
                if process_single_image(img, name):
                    success_count += 1
        except Exception as e:
            print(f"Face Registration Error: {e}")
            
    if success_count > 0:
        with data_lock:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        return True
    return False

def process_single_image(img, name):
    global known_face_encodings, known_face_names
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        encodings = face_recognition.face_encodings(rgb, boxes)
        if len(encodings) > 0:
            with data_lock:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            return True
    except: pass
    return False

# ==================== Database Setup ====================
DB_PATH = "monitor_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for daily activity summaries
    c.execute('''CREATE TABLE IF NOT EXISTS activity
                 (date TEXT, person_id TEXT, walking REAL, standing REAL, sitting REAL, sleeping REAL, PRIMARY KEY(date, person_id))''')
    
    # Migration: Add standing column if it doesn't exist
    try:
        c.execute("ALTER TABLE activity ADD COLUMN standing REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Already exists

    # Table for fall events
    c.execute('''CREATE TABLE IF NOT EXISTS falls
                 (timestamp DATETIME, person_id TEXT, type TEXT)''')
    
    # Migration: Add unix_timestamp column if it doesn't exist
    try:
        c.execute("ALTER TABLE falls ADD COLUMN unix_timestamp REAL")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

init_db()

def log_activity_to_db():
    """Sync current in-memory stats to DB and save ReID banks every minute"""
    while True:
        time.sleep(60)
        # Snapshot the data while holding lock to minimize contention
        with data_lock:
            stats_snapshot = []
            for pid in list(all_tracked_people):
                stats_snapshot.append((str(pid), walking_time.get(pid, 0), standing_time.get(pid, 0),
                                     sitting_time.get(pid, 0), sleeping_time.get(pid, 0)))
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            today = str(date.today())
            for pid, w, st, s, sl in stats_snapshot:
                c.execute('''INSERT OR REPLACE INTO activity (date, person_id, walking, standing, sitting, sleeping)
                             VALUES (?, ?, ?, ?, ?, ?)''', (today, pid, w, st, s, sl))
            conn.commit()
            conn.close()
            
            # Save ReID and Identity mapping
            with data_lock:
                protected = list(manual_id_map.keys())
                pruned_ids = reid_manager.prune_bank(protected_ids=protected)
                for pid in pruned_ids:
                    if pid in all_tracked_people:
                        all_tracked_people.remove(pid)
                    # Also remove from stats if they have very little time (ghosts)
                    total_time = walking_time.get(pid, 0) + standing_time.get(pid, 0) + sitting_time.get(pid, 0) + sleeping_time.get(pid, 0)
                    if total_time < 5:
                        walking_time.pop(pid, None)
                        standing_time.pop(pid, None)
                        sitting_time.pop(pid, None)
                        sleeping_time.pop(pid, None)
                
                reid_manager.save_bank()
                save_manual_id_map()
            print("✓ Database and ReID banks synchronized.")
        except Exception as e:
            print(f"Sync Error: {e}")

# Start DB sync thread (MOVED TO END OF INITIALIZATION)

# ==================== Flask Server ====================
app = Flask(__name__)
fall = False
active_alerts = []  # List of unacknowledged falls

# Track fall events with timestamps (history)
fall_events = []

# Track walking/sleeping/sitting/standing durations
walking_time = defaultdict(float)
standing_time = defaultdict(float)
sleeping_time = defaultdict(float)
sitting_time = defaultdict(float)

# Track current state per person
person_state = {}
person_last_time = {}
lying_start_time = {} # Track when a person started lying down
minor_fall_start_time = {} # Track duration of minor fall for escalation
recovery_mode = {} # pid -> expiry_time (suppress minor fall alerts while getting up)
recovery_confirm_count = {} # persistent_id -> count (frames of sustained upright activity)
active_fall_event = {} # pid -> True (prevent multiple alerts for the same fall)

# Movement tracking for static object filtering and activity refinement
person_start_pos = {}
person_last_pos = {} # Track last frame position for velocity
person_velocity = defaultdict(float) # Rolling average velocity
person_vertical_velocity = defaultdict(float) # Rolling average vertical velocity
person_frames_seen = {}
person_is_confirmed = {}

# Squelch logic for ghost IDs/phantom bodies
last_global_alert_time = 0
last_alert_coords = {} # type -> (x, y)
last_alert_pid = {}   # type -> pid
all_tracked_people = set()  # Persistent list of all detected IDs
manual_id_map = {} # Manual link: YOLO_ID -> Registered Name
person_signatures = {} # Store color histograms: Name -> Histogram

ID_MAP_FILE = "manual_id_map.pickle"

def load_manual_id_map():
    global manual_id_map
    if os.path.exists(ID_MAP_FILE):
        try:
            with open(ID_MAP_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                manual_id_map = data
                print(f"Loaded {len(manual_id_map)} manual ID mappings.")
            else:
                print(f"Warning: {ID_MAP_FILE} had invalid format. Starting fresh.")
                manual_id_map = {}
        except Exception as e:
            print(f"Error loading manual ID map: {e}")
            manual_id_map = {}

def save_manual_id_map():
    try:
        # Prevent overwriting with empty map if it's likely a load failure
        # (Only save if we have mappings or if the file didn't exist)
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(manual_id_map, f)
    except Exception as e:
        print(f"Error saving manual ID map: {e}")

load_manual_id_map()

def get_color_signature(image):
    """Calculate color histogram for Re-Identification"""
    try:
        if image is None or image.size == 0: return None
        # Focus on the torso (center of the crop)
        h, w = image.shape[:2]
        torso = image[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.8)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except: return None

def compare_signatures(sig1, sig2):
    """Compare two color histograms (0.0 to 1.0)"""
    if sig1 is None or sig2 is None: return 0
    return cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)

status_message = ""
status_expiry = 0

def load_stats_from_db():
    global walking_time, standing_time, sleeping_time, sitting_time, all_tracked_people
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        today = str(date.today())
        c.execute("SELECT person_id, walking, standing, sitting, sleeping FROM activity WHERE date=?", (today,))
        rows = c.fetchall()
        with data_lock:
            for r in rows:
                pid, w, st, s, sl = r
                # Use the person_id as stored in DB
                walking_time[pid] = w
                standing_time[pid] = st
                sitting_time[pid] = s
                sleeping_time[pid] = sl
                all_tracked_people.add(pid)
        conn.close()
        print(f"Loaded stats for {len(rows)} people from database.")
    except Exception as e:
        print(f"Error loading stats: {e}")

load_stats_from_db()

def load_fall_history():
    global fall_events
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, person_id, type, unix_timestamp FROM falls ORDER BY unix_timestamp DESC LIMIT 50")
        rows = c.fetchall()
        for r in rows:
            ts, pid, ftype, uts = r
            
            # Robust timestamp parsing
            if isinstance(ts, str):
                try:
                    # SQLite datetime.now() usually looks like '2026-01-28 14:25:10.123456'
                    # or '2026-01-28 14:25:10'
                    dt_obj = datetime.strptime(ts.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    display_time = dt_obj.strftime("%H:%M:%S")
                    derived_uts = dt_obj.timestamp()
                except:
                    display_time = ts # Fallback
                    derived_uts = uts if uts else time.time()
            else:
                display_time = ts.strftime("%H:%M:%S")
                derived_uts = uts if uts else ts.timestamp()

            with data_lock:
                fall_events.append({
                    "person": pid,
                    "type": ftype,
                    "timestamp": uts if uts else derived_uts,
                    "time_str": display_time
                })
        # Keep them in chronological order for the list (appends happened in reverse order from SELECT)
        with data_lock:
            fall_events.sort(key=lambda x: x['timestamp'])
        conn.close()
        print(f"Loaded {len(fall_events)} fall events from history.")
    except Exception as e:
        print(f"Error loading fall history: {e}")

load_fall_history()

# Start DB sync thread (Wait for all initializations to complete)
threading.Thread(target=log_activity_to_db, daemon=True).start()

@app.route("/trigger", methods=["POST"])
def trigger():
    global fall, active_alerts
    data = request.get_json(silent=True) or {}
    person_id = data.get("person_id")
    msg = data.get("message", "Unknown")
    fall_type = data.get("type", "FALL")
    
    if not person_id:
        # Backward compatibility or fallback
        person_id = msg.split(' (')[0] if ' (' in msg else msg

    now = time.time()
    
    with data_lock:
        # Update existing alert for this person to prevent duplicates (e.g. Minor -> Major)
        found = False
        for alert in active_alerts:
            # Match by the clean person_id
            if str(alert['person_id']) == str(person_id):
                alert['type'] = fall_type
                alert['message'] = msg
                alert['time_str'] = time.strftime("%H:%M:%S", time.localtime(now))
                alert['timestamp'] = now
                found = True
                break
        
        if not found:
            active_alerts.append({
                "person_id": person_id,
                "message": msg,
                "type": fall_type,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "timestamp": now
            })
    
    fall = True
    return "OK"

@app.route("/api/acknowledge/<pid>", methods=["POST"])
def acknowledge(pid):
    global active_alerts
    with data_lock:
        # Match by person_id for precise removal
        active_alerts = [a for a in active_alerts if str(a['person_id']) != str(pid) and str(a['message']) != str(pid)]
    return jsonify({"status": "success"})

@app.route("/fall")
def check():
    with data_lock:
        is_fall = len(active_alerts) > 0
    return jsonify({"fall": is_fall})

last_frame = None

def rename_person(old_id, new_name):
    """Safely rename a person and transfer all their stats and mappings."""
    with data_lock:
        _rename_person_internal(old_id, new_name)

def _rename_person_internal(old_id, new_name):
    # 1. Resolve to actual persistent_id (Person_N) if old_id is already a name
    target_persistent_id = old_id
    for k, v in manual_id_map.items():
        if v == old_id:
            target_persistent_id = k
            break
            
    manual_id_map[str(target_persistent_id)] = new_name
    
    # Transfer current stats (from both old_id and target_persistent_id to new_name)
    for source_id in [old_id, target_persistent_id]:
        if source_id != new_name:
            walking_time[new_name] += walking_time.pop(source_id, 0)
            standing_time[new_name] += standing_time.pop(source_id, 0)
            sitting_time[new_name] += sitting_time.pop(source_id, 0)
            sleeping_time[new_name] += sleeping_time.pop(source_id, 0)
    
    # Update set of all people
    if str(old_id) in all_tracked_people and str(old_id) != new_name:
        all_tracked_people.remove(old_id)
    if str(target_persistent_id) in all_tracked_people and str(target_persistent_id) != new_name:
        all_tracked_people.remove(target_persistent_id)
    all_tracked_people.add(new_name)
    
    # Update active alerts and history in memory
    for alert in active_alerts:
        if str(alert.get('person_id')) == str(old_id) or str(alert.get('person_id')) == str(target_persistent_id):
            alert['person_id'] = new_name
            alert['message'] = alert['message'].replace(str(old_id), new_name).replace(str(target_persistent_id), new_name)
            
    for event in fall_events:
        if str(event.get('person')) == str(old_id) or str(event.get('person')) == str(target_persistent_id):
            event['person'] = new_name

    # Save mappings AND bank immediately to ensure persistence
    save_manual_id_map()
    reid_manager.save_bank()
    
    # Update database: Delete old entries (stats were transferred to new name)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM activity WHERE person_id=?", (str(old_id),))
        if target_persistent_id != old_id:
            c.execute("DELETE FROM activity WHERE person_id=?", (str(target_persistent_id),))
        
        # Also update fall history to link to the new name
        c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(old_id)))
        if target_persistent_id != old_id:
            c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(target_persistent_id)))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating DB for rename: {e}")
        
    print(f"👤 Renamed {old_id} (ID: {target_persistent_id}) to {new_name} and saved.")

@app.route("/register", methods=["POST"])
def register():
    global last_frame, status_message, status_expiry
    name = request.form.get("name")
    target_id = request.form.get("yolo_id") # Register by active ID (can be persistent_id)
    
    if not name:
        return jsonify({"status": "error", "message": "Missing name"})
    
    # CASE 1: Manual ID naming (No face needed)
    with data_lock:
        if target_id and (str(target_id) in person_state or str(target_id) in all_tracked_people):
            _rename_person_internal(str(target_id), name)
            status_message = f"ID {target_id} is now {name}"
            status_expiry = time.time() + 5
            return jsonify({"status": "success", "message": f"Successfully named body {target_id} as {name}"})

        # NEW: Fallback for Navbar button - If only one person is visible or unnamed, name them
        if not target_id:
            unnamed = [pid for pid in person_state if pid not in manual_id_map]
            if len(unnamed) == 1:
                target_id = unnamed[0]
                _rename_person_internal(str(target_id), name)
                status_message = f"Auto-linked {name} to ID {target_id}"
                status_expiry = time.time() + 5
                return jsonify({"status": "success", "message": f"Successfully named visible body {target_id} as {name}"})
            elif len(unnamed) > 1:
                 return jsonify({"status": "error", "message": "Multiple unnamed bodies. Click the ID button next to the body instead."})

    # CASE 2: Uploaded files or Live frame
    front_img = request.files.get("front")
    back_img = request.files.get("back")
    
    to_process = []
    if front_img: to_process.append(front_img)
    if back_img: to_process.append(back_img)
    
    if not to_process:
        if last_frame is not None:
            to_process = [last_frame]
        else:
            return jsonify({"status": "error", "message": "No photos uploaded or live frame available"})
    
    if register_face(to_process, name, yolo_model=model if not (front_img or back_img) else None):
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        return jsonify({"status": "success", "message": f"Successfully registered {name}"})
    return jsonify({"status": "error", "message": "No face detected in provided images"})

@app.route("/api/history")
def activity_history():
    """Get history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, person_id, walking, standing, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"date": r[0], "pid": r[1], "walk": r[2], "stand": r[3], "sit": r[4], "sleep": r[5]} for r in rows])

@app.route("/api/history/monthly")
def monthly_history():
    """Get monthly aggregated history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT SUBSTR(date, 1, 7) as month, person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping) 
                 FROM activity 
                 GROUP BY month, person_id 
                 ORDER BY month DESC LIMIT 24""")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"date": r[0], "pid": r[1], "walk": r[2], "stand": r[3], "sit": r[4], "sleep": r[5]} for r in rows])

@app.route("/")
def home():
    """Enhanced modern dashboard with Monthly Analytics and Fall Messaging"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elderly Monitor Pro</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --danger-color: #ef233c;
                --success-color: #2ecc71;
                --warning-color: #f39c12;
                --dark-bg: #1e1e2f;
                --card-bg: #ffffff;
                --sidebar-bg: #f8f9fa;
            }
            body { background-color: #f0f2f5; font-family: 'Inter', sans-serif; color: #333; }
            .navbar { background: white; border-bottom: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
            .sidebar { background: var(--sidebar-bg); border-right: 1px solid #e0e0e0; min-height: 100vh; padding: 20px; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px; transition: transform 0.2s; }
            .card:hover { transform: translateY(-2px); }
            .status-badge { padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
            
            .bg-walking { background-color: #d1fae5; color: #065f46; }
            .bg-standing { background-color: #dbeafe; color: #1e40af; }
            .bg-sleeping { background-color: #f3e8ff; color: #5b21b6; }
            .bg-sitting { background-color: #fef3c7; color: #92400e; }
            .bg-major-fall { background-color: #fee2e2; color: #991b1b; animation: pulse 2s infinite; }
            .bg-minor-fall { background-color: #ffedd5; color: #9a3412; }
            .bg-recovered { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
            .bg-away { background-color: #f3f4f6; color: #4b5563; }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(239, 35, 60, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0); }
            }

            .alert-item { 
                background: white; border-radius: 10px; padding: 16px; margin-bottom: 12px; 
                border-left: 6px solid var(--danger-color); box-shadow: 0 2px 8px rgba(239, 35, 60, 0.1);
            }
            .alert-item.recovered { border-left-color: var(--success-color); box-shadow: 0 2px 8px rgba(46, 204, 113, 0.1); }
            
            .stat-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
            .icon-walk { background: #d1fae5; color: #10b981; }
            .icon-stand { background: #dbeafe; color: #3b82f6; }
            .icon-sit { background: #fef3c7; color: #f59e0b; }
            .icon-sleep { background: #f3e8ff; color: #8b5cf6; }

            #fall-message-display { position: fixed; top: 20px; right: 20px; z-index: 9999; width: 320px; }
            .toast-fall { 
                background: var(--danger-color); color: white; padding: 16px; border-radius: 8px; margin-bottom: 10px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); display: flex; align-items: center; animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
            
            .nav-link { color: #666; padding: 10px 15px; border-radius: 8px; margin-bottom: 5px; font-weight: 500; }
            .nav-link:hover, .nav-link.active { background: white; color: var(--primary-color); box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            
            .chart-container { height: 300px; }
        </style>
    </head>
    <body>
        <div id="fall-message-display"></div>

        <nav class="navbar px-4 py-2 sticky-top">
            <div class="container-fluid">
                <a class="navbar-brand fw-bold d-flex align-items-center" href="#">
                    <div class="bg-primary text-white p-2 rounded-3 me-2" style="width: 35px; height: 35px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-heartbeat"></i>
                    </div>
                    ElderlyCare <span class="text-primary ms-1">Pro</span>
                </a>
                <div class="d-flex align-items-center gap-3">
                    <div class="input-group input-group-sm shadow-sm" style="width: 250px;">
                        <input type="text" id="reg-name" class="form-control border-0 px-3" placeholder="Register Name">
                        <button class="btn btn-primary px-3" onclick="registerPerson()">
                            <i class="fas fa-user-plus"></i>
                        </button>
                    </div>
                    <div id="live-time" class="badge bg-light text-dark border p-2 fw-normal"></div>
                </div>
            </div>
        </nav>

        <div class="container-fluid">
            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-2 d-none d-md-block sidebar">
                    <div class="nav flex-column">
                        <a href="#" class="nav-link active"><i class="fas fa-chart-line me-2"></i> Dashboard</a>
                        <a href="#" class="nav-link"><i class="fas fa-history me-2"></i> Event History</a>
                        <a href="#" class="nav-link"><i class="fas fa-users me-2"></i> Managed People</a>
                        <a href="#" class="nav-link"><i class="fas fa-cog me-2"></i> Settings</a>
                    </div>
                    <hr>
                    <div class="p-3 bg-white rounded-3 shadow-sm mt-4">
                        <div class="small text-muted mb-2">System Status</div>
                        <div class="d-flex align-items-center">
                            <div class="spinner-grow spinner-grow-sm text-success me-2"></div>
                            <span class="small fw-bold">Live Monitoring</span>
                        </div>
                    </div>
                </div>

                <!-- Main Content -->
                <div class="col-md-10 p-4">
                    <div id="alert-container"></div>

                    <div class="row g-4">
                        <!-- Activity Grid -->
                        <div class="col-lg-8">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <h5 class="fw-bold mb-0">Daily Activity Tracker</h5>
                                <div id="unnamed-container" class="d-none">
                                     <div id="unnamed-list" class="d-flex gap-2"></div>
                                </div>
                            </div>
                            
                            <div id="people-grid" class="row"></div>

                            <!-- Analytics Section -->
                            <div class="card p-4">
                                <div class="d-flex justify-content-between align-items-center mb-4">
                                    <div>
                                        <h5 class="fw-bold mb-0" id="chart-title">Activity Analytics</h5>
                                        <p class="text-muted small mb-0">Compare walking, sitting, and resting patterns</p>
                                    </div>
                                    <div class="btn-group p-1 bg-light rounded-3">
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-daily" onclick="setViewMode('daily')">Daily</button>
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-monthly" onclick="setViewMode('monthly')">Monthly</button>
                                    </div>
                                </div>
                                <div class="chart-container">
                                    <canvas id="activityChart"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Side Panel: Fall Monitor -->
                        <div class="col-lg-4">
                            <h5 class="fw-bold mb-4">Fall Monitor Feed</h5>
                            <div class="card shadow-sm h-100">
                                <div class="card-header bg-white py-3">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span class="small fw-bold text-uppercase text-muted letter-spacing-1">Recent Events</span>
                                        <i class="fas fa-bell text-muted"></i>
                                    </div>
                                </div>
                                <div class="card-body p-0" style="max-height: 700px; overflow-y: auto;">
                                    <div id="fall-history" class="list-group list-group-flush"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let activityChart = null;
            let currentViewMode = 'daily';
            let seenAlerts = new Set();

            function setViewMode(mode) {
                currentViewMode = mode;
                document.getElementById('btn-daily').classList.toggle('bg-white', mode === 'daily');
                document.getElementById('btn-daily').classList.toggle('shadow-sm', mode === 'daily');
                document.getElementById('btn-monthly').classList.toggle('bg-white', mode === 'monthly');
                document.getElementById('btn-monthly').classList.toggle('shadow-sm', mode === 'monthly');
                window.lastChartUpdate = 0;
                update();
            }

            function registerPerson(yoloId = null) {
                const name = document.getElementById('reg-name').value;
                if (!name) { alert('Enter a name first'); return; }
                const formData = new FormData();
                formData.append('name', name);
                if (yoloId) formData.append('yolo_id', yoloId);
                fetch('/register', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        if(data.status === 'success') {
                            document.getElementById('reg-name').value = '';
                        }
                        alert(data.message);
                    });
            }

            function initChart(data) {
                const ctx = document.getElementById('activityChart').getContext('2d');
                if (activityChart) activityChart.destroy();
                
                const labels = [...new Set(data.map(d => d.date))].reverse();
                const datasets = [];
                const colors = { 
                    walk: '#10b981', 
                    stand: '#3b82f6', 
                    sit: '#f59e0b', 
                    sleep: '#8b5cf6' 
                };

                ['walk', 'stand', 'sit', 'sleep'].forEach(type => {
                    datasets.push({
                        label: type.charAt(0).toUpperCase() + type.slice(1),
                        data: labels.map(l => {
                            const entries = data.filter(d => d.date === l);
                            const val = entries.reduce((acc, curr) => acc + curr[type], 0);
                            // Convert to hours for monthly, minutes for daily
                            return currentViewMode === 'monthly' ? (val / 3600).toFixed(1) : (val / 60).toFixed(1);
                        }),
                        backgroundColor: colors[type],
                        borderColor: colors[type],
                        borderWidth: 1,
                        borderRadius: 4
                    });
                });

                activityChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels, datasets },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        plugins: { 
                            legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 6 } },
                            tooltip: { backgroundColor: '#1e1e2f', padding: 12 }
                        },
                        scales: { 
                            y: { 
                                beginAtZero: true, 
                                title: { display: true, text: currentViewMode === 'monthly' ? 'Hours' : 'Minutes' },
                                grid: { color: '#f0f0f0' }
                            },
                            x: { grid: { display: false } }
                        } 
                    }
                });
            }

            function showFallToast(msg, type) {
                const container = document.getElementById('fall-message-display');
                const toast = document.createElement('div');
                toast.className = 'toast-fall';
                if (type === 'RECOVERED') {
                    toast.style.background = '#10b981';
                    toast.innerHTML = `<i class="fas fa-check-circle me-3"></i> <div><strong>RECOVERY</strong><br><small>${msg}</small></div>`;
                } else {
                    toast.innerHTML = `<i class="fas fa-exclamation-triangle me-3"></i> <div><strong>FALL ALERT</strong><br><small>${msg}</small></div>`;
                }
                container.appendChild(toast);
                setTimeout(() => {
                    toast.style.opacity = '0';
                    setTimeout(() => toast.remove(), 300);
                }, 6000);
            }

            function ackFall(pid) {
                fetch(`/api/acknowledge/${pid}`, { method: 'POST' }).then(() => update());
            }

            function update() {
                fetch('/api/report')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('live-time').innerHTML = `<i class="far fa-clock me-1 text-muted"></i> ${new Date().toLocaleTimeString()}`;
                        
                        // Active Alerts
                        let alertHtml = '';
                        for (let l of data.active_alerts) {
                            const isRecovered = l.type === 'RECOVERED';
                            const alertKey = `${l.person_id}-${l.type}-${l.timestamp}`;
                            if (!seenAlerts.has(alertKey)) {
                                showFallToast(`${l.message}`, l.type);
                                seenAlerts.add(alertKey);
                            }
                            
                            alertHtml += `
                                <div class="alert-item shadow-sm ${isRecovered ? 'recovered' : ''}">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h6 class="mb-1 ${isRecovered ? 'text-success' : 'text-danger'} fw-bold">
                                                ${isRecovered ? '<i class="fas fa-check-circle"></i> RECOVERY' : '<i class="fas fa-exclamation-circle"></i> FALL ALERT'}
                                            </h6>
                                            <p class="mb-0 small text-muted"><strong>${l.message}</strong> - Detected at ${l.time_str}</p>
                                        </div>
                                        <button class="btn btn-outline-dark btn-sm rounded-pill px-3" onclick="ackFall('${l.person_id}')">Dismiss</button>
                                    </div>
                                </div>`;
                        }
                        document.getElementById('alert-container').innerHTML = alertHtml;
                        
                        // Unnamed IDs
                        const unnamedContainer = document.getElementById('unnamed-container');
                        if (data.unnamed_ids && data.unnamed_ids.length > 0) {
                            unnamedContainer.classList.remove('d-none');
                            document.getElementById('unnamed-list').innerHTML = data.unnamed_ids.map(id => 
                                `<button class="btn btn-warning btn-sm border-0 rounded-2 px-3 shadow-sm" onclick="registerPerson('${id}')">Name ID ${id}</button>`
                            ).join('');
                        } else {
                            unnamedContainer.classList.add('d-none');
                        }

                        // People Grid
                        document.getElementById('people-grid').innerHTML = data.people.map(p => {
                            let activity = p.current_activity;
                            let badgeCls = 'bg-' + activity.toLowerCase().replace(' ', '-');
                            return `
                                <div class="col-md-6 mb-4" style="opacity: ${p.is_active ? '1.0' : '0.6'}">
                                    <div class="card p-4 h-100">
                                        <div class="d-flex justify-content-between align-items-center mb-4">
                                            <div class="d-flex align-items-center">
                                                <div class="bg-light p-2 rounded-circle me-3" style="width: 45px; height: 45px; display: flex; align-items: center; justify-content: center;">
                                                    <i class="fas fa-user text-muted"></i>
                                                </div>
                                                <h6 class="fw-bold mb-0">${p.person}</h6>
                                            </div>
                                            <span class="status-badge ${badgeCls}">${activity}</span>
                                        </div>
                                        <div class="row g-2">
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-walk mx-auto"><i class="fas fa-walking"></i></div>
                                                <div class="fw-bold small">${p.walking_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-stand mx-auto"><i class="fas fa-male"></i></div>
                                                <div class="fw-bold small">${p.standing_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-sit mx-auto"><i class="fas fa-chair"></i></div>
                                                <div class="fw-bold small">${p.sitting_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-sleep mx-auto"><i class="fas fa-bed"></i></div>
                                                <div class="fw-bold small">${p.sleeping_dur}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>`;
                        }).join('') || '<div class="col-12"><div class="card p-5 text-center text-muted">No active detections...</div></div>';
                        
                        // Fall History
                        document.getElementById('fall-history').innerHTML = data.falls.map(f => {
                            let icon = f.type === 'MAJOR FALL' ? 'fa-exclamation-circle text-danger' : 
                                      (f.type === 'RECOVERED' ? 'fa-check-circle text-success' : 'fa-exclamation-triangle text-warning');
                            return `
                                <div class="list-group-item px-4 py-3 border-0 border-bottom">
                                    <div class="d-flex align-items-center">
                                        <i class="fas ${icon} fs-5 me-3"></i>
                                        <div class="flex-grow-1">
                                            <div class="fw-bold small">${f.person}</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">${f.type} • ${f.time_str}</div>
                                        </div>
                                    </div>
                                </div>`;
                        }).join('') || '<div class="p-5 text-center text-muted small">No fall history yet.</div>';
                    });

                // Update charts
                if (!window.lastChartUpdate || (Date.now() - window.lastChartUpdate > 30000)) {
                    const endpoint = currentViewMode === 'daily' ? '/api/history' : '/api/history/monthly';
                    fetch(endpoint).then(r => r.json()).then(data => {
                        initChart(data);
                        window.lastChartUpdate = Date.now();
                    });
                }
            }
            setInterval(update, 2000);
            update();
            setViewMode('daily');
        </script>
    </body>
    </html>
    """
    return html

def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

@app.route("/api/report")
def api_report():
    """API endpoint for JSON report with sorting and limiting"""
    with data_lock:
        # Snapshot current state for reporting
        current_state_snapshot = person_state.copy()
        # Unnamed IDs are those in current state that don't have a manual mapping
        unnamed_ids = [pid for pid in current_state_snapshot if pid not in manual_id_map]
        
        # Sort people: currently active first, then by total monitored time
        all_pids = list(all_tracked_people)
        
        def get_activity_score(display_id):
            # Check if this person (by name or ID) is currently active
            is_active = display_id in current_state_snapshot # If it's a persistent_id
            if not is_active:
                # Check if any persistent_id mapped to this name is active
                is_active = any(k in current_state_snapshot for k, v in manual_id_map.items() if v == display_id)
            
            total_time = walking_time.get(display_id, 0) + standing_time.get(display_id, 0) + sitting_time.get(display_id, 0) + sleeping_time.get(display_id, 0)
            return (is_active, total_time)

        sorted_pids = sorted(all_pids, key=get_activity_score, reverse=True)
        
        report_data = []
        for display_id in sorted_pids[:10]:
            # Try to find an ACTIVE persistent ID for this name to get the current activity
            internal_id = display_id
            for k, v in manual_id_map.items():
                if v == display_id:
                    internal_id = k
                    if k in current_state_snapshot:
                        break # Prioritize the active one

            report_data.append({
                "person": display_id,
                "walking_dur": format_duration(walking_time.get(display_id, 0)),
                "standing_dur": format_duration(standing_time.get(display_id, 0)),
                "sleeping_dur": format_duration(sleeping_time.get(display_id, 0)),
                "sitting_dur": format_duration(sitting_time.get(display_id, 0)),
                "current_activity": current_state_snapshot.get(internal_id, "AWAY"),
                "is_active": internal_id in current_state_snapshot
            })
            
        # Ensure alerts and history are sorted by high-precision float timestamp
        alerts_copy = sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)
        falls_copy = sorted(fall_events, key=lambda x: x['timestamp'], reverse=True)[:10]

    return jsonify({
        "people": report_data,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids
    })

def run_server():
    # Runs Flask server in background thread
    try:
        print("Flask: Starting server...")
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Flask Error: {e}")
        import traceback
        traceback.print_exc()

print("Starting Flask server thread...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(1)  # Give Flask time to start
print("✓ Flask server should be running at http://127.0.0.1:5000")
print("  Access the report at: http://127.0.0.1:5000/")

# ==================== YOLO11 Pose Model ====================
model = YOLO("yolo11n-pose.pt")  # latest & fast
FALL_URL = "http://127.0.0.1:5000/trigger"
fall_cooldown = {}  # Prevent spam alerts for same person

def classify_activity(keypoints, conf, velocity=0, v_velocity=0, aspect_ratio=1.0):
    """
    Advanced skeleton-based activity classification with Impact Detection.
    """
    try:
        # Check confidence of critical joints
        critical_joints = [5, 6, 11, 12] # Shoulders and Hips
        
        # Calculate Midpoints and distances
        sho_y = (keypoints[5][1] + keypoints[6][1]) / 2
        sho_x = (keypoints[5][0] + keypoints[6][0]) / 2
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
        
        dy = hip_y - sho_y
        dx = hip_x - sho_x
        angle = abs(np.degrees(np.arctan2(dx, dy)))

        # Impact Detection: High vertical velocity (downward) + High angle or wide box
        # Threshold: > 8 pixels/frame downward is usually a fall
        if v_velocity > 8.0 and (angle > 30 or aspect_ratio > 1.2):
            return "MAJOR FALL"

        if any(conf[j] < 0.5 for j in critical_joints):
            if conf[0] > 0.5 and (conf[11] > 0.5 or conf[12] > 0.5):
                nose_y = keypoints[0][1]
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (conf[11] > 0.5 and conf[12] > 0.5) else (keypoints[11][1] if conf[11] > 0.5 else keypoints[12][1])
                if nose_y > hip_y - 10:
                    return "LYING"
            
            # If joints are hidden but the box is very wide, likely lying down
            if aspect_ratio > 1.8:
                return "LYING"
            return "UNKNOWN"
        
        # 1. LYING (Horizontal and low)
        # Relaxed thresholds for better fall detection
        if angle > 65 or aspect_ratio > 1.8:
            return "LYING"
        
        # Nose-to-ground ratio: If head is significantly lower than normal relative to torso
        if conf[0] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            nose_y = keypoints[0][1]
            # Height of head from "floor" (ankles)
            head_height = abs(ank_y - nose_y)
            # Length of torso
            torso_len = np.sqrt(dx**2 + dy**2)
            # If head is very low (less than torso length away from floor), they are down
            if head_height < torso_len * 0.8:
                return "LYING"

        if angle > 45 and conf[0] > 0.5 and keypoints[0][1] > hip_y:
            return "LYING"
        
        # 3. SITTING vs UPRIGHT (STANDING/WALKING)
        # Move sitting logic up to prevent "sitting on floor" from being "Minor Fall"
        is_sitting = False
        if conf[13] > 0.5 and conf[14] > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            if conf[15] > 0.5 and conf[16] > 0.5:
                ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
                upper_leg = abs(knee_y - hip_y)
                lower_leg = abs(ank_y - knee_y)
                if upper_leg < lower_leg * 0.5: 
                    is_sitting = True
        
        if not is_sitting and conf[15] > 0.5 and conf[16] > 0.5:
            torso_len = np.sqrt(dx**2 + dy**2)
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            total_h = abs(ank_y - sho_y)
            # Sitting on floor with spread legs often results in torso_len / total_h > 0.6
            if torso_len / total_h > 0.6: 
                is_sitting = True
        
        # Additional floor sitting check: Hips close to floor (ankles) but torso not horizontal
        if not is_sitting and conf[11] > 0.5 and conf[12] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            hip_to_floor = abs(ank_y - hip_y)
            torso_len = np.sqrt(dx**2 + dy**2)
            if hip_to_floor < torso_len * 0.4 and angle < 50:
                is_sitting = True

        if is_sitting:
            return "SITTING"
        
        # 2. MINOR FALL (Significant tilt but not fully flat)
        # Refined: Increased angle/aspect ratio and ensure not actively walking
        if (angle > 45 or aspect_ratio > 1.5) and velocity < 3.0:
            return "MINOR FALL"
        
        # 4. STANDING vs WALKING (Using velocity + Pose)
        # Increased threshold to 3.0 for WALKING to reduce noise
        if velocity > 3.0:
            return "WALKING"
        
        # Fallback to pose-based walking detection if velocity is moderate
        if velocity > 0.8 and conf[15] > 0.5 and conf[16] > 0.5:
            feet_dist = abs(keypoints[15][0] - keypoints[16][0])
            shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
            # Feet must be wider than shoulders to be 'walking' if velocity is low
            if feet_dist > shoulder_width * 1.1:
                return "WALKING"

        return "STANDING"

    except Exception:
        return "UNKNOWN"

def send_fall_alert(alert_msg, pid, fall_type, coords=None):
    global last_global_alert_time, last_alert_coords, last_alert_pid, status_message, status_expiry
    try:
        now = time.time()
        
        # Spatial-Temporal Squelch:
        # If we sent an alert of this type recently (< 5s) and it was in the same area (< 150px)
        # then it's likely a phantom ID/tracker drift for the same person.
        if coords and fall_type in last_alert_coords:
            prev_coords = last_alert_coords[fall_type]
            dist = np.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2)
            if dist < 150 and (now - last_global_alert_time) < 5:
                # If it's the SAME person (resolved name), definitely skip
                # If it's a DIFFERENT person but very close/recent, it's likely a ghost ID
                print(f"🤫 Squelching redundant {fall_type} for {pid} (likely ghost ID)")
                return
        
        # Update last alert state
        last_global_alert_time = now
        if coords: last_alert_coords[fall_type] = coords
        last_alert_pid[fall_type] = pid

        # Update status message for visual feedback on frame
        status_message = alert_msg
        status_expiry = now + 5

        print(f"⚠️  {alert_msg}! Sending alert...")
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO falls (timestamp, person_id, type, unix_timestamp) VALUES (?, ?, ?, ?)",
                  (datetime.now(), str(pid), fall_type, now))
        conn.commit()
        conn.close()
        
        requests.post(FALL_URL, json={"person_id": str(pid), "message": alert_msg, "type": fall_type}, timeout=1)
        print("✓ Fall alert sent and logged successfully")
    except Exception as e:
        print(f"✗ Failed to send/log fall alert: {e}")

# ==================== Camera Loop ====================
def open_camera():
    """Robust camera opener for Windows"""
    for _ in range(3): # Try up to 3 times
        # Using DSHOW (DirectShow) on Windows is much more stable for resolution changes
        c = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if c.isOpened():
            # Set buffer size to 1 for lowest latency
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c
        time.sleep(0.5)
    return None

cap = open_camera()
if cap is None:
    print("Error: Cannot open camera")
    exit()

frame_count = 0
start_time = time.time()
last_detection = {}  # Track last frame when person was detected
last_motion_time = time.time()
prev_gray = None
system_sleeping = False

while True:
    frame_count += 1
    now = time.time()
    
    # --- 1. Handle Sleep Mode Lifecycle ---
    if system_sleeping:
        time.sleep(2.0)
        cap = open_camera()
        if cap is None: continue
        
        # Peek at low resolution to save power
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, peek_frame = cap.read()
        
        if not ret or peek_frame is None:
            if cap: cap.release()
            cap = None
            continue
            
        gray = cv2.cvtColor(peek_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            if np.sum(cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]) > 15000:
                print("☀️ Motion detected! Webcam and AI Reopened.")
                system_sleeping = False
                last_motion_time = now
                # Restore full resolution for AI
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # We'll take a fresh full-res frame in the next step
                ret, frame = cap.read()
                if not ret: continue
            else:
                prev_gray = gray
                if cap: cap.release()
                cap = None
                continue
        else:
            prev_gray = gray
            if cap: cap.release()
            cap = None
            continue

    # --- 2. Normal Camera Operation ---
    if not system_sleeping:
        if cap is None or not cap.isOpened():
            cap = open_camera()
            if cap is None:
                time.sleep(1)
                continue

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Camera read failed. Retrying...")
            if cap: cap.release()
            cap = None
            continue
        
        # Motion detection to keep system awake
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            if np.sum(cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]) > 20000:
                last_motion_time = now
            prev_gray = cv2.addWeighted(prev_gray, 0.9, gray, 0.1, 0)
        else:
            prev_gray = gray
            continue

        # Check for idle timeout (Sleep Transition)
        # BUGFIX: Don't sleep if people are currently being tracked
        if now - last_motion_time > 12 and not person_state:
            print("💤 Low Power Mode: No motion detected. Turning off webcam...")
            system_sleeping = True
            cv2.destroyAllWindows()
            if cap: cap.release()
            cap = None
            continue

    last_frame = frame.copy() # Store for registration
    try:
        h, w = frame.shape[:2]
        # Make a copy to display
        display_frame = frame.copy()

        # Run YOLO tracking with higher confidence to reduce false positives
        results = model.track(frame, persist=True, conf=0.65, verbose=False)

        detected_ids = set()  # Track which people are in this frame
        
        if results[0].keypoints is not None and results[0].boxes.id is not None:
            # Keep system awake if people are detected
            last_motion_time = now
            for i, kp in enumerate(results[0].keypoints.xy):
                # 1. Get YOLO track ID
                yolo_id = int(results[0].boxes.id[i])
                yolo_id_str = str(yolo_id)
                
                # Bounding box for movement check
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                center_coords = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                # 2. Map YOLO ID to Persistent ID (ReID)
                if yolo_id_str not in tracker_to_persistent:
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        embedding = reid_manager.get_embedding(person_img)
                        with data_lock:
                            persistent_id = reid_manager.match_identity(embedding)
                        tracker_to_persistent[yolo_id_str] = persistent_id
                        
                        # Initialize movement tracking
                        person_start_pos[persistent_id] = center_coords
                        person_frames_seen[persistent_id] = 0
                        person_is_confirmed[persistent_id] = False
                    else:
                        continue
                
                persistent_id = tracker_to_persistent[yolo_id_str]
                
                # IMPORTANT: Always mark as detected so they aren't pruned while being confirmed
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                
                person_frames_seen[persistent_id] = person_frames_seen.get(persistent_id, 0) + 1

                # 3. Static Object Filtering (e.g., clothes on wall)
                # If a person hasn't moved at all in 60 frames (~2s), it's likely a static object
                is_static_object = False
                if not person_is_confirmed.get(persistent_id, False):
                    start_pos = person_start_pos.get(persistent_id, center_coords)
                    dist = np.sqrt((center_coords[0]-start_pos[0])**2 + (center_coords[1]-start_pos[1])**2)
                    
                    if dist > 30: # Moved 30 pixels? Confirmed human
                        person_is_confirmed[persistent_id] = True
                    elif person_frames_seen[persistent_id] > 60: 
                        is_static_object = True
                
                # Draw skeleton for ALL detections (including unconfirmed) so user sees tracking
                keypoints = kp.cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                
                # --- Draw Skeleton ---
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
                    (5, 11), (6, 12), (11, 12),              # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
                ]
                for start_idx, end_idx in connections:
                    if confidences[start_idx] > 0.5 and confidences[end_idx] > 0.5:
                        pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
                for j in range(5, 17):
                    if confidences[j] > 0.5:
                        cv2.circle(display_frame, (int(keypoints[j][0]), int(keypoints[j][1])), 4, (255, 0, 0), -1)

                if is_static_object:
                    continue
                
                # Stabilization delay for logic processing (still show skeleton above)
                if not person_is_confirmed.get(persistent_id, False) and person_frames_seen[persistent_id] < 10:
                    continue

                # 4. Resolve Display Name (Face/Manual Name > Persistent ID)
                pid = persistent_id # Fallback
                if persistent_id in manual_id_map:
                    pid = manual_id_map[persistent_id]
                
                # 4. Periodically try to "Name" the Persistent ID using Face Recognition
                if frame_count % 60 == 0 and pid == persistent_id: # Only if not already named
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        with data_lock:
                            target_encodings = known_face_encodings[:]
                            target_names = known_face_names[:]
                        
                        if target_encodings:
                            rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_person, number_of_times_to_upsample=1)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                                for fe in face_encodings:
                                    matches = face_recognition.compare_faces(target_encodings, fe, tolerance=0.6)
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        real_name = str(target_names[first_match_index])
                                        rename_person(persistent_id, real_name)
                                        pid = real_name
                                        break
                
                # Update tracking metadata
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                
                # 5. Classify Activity
                keypoints = kp.cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                now = time.time()
                
                # Calculate velocity (rolling average displacement)
                if persistent_id in person_last_pos:
                    last_pos = person_last_pos[persistent_id]
                    dist = np.sqrt((center_coords[0]-last_pos[0])**2 + (center_coords[1]-last_pos[1])**2)
                    v_dist = center_coords[1] - last_pos[1] # Positive is downward
                    person_velocity[persistent_id] = person_velocity[persistent_id] * 0.8 + dist * 0.2
                    person_vertical_velocity[persistent_id] = person_vertical_velocity[persistent_id] * 0.8 + v_dist * 0.2
                
                person_last_pos[persistent_id] = center_coords
                current_velocity = person_velocity[persistent_id]
                current_v_velocity = person_vertical_velocity[persistent_id]
                
                # Aspect Ratio of bounding box (width/height)
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                aspect_ratio = bw / bh if bh > 0 else 0
                
                activity = classify_activity(keypoints, confidences, velocity=current_velocity, v_velocity=current_v_velocity, aspect_ratio=aspect_ratio)

                # --- 6. Body Scanning (Capture multi-angle signatures) ---
                # During the first 10 seconds of seeing a person, periodically capture different angles
                with data_lock:
                    first_seen = reid_manager.identity_bank.get(persistent_id, {}).get('first_seen', 0)
                    gallery_len = len(reid_manager.identity_bank.get(persistent_id, {}).get('embeddings', []))
                
                if (now - first_seen) < 10 and frame_count % 15 == 0 and gallery_len < 10:
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        emb = reid_manager.get_embedding(person_img)
                        with data_lock:
                            reid_manager.add_to_gallery(persistent_id, emb)
                        
                        # Visual feedback for scanning
                        cv2.putText(display_frame, "Scanning Body...", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                with data_lock:
                    if persistent_id not in person_state:
                        person_state[persistent_id] = "UNKNOWN"
                        person_last_time[persistent_id] = now
                        all_tracked_people.add(pid) # Store resolved name for DB/Reporting
                        print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")

                # State Machine logic for ESCALATION and RECOVERY
                new_state = activity
                if activity == "UNKNOWN":
                    new_state = person_state.get(persistent_id, "UNKNOWN")
                
                # --- State Transition Refinements ---
                prev_s = person_state.get(persistent_id, "UNKNOWN")
                
                # If already in a confirmed MAJOR FALL, stay there until recovery
                if prev_s == "MAJOR FALL" and new_state in ["MINOR FALL", "LYING"]:
                    new_state = "MAJOR FALL"

                # SUPPRESS "getting up" misclassification:
                # If they are moving UP (negative v_velocity) and were previously down, 
                # they are likely getting up. Force upright state to prevent false Minor Fall.
                if current_v_velocity < -1.0 and prev_s in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]:
                    if new_state in ["MINOR FALL", "LYING"]:
                        new_state = "STANDING" 

                # 4. Special case: If in recovery mode, show RECOVERED label briefly
                if persistent_id in recovery_mode:
                    if now > recovery_mode[persistent_id]: 
                        del recovery_mode[persistent_id]
                    else:
                        # If they briefly tilt while getting up/stabilizing, don't trigger a new fall
                        if new_state in ["MINOR FALL", "LYING"]:
                            new_state = "STANDING" # Keep them upright during stabilization
                        
                        # Show RECOVERED label for 5 seconds (of the 10s recovery window)
                        if now < (recovery_mode[persistent_id] - 5.0):
                            new_state = "RECOVERED"

                # 1. Handle Risk States (Lying or Minor Fall)
                is_currently_down = (new_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"])
                
                # INITIAL FALL DETECTION
                if is_currently_down and persistent_id not in active_fall_event:
                    # Trigger alert immediately if they transitioned from WALKING/STANDING
                    # Exclude SITTING: Sitting -> Lying/Minor Fall is considered intentional/sleeping
                    if prev_s in ["WALKING", "STANDING", "UNKNOWN", "RECOVERED"]:
                        active_fall_event[persistent_id] = "MINOR"
                        send_fall_alert(f"MINOR FALL (ID {pid})", pid, "MINOR FALL", coords=center_coords)
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MINOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })

                if is_currently_down:
                    recovery_confirm_count[persistent_id] = 0 # Reset recovery counter
                    if persistent_id not in minor_fall_start_time:
                        # Start timer to track duration for MAJOR FALL escalation
                        # Only start if transitioning from an upright state
                        if prev_s in ["WALKING", "STANDING", "UNKNOWN", "RECOVERED"]:
                            minor_fall_start_time[persistent_id] = now
                    
                    # Escalation check: 10 seconds after being down (Only if it was a fall, not sleep)
                    # USER REQUEST: Only if minor fall AND lying down for 10s
                    if active_fall_event.get(persistent_id) == "MINOR" and new_state in ["LYING", "MAJOR FALL"] and (now - minor_fall_start_time.get(persistent_id, now) > 10.0):
                        send_fall_alert(f"MAJOR FALL (ID {pid}) - No recovery after 10s", pid, "MAJOR FALL", coords=center_coords)
                        active_fall_event[persistent_id] = "MAJOR"
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MAJOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })
                
                # 2. Handle Potential Recovery (Upright: WALKING, STANDING, SITTING)
                elif new_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"]:
                    # Require 30 frames (~1s at 30fps) of consistent upright pose before confirming recovery
                    recovery_confirm_count[persistent_id] = recovery_confirm_count.get(persistent_id, 0) + 1
                    
                    if recovery_confirm_count[persistent_id] > 30:
                        if persistent_id in active_fall_event:
                            recovery_mode[persistent_id] = now + 10.0 # 10s stabilization window
                            send_fall_alert(f"RECOVERED (ID {pid})", pid, "RECOVERED", coords=center_coords)
                            with data_lock:
                                fall_events.append({
                                    "person": pid, "type": "RECOVERED", "timestamp": now,
                                    "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                                })
                            del active_fall_event[persistent_id]
                        
                        # Always clear "down" timers if confirmed upright
                        if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                        if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                        recovery_confirm_count[persistent_id] = 0

                # 3. SLEEPING logic (sustained lying or sitting-to-lying)
                if new_state == "LYING":
                    if persistent_id not in lying_start_time: 
                        lying_start_time[persistent_id] = now
                    
                    # USER REQUEST: Sitting -> Lying is considered sleeping/intentional
                    if prev_s == "SITTING" or prev_s == "SLEEPING":
                        new_state = "SLEEPING"
                    elif now - lying_start_time[persistent_id] > 300: # 5 minutes default
                        new_state = "SLEEPING"
                elif new_state != "SLEEPING":
                    # Clear lying timer if they are not lying or already sleeping
                    if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                
                # Accumulate time for CURRENT activity (every frame)
                duration = now - person_last_time[persistent_id]
                if duration > 0:
                    with data_lock:
                        prev_state = person_state.get(persistent_id, "UNKNOWN")
                        if prev_state == "WALKING": walking_time[pid] += duration
                        elif prev_state == "STANDING": standing_time[pid] += duration
                        elif prev_state == "SITTING": sitting_time[pid] += duration
                        elif prev_state == "SLEEPING": sleeping_time[pid] += duration
                    
                person_last_time[persistent_id] = now

                # Update state if changed
                if new_state != "UNKNOWN" and new_state != prev_s:
                    with data_lock:
                        person_state[persistent_id] = new_state

                # Overlay text
                with data_lock:
                    walk_str = format_duration(walking_time[pid])
                    stand_str = format_duration(standing_time[pid])
                    sit_str = format_duration(sitting_time[pid])
                    sleep_str = format_duration(sleeping_time[pid])
                    
                    # Color code based on state
                    current_s = person_state.get(persistent_id, "UNKNOWN")
                    if "FALL" in current_s: color = (0, 0, 255) # Red for fall
                    elif current_s == "RECOVERED": color = (0, 255, 0) # Green for recovery
                    elif current_s in ["WALKING", "STANDING"]: color = (0, 255, 255) # Yellow for active
                    else: color = (255, 0, 0) # Blue for sitting/sleeping/unknown
                    
                    cv2.putText(display_frame, f"ID {pid}: {current_s}", (20, 60 + (i % 5) * 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_frame, f"W:{walk_str} St:{stand_str} S:{sit_str} Sl:{sleep_str}", (20, 90 + (i % 5) * 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No keypoints detected
            cv2.putText(display_frame, "No pose detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Clean up people not detected for 30 frames (about 1 second at 30fps)
        ids_to_remove = []
        for persistent_id in list(person_state.keys()):
            if persistent_id not in detected_ids and (frame_count - last_detection.get(persistent_id, 0)) > 30:
                ids_to_remove.append(persistent_id)
        
        with data_lock:
            for persistent_id in ids_to_remove:
                if persistent_id in person_state: del person_state[persistent_id]
                if persistent_id in person_last_time: del person_last_time[persistent_id]
                if persistent_id in last_detection: del last_detection[persistent_id]
                if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                if persistent_id in recovery_mode: del recovery_mode[persistent_id]
                if persistent_id in active_fall_event: del active_fall_event[persistent_id]
                if persistent_id in recovery_confirm_count: del recovery_confirm_count[persistent_id]
                if persistent_id in person_start_pos: del person_start_pos[persistent_id]
                if persistent_id in person_frames_seen: del person_frames_seen[persistent_id]
                if persistent_id in person_is_confirmed: del person_is_confirmed[persistent_id]
                if persistent_id in person_velocity: del person_velocity[persistent_id]
                if persistent_id in person_last_pos: del person_last_pos[persistent_id]
                
                # Clean up tracker mapping to prevent stale entries
                yolo_keys = [k for k, v in tracker_to_persistent.items() if v == persistent_id]
                for k in yolo_keys: del tracker_to_persistent[k]
                
                print(f"Removed internal ID {persistent_id} from active tracking")

        # Log progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count} | FPS: {fps:.1f} | People tracked: {len(person_state)}")

        # Show status message if active
        if time.time() < status_expiry:
            cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 255, 0), -1)
            cv2.putText(display_frame, status_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Always show camera feed
        cv2.imshow("Elderly Monitor", display_frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    except Exception as e:
        print(f"Error at frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Program exited.")
sys.exit(0)

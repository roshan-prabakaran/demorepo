import os
import json
import uuid
import traceback
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from sqlalchemy import create_engine
from PIL import Image
import joblib
import tensorflow as tf
import google.generativeai as genai

# ===========================
#  ENV + DATABASE CONFIG
# ===========================
load_dotenv()
db_url = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or "sqlite:///local.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
db = SQLAlchemy(app)

# ===========================
#  DATABASE MODELS
# ===========================
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    predicted_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_text = db.Column(db.Text)
    bot_reply = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# ===========================
#  LOAD MODEL + CLASSES
# ===========================
MODEL_DIR = "models"
cv_model_path = os.path.join(MODEL_DIR, "waste_classifier.h5")
classes_path = os.path.join(MODEL_DIR, "classes.json")

cv_model = None
idx2class = {0: "Organic", 1: "Recyclable"}
target_size = (160, 160)

if os.path.exists(cv_model_path):
    try:
        cv_model = tf.keras.models.load_model(cv_model_path)
        input_shape = cv_model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) == 4:
            target_size = (input_shape[1], input_shape[2])
        print(f"✅ Model loaded: {target_size}")
    except Exception as e:
        print("❌ Model load failed:", e)

if os.path.exists(classes_path):
    try:
        with open(classes_path, "r") as f:
            class_indices = json.load(f)
        if all(k.isdigit() for k in class_indices.keys()):
            idx2class = {int(k): v for k, v in class_indices.items()}
        else:
            idx2class = {int(v): k for k, v in class_indices.items()}
    except Exception:
        pass

print("🎯 Class mappings:", idx2class)

# ===========================
#  GEMINI CONFIG
# ===========================
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print("⚠️ Gemini config failed:", e)
else:
    print("⚠️ GEMINI_API_KEY not found")

# ===========================
#  HELPER FUNCTIONS
# ===========================
def preprocess_image(img_pil, size):
    img = img_pil.convert("RGB").resize(size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def validate_image(file_stream):
    try:
        img = Image.open(file_stream)
        img.verify()
        file_stream.seek(0)
        return True
    except Exception:
        return False

def normalize_class_label(raw_label):
    label = str(raw_label).lower()
    if "recycle" in label:
        return "Recyclable", "recyclable"
    elif "organic" in label:
        return "Organic", "organic"
    return "Unknown", "unknown"

def get_gemini_response(user_input):
    if not gemini_model:
        return "Gemini model not available. Please try again later."
    try:
        prompt = f"You are EcoMind, an AI recycling assistant. Question: {user_input}"
        res = gemini_model.generate_content(prompt)
        return res.text if res and res.text else "No response."
    except Exception as e:
        return f"Gemini error: {str(e)}"

# ===========================
#  ROUTES
# ===========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not validate_image(file.stream):
        return jsonify({"error": "Invalid image"}), 400

    os.makedirs("static/uploads", exist_ok=True)
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    fpath = os.path.join("static/uploads", fname)
    file.save(fpath)

    if not cv_model:
        return jsonify({"error": "Model not loaded"}), 500

    img = Image.open(fpath)
    x = preprocess_image(img, target_size)
    preds = cv_model.predict(x)
    idx = int(np.argmax(preds))
    raw_label = idx2class.get(idx, "Unknown")
    conf = float(np.max(preds) * 100)
    label, class_type = normalize_class_label(raw_label)

    pred = Prediction(filename=fname, predicted_class=class_type, confidence=conf)
    db.session.add(pred)
    db.session.commit()

    return jsonify({
        "prediction": label,
        "confidence": f"{conf:.2f}%",
        "image_url": f"/{fpath}"
    })

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_input = data.get("text", "")
    if not user_input:
        return jsonify({"reply": "Please ask something about recycling."})
    reply = get_gemini_response(user_input)
    log = ChatLog(user_text=user_input, bot_reply=reply)
    db.session.add(log)
    db.session.commit()
    return jsonify({"reply": reply})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found"}), 404

# ===========================
#  MAIN ENTRY POINT
# ===========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)

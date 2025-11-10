# app.py
import os
import json
import uuid
import traceback
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from PIL import Image
import joblib
import tensorflow as tf
import google.generativeai as genai

# Load env with better error handling
load_dotenv()

# Check if .env file exists and show what's loaded
env_path = '.env'
if os.path.exists(env_path):
    print(f"✅ .env file found at: {os.path.abspath(env_path)}")
    # Show what keys are loaded
    with open(env_path, 'r') as f:
        env_content = f.read()
        print("📋 .env content:")
        for line in env_content.split('\n'):
            if line.strip() and not line.startswith('#'):
                key = line.split('=')[0] if '=' in line else line
                print(f"   {key}")
else:
    print(f"❌ .env file NOT found at: {os.path.abspath(env_path)}")

db_url = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("🔗 Connecting to PostgreSQL Database...")
print(f"🔑 GEMINI_API_KEY found: {GEMINI_API_KEY is not None}")

# Quick DB test
try:
    if db_url:
        # Fix for PostgreSQL URL format if needed
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("✅ PostgreSQL connection successful!")
    else:
        print("⚠️ Using SQLite as fallback")
        db_url = 'sqlite:///ecomind_local.db'
except Exception as e:
    print("❌ Connection failed:", e)
    db_url = 'sqlite:///ecomind_local.db'

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
db = SQLAlchemy(app)


# DB models optimized for PostgreSQL
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    predicted_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class ChatLog(db.Model):
    __tablename__ = 'chat_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_text = db.Column(db.Text)
    bot_reply = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# Load model
MODEL_DIR = 'models'
cv_model_path = os.path.join(MODEL_DIR, 'waste_classifier.h5')
classes_path = os.path.join(MODEL_DIR, 'classes.json')

cv_model = None
idx2class = {}
target_size = (160, 160)  # Based on your model's expected input

if os.path.exists(cv_model_path):
    try:
        cv_model = tf.keras.models.load_model(cv_model_path)
        input_shape = cv_model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) == 4:
            target_size = (input_shape[1], input_shape[2])
        print("✅ Model loaded. Expected input:", target_size)
    except Exception as e:
        print("❌ Failed to load model:", e)
        print(traceback.format_exc())
else:
    print("❌ Model file not found at:", cv_model_path)

# Default class mappings - fallback if classes.json is corrupted
default_class_mappings = {
    0: 'Organic',
    1: 'Recyclable'
}

# Try to load classes.json, fallback to default if corrupted
if os.path.exists(classes_path):
    try:
        with open(classes_path, 'r') as f:
            class_indices = json.load(f)
            print("✅ Classes.json loaded successfully")

        # Handle different class mapping formats
        if all(k.isdigit() for k in map(str, class_indices.keys())):
            idx2class = {int(k): v for k, v in class_indices.items()}
        else:
            idx2class = {int(v): k for k, v in class_indices.items()}

        print("✅ Class mappings:", idx2class)
    except (json.JSONDecodeError, Exception) as e:
        print("❌ classes.json is corrupted or empty, using default mappings")
        print("Error details:", e)
        idx2class = default_class_mappings
else:
    print("❌ classes.json not found, using default mappings")
    idx2class = default_class_mappings

print("🎯 Final class mappings:", idx2class)

# Gemini API configuration - IMPROVED VERSION
# Replace your Gemini configuration with this:

# Gemini API configuration - USING AVAILABLE MODELS
gemini_model = None
if GEMINI_API_KEY:
    try:
        print(f"🔑 Gemini API Key found: {GEMINI_API_KEY[:10]}...")
        genai.configure(api_key=GEMINI_API_KEY)

        # Use available models that should work
        available_models_to_try = [
            'models/gemini-2.0-flash',  # Fast and available
            'models/gemini-2.0-flash-lite',  # Lightweight
            'models/gemini-flash-latest',  # Latest flash model
            'models/gemini-pro-latest',  # Latest pro model
            'models/gemini-2.0-flash-001',  # Specific version
            'models/gemma-3-4b-it',  # Gemma model as fallback
        ]

        for model_name in available_models_to_try:
            try:
                print(f"🧪 Testing: {model_name}")
                gemini_model = genai.GenerativeModel(model_name)
                # Use a very simple test to avoid quota issues
                test_response = gemini_model.generate_content("Say 'ok'")
                if test_response.text:
                    print(f"✅ Success with: {model_name}")
                    print(f"✅ Response: {test_response.text}")
                    break
                else:
                    gemini_model = None
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    print(f"⚠️ Quota exceeded for {model_name}, trying next...")
                    continue
                elif "not found" in error_msg.lower() or "404" in error_msg:
                    print(f"❌ Model not available: {model_name}")
                    continue
                else:
                    print(f"❌ {model_name} failed: {error_msg[:100]}...")
                gemini_model = None
                continue

        if gemini_model:
            print("✅ Gemini API configured successfully")
        else:
            print("❌ All models failed - likely quota exceeded")
            print("💡 Wait a few minutes or check your billing at: https://ai.google.dev/gemini-api/docs/rate-limits")

    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {str(e)}")
        gemini_model = None
else:
    print("❌ GEMINI_API_KEY not found in environment variables")
# Chatbot (optional fallback)
chat_pipeline = None
chat_le = None
intents = None
chat_pipeline_path = os.path.join(MODEL_DIR, 'chatbot_pipeline.joblib')
chat_label_path = os.path.join(MODEL_DIR, 'chatbot_label_encoder.joblib')
if os.path.exists(chat_pipeline_path) and os.path.exists(chat_label_path):
    try:
        chat_pipeline = joblib.load(chat_pipeline_path)
        chat_le = joblib.load(chat_label_path)
        with open('nlp/intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        print("✅ Fallback chatbot loaded")
    except Exception as e:
        print("⚠️ Failed to load fallback chatbot models:", e)


# Helper functions
def preprocess_image_pil(img_pil, target_size):
    img = img_pil.convert('RGB').resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_class_counts():
    """Get counts for each waste classification category"""
    try:
        counts = {}
        for class_name in ['organic', 'recyclable', 'unknown', 'error']:
            count = Prediction.query.filter(
                Prediction.predicted_class == class_name
            ).count()
            counts[class_name] = count

        # Also get total predictions
        counts['total'] = Prediction.query.count()

        # Get weekly trends
        week_ago = datetime.now() - timedelta(days=7)
        weekly_counts = {}
        for class_name in ['organic', 'recyclable']:
            count = Prediction.query.filter(
                Prediction.predicted_class == class_name,
                Prediction.created_at >= week_ago
            ).count()
            weekly_counts[class_name] = count

        counts['weekly_trends'] = weekly_counts

        return counts
    except Exception as e:
        print("Error getting class counts:", e)
        return {"organic": 0, "recyclable": 0, "unknown": 0, "error": 0, "total": 0, "weekly_trends": {}}


def get_recent_activity(limit=5):
    """Get recent classification activity"""
    try:
        recent = Prediction.query.filter(
            Prediction.predicted_class.in_(['organic', 'recyclable', 'unknown'])
        ).order_by(Prediction.created_at.desc()).limit(limit).all()

        activity = []
        for pred in recent:
            activity.append({
                'type': pred.predicted_class,
                'filename': pred.filename,
                'confidence': pred.confidence,
                'time': pred.created_at.strftime('%H:%M'),
                'date': pred.created_at.strftime('%Y-%m-%d')
            })
        return activity
    except Exception as e:
        print("Error getting recent activity:", e)
        return []


def validate_image(file_stream):
    """Validate uploaded image"""
    try:
        image = Image.open(file_stream)
        image.verify()
        file_stream.seek(0)
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def normalize_class_label(raw_label):
    """Normalize class labels to consistent format"""
    raw_label_str = str(raw_label).lower()

    if any(key in raw_label_str for key in ['r', 'recyclable']):
        return "Recyclable", "recyclable"
    elif any(key in raw_label_str for key in ['o', 'organic']):
        return "Organic", "organic"
    else:
        return "Unknown", "unknown"


def get_gemini_response(user_input):
    """Get response from Gemini API with waste management context"""
    try:
        prompt = f"""You are EcoMind, a friendly waste management assistant. Answer clearly and helpfully in 1-2 paragraphs.

Question: {user_input}

Provide practical advice about recycling, composting, or waste management:"""

        print(f"📤 Sending to Gemini: {user_input}")
        response = gemini_model.generate_content(prompt)

        if response.text:
            print(f"📥 Gemini response received ({len(response.text)} chars)")
            return response.text
        else:
            print("❌ Empty response from Gemini")
            return "I'm not sure how to help with that. Could you ask about recycling or waste management?"

    except Exception as e:
        print(f"❌ Gemini API error in get_gemini_response: {str(e)}")
        raise e


def get_ml_chatbot_response(text):
    """Get response from ML chatbot model"""
    if chat_pipeline and chat_le and intents:
        try:
            y = chat_pipeline.predict([text])
            tag_idx = int(y[0])
            tag = chat_le.inverse_transform([tag_idx])[0]
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    return np.random.choice(
                        intent.get('responses', ["I'm not sure about that. Could you rephrase your question?"]))
        except Exception as e:
            print("ML chatbot error:", e)

    return generate_fallback_response(text)


def generate_fallback_response(text):
    """Enhanced fallback responses when ML model is not available"""
    text_lower = text.lower()

    waste_responses = {
        'recycle': "Recyclable items include clean plastic bottles, glass containers, metal cans, cardboard, and paper. Always rinse containers and check local guidelines! ♻️",
        'organic': "Organic waste includes food scraps, yard trimmings, coffee grounds, and paper towels. These can be composted to create nutrient-rich soil for plants! 🌱",
        'plastic': "Most plastic bottles and containers (#1 & #2) are recyclable. Plastic bags and wrappers usually need special drop-off locations. Reduce plastic use when possible!",
        'glass': "Glass bottles and jars are highly recyclable! Rinse them and remove lids. Broken glass should be wrapped and placed in general waste for safety.",
        'metal': "Aluminum cans, tin cans, and clean foil are recyclable. Recycling metals saves 95% of the energy needed to make new ones!",
        'compost': "Great for the environment! Compost fruit/vegetable scraps, eggshells, coffee grounds, and yard waste. Avoid meat, dairy, and oily foods in home compost.",
        'battery': "Batteries are hazardous waste! Never put them in regular trash. Many stores have battery recycling bins. Check your local recycling center.",
        'electronic': "E-waste contains valuable materials but also toxins. Many electronics stores offer recycling programs. Never throw electronics in regular trash.",
        'pizza box': "Greasy pizza boxes go in compost/organic waste. If parts are clean, you can tear those off for recycling. When in doubt, compost the whole box!",
        'reduce': "Excellent question! Use reusable bags, bottles, and containers. Buy in bulk, repair items, and donate what you don't need. Every small change helps our planet! 🌍"
    }

    for keyword, response in waste_responses.items():
        if keyword in text_lower:
            return response

    # Default response for unrecognized questions
    return "I'm EcoMind, your waste management assistant! I can help with recycling guidelines, composting tips, waste classification, and sustainable practices. Try asking about specific items like 'Can I recycle plastic bottles?' or 'How to compost at home?'"


# Routes
@app.route('/')
def index():
    return render_template('index.html', prediction=None, confidence=None, image_url=None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate image
    is_valid, message = validate_image(file.stream)
    if not is_valid:
        return jsonify({'error': message}), 400

    file.stream.seek(0)  # Reset stream position

    os.makedirs('static/uploads', exist_ok=True)
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    fpath = os.path.join('static', 'uploads', fname)

    try:
        file.save(fpath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    if not cv_model:
        return jsonify({
            'prediction': 'Model not loaded',
            'confidence': '0.00%',
            'image_url': fpath
        })

    try:
        img = Image.open(fpath)
        x = preprocess_image_pil(img, target_size)
        preds = cv_model.predict(x, verbose=0)

        preds = np.asarray(preds, dtype=float)
        idx = int(np.argmax(preds, axis=1)[0])
        raw_label = idx2class.get(idx, "Unknown")
        confidence_value = float(np.max(preds) * 100)

        print(f"🔍 Prediction details - Index: {idx}, Raw label: {raw_label}, Confidence: {confidence_value:.2f}%")

        # Normalize the label output
        display_label, class_type = normalize_class_label(raw_label)

        # Confidence-based final label
        if confidence_value >= 70:
            final_label = display_label
        elif confidence_value >= 50:
            final_label = f"Likely {display_label}"
        else:
            final_label = "Unknown"
            class_type = "unknown"

        # Save to database
        p = Prediction(
            filename=fname,
            predicted_class=class_type,
            confidence=confidence_value
        )
        db.session.add(p)
        db.session.commit()

        return jsonify({
            'success': True,
            'prediction': final_label,
            'confidence': f"{confidence_value:.2f}%",
            'image_url': fpath,
            'class_type': class_type
        })

    except Exception as e:
        print("Prediction error:", e)
        print(traceback.format_exc())
        try:
            err_p = Prediction(filename=fname, predicted_class="error", confidence=0.0)
            db.session.add(err_p)
            db.session.commit()
        except Exception:
            pass
        return jsonify({
            'error': 'Error during prediction',
            'prediction': 'Error',
            'confidence': '0.00%',
            'image_url': fpath
        }), 500


@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


# FIXED: Removed duplicate route decorator
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    text = data.get('text', '').strip()

    print(f"=== CHAT REQUEST ===")
    print(f"Input: '{text}'")
    print(f"Gemini available: {gemini_model is not None}")

    if not text:
        return jsonify({'reply': 'Please type a question about waste management, recycling, or composting.'})

    reply = "I'm here to help with waste management questions!"

    if gemini_model:
        print("🚀 Using Gemini API...")
        try:
            reply = get_gemini_response(text)
            print(f"✅ Gemini reply length: {len(reply)}")
        except Exception as e:
            print(f"❌ Gemini failed: {str(e)}")
            print("🔄 Using fallback...")
            reply = get_ml_chatbot_response(text)
    else:
        print("⚠️ Using fallback chatbot")
        reply = get_ml_chatbot_response(text)

    print(f"📝 Final reply: {reply[:100]}...")
    print("=== END REQUEST ===\n")

    # Save chat log
    try:
        c = ChatLog(user_text=text, bot_reply=reply)
        db.session.add(c)
        db.session.commit()
    except Exception as e:
        print(f"💾 Failed to save chat log: {e}")

    return jsonify({'reply': reply})


@app.route('/dashboard')
def dashboard():
    try:
        counts = get_class_counts()
        recent_activity = get_recent_activity(5)

        # Calculate percentages for the chart
        total = counts.get('total', 1)  # Avoid division by zero
        organic_pct = round((counts.get('organic', 0) / total) * 100) if total > 0 else 0
        recyclable_pct = round((counts.get('recyclable', 0) / total) * 100) if total > 0 else 0
        unknown_pct = round((counts.get('unknown', 0) / total) * 100) if total > 0 else 0

        return render_template('dashboard.html',
                               counts=counts,
                               recent_activity=recent_activity,
                               organic_pct=organic_pct,
                               recyclable_pct=recyclable_pct,
                               unknown_pct=unknown_pct)
    except Exception as e:
        print("Dashboard error:", e)
        counts = {"organic": 0, "recyclable": 0, "unknown": 0, "total": 0, "weekly_trends": {}}
        return render_template('dashboard.html', counts=counts, recent_activity=[])


@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        counts = get_class_counts()
        recent_activity = get_recent_activity(5)

        return jsonify({
            'success': True,
            'counts': counts,
            'recent_activity': recent_activity
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for AJAX file uploads"""
    return predict()


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created/verified")
        except Exception as e:
            print("❌ Database initialization error:", e)

        print("🚀 Starting EcoMind Waste Classifier...")
        print("📊 Dashboard available at /dashboard")
        print("🤖 Chatbot available at /chatbot")
        print("📸 Classifier available at /")

        if gemini_model:
            print("✅ Gemini AI Chatbot: ACTIVE")
        else:
            print("❌ Gemini AI Chatbot: INACTIVE")
            print("💡 To activate Gemini:")
            print("   1. Go to: https://aistudio.google.com/app/apikey")
            print("   2. Create an API key")
            print("   3. Add to .env: GEMINI_API_KEY=your_key_here")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# ml/analytics.py
from app import db, Prediction

def get_class_counts():
    try:
        organic = db.session.query(Prediction).filter_by(predicted_class="organic").count()
        recyclable = db.session.query(Prediction).filter_by(predicted_class="recyclable").count()
        unknown = db.session.query(Prediction).filter_by(predicted_class="Unknown").count()
        return {"organic": organic, "recyclable": recyclable, "unknown": unknown}
    except Exception as e:
        print("analytics error:", e)
        return {"organic": 0, "recyclable": 0, "unknown": 0}

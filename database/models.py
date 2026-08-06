from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    caption = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    prediction_time = db.Column(db.Float, nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

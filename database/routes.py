from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for, make_response
import csv
from io import StringIO
from flask_login import login_required, current_user
from database.models import Prediction, db
from werkzeug.utils import secure_filename
from utils.predict import generate_caption
import os
import time

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date_created.desc()).limit(5).all()
    total_preds = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', predictions=preds, total=total_preds)

@main.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict
                start_time = time.time()
                caption, confidence = generate_caption(filepath)
                pred_time = time.time() - start_time
                
                # Save to DB
                new_pred = Prediction(
                    image_filename=filename,
                    caption=caption,
                    confidence=confidence,
                    prediction_time=pred_time,
                    user_id=current_user.id
                )
                db.session.add(new_pred)
                db.session.commit()
                
                return jsonify({'caption': caption, 'confidence': confidence, 'time': pred_time, 'filename': filename})
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG or WEBP.'}), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    return render_template('upload.html')

@main.route('/history')
@login_required
def history():
    preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date_created.desc()).all()
    return render_template('history.html', predictions=preds)

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@main.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return render_template('error/404.html'), 404
    total_users = User.query.count()
    total_preds = Prediction.query.count()
    return render_template('admin.html', total_users=total_users, total_preds=total_preds)

@main.route('/export/csv')
@login_required
def export_csv():
    preds = Prediction.query.filter_by(user_id=current_user.id).all()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Image Filename', 'Caption', 'Confidence (%)', 'Prediction Time (s)'])
    for p in preds:
        cw.writerow([p.date_created.strftime('%Y-%m-%d %H:%M'), p.image_filename, p.caption, f"{p.confidence*100:.1f}", f"{p.prediction_time:.2f}"])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=caption_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output

from flask import Flask, render_template
from config import Config
from database.models import db, User
from flask_login import LoginManager
import os

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register Blueprints
from database.routes import main as main_blueprint
app.register_blueprint(main_blueprint)
from database.auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint, url_prefix='/auth')

_models_loaded = False

@app.before_request
def startup():
    global _models_loaded
    db.create_all()
    if not _models_loaded:
        _models_loaded = True
        # Pre-load models in background thread so first upload doesn't hang
        import threading
        from utils.predict import load_models
        t = threading.Thread(target=load_models, daemon=True)
        t.start()

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error/500.html'), 500

if __name__ == '__main__':
    # Port 7860 is required for Hugging Face Spaces compatibility
    app.run(host='0.0.0.0', port=7860, debug=False)

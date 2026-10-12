import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session, send_file, url_for, flash
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.h5"
CLASSES_PATH = BASE_DIR / "model" / "classes.json"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
LOGS_PATH = BASE_DIR / "logs.json"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-only-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
    raise FileNotFoundError(
        "Model files are missing. Expected model/model.h5 and model/classes.json."
    )

model = tf.keras.models.load_model(MODEL_PATH)
with CLASSES_PATH.open("r", encoding="utf-8") as f:
    classes = json.load(f)

descriptions = {
    "agriculture": "Land used for farming and cultivation.",
    "forest": "Area covered with dense trees and vegetation.",
    "water": "Water bodies such as rivers, lakes, or ponds.",
    "barren": "Dry land with little or no vegetation.",
    "crop": "Land actively used for growing crops.",
}

def read_logs():
    if not LOGS_PATH.exists():
        return []
    try:
        with LOGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def write_log(entry):
    logs = read_logs()
    logs.append(entry)
    with LOGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def canonical_label(raw_label):
    key = raw_label.strip().lower()
    if key in {"agriculture", "crop"}:
        return "Agriculture", descriptions["agriculture"]
    if key == "forest":
        return "Forest", descriptions["forest"]
    if key == "water":
        return "Water", descriptions["water"]
    if key == "barren":
        return "Barren", descriptions["barren"]
    return raw_label.title(), descriptions.get(key, "Predicted land category.")

def allowed_image(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/architecture")
def architecture():
    return render_template("architecture.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        # Optional local demo admin. Configure credentials via environment variables.
        admin_user = os.environ.get("AGRIVISION_ADMIN_USER", "admin")
        admin_password = os.environ.get("AGRIVISION_ADMIN_PASSWORD")
        if admin_password and username == admin_user and password == admin_password:
            session["user"] = "admin"
            return redirect(url_for("admin"))
        # User registration/authentication is intentionally not enabled in this
        # cleaned portfolio build because the supplied project stored passwords
        # in plaintext. Use a proper database and password hashing before adding it.
        return render_template("login.html"), 401
    return render_template("login.html")

@app.route("/admin")
def admin():
    if session.get("user") != "admin":
        return redirect(url_for("login"))
    logs = read_logs()
    counts = {"Agriculture": 0, "Forest": 0, "Water": 0, "Barren": 0}
    invalid_count = 0
    for item in logs:
        label = item.get("prediction", "")
        if label in counts:
            counts[label] += 1
        else:
            invalid_count += 1
    return render_template(
        "admin.html",
        logs=logs[::-1],
        total_predictions=len(logs),
        total_users="—",
        last_upload=logs[-1].get("time", "No uploads") if logs else "No uploads",
        agriculture_count=counts["Agriculture"],
        forest_count=counts["Forest"],
        water_count=counts["Water"],
        barren_count=counts["Barren"],
        invalid_count=invalid_count,
    )

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("history.html", logs=read_logs()[::-1])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return "Please select an image.", 400
    if not allowed_image(uploaded.filename):
        return "Unsupported image type. Upload JPG, JPEG, PNG, or WEBP.", 400

    safe_name = secure_filename(uploaded.filename)
    unique_filename = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = UPLOAD_FOLDER / unique_filename

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        image.save(image_path, format={".png": "PNG", ".webp": "WEBP"}.get(image_path.suffix.lower(), "JPEG"))
        image = image.resize(IMG_SIZE)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]

        top_indices = np.argsort(probabilities)[-3:][::-1]
        top3 = [(classes[i], round(float(probabilities[i]) * 100, 2)) for i in top_indices]
        raw_label = classes[int(np.argmax(probabilities))]
        confidence = round(float(np.max(probabilities)) * 100, 2)
        prediction, description = canonical_label(raw_label)

        # Confidence is shown as the model's score; it is not an OOD detector.
        write_log({
            "filename": unique_filename,
            "prediction": prediction,
            "confidence": confidence,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        session["prediction"] = prediction
        session["confidence"] = confidence
        session["description"] = description

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            description=description,
            image_file=unique_filename,
            top3=top3,
        )
    except (OSError, ValueError) as exc:
        image_path.unlink(missing_ok=True)
        return f"Could not process the uploaded image: {exc}", 400

@app.route("/download_report")
def download_report():
    if "prediction" not in session:
        return redirect(url_for("home"))
    report_path = BASE_DIR / "prediction_report.pdf"
    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    pdf.setTitle("Agrivision AI Prediction Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 740, "Agrivision AI - Prediction Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 700, f"Prediction: {session.get('prediction', 'N/A')}")
    pdf.drawString(72, 675, f"Model score: {session.get('confidence', 'N/A')}%")
    pdf.drawString(72, 650, f"Description: {session.get('description', 'N/A')[:90]}")
    pdf.drawString(72, 625, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    pdf.save()
    return send_file(str(report_path), as_attachment=True, download_name="agrivision_prediction_report.pdf")

@app.errorhandler(413)
def upload_too_large(_error):
    return "Image is too large. Maximum upload size is 10 MB.", 413

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")

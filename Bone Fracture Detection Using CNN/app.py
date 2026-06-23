from flask import Flask, render_template, request
import os
import sys

# Ensure local project modules can be imported when running app.py directly.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from werkzeug.utils import secure_filename
from model.predict import predict_fracture, generate_pdf, get_model_performance # ✅ Import both from predict.py

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

LOGO_PATH = os.path.join("static", "images", "project_logo.png")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # 🔐 Secure filename
    file_name = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)

    # 🔥 Prediction
    result_data = predict_fracture(file_path)

    # -----------------------------
    # 🔥 FIX: Grad-CAM path handling
    # -----------------------------
    gradcam_path = result_data.get("gradcam")

    if gradcam_path and os.path.exists(gradcam_path):
        # Convert to relative path for HTML
        gradcam_path = gradcam_path.replace("\\", "/")
    else:
        gradcam_path = None

    # -----------------------------
    # 📄 Generate PDF
    # -----------------------------
    report_filename = file_name.rsplit(".", 1)[0] + "_report.pdf"
    report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)

    generate_pdf(
        report_path=report_path,
        original_img_path=file_path,
        gradcam_img_path=gradcam_path,
        result=result_data.get("result"),
        confidence=result_data.get("confidence"),
        severity=result_data.get("severity"),
        logo_path=LOGO_PATH
    )
    # -----------------------------
    # 📊 Get Model Performance Data
    # -----------------------------
    performance = get_model_performance()

    # -----------------------------
    # 🔥 PASS EVERYTHING TO UI
    # -----------------------------
    return render_template(
      "result.html",
       result=result_data.get("result"),
       confidence=result_data.get("confidence"),
       image_path=file_path.replace("\\", "/"),
       gradcam_path=gradcam_path,
       severity=result_data.get("severity"),
       recommendation=result_data.get("recommendation"),  # 🧾 NEW
       pdf_report=report_filename,

       # 📊 Chart Data
       epochs=performance["epochs"],
       accuracy=performance["accuracy"],
       val_accuracy=performance["val_accuracy"]
    )


if __name__ == "__main__":
    app.run(debug=True)
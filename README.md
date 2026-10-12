# Agrivision AI — Agriculture Land Classification

Agrivision AI is a Flask web application that uses a TensorFlow/Keras image-classification model to classify uploaded images into five dataset categories: Agriculture, Forest, barren, crop, and water. The app displays the leading prediction, its model score, and the top-three class scores.

## Application Preview

![Agrivision AI prediction result](Screenshot%202026-10-12%20192329.png)

## Features

- Image upload and land-category prediction
- Top-three model scores
- Class descriptions
- Local prediction history
- Admin dashboard (admin login must be configured through environment variables)
- Downloadable PDF prediction report
- Project information and architecture pages

## Technology stack

- Python
- Flask
- TensorFlow / Keras
- MobileNetV2-based transfer-learning model (as implemented in the training script)
- NumPy and Pillow
- ReportLab
- HTML, CSS, Bootstrap

## Project structure

```text
Agrivision_AI/
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
├── README.md
├── model/
│   ├── model.h5
│   └── classes.json
├── templates/
│   ├── about.html
│   ├── admin.html
│   ├── architecture.html
│   ├── history.html
│   ├── index.html
│   ├── login.html
│   └── result.html
└── static/
    └── uploads/  # created/used locally; uploaded files are ignored by Git
```

## Run locally

Use Python compatible with your installed TensorFlow version.

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set a secret key and optional admin credentials.

   PowerShell:

   ```powershell
   $env:FLASK_SECRET_KEY = "replace-with-a-long-random-secret"
   $env:AGRIVISION_ADMIN_USER = "admin"
   $env:AGRIVISION_ADMIN_PASSWORD = "choose-a-strong-password"
   ```

4. Start the app:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000`.

The admin dashboard is disabled unless `AGRIVISION_ADMIN_PASSWORD` is configured. The login page intentionally does not support user signup in this cleaned portfolio build because the supplied project stored passwords in plaintext. A production authentication system should use password hashing, a database, CSRF protection, and proper session management.

## Model notes and limitations

- The saved model expects 224 × 224 RGB images and this application scales pixel values to `[0, 1]`, matching the supplied training script.
- Displayed class scores are model outputs, not a guarantee of calibrated confidence or real-world accuracy.
- The project has not been independently benchmarked as part of this repository preparation. Do not claim a specific accuracy until it is measured on a clearly separated test set.
- The current image-classification approach is not a reliable out-of-distribution detector. Use suitable land images and treat predictions on unrelated images as unreliable.
- The training script requires the original dataset under `dataset/` with one subfolder per class. The dataset is intentionally not included in this repository.
- The app stores prediction metadata in local `logs.json` and uploads images in `static/uploads/`; those local data files are excluded from Git.

## Security and deployment

This is a portfolio/local-demo project, not a production-hardened service. Before public deployment, use secure configuration, production WSGI hosting, request-rate limits, robust file validation, proper authentication, and privacy-conscious retention of uploaded images and logs.

## License

Add a license if you want others to reuse or modify this project.

from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "laptop_price_model.joblib"
app = Flask(__name__)

def load_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_FILE)

@app.route("/", methods=["GET", "POST"])
def index():
    bundle = load_bundle()
    prediction = None
    error = None
    form = {"brand": "", "ram": "8", "processor": "", "storage": "512"}
    if request.method == "POST":
        try:
            form = {
                "brand": request.form.get("brand", "").strip(),
                "ram": request.form.get("ram", "").strip(),
                "processor": request.form.get("processor", "").strip(),
                "storage": request.form.get("storage", "").strip()
            }
            row = pd.DataFrame([{
                "Brand": form["brand"],
                "RAM_GB": float(form["ram"]),
                "Processor_Family": form["processor"],
                "Storage_GB": float(form["storage"])
            }])
            if not form["brand"] or not form["processor"]:
                raise ValueError("Please select a brand and processor family.")
            if row.loc[0, "RAM_GB"] <= 0 or row.loc[0, "Storage_GB"] < 0:
                raise ValueError("RAM must be positive and storage cannot be negative.")
            prediction = float(bundle["model"].predict(row)[0])
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html",
        brands=bundle["brand_options"],
        processors=bundle["processor_options"],
        prediction=prediction,
        error=error,
        form=form,
        metrics=bundle.get("metrics", {})
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

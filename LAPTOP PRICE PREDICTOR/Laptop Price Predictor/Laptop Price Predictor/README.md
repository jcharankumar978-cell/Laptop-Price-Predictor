# Laptop Price Predictor

A Machine Learning web application that predicts an estimated laptop price from Brand, RAM, Processor, and Storage.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Linear Regression
- Flask
- HTML/CSS

## Run locally

1. Open a terminal in this folder.
2. Install packages:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Start Flask:

```bash
python app.py
```

5. Open the address shown by Flask, normally:
http://127.0.0.1:5000/

## Important
The included CSV is a project/demo dataset generated for this application. For a real-world submission, you can replace it with a public laptop-price dataset and retrain the model.

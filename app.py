import numpy as np
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load Model/Scaler
def load_model():
    # Check if we need to merge chunks
    if not os.path.exists('model.pkl') and os.path.exists('model.pkl.part0'):
        print("Merging model chunks...")
        with open('model.pkl', 'wb') as outfile:
            chunk = 0
            while True:
                chunk_filename = f"model.pkl.part{chunk}"
                if os.path.exists(chunk_filename):
                    with open(chunk_filename, 'rb') as infile:
                        outfile.write(infile.read())
                    chunk += 1
                else:
                    break
        print("Model merged successfully.")

    try:
        with open('model.pkl', 'rb') as m:
            model = pickle.load(m)
        with open('scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)
        return model, scaler
    except FileNotFoundError:
        print("Warning: model.pkl or scaler.pkl not found. Predictions will fail.")
        return None, None

model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about-model')
def about_model():
    return render_template('about-model.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract raw form data
            age_yrs = float(request.form['age_yrs'])
            gender = int(request.form['gender'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            ap_hi = float(request.form['ap_hi'])
            ap_lo = float(request.form['ap_lo'])
            chol = int(request.form['chol'])
            gluc = int(request.form['gluc'])
            smoke = int(request.form['smoke'])
            alco = int(request.form['alco'])
            active = int(request.form['active'])

            # Calculate BMI: weight (kg) / (height (m) ^ 2)
            # Height is in cm, convert to meters
            bmi = weight / ((height / 100) ** 2)

            # Feature alignment: gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, bmi
            input_data = [
                gender,
                ap_hi,
                ap_lo,
                chol,
                gluc,
                smoke,
                alco,
                active,
                age_yrs,
                bmi
            ]

            final_features = np.array([input_data])
            
            # Scale features
            if scaler:
                final_features = scaler.transform(final_features)
            
            # Predict
            if model:
                prediction = model.predict(final_features)
                # Get probability
                prob = model.predict_proba(final_features)[0][1] * 100
                
                result_text = "High Risk Detected" if prediction[0] == 1 else "Low Risk Detected"
                result_class = "danger" if prediction[0] == 1 else "success" # for css styling
                
                return render_template('result.html', 
                                     result=result_text, 
                                     probability=f"{prob:.1f}%",
                                     result_class=result_class,
                                     bmi=f"{bmi:.2f}")
            else:
                return render_template('result.html', result="Model not loaded", probability="N/A", result_class="warning")

        except Exception as e:
            return render_template('result.html', result=f"Error: {str(e)}", probability="N/A", result_class="danger")

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
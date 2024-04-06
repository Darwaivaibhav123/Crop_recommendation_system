from flask import Flask, request, render_template
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create Flask app
app = Flask(__name__)

# Load crop data
crop = pd.read_csv(r"C:\Users\DELL\Desktop\vai\Crop_recommendation.csv")

# Dummy training data (you should replace this with your actual training data)
X_train = np.random.rand(100, 7)  # Assuming you have 100 samples with 7 features
y_train = np.random.randint(1, 23, size=100)  # Assuming 100 labels

# Instantiate model and scaler
model = RandomForestClassifier()
scaler = StandardScaler()

# Train the model
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)

# Define routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Extract features from form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Transform features into numpy array
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Transform features using scaler
    single_pred_scaled = scaler.transform(single_pred)

    # Make prediction using model
    prediction = model.predict(single_pred_scaled)

    # Map prediction to crop name
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Prepare result message
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

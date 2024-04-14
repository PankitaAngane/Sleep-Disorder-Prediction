from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib



def final_prediction(model,scaler,data):
    c = list(data.values())
    e = np.array(c, dtype=int)
    w = scaler.transform([e])
    res = model.predict(w)
    label = ["Insomnia","None","Sleep Apnea",]
    return label[res[0]]

scaler_model = joblib.load("Models/SCALER.pkl")
model = joblib.load("Models/RFC.pkl")




app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'

@app.route('/prediction', methods=['POST'])
def predict_flower():
    # RECIEVE THE REQUEST
    content = request.json

    # PRINT THE DATA PRESENT IN THE REQUEST
    print("[INFO] Request: ", content)

    # PREDICT THE CLASS USING HELPER FUNCTION
    results = final_prediction(model=model,
                                scaler=scaler_model,
                                data=content)

    # PRINT THE RESULT
    print("[INFO] Responce: ", results)

    # SEND THE RESULT AS JSON OBJECT
    return jsonify(results)


if __name__ == '__main__':
    app.run("0.0.0.0")
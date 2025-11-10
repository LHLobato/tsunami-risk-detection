from flask import Flask, jsonify, request
import joblib
import numpy as np
import os
import urllib.parse
from global_land_mask import globe

app = Flask(__name__)

try:
    print("Initializing Tsunami Detection API..")
    print("Loading Machine Learning models..")
    model = joblib.load("../model/best_model.joblib")
    scaler = joblib.load("../model/scaler.joblib")
    print("Success, models loaded!")

except FileNotFoundError as e:
    model = scaler = None

except Exception as e:
    model = scaler = None 
    
@app.route('/classify/tsunami', methods=['POST'])
def predict():
    
    if not all([model, scaler]):
        return jsonify({"error": "Models are not available on the server."}), 503
    
    input_data = request.get_json()
    
    if not input_data:
        return jsonify({"error": "Invalid Request"}), 400
    
    try:
        
        
        # 1. Definir ONDE encontrar cada feature
        ROOT_FEATURES = [
            'profundidade', 'longitude', 'latitude'
        ]
        
        PROPERTY_FEATURES = [
            'mag', 'sig',  
            'magType_mb', 'magType_mb_lg', 'magType_md',
            'magType_mh', 'magType_ml', 'magType_mlv', 
            'magType_ms_vx', 'magType_mw', 'magType_mwb',
            'magType_mwr', 'magType_mww', 'type_earthquake', 
            'type_explosion', 'type_ice quake', 'type_landslide',
            'type_mine collapse', 'type_mining explosion',
            'type_other event', 'type_quarry blast',
            'type_volcanic eruption'
        ]

        root_values = {f: input_data[f] for f in ROOT_FEATURES}
        lat = root_values['latitude']
        lon = root_values['longitude']
        

        is_land_feature = 1 if globe.is_land(lat, lon) else 0
        safe_depth = root_values['profundidade'] + 1
       
        if is_land_feature == 1 or safe_depth - 1 > 100:
            response = {
            "predicition_class": 0, 
            "is_tsunami_risk": 0, 
            "probability_no_tsunami": 1.0, 
            "probability_tsunami_risk": 0}
            return jsonify(response), 200
        

# Outra feature: Risco em terra


        properties_data = input_data.get('properties', {})
        property_values = {f: properties_data[f] for f in PROPERTY_FEATURES}
        risco_mag_prof = properties_data['mag'] / safe_depth
        risco_terra = properties_data['mag'] * (1 - is_land_feature)
        
        feature_values = [
            property_values['mag'],
            property_values['sig'],
            root_values['profundidade'],
            root_values['longitude'],
            root_values['latitude'],
            is_land_feature,
            property_values['magType_mb'],
            property_values['magType_mb_lg'],
            property_values['magType_md'],
            property_values['magType_mh'],
            property_values['magType_ml'],
            property_values['magType_mlv'],
            property_values['magType_ms_vx'],
            property_values['magType_mw'],
            property_values['magType_mwb'],
            property_values['magType_mwr'],
            property_values['magType_mww'],
            property_values['type_earthquake'],
            property_values['type_explosion'],
            property_values['type_ice quake'],
            property_values['type_landslide'],
            property_values['type_mine collapse'],
            property_values['type_mining explosion'],
            property_values['type_other event'],
            property_values['type_quarry blast'],
            property_values['type_volcanic eruption'],
            safe_depth, risco_mag_prof, risco_terra
        ]
        
        feature_values = np.array([feature_values])
        data_scaled = scaler.transform(feature_values)
        
        prediction = model.predict(data_scaled)
        probability = model.predict_proba(data_scaled)
        
        response = {
            "predicition_class": int(prediction[0]), 
            "is_tsunami_risk": bool(prediction[0] == 1), 
            "probability_no_tsunami": float(probability[0][0]), 
            "probability_tsunami_risk": float(probability[0][1])
        }
        return jsonify(response), 200
    
    except KeyError as e:
        return jsonify({"error": f"Missing feature in request: {str(e)}"}), 400
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)
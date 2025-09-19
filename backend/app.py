from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import requests

app = Flask(__name__)

# Load trained ML pipeline
pipeline = joblib.load("crop_yield_pipeline.joblib")

# Load merged dataset to fetch unique dropdown values
df = pd.read_csv("crop_data.csv")

# Weather API config
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your key
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Unique values for dropdowns
def get_unique_values():
    return {
        "Crops": sorted(df['Crop'].dropna().unique().tolist()),
        "States": sorted(df['State Name'].dropna().unique().tolist()),
        "Districts": sorted(df['Dist Name'].dropna().unique().tolist())
    }

# Season mapping
KHARIF = {'rice','maize','cotton','soybean','groundnut','bajra','jowar','sorghum'}
RABI = {'wheat','barley','gram','mustard','linseed','pea','rapeseed'}
ZAID = {'watermelon','muskmelon','cucumber','vegetables','fodder'}

def get_season(crop):
    if not crop: return 'Other'
    s = crop.strip().lower()
    if s in KHARIF: return 'Kharif'
    if s in RABI: return 'Rabi'
    if s in ZAID: return 'Zaid'
    return 'Other'

# Get weather data for a district/state
def get_weather(state, district):
    try:
        q = f"{district},{state},IN"  # Query format for India
        params = {"q": q, "appid": WEATHER_API_KEY, "units": "metric"}
        resp = requests.get(WEATHER_API_URL, params=params, timeout=5)
        data = resp.json()
        return {
            "Temperature_C": data['main']['temp'],
            "Humidity_%": data['main']['humidity'],
            "Wind_Speed_m_s": data['wind']['speed'],
            "Rainfall_mm": data.get('rain', {}).get('1h', 5),  # fallback to 5mm if missing
            "Solar_Radiation_MJ_m2_day": 18  # default static
        }
    except Exception as e:
        # fallback defaults if API fails
        return {
            "Temperature_C": 28,
            "Humidity_%": 65,
            "Wind_Speed_m_s": 2,
            "Rainfall_mm": 800,
            "Solar_Radiation_MJ_m2_day": 18
        }

# Prepare input row (fill defaults)
def prepare_input(user_input):
    row = user_input.copy()
    
    # Required categorical features
    row['Crop'] = row.get('Crop','Other')
    row['State Name'] = row.get('State Name','missing')
    row['Dist Name'] = row.get('Dist Name','missing')
    row['Season'] = get_season(row['Crop'])
    
    # Default numeric features if not provided
    defaults = {
        'Area_ha':1,
        'N_req_kg_per_ha':50, 'P_req_kg_per_ha':25, 'K_req_kg_per_ha':20,
        'applied_N_per_ha':30, 'applied_P_per_ha':15, 'applied_K_per_ha':10,
        'pH':6.5,
        'Year':2025
    }
    for k,v in defaults.items():
        if k not in row or row[k] is None:
            row[k] = v
    
    # Fetch weather data dynamically
    weather = get_weather(row['State Name'], row['Dist Name'])
    row.update(weather)
    
    # Derived features
    for nutrient in ['N','P','K']:
        row[f'{nutrient}_deficit_kg_per_ha'] = row[f'applied_{nutrient}_per_ha'] - row[f'{nutrient}_req_kg_per_ha']
        row[f'{nutrient}_frac_of_req'] = row[f'applied_{nutrient}_per_ha'] / (row[f'{nutrient}_req_kg_per_ha']+1e-9)
    
    return row

# Apply agronomic suggestions
def apply_suggestions(row):
    improved = row.copy()
    for nutrient in ['N','P','K']:
        improved[f'applied_{nutrient}_per_ha'] = max(improved[f'applied_{nutrient}_per_ha'], improved[f'{nutrient}_req_kg_per_ha'])
    improved['pH'] = min(max(row.get('pH',6.5),6.5),7.5)
    improved['Rainfall_mm'] = max(row.get('Rainfall_mm',400),400)
    improved['Temperature_C'] = min(max(row.get('Temperature_C',20),20),30)
    return improved

# Recommendation generator
def generate_recommendations(row):
    recs = []
    for nutrient in ['N','P','K']:
        if row[f'{nutrient}_req_kg_per_ha'] > row[f'applied_{nutrient}_per_ha']*1.05:
            recs.append(f"{nutrient} deficit → Add fertilizer")
        elif row[f'applied_{nutrient}_per_ha'] > row[f'{nutrient}_req_kg_per_ha']*1.2:
            recs.append(f"Excess {nutrient} → Reduce fertilizer")
    if row['pH'] < 5.5: recs.append("Soil acidic → Apply lime")
    if row['pH'] > 8.0: recs.append("Soil alkaline → Apply gypsum/organic amendments")
    improved_row = apply_suggestions(row)
    base_yield = pipeline.predict(pd.DataFrame([row]))[0]
    improved_yield = pipeline.predict(pd.DataFrame([improved_row]))[0]
    recs.append(f"Predicted yield: {base_yield:.2f} kg/ha")
    recs.append(f"Predicted yield with suggestions: {improved_yield:.2f} kg/ha (+{((improved_yield-base_yield)/base_yield*100):.1f}%)")
    return recs

# Endpoint: Predict yield & recommendations
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error":"No input data provided"}),400
    full_input = prepare_input(data)
    df_input = pd.DataFrame([full_input])
    predicted_yield = pipeline.predict(df_input)[0]
    recommendations = generate_recommendations(full_input)
    return jsonify({
        "predicted_yield_kg_per_ha": round(predicted_yield,2),
        "recommendations": recommendations
    })

# Endpoint: Get unique dropdown values
@app.route('/dropdown-values', methods=['GET'])
def dropdown_values():
    return jsonify(get_unique_values())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
 
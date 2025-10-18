# 🌴 Kerala Smart Weather Predictor (District + Future Date Only)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import datetime
import math

st.set_page_config(page_title="Kerala Smart Weather Predictor", page_icon="🌦️")

st.title("🌴 Kerala Weather Prediction System")
st.markdown("""
### Enter only the **District** and **Date**, and get the predicted weather 🌤️  
Model trained on 20,000+ Kerala weather samples across all 14 districts.
""")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("kerala_district_weather_20000.csv")

data = load_data()
st.success(f"✅ Loaded {len(data)} Kerala weather records successfully!")

# -------------------- ENCODE --------------------
le_weather = LabelEncoder()
le_monsoon = LabelEncoder()
le_district = LabelEncoder()

data["Weather"] = le_weather.fit_transform(data["Weather"])
data["MonsoonPhase"] = le_monsoon.fit_transform(data["MonsoonPhase"])
data["District"] = le_district.fit_transform(data["District"])

# -------------------- TRAIN MODEL --------------------
X = data.drop("Weather", axis=1)
y = data["Weather"]

model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X, y)

# -------------------- HELPER FUNCTIONS --------------------
def get_monsoon_phase(day):
    if 150 <= day <= 270:
        return "SouthWest Monsoon"
    elif 271 <= day <= 330:
        return "NorthEast Monsoon"
    elif 90 <= day < 150:
        return "Pre-Monsoon"
    else:
        return "Post-Monsoon"

def generate_district_weather(district, day):
    """Simulate realistic feature values for given district & day"""
    # district-based baseline climate
    coastal = ["Thiruvananthapuram", "Kollam", "Alappuzha", "Ernakulam", "Kozhikode", "Kannur", "Kasaragod"]
    highland = ["Idukki", "Wayanad"]
    midland = ["Kottayam", "Pathanamthitta", "Thrissur", "Palakkad", "Malappuram"]

    monsoon = get_monsoon_phase(day)
    base_temp = 29
    base_hum = 80
    base_precip = 5
    base_cloud = 60
    base_wind = 10

    if district in coastal:
        base_temp -= 1
        base_hum += 5
        base_wind += 2
    elif district in highland:
        base_temp -= 3
        base_hum -= 2
        base_precip += 5
    elif district in midland:
        base_temp += 0.5
        base_hum += 1

    # Adjust for monsoon phase
    if monsoon == "SouthWest Monsoon":
        base_precip += 10
        base_cloud += 15
        base_hum += 5
    elif monsoon == "NorthEast Monsoon":
        base_precip += 6
        base_cloud += 10
    elif monsoon == "Pre-Monsoon":
        base_temp += 2
        base_hum -= 5
        base_precip -= 3
    else:  # Post-Monsoon
        base_precip -= 2
        base_temp -= 1

    # Add small randomness for realism
    temperature = np.random.normal(base_temp, 1.5)
    humidity = np.random.normal(base_hum, 5)
    pressure = np.random.normal(1010, 5)
    wind_speed = np.random.normal(base_wind, 2)
    precipitation = max(0, np.random.normal(base_precip, 4))
    cloud_cover = np.clip(np.random.normal(base_cloud, 10), 0, 100)

    return {
        "Temperature": round(temperature, 2),
        "Humidity": round(humidity, 2),
        "Pressure": round(pressure, 2),
        "WindSpeed": round(wind_speed, 2),
        "Precipitation": round(precipitation, 2),
        "CloudCover": round(cloud_cover, 2),
        "MonsoonPhase": monsoon
    }

# -------------------- USER INPUT --------------------
st.subheader("🌏 Enter Prediction Details")
districts = le_district.classes_
selected_district = st.selectbox("🏙️ Select District", districts)
future_date = st.date_input("📅 Select Date", datetime.date.today())

# -------------------- PREDICT --------------------
if st.button("🔮 Predict Weather"):
    day_of_year = future_date.timetuple().tm_yday
    weather_data = generate_district_weather(selected_district, day_of_year)
    monsoon_value = le_monsoon.transform([weather_data["MonsoonPhase"]])[0]
    district_value = le_district.transform([selected_district])[0]

    new_data = pd.DataFrame({
        "District": [district_value],
        "Temperature": [weather_data["Temperature"]],
        "Humidity": [weather_data["Humidity"]],
        "Pressure": [weather_data["Pressure"]],
        "WindSpeed": [weather_data["WindSpeed"]],
        "Precipitation": [weather_data["Precipitation"]],
        "CloudCover": [weather_data["CloudCover"]],
        "DayOfYear": [day_of_year],
        "MonsoonPhase": [monsoon_value]
    })

    prediction = model.predict(new_data)
    predicted_label = le_weather.inverse_transform(prediction)[0]

    st.success(f"🌆 **District:** {selected_district}")
    st.info(f"📅 **Date:** {future_date.strftime('%d %B %Y')} (Day {day_of_year})")
    st.write(f"🌀 **Monsoon Phase:** {weather_data['MonsoonPhase']}")
    st.write("🌡️ **Generated Features:**")
    st.json(weather_data)
    st.subheader(f"🌈 **Predicted Weather:** {predicted_label}")

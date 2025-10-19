import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Kerala Smart Weather Predictor",
    page_icon="🌦️",
    layout="wide"
)

# -------------------- CUSTOM CSS (Glassmorphism + Kerala Theme) --------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://opendigest.in/wp-content/uploads/2025/01/Weather-App-.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
}

h1, h2, h3 {
    color: #064663;
    text-shadow: 1px 1px 2px #ffffff;
}

.stButton button {
    background-color: #0E7490;
    color: white;
    border-radius: 12px;
    font-weight: bold;
    padding: 0.6rem 1.2rem;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #0C4A6E;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<h1>🌴 Kerala Smart Weather Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Predict Kerala’s weather by selecting a district & future date 🌤️</h3>", unsafe_allow_html=True)

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
    coastal = ["Thiruvananthapuram", "Kollam", "Alappuzha", "Ernakulam", "Kozhikode", "Kannur", "Kasaragod"]
    highland = ["Idukki", "Wayanad"]
    midland = ["Kottayam", "Pathanamthitta", "Thrissur", "Palakkad", "Malappuram"]

    monsoon = get_monsoon_phase(day)
    base_temp, base_hum, base_precip, base_cloud, base_wind = 29, 80, 5, 60, 10

    if district in coastal:
        base_temp -= 1; base_hum += 5; base_wind += 2
    elif district in highland:
        base_temp -= 3; base_hum -= 2; base_precip += 5
    elif district in midland:
        base_temp += 0.5; base_hum += 1

    if monsoon == "SouthWest Monsoon":
        base_precip += 10; base_cloud += 15; base_hum += 5
    elif monsoon == "NorthEast Monsoon":
        base_precip += 6; base_cloud += 10
    elif monsoon == "Pre-Monsoon":
        base_temp += 2; base_hum -= 5; base_precip -= 3
    else:
        base_precip -= 2; base_temp -= 1

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
st.markdown("<h2 style='color:white;'>🌏 Enter Prediction Details</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
districts = le_district.classes_

with col1:
    selected_district = st.selectbox("🏙️ Select District", districts)
with col2:
    future_date = st.date_input("📅 Select Date", datetime.date.today())

# -------------------- PREDICTION --------------------
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

    st.markdown("<hr>", unsafe_allow_html=True)
    st.success(f"🌆 **District:** {selected_district}")
    st.info(f"📅 **Date:** {future_date.strftime('%d %B %Y')} (Day {day_of_year})")
    st.write(f"🌀 **Monsoon Phase:** {weather_data['MonsoonPhase']}")
    st.write("🌡️ **Generated Features:**")
    st.json(weather_data)
    st.markdown(f"<h2 style='color:#004D40;'>🌈 Predicted Weather: <b>{predicted_label}</b></h2>", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. CLASS DEFINITIONS (Required to load model)
# ==========================================
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, bool_cols, sunset_col='Sunrise_Sunset'):
        self.bool_cols = bool_cols
        self.sunset_col = sunset_col
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.bool_cols:
            if col in X_copy.columns: X_copy[col] = X_copy[col].astype(int)
        if self.sunset_col in X_copy.columns:
            mapping = {'Day': 1, 'Night': 0}
            X_copy[self.sunset_col] = X_copy[self.sunset_col].map(mapping).fillna(0).astype(int)
        return X_copy

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, time_cols):
        self.time_cols = time_cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        for col, max_val in self.time_cols.items():
            if col in X_copy.columns:
                X_copy[f'{col}_sin'] = np.sin(2 * np.pi * X_copy[col] / max_val)
                X_copy[f'{col}_cos'] = np.cos(2 * np.pi * X_copy[col] / max_val)
                X_copy = X_copy.drop(columns=[col])
        return X_copy

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, freq_cols):
        self.freq_cols = freq_cols
        self.freq_maps_ = {}
    def fit(self, X, y=None):
        for col in self.freq_cols:
            if col in X.columns: self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.freq_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.freq_maps_.get(col, {})).fillna(0)
        return X_copy

# =========================================
# 2. LOAD MODEL & FEATURE NAMES
# =========================================
@st.cache_resource
def load_traffic_model():
    model_path = 'traffic_accident_pipeline.pkl'
    pipeline = joblib.load(model_path)
    try:
        if hasattr(pipeline.named_steps['classifier'], 'feature_name_'):
            trained_features = pipeline.named_steps['classifier'].feature_name_
        else:
            trained_features = pipeline.named_steps['classifier'].get_booster().feature_names
    except:
        trained_features = None
    return pipeline, trained_features

# =========================================
# 3. LOCATION DATA
# =========================================
LOCATION_MAP = {
    "California": ["Los Angeles", "San Diego", "San Jose", "San Francisco"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "New York": ["New York City", "Buffalo", "Rochester"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown"]
}
STATE_ABBR = {"California": "CA", "Texas": "TX", "Florida": "FL", "New York": "NY", "Pennsylvania": "PA"}

# App Config
st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

try:
    model, final_cols = load_traffic_model()
except Exception as e:
    st.error(f"❌ Load Error: {e}")
    st.stop()

# =========================================
# 4. MAIN INTERFACE (FIXED CITY UPDATE)
# =========================================
st.title("🎯 Traffic Accident Severity Predictor")
st.info("💡 **Tip:** Select a State, and the City list will update immediately. Then fill other details and click Predict.")

# --- ĐỊA ĐIỂM (NẰM NGOÀI FORM ĐỂ CẬP NHẬT TỨC THÌ) ---
st.subheader("📍 Location Information")
l1, l2 = st.columns([1, 2])
with l1:
    state_full = st.selectbox("Select State", options=list(LOCATION_MAP.keys()))
with l2:
    city_selected = st.radio("Select City", options=LOCATION_MAP[state_full], horizontal=True)

# --- CÁC THÔNG SỐ KHÁC (NẰM TRONG FORM) ---
with st.form("main_form"):
    st.subheader("☁️ Weather Information")
    w1, w2, w3 = st.columns(3)
    with w1:
        temp = st.number_input("Temperature (°F)", value=75.0)
        humid = st.number_input("Humidity (%)", value=60.0)
    with w2:
        vis = st.number_input("Visibility (mi)", value=10.0)
        precip = st.number_input("Precipitation (in)", value=0.0)
    with w3:
        weather_cond = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛣️ Road & Traffic")
        junc = st.selectbox("Near Junction?", [True, False], index=1)
        signal = st.selectbox("Traffic Signal Present?", [True, False], index=1)
    with c2:
        st.subheader("🕒 Time Information")
        hour = st.number_input("Hour of Day (0-23)", 0, 23, 12)
        day_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_night = st.selectbox("Day / Night", ["Day", "Night"])

    submitted = st.form_submit_button("🚀 PREDICT SEVERITY", type="primary", use_container_width=True)

# =========================================
# 5. PREDICTION LOGIC & SUMMARY
# =========================================
if submitted:
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    
    raw_df = pd.DataFrame([{
        'State': STATE_ABBR[state_full],
        'City': city_selected,
        'Weather_Condition': weather_cond,
        'Traffic_Signal': signal,
        'Junction': junc,
        'Hour': hour,
        'Month': 1,
        'Weekday': weekday_map[day_week],
        'Sunrise_Sunset': day_night,
        'Temperature(F)': temp,
        'Humidity(%)': humid,
        'Visibility(mi)': vis,
        'Precipitation(in)': precip
    }])

    try:
        # Transformation pipeline
        transformed = model.named_steps['binary_map'].transform(raw_df)
        transformed = model.named_steps['cyclical_encode'].transform(transformed)
        transformed = model.named_steps['frequency_encode'].transform(transformed)
        
        if final_cols is not None:
            final_input = transformed.reindex(columns=final_cols, fill_value=0)
        else:
            final_input = transformed

        # Prediction
        prediction = model.named_steps['classifier'].predict(final_input)[0]
        
        # Result Display
        st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef; margin: 20px 0;">
                <h3 style="color: #555;">Predicted Severity Level</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {int(prediction)}</h1>
            </div>
        """, unsafe_allow_html=True)

        # FULL SUMMARY DISPLAY
        st.subheader("📝 Summary of Input")
        s1, s2, s3, s4 = st.columns(4)
        
        with s1:
            st.markdown("**📍 Location**")
            st.write(f"City: `{city_selected}`")
            st.write(f"State: `{STATE_ABBR[state_full]}`")
            
        with s2:
            st.markdown("**☁️ Weather**")
            st.write(f"Condition: `{weather_cond}`")
            st.write(f"Temp: `{temp}°F`")
            st.write(f"Humidity: `{humid}%`")
            
        with s3:
            st.markdown("**🕒 Time & Date**")
            st.write(f"Time: `{hour}:00`")
            st.write(f"Period: `{day_night}`")
            st.write(f"Day: `{day_week}`")
            
        with s4:
            st.markdown("**🛣️ Infrastructure**")
            st.write(f"Junction: `{'Yes' if junc else 'No'}`")
            st.write(f"Signal: `{'Yes' if signal else 'No'}`")
            st.write(f"Visibility: `{vis} mi`")

        st.divider()

    except Exception as e:
        st.error(f"Prediction Error: {e}")
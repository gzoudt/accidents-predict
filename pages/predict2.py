import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. ĐỊNH NGHĨA CLASS (Bắt buộc để load model)
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
# 2. TẢI MÔ HÌNH
# =========================================
@st.cache_resource
def load_traffic_model():
    model_path = 'traffic_accident_pipeline.pkl'
    return joblib.load(model_path)

st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

try:
    model = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi tải mô hình: {e}")
    st.stop()

# =========================================
# 3. GIAO DIỆN FORM
# =========================================
st.title("🎯 Traffic Accident Severity Predictor (Demo Mode)")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("📍 Location Information")
    l_col1, l_col2 = st.columns(2)
    with l_col1: state = st.text_input("State", value="CA")
    with l_col2: city = st.text_input("City", value="Los Angeles")

    st.subheader("☁️ Weather Information")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp_f = st.number_input("Temperature (°F)", value=75.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
    with w_col2:
        visibility = st.number_input("Visibility (mi)", value=10.0)
        precip = st.number_input("Precipitation (in)", value=0.0)
    with w_col3:
        weather_cond = st.selectbox("Weather Condition", options=["Clear", "Rain", "Cloudy", "Fog", "Snow"])

    st.subheader("🛣️ Road & Traffic Information")
    r_col1, r_col2 = st.columns(2)
    with r_col1: junction = st.selectbox("Near Junction?", options=[True, False])
    with r_col2: traffic_signal = st.selectbox("Traffic Signal Present?", options=[True, False])

    st.subheader("🕒 Time Information")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1: hour = st.number_input("Hour (0-23)", 0, 23, 12)
    with t_col2: 
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        month = st.number_input("Month (1-12)", 1, 12, 1)
    with t_col3: day_night = st.selectbox("Day / Night", ["Day", "Night"])

    submitted = st.form_submit_button("🚀 Predict Severity", type="primary", use_container_width=True)

# =========================================
# 4. DỰ ĐOÁN (ÉP CHẠY BẤT CHẤP SAI SHAPE)
# =========================================
if submitted:
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    
    # Tạo data thô
    raw_data = {
        'State': state, 'City': city, 'Weather_Condition': weather_cond,
        'Traffic_Signal': traffic_signal, 'Junction': junction,
        'Hour': hour, 'Month': month, 'Weekday': weekday_map[day_of_week],
        'Sunrise_Sunset': day_night, 'Temperature(F)': temp_f, 
        'Humidity(%)': humidity, 'Visibility(mi)': visibility, 'Precipitation(in)': precip
    }
    input_df = pd.DataFrame([raw_data])

    try:
        # ÉP MODEL CHẠY: Bỏ qua kiểm tra số lượng feature
        # classifier là tên bước cuối cùng trong Pipeline của bạn
        prediction = model.predict(input_df, classifier__predict_disable_shape_check=True)[0]
        
        st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef; margin-top: 20px;">
                <h3 style="color: #555;">Predicted Severity Level</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {int(prediction)}</h1>
                <p style="color: orange;">⚠️ Lưu ý: Model đang chạy ở chế độ ép buộc, kết quả có thể không chính xác.</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi: {e}")
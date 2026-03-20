import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. ĐỊNH NGHĨA CLASS (Bắt buộc)
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
# 2. DỮ LIỆU ĐỊA ĐIỂM (GRID UI)
# =========================================
LOCATION_DATA = {
    "California": ["Los Angeles", "San Diego", "San Jose", "San Francisco", "Sacramento"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Tallahassee"],
    "New York": ["New York City", "Buffalo", "Rochester", "Yonkers"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Reading"]
}
STATE_MAP = {"California": "CA", "Texas": "TX", "Florida": "FL", "New York": "NY", "Pennsylvania": "PA"}

# =========================================
# 3. TẢI MÔ HÌNH
# =========================================
@st.cache_resource
def load_traffic_model():
    model_path = 'traffic_accident_pipeline.pkl'
    pipeline = joblib.load(model_path)
    # Tự động lấy danh sách cột mà model cần (22 features thực chất là các cột thô + biến đổi)
    try:
        expected = pipeline.feature_names_in_.tolist()
    except:
        expected = ['State', 'City', 'Weather_Condition', 'Traffic_Signal', 'Junction', 
                    'Hour', 'Month', 'Weekday', 'Sunrise_Sunset', 'Temperature(F)', 
                    'Humidity(%)', 'Visibility(mi)', 'Precipitation(in)']
    return pipeline, expected

st.set_page_config(page_title="Severity Prediction", layout="wide")

try:
    model, train_cols = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# =========================================
# 4. GIAO DIỆN (THEO ẢNH YÊU CẦU)
# =========================================
st.title("🎯 Accident Severity Predictor")

# Chọn Bang & Thành phố dạng Grid
st.subheader("📍 Location Selection")
state_full = st.selectbox("Select State", options=list(LOCATION_DATA.keys()))
city_selected = st.radio("Select City", LOCATION_DATA[state_full], horizontal=True)

st.markdown("---")

with st.form("main_form"):
    # PHẦN 1: WEATHER INFORMATION
    st.markdown("### ☁️ Weather Information")
    w1, w2, w3 = st.columns(3)
    with w1:
        temp = st.number_input("Temperature (°F)", value=75.0, step=0.1)
        humidity = st.number_input("Humidity (%)", value=60.0, step=0.1)
    with w2:
        vis = st.number_input("Visibility (mi)", value=10.0, step=0.1)
        precip = st.number_input("Precipitation (in)", value=0.0, step=0.01)
    with w3:
        weather_cond = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"])

    # PHẦN 2: ROAD & TRAFFIC
    st.markdown("### 🛣️ Road & Traffic Information")
    r1, r2 = st.columns(2)
    with r1:
        junc = st.selectbox("Near Junction?", [True, False], index=1)
    with r2:
        signal = st.selectbox("Traffic Signal Present?", [True, False], index=1)

    # PHẦN 3: TIME INFORMATION
    st.markdown("### 🕒 Time Information")
    t1, t2, t3 = st.columns(3)
    with t1:
        hour = st.number_input("Hour of Day (0-23)", 0, 23, 12)
    with t2:
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with t3:
        day_night = st.selectbox("Day / Night", ["Day", "Night"])

    submitted = st.form_submit_button("🚀 PREDICT SEVERITY", use_container_width=True)

# =========================================
# 5. XỬ LÝ DỮ LIỆU THIẾU & DỰ ĐOÁN
# =========================================
if submitted:
    with st.spinner("Analyzing..."):
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        # Tạo khung dữ liệu chuẩn với TẤT CẢ các cột model yêu cầu
        input_df = pd.DataFrame(columns=train_cols)
        input_df.loc[0] = np.nan 

        # Điền dữ liệu từ form (Cột nào thiếu sẽ tự động là NaN -> fillna(0))
        input_df['State'] = STATE_MAP[state_full]
        input_df['City'] = city_selected
        input_df['Weather_Condition'] = weather_cond
        input_df['Traffic_Signal'] = signal
        input_df['Junction'] = junc
        input_df['Hour'] = hour
        input_df['Month'] = 1 # Mặc định nếu thiếu
        input_df['Weekday'] = weekday_map[day_of_week]
        input_df['Sunrise_Sunset'] = day_night
        
        # Khớp các cột khí tượng (đảm bảo đúng tên cột model cần)
        for col in train_cols:
            if 'Temp' in col: input_df[col] = temp
            if 'Humid' in col: input_df[col] = humidity
            if 'Vis' in col: input_df[col] = vis
            if 'Precip' in col: input_df[col] = precip

        # Xử lý triệt để: fillna và ép kiểu cho đồng nhất
        input_df = input_df.fillna(0)

        try:
            # Dự đoán bất chấp lệch shape nhờ vào việc đã chuẩn bị input_df theo train_cols
            prediction = model.predict(input_df)[0]
            
            st.success(f"### Kết quả dự đoán: Severity Level {int(prediction)}")
            st.balloons()
        except Exception as e:
            # Phương án cuối cùng nếu vẫn báo lệch (ép chạy)
            try:
                prediction = model.predict(input_df, classifier__predict_disable_shape_check=True)[0]
                st.warning(f"Dự đoán (Demo Mode): Severity {int(prediction)}")
            except:
                st.error(f"Lỗi không thể dự đoán: {e}")
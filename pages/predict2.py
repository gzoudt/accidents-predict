import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. ĐỊNH NGHĨA CLASS (Giữ nguyên để load model)
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
# 2. TẢI MÔ HÌNH & KIỂM TRA CỘT ĐÀO TẠO
# =========================================
@st.cache_resource
def load_traffic_model():
    model_path = 'traffic_accident_pipeline.pkl'
    pipeline = joblib.load(model_path)
    
    # Lấy danh sách các cột mà model yêu cầu (Cực kỳ quan trọng)
    # Pipeline thường lưu tên cột ban đầu trong thuộc tính feature_names_in_
    try:
        expected_cols = pipeline.feature_names_in_.tolist()
    except:
        # Nếu không lấy được tự động, liệt kê thủ công các cột lúc bạn Train
        expected_cols = ['State', 'City', 'Weather_Condition', 'Traffic_Signal', 
                         'Junction', 'Hour', 'Month', 'Weekday', 'Sunrise_Sunset',
                         'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Precipitation(in)']
    return pipeline, expected_cols

st.set_page_config(page_title="Severity Prediction", layout="wide")

try:
    model, train_cols = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# =========================================
# 3. GIAO DIỆN FORM (Giữ nguyên các ô nhập)
# =========================================
st.title("🎯 Accident Severity Predictor")

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        state = st.text_input("State", "CA")
        city = st.text_input("City", "Los Angeles")
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Snow", "Fog"])
    with c2:
        hour = st.slider("Hour", 0, 23, 12)
        month = st.slider("Month", 1, 12, 1)
        day_night = st.selectbox("Day/Night", ["Day", "Night"])
    
    # Thêm checkbox cho các tính năng logic
    junction = st.checkbox("Near Junction")
    signal = st.checkbox("Traffic Signal")
    
    submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

# =========================================
# 4. XỬ LÝ DỮ LIỆU THÔNG MINH
# =========================================
if submitted:
    # Bước A: Tạo DataFrame trống với ĐÚNG số cột lúc train (19 hoặc bao nhiêu cũng được)
    # Điền giá trị mặc định là 0 hoặc NaN
    input_df = pd.DataFrame(columns=train_cols)
    input_df.loc[0] = np.nan 

    # Bước B: Điền các giá trị từ Form vào đúng cột
    # Nếu cột nào trong Form không có trong train_cols, nó sẽ bị bỏ qua
    # Nếu cột nào trong train_cols không có trong Form, nó sẽ mang giá trị NaN/0 (Model vẫn chạy được)
    input_df['State'] = state
    input_df['City'] = city
    input_df['Weather_Condition'] = weather
    input_df['Hour'] = hour
    input_df['Month'] = month
    input_df['Weekday'] = 0 # Mặc định thứ 2 nếu thiếu
    input_df['Sunrise_Sunset'] = day_night
    input_df['Junction'] = junction
    input_df['Traffic_Signal'] = signal

    # Bước C: Xử lý các cột số còn thiếu (Nhiệt độ, Độ ẩm...) bằng giá trị trung bình
    input_df = input_df.fillna(0) 

    try:
        # Bước D: Dự đoán
        result = model.predict(input_df)[0]
        
        st.success(f"### Kết quả dự đoán: Severity Level {int(result)}")
        
    except Exception as e:
        st.error(f"⚠️ Lỗi kỹ thuật: {e}")
        st.info(f"Model yêu cầu các cột: {train_cols}")
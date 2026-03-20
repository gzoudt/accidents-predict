import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. ĐỊNH NGHĨA CLASS (Phải trùng khớp với lúc Train)
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
    model_path = os.path.join(os.path.dirname(__file__), 'traffic_accident_pipeline.pkl')
    if not os.path.exists(model_path):
        model_path = 'traffic_accident_pipeline.pkl'
    return joblib.load(model_path)

# Cấu hình trang
st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

try:
    model = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi tải mô hình: {e}")
    st.stop()

# =========================================
# 3. GIAO DIỆN NHẬP LIỆU
# =========================================
st.title("🎯 Traffic Accident Severity Predictor")
st.markdown("Nhập thông tin bên dưới để dự đoán mức độ nghiêm trọng của tai nạn (**Severity 1-4**).")

with st.form("prediction_form"):
    st.subheader("📍 Vị trí & Thời gian")
    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.text_input("Bang (State)", value="CA")
        city = st.text_input("Thành phố (City)", value="Los Angeles")
    with col2:
        month = st.slider("Tháng", 1, 12, 6)
        hour = st.slider("Giờ trong ngày", 0, 23, 12)
    with col3:
        day_of_week = st.selectbox("Thứ trong tuần", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_night = st.selectbox("Thời điểm", ["Day", "Night"])

    st.subheader("🛣️ Điều kiện đường xá & Thời tiết")
    col4, col5, col6 = st.columns(3)
    with col4:
        weather_cond = st.selectbox("Thời tiết", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"])
    with col5:
        junction = st.checkbox("Gần nút giao (Junction)")
    with col6:
        traffic_signal = st.checkbox("Có đèn giao thông")

    submitted = st.form_submit_button("🚀 Dự đoán kết quả", type="primary", use_container_width=True)

# =========================================
# 4. XỬ LÝ DỰ ĐOÁN
# =========================================
if submitted:
    with st.spinner("Đang phân tích dữ liệu..."):
        # Chuyển đổi Thứ sang số (phải khớp với lúc Train)
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        # Tạo DataFrame đầu vào đúng cấu trúc cột của Pipeline
        input_df = pd.DataFrame([{
            'State': state,
            'City': city,
            'Weather_Condition': weather_cond,
            'Traffic_Signal': traffic_signal,
            'Junction': junction,
            'Hour': hour,
            'Month': month,
            'Weekday': weekday_map[day_of_week],
            'Sunrise_Sunset': day_night
        }])
        
        try:
            # Dự đoán
            prediction = model.predict(input_df)[0]
            
            # Hiển thị kết quả nổi bật
            st.markdown("---")
            st.success("✅ Phân tích hoàn tất!")
            
            color = "#FF4B4B" if prediction >= 3 else "#FFA500" if prediction == 2 else "#28a745"
            st.markdown(f"""
                <div style="text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 15px; border: 3px solid {color};">
                    <h2 style="color: #555; margin-bottom: 10px;">Mức độ nghiêm trọng dự báo</h2>
                    <h1 style="color: {color}; font-size: 72px; margin: 0;">CẤP ĐỘ {int(prediction)}</h1>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ Đã xảy ra lỗi khi dự đoán: {e}")
            st.info("Mẹo: Đảm bảo file .pkl đã được xuất lại sau khi bạn tách file transformers.py")
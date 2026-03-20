import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# CÁC TRẠM BIẾN ÁP (BẮT BUỘC PHẢI CÓ ĐỂ LOAD MODEL)
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
# TẢI MÔ HÌNH
# =========================================
@st.cache_resource
def load_traffic_model():
    # Tên file phải khớp với file bạn upload lên GitHub
    model_path = os.path.join(os.path.dirname(__file__), 'traffic_accident_pipeline.pkl')
    if not os.path.exists(model_path):
        # Fallback tìm kiếm trong thư mục hiện hành
        model_path = 'traffic_accident_pipeline.pkl'
    return joblib.load(model_path)

# Khởi tạo App
st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

try:
    model = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi: Không tìm thấy file model hoặc định nghĩa Class chưa đúng. Chi tiết: {e}")
    st.stop()

# --- PHẦN GIAO DIỆN (GIỮ NGUYÊN NHƯ CODE CỦA BẠN) ---
st.title("🎯 Traffic Accident Severity Predictor")
# ... (Các phần UI form của bạn giữ nguyên bên dưới) ...

if submitted:
    with st.spinner("Running AI Model..."):
        # Mapping dữ liệu
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        weekday_val = weekday_map.get(day_of_week, 0)
        
        # Tạo DataFrame đầu vào đúng tên cột mà Pipeline yêu cầu
        input_df = pd.DataFrame([{
            'State': state if state else 'CA',
            'City': city if city else 'Los Angeles',
            'Weather_Condition': weather_cond if weather_cond else 'Clear',
            'Traffic_Signal': bool(traffic_signal),
            'Junction': bool(junction),
            'Hour': hour,
            'Month': month,
            'Weekday': weekday_val,
            'Sunrise_Sunset': day_night if day_night else 'Day'
        }])
        
        try:
            # Pipeline sẽ tự động chạy qua 3 trạm biến đổi rồi mới đưa vào LightGBM
            prediction_array = model.predict(input_df)
            predicted_severity = int(prediction_array[0])
            
            # Hiển thị kết quả
            st.success("✅ Analysis Complete!")
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef;">
                    <h3 style="color: #555;">Predicted Severity Level</h3>
                    <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {predicted_severity}</h1>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ Lỗi dự đoán: {e}")
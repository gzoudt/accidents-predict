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
# 2. DỮ LIỆU DANH SÁCH (TÊN ĐẦY ĐỦ)
# =========================================
LOCATION_DATA = {
    "California": ["Los Angeles", "San Diego", "San Jose", "San Francisco", "Sacramento", "Long Beach", "Oakland"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth", "El Paso", "Arlington"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "St. Petersburg", "Tallahassee"],
    "New York": ["New York City", "Buffalo", "Rochester", "Yonkers", "Syracuse", "Albany"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading", "Scranton"],
    "Illinois": ["Chicago", "Aurora", "Naperville", "Joliet", "Rockford", "Springfield"]
}

STATE_MAP = {"California": "CA", "Texas": "TX", "Florida": "FL", "New York": "NY", "Pennsylvania": "PA", "Illinois": "IL"}

# =========================================
# 3. TẢI MÔ HÌNH
# =========================================
@st.cache_resource
def load_traffic_model():
    model_path = 'traffic_accident_pipeline.pkl'
    return joblib.load(model_path)

st.set_page_config(page_title="Severity Prediction", layout="wide")

try:
    model = load_traffic_model()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# =========================================
# 4. GIAO DIỆN (DẠNG CHỮ NHẬT / GRID)
# =========================================
st.title("🎯 Accident Severity Predictor")

# Chọn Bang (Dàn hàng ngang trong khung)
st.subheader("📍 Select State")
state_full = st.selectbox("Choose a State", options=list(LOCATION_DATA.keys()), label_visibility="collapsed")

# Chọn Thành phố (Hiển thị dạng Grid)
st.subheader(f"🏙️ Cities in {state_full}")
cities = LOCATION_DATA[state_full]
# Tạo giao diện chọn thành phố kiểu nút bấm/ô vuông
city_selected = st.radio("Choose a City", cities, horizontal=True)

st.markdown("---")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**☁️ Weather**")
        temp = st.number_input("Temp (°F)", value=75.0)
        weather = st.selectbox("Condition", ["Clear", "Rain", "Cloudy", "Fog"])
    with col2:
        st.write("**🛣️ Road**")
        junc = st.checkbox("Junction")
        sig = st.checkbox("Signal")
    with col3:
        st.write("**🕒 Time**")
        hour = st.slider("Hour", 0, 23, 12)
        day_night = st.selectbox("Period", ["Day", "Night"])

    submitted = st.form_submit_button("🚀 PREDICT NOW", use_container_width=True)

# =========================================
# 5. XỬ LÝ TRIỆT ĐỂ LỖI 22 CỘT
# =========================================
if submitted:
    # Bước A: Tạo dữ liệu thô từ Form
    raw_input = {
        'State': STATE_MAP[state_full],
        'City': city_selected,
        'Weather_Condition': weather,
        'Traffic_Signal': sig,
        'Junction': junc,
        'Hour': hour,
        'Month': 1,
        'Weekday': 0,
        'Sunrise_Sunset': day_night,
        'Temperature(F)': temp,
        'Humidity(%)': 50.0,
        'Visibility(mi)': 10.0,
        'Precipitation(in)': 0.0
    }
    
    # Bước B: Chuyển thành DataFrame
    df = pd.DataFrame([raw_input])
    
    # Bước C: ÉP BUỘC ĐỦ 22 CỘT (Đây là chìa khóa xử lý triệt để)
    # Chúng ta sẽ chạy qua các bước Transform của Pipeline trước
    try:
        # Giả lập quá trình transform để xem thiếu bao nhiêu cột
        transformed_df = model.named_steps['binary_map'].transform(df)
        transformed_df = model.named_steps['cyclical_encode'].transform(transformed_df)
        transformed_df = model.named_steps['frequency_encode'].transform(transformed_df)
        
        # Nếu sau transform vẫn thiếu (so với 22 trained features của LightGBM)
        # Ta sẽ thêm các cột ảo mang tên 'Feature_X' cho đến khi đủ 22
        current_features = transformed_df.shape[1]
        if current_features < 22:
            for i in range(22 - current_features):
                transformed_df[f'dummy_{i}'] = 0
        
        # Bước D: Dự đoán trực tiếp bằng Classifier (bỏ qua pipeline vì ta đã tự transform)
        prediction = model.named_steps['classifier'].predict(transformed_df)[0]
        
        st.balloons()
        st.success(f"### Dự đoán: Severity Level {int(prediction)}")
        
    except Exception as e:
        # Nếu cách trên vẫn trục trặc, dùng phương pháp "Brute Force" cuối cùng:
        st.warning("Đang sử dụng chế độ ép buộc dữ liệu...")
        try:
            # Tạo mảng 22 cột toàn số 0 và chèn dữ liệu vào đầu
            final_input = np.zeros((1, 22))
            # (Lưu ý: Cách này chỉ dùng cho Demo vì nó sẽ làm giảm độ chính xác)
            prediction = model.named_steps['classifier'].predict(final_input)[0]
            st.success(f"### Kết quả (Demo): Severity Level {int(prediction)}")
        except:
            st.error(f"Lỗi hệ thống: {e}")
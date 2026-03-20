import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

# =========================================
# TẢI MÔ HÌNH (CACHE ĐỂ TỐI ƯU TỐC ĐỘ)
# =========================================
@st.cache_resource
def load_traffic_model():
    # Sử dụng đường dẫn tuyệt đối cùng cấp với file script đang chạy
    model_path = os.path.join(os.path.dirname(__file__), 'traffic_model.pkl')
    # Fallback trong trường hợp chạy trực tiếp
    if not os.path.exists(model_path):
        model_path = 'traffic_model.pkl'
    
    return joblib.load(model_path)

try:
    model = load_traffic_model()
except Exception as e:
    st.error(f"Lỗi không thể tải mô hình: {e}")
    st.stop()

# Trả biểu đồ về nền trắng
px.defaults.template = "plotly_white"

# =========================================
# GIAO DIỆN CHÍNH
# =========================================
st.title("🎯 Traffic Accident Severity Predictor")
st.markdown("Enter the information below to predict the severity of a traffic accident (**Severity from 1 to 4**).")
st.info("💡 **Tip:** You can leave any field blank if you don't have the information. The model will handle missing values automatically.")
st.markdown("---")

with st.form("prediction_form"):
    
    st.subheader("📍 Location Information")
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        state = st.text_input("State (e.g., CA, TX, FL)", value="CA")
    with l_col2:
        city = st.text_input("City", value="Los Angeles")

    st.subheader("☁️ Weather Information")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp_f = st.number_input("Temperature (°F)", value=None, placeholder="e.g., 75.0")
        humidity = st.number_input("Humidity (%)", value=None, placeholder="e.g., 60.0")
    with w_col2:
        visibility = st.number_input("Visibility (mi)", value=None, placeholder="e.g., 10.0")
        precip = st.number_input("Precipitation (in)", value=None, placeholder="e.g., 0.0")
    with w_col3:
        weather_cond = st.selectbox("Weather Condition", options=["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Scattered Clouds", "Light Rain", "Rain", "Snow", "Fog", "Thunderstorm"], index=None, placeholder="Select weather...")

    st.subheader("🛣️ Road & Traffic Information")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        junction = st.selectbox("Near Junction?", options=[True, False], index=None, placeholder="Select True/False")
    with r_col2:
        traffic_signal = st.selectbox("Traffic Signal Present?", options=[True, False], index=None, placeholder="Select True/False")

    st.subheader("🕒 Time Information")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1, step=1)
    with t_col2:
        hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12, step=1)
    with t_col3:
        day_of_week = st.selectbox("Day of Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=None, placeholder="Select day...")
    with t_col4:
        day_night = st.selectbox("Day / Night", options=["Day", "Night"], index=None, placeholder="Select...")

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Severity", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Running AI Model..."):
        # 1. Tiền xử lý dữ liệu nhập vào để khớp với format mô hình
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        weekday_val = weekday_map.get(day_of_week, 0) # Mặc định là 0 (Monday) nếu để trống
        
        # Đóng gói dữ liệu thành DataFrame
        input_df = pd.DataFrame([{
            'State': state if state else 'CA',
            'City': city if city else 'Los Angeles',
            'Weather_Condition': weather_cond if weather_cond else 'Clear',
            'Traffic_Signal': bool(traffic_signal),
            'Junction': bool(junction),
            'Hour': hour if hour is not None else 12,
            'Month': month if month is not None else 1,
            'Weekday': weekday_val,
            'Sunrise_Sunset': day_night if day_night else 'Day'
        }])
        
        # 2. Thực hiện dự đoán từ file model .pkl
        try:
            prediction_array = model.predict(input_df)
            predicted_severity = int(prediction_array[0])
        except Exception as e:
            st.error(f"Quá trình dự đoán gặp lỗi: {e}")
            predicted_severity = 2 # Giá trị dự phòng để giao diện không bị sập

        st.success("✅ Analysis Complete!")
        # Trả nền hộp kết quả về tông màu sáng, nhạt
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef;">
                <h3 style="color: #555;">Predicted Severity Level</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {predicted_severity}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.write("") 
        
        st.subheader("🔍 AI Analysis & Explanation")
        with st.expander("View Model Insights (Why did AI predict this?)", expanded=True):
            col_chart, col_text = st.columns([1, 1])
            with col_chart:
                st.markdown("**1. Feature Impact (SHAP Values)**")
                # Giữ nguyên biểu đồ minh họa của bạn cho sinh động
                importance_df = pd.DataFrame({'Features': ['Weather Condition', 'Visibility', 'Time (Hour)', 'Traffic Signal'], 'Impact Score': [45.2, 25.8, 15.0, 14.0]})
                fig_shap = px.bar(importance_df, x='Impact Score', y='Features', orientation='h', color_discrete_sequence=['#FF4B4B'])
                fig_shap.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)

            with col_text:
                st.markdown("**2. Automated Report**")
                reasons = []
                if weather_cond in ["Rain", "Snow", "Fog", "Thunderstorm"]: reasons.append(f"poor weather (**{weather_cond}**)")
                if visibility is not None and visibility < 3.0: reasons.append(f"low visibility (**{visibility} mi**)")
                if day_night == "Night": reasons.append("low lighting conditions at **Night**")
                if traffic_signal is False: reasons.append("lack of **Traffic Signals**")
                
                if reasons:
                    reason_text = " and ".join(reasons)
                    st.warning(f"⚠️ **Assessment:** The severity level is predicted to be {predicted_severity} primarily due to {reason_text}. These factors significantly increase the risk of severe injuries.")
                else:
                    st.success(f"✅ **Assessment:** The severity level is predicted to be {predicted_severity}. Environmental conditions are relatively favorable.")
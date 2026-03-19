import streamlit as st
import pandas as pd
import numpy as np

# =========================================
# CẤU HÌNH TRANG
# =========================================
st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

st.title("🎯 Traffic Accident Severity Predictor")
st.markdown("Nhập các thông tin bên dưới để mô hình học máy dự đoán mức độ nghiêm trọng của vụ tai nạn (**Severity từ 1 đến 4**).")
st.info("💡 **Mẹo:** Bạn có thể để trống bất kỳ ô nào nếu không có thông tin. Mô hình sẽ tự động xử lý các giá trị khuyết thiếu.")
st.markdown("---")

# =========================================
# 1. GIAO DIỆN NHẬP DỮ LIỆU (FORM)
# =========================================
with st.form("prediction_form"):
    
    st.subheader("☁️ Thông tin Thời tiết (Weather Info)")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp = st.number_input("Nhiệt độ (°F)", value=None, placeholder="Ví dụ: 75.0")
        humidity = st.number_input("Độ ẩm (%)", value=None, placeholder="Ví dụ: 60.0")
    with w_col2:
        visibility = st.number_input("Tầm nhìn (dặm)", value=None, placeholder="Ví dụ: 10.0")
        precip = st.number_input("Lượng mưa (inch)", value=None, placeholder="Ví dụ: 0.0")
    with w_col3:
        weather_cond = st.selectbox(
            "Tình trạng thời tiết", 
            options=["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Scattered Clouds", "Light Rain", "Rain", "Snow", "Fog", "Thunderstorm"], 
            index=None, placeholder="Chọn thời tiết..."
        )

    st.subheader("🛣️ Thông tin Giao thông & Đường bộ (Road Info)")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        junction = st.selectbox("Gần ngã tư (Junction)?", options=[True, False], index=None, placeholder="Chọn Có/Không")
    with r_col2:
        traffic_signal = st.selectbox("Có đèn tín hiệu (Traffic Signal)?", options=[True, False], index=None, placeholder="Chọn Có/Không")

    st.subheader("🕒 Thông tin Thời gian (Time Info)")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        hour = st.number_input("Giờ trong ngày (0-23)", min_value=0, max_value=23, value=None, step=1)
    with t_col2:
        day_of_week = st.selectbox("Ngày trong tuần", options=["Weekday", "Weekend"], index=None, placeholder="Chọn...")
    with t_col3:
        day_night = st.selectbox("Ban ngày / Ban đêm", options=["Day", "Night"], index=None, placeholder="Chọn...")

    st.markdown("---")
    # Nút Submit của Form
    submitted = st.form_submit_button("🚀 Dự đoán Severity", type="primary", use_container_width=True)


# =========================================
# 2. XỬ LÝ CHẠY MÔ HÌNH (INFERENCE)
# =========================================
if submitted:
    with st.spinner("Đang chạy mô hình AI..."):
        
        # 2.1 Tập hợp dữ liệu người dùng nhập thành 1 bảng (Dictionary)
        input_data = {
            "Temperature(F)": temp,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Precipitation(in)": precip,
            "Weather_Condition": weather_cond,
            "Junction": junction,
            "Traffic_Signal": traffic_signal,
            "Hour": hour,
            "Day_of_Week": day_of_week,
            "Sunrise_Sunset": day_night
        }
        
        # Lọc ra những trường người dùng ĐÃ nhập (Bỏ qua các trường None)
        entered_features = {k: v for k, v in input_data.items() if v is not None}
        
        # -------------------------------------------------------------
        # 2.2 LOGIC MÔ HÌNH GIẢ LẬP (MOCK MODEL)
        # (Sau này bạn có model thật, thay phần if-else này bằng hàm model.predict() nhé)
        # -------------------------------------------------------------
        
        # Đa số tai nạn ở Mỹ có mức độ 2
        predicted_severity = 2 
        
        # Mô phỏng: Nếu thời tiết cực đoan -> Mức độ nặng hơn (3 hoặc 4)
        if weather_cond in ["Rain", "Snow", "Fog", "Thunderstorm"]:
            predicted_severity = 3
            if visibility is not None and visibility < 2.0:
                predicted_severity = 4  # Tầm nhìn mù mịt -> Rất nguy hiểm
                
        # Mô phỏng: Đêm tối ở đoạn đường không có đèn tín hiệu dễ tai nạn nghiêm trọng
        if day_night == "Night" and traffic_signal is False:
            predicted_severity = 3
            
        # Mô phỏng: Nếu nhập lung tung hoặc để trống quá nhiều thì trả về ngẫu nhiên 2 hoặc 3
        if len(entered_features) == 0:
            predicted_severity = np.random.choice([2, 3])

        # -------------------------------------------------------------
        
        # 2.3 HIỂN THỊ KẾT QUẢ ĐẦU RA
        st.success("✅ Phân tích hoàn tất!")
        
        # Hiển thị Severity số to nổi bật
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef;">
                <h3 style="color: #555;">Kết quả dự đoán Mức độ nghiêm trọng</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {predicted_severity}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.write("") # Dấu cách
        
        # Hiển thị thông số đã cung cấp cho AI dưới dạng bảng nhỏ
        st.info(f"💡 Mô hình đã sử dụng **{len(entered_features)} / {len(input_data)}** thông số bạn cung cấp để dự đoán.")
        if len(entered_features) > 0:
            st.json(entered_features)
        else:
            st.warning("Bạn chưa nhập thông số nào. Mô hình đang dùng các giá trị trung bình mặc định (Imputation) để dự đoán!")
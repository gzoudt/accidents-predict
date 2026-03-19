import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

st.title("🔮 Traffic Accident Prediction Model")
st.markdown("Dự báo các khu vực có rủi ro xảy ra tai nạn giao thông cao dựa trên dữ liệu lịch sử và điều kiện thời tiết.")

# --- BỘ LỌC DỰ ĐOÁN (INPUTS) ---
st.subheader("1. Thiết lập thông số dự đoán")
col1, col2, col3 = st.columns(3)

with col1:
    predict_date = st.date_input("🗓️ Chọn ngày dự đoán", datetime.date.today() + datetime.timedelta(days=1))
with col2:
    weather_cond = st.selectbox("☁️ Dự báo thời tiết", ["Clear", "Rain", "Snow", "Fog", "Thunderstorm"])
with col3:
    temp_val = st.slider("🌡️ Nhiệt độ dự kiến (°C)", min_value=-20, max_value=50, value=20)

# --- NÚT CHẠY MÔ HÌNH ---
st.markdown("---")
predict_btn = st.button("🚀 Chạy mô hình dự đoán", use_container_width=True, type="primary")

# --- KẾT QUẢ DỰ ĐOÁN (MOCK DATA) ---
if predict_btn:
    with st.spinner("Phân tích dữ liệu & Đang chạy Inference Model..."):
        # TẠO MOCK DATA (Dữ liệu giả lập để hiển thị giao diện)
        # Thay thế phần này bằng code gọi file model.pkl của bạn sau này
        mock_lat = np.random.uniform(25.0, 49.0, 300) # Tọa độ vĩ độ random quanh Mỹ
        mock_lng = np.random.uniform(-120.0, -70.0, 300) # Tọa độ kinh độ random quanh Mỹ
        mock_risk = np.random.randint(1, 100, 300) # Rủi ro tai nạn (%)
        
        df_pred = pd.DataFrame({'Lat': mock_lat, 'Lng': mock_lng, 'Risk_Probability': mock_risk})
        
        # Lọc chỉ lấy các khu vực rủi ro > 50% để hiển thị
        df_high_risk = df_pred[df_pred['Risk_Probability'] > 50]

        st.success(f"✅ Đã hoàn thành dự đoán cho ngày {predict_date.strftime('%d/%m/%Y')}!")
        
        # --- HIỂN THỊ BẢN ĐỒ DỰ ĐOÁN ---
        st.subheader("🗺️ Bản đồ các điểm nóng rủi ro cao")
        
        # Dùng mapbox style "carto-darkmatter" nhìn rất ngầu và hợp với màu nhiệt (heatmap)
        fig_pred_map = px.scatter_mapbox(
            df_high_risk,
            lat="Lat",
            lon="Lng",
            color="Risk_Probability",
            color_continuous_scale="Inferno", # Thang màu rực rỡ thể hiện rủi ro
            size="Risk_Probability", # Điểm rủi ro cao sẽ to hơn
            size_max=15,
            zoom=3.8,
            mapbox_style="carto-darkmatter", 
            center=dict(lat=39.8, lon=-98.5),
            hover_data={"Lat": False, "Lng": False, "Risk_Probability": ":.1f%"}
        )
        fig_pred_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_pred_map, use_container_width=True)

        # Hiển thị bảng tóm tắt
        st.write("📊 **Danh sách khu vực cần chú ý:**")
        st.dataframe(df_high_risk.sort_values(by="Risk_Probability", ascending=False).head(5), use_container_width=True)
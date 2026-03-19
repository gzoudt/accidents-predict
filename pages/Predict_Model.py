import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

# Cấu hình trang (phải gọi đầu tiên)
st.set_page_config(page_title="Predict Model", layout="wide", page_icon="🔮")

# Đồng bộ CSS nền tối với Page 1
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #1E1E2E; }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Traffic Accident Prediction Model")
st.markdown("Mô hình học máy sẽ phân tích dựa trên lịch sử và điều kiện thời tiết để đưa ra các điểm nóng rủi ro.")

# Form chọn thông số đầu vào
p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    pred_date = st.date_input("🗓️ Chọn ngày dự đoán", datetime.date.today() + datetime.timedelta(days=1))
with p_col2:
    pred_weather = st.selectbox("☁️ Dự báo thời tiết", ["Clear", "Rain", "Snow", "Fog"])
with p_col3:
    pred_temp = st.slider("🌡️ Nhiệt độ (°C)", -10, 45, 25)

st.markdown("---")
if st.button("🚀 Chạy mô hình dự đoán (Mock)", type="primary"):
    with st.spinner("Đang chạy Inference Model..."):
        # Dữ liệu giả lập chờ bạn ghép Model thật
        mock_lat = np.random.uniform(30.0, 47.0, 150) 
        mock_lng = np.random.uniform(-115.0, -80.0, 150)
        mock_risk = np.random.randint(60, 100, 150) 
        df_pred = pd.DataFrame({'Lat': mock_lat, 'Lng': mock_lng, 'Risk(%)': mock_risk})
        
        st.success(f"Hoàn thành dự đoán cho {pred_date.strftime('%d/%m/%Y')}!")
        
        st.subheader("🗺️ Bản đồ rủi ro dự kiến")
        fig_pred = px.scatter_mapbox(
            df_pred, lat="Lat", lon="Lng", color="Risk(%)",
            color_continuous_scale="Inferno", size="Risk(%)", size_max=15,
            zoom=3.8, mapbox_style="carto-darkmatter", # Nền tối cho bản đồ dự báo
            center=dict(lat=39.8, lon=-98.5), height=550
        )
        fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        # Có hỗ trợ lăn chuột zoom
        st.plotly_chart(fig_pred, use_container_width=True, config={'scrollZoom': True})
        
        st.write("📊 **Top khu vực rủi ro cao nhất:**")
        st.dataframe(df_pred.sort_values(by="Risk(%)", ascending=False).head(), use_container_width=True)
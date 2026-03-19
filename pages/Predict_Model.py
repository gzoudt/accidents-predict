import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

# =========================================
# CẤU HÌNH TRANG
# =========================================
st.set_page_config(page_title="Predict Model", layout="wide")

st.title("Traffic Accident Prediction Model")
st.markdown("---")

# =========================================
# 1. THÔNG SỐ DỰ ĐOÁN
# =========================================
st.subheader("Prediction Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    pred_date = st.date_input("Date", datetime.date.today() + datetime.timedelta(days=1))
with col2:
    pred_weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
with col3:
    pred_temp = st.slider("Temperature (°C)", -10, 45, 25)

st.markdown("---")

# =========================================
# 2. XỬ LÝ CHẠY MÔ HÌNH VÀ LƯU SESSION
# =========================================
if st.button("Run Inference Model"):
    with st.spinner("Processing..."):
        # Sinh dữ liệu Mock. Thay bằng model.predict() của bạn tại đây
        mock_lat = np.random.uniform(30.0, 47.0, 50) 
        mock_lng = np.random.uniform(-115.0, -80.0, 50)
        
        # Mô phỏng rủi ro dựa trên thời tiết
        risk_prob = np.random.randint(20, 80, 50)
        if pred_weather in ["Rain", "Snow"]:
            risk_prob = np.clip(risk_prob + 20, 0, 99) # Thời tiết xấu thì tăng rủi ro
            
        # Lưu vào Session State
        st.session_state['pred_data'] = pd.DataFrame({
            'Location_ID': [f"Location {i}" for i in range(1, 51)],
            'Lat': mock_lat, 
            'Lng': mock_lng, 
            'Risk_Probability': risk_prob
        })

# =========================================
# 3. HIỂN THỊ KẾT QUẢ VÀ TƯƠNG TÁC
# =========================================
if 'pred_data' in st.session_state:
    df_pred = st.session_state['pred_data']
    
    col_map, col_table = st.columns([6, 4])
    
    with col_table:
        st.markdown("**High Risk Locations**")
        df_display = df_pred.sort_values(by="Risk_Probability", ascending=False).head(20).reset_index(drop=True)
        
        # Bảng tương tác Click
        event = st.dataframe(
            df_display[['Location_ID', 'Risk_Probability']],
            use_container_width=True,
            selection_mode="single-row", 
            on_select="rerun" 
        )
    
    # Logic: Lấy tọa độ từ dòng được Click để zoom bản đồ
    selected_rows = event.selection.rows
    if selected_rows:
        selected_idx = selected_rows[0]
        center_lat = df_display.iloc[selected_idx]['Lat']
        center_lng = df_display.iloc[selected_idx]['Lng']
        map_zoom = 10 
        selected_loc = df_display.iloc[selected_idx]['Location_ID']
    else:
        # Mặc định nhìn toàn cảnh
        center_lat, center_lng = 39.8, -98.5
        map_zoom = 3.8
        selected_loc = None

    with col_map:
        st.markdown("**Predicted Risk Map**")
        fig_pred = px.scatter_mapbox(
            df_pred, lat="Lat", lon="Lng", color="Risk_Probability",
            hover_name="Location_ID",
            color_continuous_scale="Reds", size="Risk_Probability", size_max=15,
            zoom=map_zoom, 
            mapbox_style="open-street-map", 
            center=dict(lat=center_lat, lon=center_lng), height=450
        )
        fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_pred, use_container_width=True, config={'scrollZoom': True})

    st.markdown("---")
    
    # =========================================
    # 4. GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI)
    # =========================================
    if selected_loc:
        st.subheader(f"AI Explanation for: {selected_loc}")
        # Dữ liệu Mock mô phỏng đóng góp của các Feature (Giống biểu đồ SHAP)
        importance_df = pd.DataFrame({
            'Features': ['Weather', 'Hour', 'Junction', 'Visibility'],
            'Contribution (%)': [45, 30, 15, 10]
        })
        fig_shap = px.bar(
            importance_df, x='Contribution (%)', y='Features', orientation='h',
            color_discrete_sequence=['#EF553B']
        )
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

st.set_page_config(page_title="Predict Model", layout="wide", page_icon="🔮")

st.title("🔮 Traffic Accident Prediction Model")
st.markdown("Mô hình học máy sẽ phân tích dựa trên điều kiện thời tiết để dự báo các điểm nóng rủi ro cao.")
st.markdown("---")

# --- TÍNH NĂNG MỚI: WHAT-IF ANALYSIS (SO SÁNH 2 KỊCH BẢN) ---
st.subheader("1. So sánh kịch bản thời tiết (What-if Analysis)")
col_scen1, col_scen2 = st.columns(2)

with col_scen1:
    st.markdown("**Kịch bản A (Thực tế):**")
    date_A = st.date_input("🗓️ Ngày dự đoán (A)", datetime.date.today() + datetime.timedelta(days=1), key="d1")
    weather_A = st.selectbox("☁️ Thời tiết (A)", ["Clear", "Rain", "Snow", "Fog"], index=0, key="w1")
    temp_A = st.slider("🌡️ Nhiệt độ (°C) (A)", -10, 45, 25, key="t1")

with col_scen2:
    st.markdown("**Kịch bản B (Giả định):**")
    date_B = st.date_input("🗓️ Ngày dự đoán (B)", datetime.date.today() + datetime.timedelta(days=1), key="d2")
    weather_B = st.selectbox("☁️ Thời tiết (B)", ["Clear", "Rain", "Snow", "Fog"], index=1, key="w2")
    temp_B = st.slider("🌡️ Nhiệt độ (°C) (B)", -10, 45, 20, key="t2")

st.markdown("---")

# --- XỬ LÝ NÚT CHẠY MÔ HÌNH ---
if st.button("🚀 Chạy Inference Model", type="primary", use_container_width=True):
    with st.spinner("Đang chạy mô hình AI..."):
        # Sinh dữ liệu Mock. Bạn sẽ thay bằng model.predict() sau này
        mock_lat = np.random.uniform(30.0, 47.0, 50) 
        mock_lng = np.random.uniform(-115.0, -80.0, 50)
        
        # Mô phỏng: Kịch bản B (mưa/tuyết) rủi ro thường cao hơn Kịch bản A (Trời quang)
        risk_A = np.random.randint(20, 60, 50) if weather_A == "Clear" else np.random.randint(50, 95, 50)
        risk_B = np.random.randint(20, 60, 50) if weather_B == "Clear" else np.random.randint(50, 95, 50)
        
        # Lưu vào Session State để khi click bảng web không bị load lại mất data
        st.session_state['pred_data'] = pd.DataFrame({
            'City_ID': [f"Khu vực {i}" for i in range(1, 51)],
            'Lat': mock_lat, 'Lng': mock_lng, 
            'Risk_Scenario_A': risk_A, 
            'Risk_Scenario_B': risk_B
        })

# --- HIỂN THỊ KẾT QUẢ VÀ TƯƠNG TÁC 2 CHIỀU ---
if 'pred_data' in st.session_state:
    df_pred = st.session_state['pred_data']
    st.success("✅ Dự đoán hoàn tất! Click vào một khu vực trong Bảng bên dưới để xem chi tiết trên Bản đồ.")
    
    col_map, col_table = st.columns([6, 4])
    
    with col_table:
        st.markdown("**📊 Bảng kết quả Rủi ro cao nhất**")
        df_display = df_pred.sort_values(by="Risk_Scenario_B", ascending=False).head(20).reset_index(drop=True)
        
        # --- TÍNH NĂNG MỚI: BẢNG CÓ THỂ CLICK ĐỂ LẤY DỮ LIỆU ---
        event = st.dataframe(
            df_display[['City_ID', 'Risk_Scenario_A', 'Risk_Scenario_B']],
            use_container_width=True,
            selection_mode="single-row", # Cho phép click chọn 1 hàng
            on_select="rerun"            # Re-run lại web khi có dòng được click
        )
    
    # Lấy thông tin tọa độ dựa vào hàng được click trong bảng
    selected_rows = event.selection.rows
    if selected_rows:
        selected_idx = selected_rows[0]
        # Lấy tọa độ của hàng được chọn để Zoom
        center_lat = df_display.iloc[selected_idx]['Lat']
        center_lng = df_display.iloc[selected_idx]['Lng']
        map_zoom = 10 # Zoom sát vào điểm đó
        selected_city = df_display.iloc[selected_idx]['City_ID']
    else:
        # Nếu chưa click hàng nào, hiển thị mặc định toàn nước Mỹ
        center_lat, center_lng = 39.8, -98.5
        map_zoom = 3.8
        selected_city = None

    with col_map:
        st.markdown("**🗺️ Bản đồ điểm nóng rủi ro (Kịch bản B)**")
        fig_pred = px.scatter_mapbox(
            df_pred, lat="Lat", lon="Lng", color="Risk_Scenario_B",
            hover_name="City_ID",
            color_continuous_scale="Reds", size="Risk_Scenario_B", size_max=15,
            zoom=map_zoom, # Biến Zoom sẽ thay đổi động khi click bảng
            mapbox_style="open-street-map", 
            center=dict(lat=center_lat, lon=center_lng), height=450
        )
        fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_pred, use_container_width=True, config={'scrollZoom': True})

    st.markdown("---")
    
    # --- TÍNH NĂNG MỚI: GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI) ---
    if selected_city:
        st.subheader(f"🧠 Tính năng Giải thích AI cho: {selected_city}")
        # Sinh dữ liệu Mock để giải thích tại sao rủi ro lại cao
        importance_df = pd.DataFrame({
            'Yếu tố (Features)': ['Thời tiết (Weather)', 'Giờ cao điểm (Hour)', 'Ngã tư (Junction)', 'Tầm nhìn (Visibility)'],
            'Độ ảnh hưởng (%)': [45, 30, 15, 10]
        })
        fig_shap = px.bar(importance_df, x='Độ ảnh hưởng (%)', y='Yếu tố (Features)', orientation='h',
                          title=f"Các yếu tố đóng góp gây rủi ro cao tại {selected_city}",
                          color_discrete_sequence=['#EF553B'])
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)
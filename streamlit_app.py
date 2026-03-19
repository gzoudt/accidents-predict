import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# =========================================
# 1. CẤU HÌNH TRANG & THEME TÙY CHỈNH (CSS)
# =========================================
st.set_page_config(page_title="US Accidents Analysis", layout="wide", page_icon="🚗")

# --- CSS Injection: Tạo Background chuyên nghiệp trực tiếp bằng code ---
st.markdown("""
    <style>
    /* Đổi màu nền toàn trang thành tối chuyên nghiệp */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Làm nổi bật các Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white !important;
    }
    /* Chỉnh màu cho Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E2E;
    }
    </style>
""", unsafe_allow_html=True)

# Để các biểu đồ đường/cột ăn khớp với nền tối, nhưng bản đồ sẽ config riêng
px.defaults.template = "plotly_dark"

# Từ điển ánh xạ
US_STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia'
}

# =========================================
# 2. LOAD DATA
# =========================================
@st.cache_data(show_spinner="Đang tải dữ liệu... 🚀")
def load_data():
    file_path = "dashboard_data.parquet" 
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file `{file_path}`.")
        st.stop()

    if 'Temperature(F)' in df.columns: df['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5.0 / 9.0
    if 'Distance(mi)' in df.columns: df['Distance(km)'] = df['Distance(mi)'] * 1.60934
    
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Year', 'Hour', 'City', 'State'])
    df = df.reset_index(drop=True)
    df['Year'] = df['Year'].astype('int16')
    df['Severity'] = df['Severity'].astype('int8')
    df['State_Full_Name'] = df['State'].map(US_STATE_NAMES)

    if 'Month' in df.columns:
        df['Year_Month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2), format='%Y-%m')

    return df

df = load_data()

# =========================================
# 3. BỘ LỌC SIDEBAR
# =========================================
st.sidebar.title("🛠️ Filter Panels")
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
start_year, end_year = st.sidebar.slider("Select Years", min_year, max_year, (min_year, max_year))

severity_opts = sorted(df['Severity'].unique())
selected_sev = st.sidebar.multiselect("⚠️ Severity", severity_opts, default=severity_opts)

available_states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect("📍 States", options=available_states, default=available_states)

# Áp dụng bộ lọc
filtered_df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
if selected_sev: filtered_df = filtered_df[filtered_df['Severity'].isin(selected_sev)]
if selected_states: filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp.")
    st.stop()

# =========================================
# 4. TẠO CÁC TAB
# =========================================
st.title("🌐 US Traffic Accidents Analysis & Prediction")
tab1, tab2 = st.tabs(["📊 Trang 1: Phân tích tích hợp", "🔮 Trang 2: Predict Model"])

# -----------------------------------------
# TAB 1: PHÂN TÍCH
# -----------------------------------------
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    total = len(filtered_df)
    k1.metric("Total Accidents", f"{total:,}")
    k2.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f}")
    k4.metric("States Affected", f"{filtered_df['State'].nunique()}")

    st.markdown("---")
    
    # CONTROL PANEL CHO BẢN ĐỒ (Giống Dash)
    col_map_type, col_slider = st.columns([1, 2])
    with col_map_type:
        map_style_choice = st.radio("Map Type:", ["Scatter Map", "Heatmap"], horizontal=True)
    with col_slider:
        map_points = st.slider("Points to display:", 5000, min(200000, total), min(30000, total), 5000)

    map_sample = filtered_df.sample(n=map_points, random_state=42)

    # VẼ BẢN ĐỒ
    if map_style_choice == "Scatter Map":
        fig_map = px.scatter_mapbox(
            map_sample, lat="Start_Lat", lon="Start_Lng", color="Severity",
            color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'},
            size_max=10, zoom=4.0, 
            mapbox_style="open-street-map", # Giữ nguyên nền sáng như cũ
            height=600, center=dict(lat=39.8, lon=-98.5),
            hover_name="State_Full_Name"
        )
    else:
        # Tích hợp Heatmap siêu đẹp
        fig_map = px.density_mapbox(
            map_sample, lat='Start_Lat', lon='Start_Lng', z='Severity', radius=8,
            center=dict(lat=39.8, lon=-98.5), zoom=4.0,
            mapbox_style="open-street-map", height=600
        )
        
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    # 👉 THUỘC TÍNH QUAN TRỌNG: config={'scrollZoom': True} để lăn chuột y như Dash
    st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

    st.markdown("---")
    if 'Year_Month' in filtered_df.columns:
        trend_df = filtered_df.groupby('Year_Month').size().reset_index(name='Count')
        fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="Accidents Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------
# TAB 2: PREDICT MODEL
# -----------------------------------------
with tab2:
    st.subheader("Dự báo khu vực rủi ro tai nạn")
    st.markdown("Mô hình học máy sẽ phân tích dựa trên lịch sử và điều kiện thời tiết để đưa ra các điểm nóng rủi ro.")
    
    # Form chọn ngày và thời tiết
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        pred_date = st.date_input("🗓️ Chọn ngày dự đoán", datetime.date.today() + datetime.timedelta(days=1))
    with p_col2:
        pred_weather = st.selectbox("☁️ Dự báo thời tiết", ["Clear", "Rain", "Snow", "Fog"])
    with p_col3:
        pred_temp = st.slider("🌡️ Nhiệt độ (°C)", -10, 45, 25)

    if st.button("🚀 Chạy mô hình dự đoán (Mock)", type="primary"):
        with st.spinner("Đang chạy Inference Model..."):
            # Dữ liệu giả lập (Mock data) chờ bạn ghép Model thật
            mock_lat = np.random.uniform(30.0, 47.0, 150) 
            mock_lng = np.random.uniform(-115.0, -80.0, 150)
            mock_risk = np.random.randint(60, 100, 150) 
            df_pred = pd.DataFrame({'Lat': mock_lat, 'Lng': mock_lng, 'Risk(%)': mock_risk})
            
            st.success(f"Hoàn thành dự đoán cho {pred_date.strftime('%d/%m/%Y')}!")
            
            # Bản đồ dự đoán: Nền tối, điểm nóng đỏ rực
            fig_pred = px.scatter_mapbox(
                df_pred, lat="Lat", lon="Lng", color="Risk(%)",
                color_continuous_scale="Inferno", size="Risk(%)", size_max=15,
                zoom=3.8, mapbox_style="carto-darkmatter", # Bản đồ predict dùng nền tối cho "ngầu"
                center=dict(lat=39.8, lon=-98.5), height=550
            )
            fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            
            # Vẫn bật scrollZoom cho bản đồ Predict
            st.plotly_chart(fig_pred, use_container_width=True, config={'scrollZoom': True})
            
            st.write("📊 **Top khu vực rủi ro cao nhất:**")
            st.dataframe(df_pred.sort_values(by="Risk(%)", ascending=False).head(), use_container_width=True)
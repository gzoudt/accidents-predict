import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================
# 1. CẤU HÌNH TRANG & THEME TÙY CHỈNH (CSS)
# =========================================
st.set_page_config(page_title="US Accidents Analysis", layout="wide", page_icon="🚗")

# --- CSS Injection: Tạo Background chuyên nghiệp ---
st.markdown("""
    <style>
    /* Đổi màu nền toàn trang thành tối chuyên nghiệp */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Chỉnh màu cho Sidebar */
    [data-testid="stSidebar"] { background-color: #1E1E2E; }
    </style>
""", unsafe_allow_html=True)

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
# 4. GIAO DIỆN CHÍNH (PAGE 1)
# =========================================
st.title("🌐 US Traffic Accidents Analysis")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
total = len(filtered_df)
k1.metric("Total Accidents", f"{total:,}")
k2.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f}")
k4.metric("States Affected", f"{filtered_df['State'].nunique()}")

st.markdown("---")
st.subheader("🗺️ Accident Location Map")

# CONTROL PANEL CHO BẢN ĐỒ
col_map_type, col_slider = st.columns([1, 2])
with col_map_type:
    map_style_choice = st.radio("Map Type:", ["Scatter Map", "Heatmap"], horizontal=True)
with col_slider:
    map_points = st.slider("Points to display:", 5000, min(200000, total), min(30000, total), 5000)

map_sample = filtered_df.sample(n=map_points, random_state=42)

# VẼ BẢN ĐỒ VỚI SCROLL ZOOM
if map_style_choice == "Scatter Map":
    fig_map = px.scatter_mapbox(
        map_sample, lat="Start_Lat", lon="Start_Lng", color="Severity",
        color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'},
        size_max=10, zoom=4.0, 
        mapbox_style="open-street-map", 
        height=600, center=dict(lat=39.8, lon=-98.5),
        hover_name="State_Full_Name"
    )
else:
    fig_map = px.density_mapbox(
        map_sample, lat='Start_Lat', lon='Start_Lng', z='Severity', radius=8,
        center=dict(lat=39.8, lon=-98.5), zoom=4.0,
        mapbox_style="open-street-map", height=600
    )
    
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Bật tính năng cuộn chuột để zoom
st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

st.markdown("---")
if 'Year_Month' in filtered_df.columns:
    trend_df = filtered_df.groupby('Year_Month').size().reset_index(name='Count')
    fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="Accidents Trend")
    st.plotly_chart(fig_trend, use_container_width=True)
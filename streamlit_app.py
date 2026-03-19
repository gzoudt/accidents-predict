import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# 1. CẤU HÌNH TRANG & THEME
# =========================================
st.set_page_config(page_title="US Accidents Analysis", layout="wide", page_icon="🚗")

# Chuyển mặc định của Plotly sang Dark Theme để biểu đồ trông chuyên nghiệp hơn
px.defaults.template = "plotly_dark"

# Từ điển ánh xạ mã bang -> tên đầy đủ (dùng cho chú thích)
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

st.title("🌐 US Accidents Analysis Dashboard")
st.markdown("---")

# =========================================
# 2. LOAD DATA (TỐI ƯU HÓA & CACHE PARQUET)
# =========================================
@st.cache_data(show_spinner="Đang tải dữ liệu tốc độ cao... 🚀")
def load_data():
    file_path = "dashboard_data.parquet" 
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file dữ liệu `{file_path}` ở thư mục gốc.")
        st.stop()

    if 'Temperature(F)' in df.columns:
        df['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5.0 / 9.0
    if 'Distance(mi)' in df.columns:
        df['Distance(km)'] = df['Distance(mi)'] * 1.60934
    if 'Visibility(mi)' in df.columns:
        df['Visibility(km)'] = df['Visibility(mi)'] * 1.60934
    if 'Precipitation(in)' in df.columns:
        df['Precipitation(mm)'] = df['Precipitation(in)'] * 25.4
    
    df = df.drop(columns=[c for c in ['Temperature(F)', 'Distance(mi)', 'Visibility(mi)', 'Precipitation(in)'] if c in df.columns])
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Year', 'Hour', 'City', 'State'])
    df = df.reset_index(drop=True)

    # Ép kiểu
    df['Year'] = df['Year'].astype('int16')
    df['Hour'] = df['Hour'].astype('int8')
    df['Severity'] = df['Severity'].astype('int8')
    df['Junction'] = df['Junction'].fillna(False).astype(bool)

    # Thêm cột tên đầy đủ của bang dùng cho hover
    df['State_Full_Name'] = df['State'].map(US_STATE_NAMES)

    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(int) 
        df['Year_Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
        df['Year_Month'] = pd.to_datetime(df['Year_Month'], format='%Y-%m')

    return df

df = load_data()

# =========================================
# 3. BỘ LỌC TƯƠNG TÁC (TOP & SIDEBAR)
# =========================================
st.markdown("#### Which years are you interested in?")
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())

start_year, end_year = st.slider(
    "Select a range of years",
    min_value=min_year, max_value=max_year,
    value=(min_year, max_year), step=1
)

# Sidebar Filter Panels
st.sidebar.header("Filter Panels")
severity_options = sorted(df['Severity'].unique())
selected_severities = st.sidebar.multiselect("⚠️ Mức độ nghiêm trọng", severity_options, default=severity_options)

available_states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect("Select States (Abbr.):", options=available_states, default=available_states)

filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["Year"] >= start_year) & (filtered_df["Year"] <= end_year)]
if selected_severities:
    filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severities)]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

if filtered_df.empty:
    st.warning("Không tìm thấy dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

# =========================================
# 4. TRANG 1: PHÂN TÍCH TÍCH HỢP
# =========================================
tabs = st.tabs(["Trang 1: Phân tích tích hợp"])

with tabs[0]:
    
    # --- 4.1 KPIs (Ảnh 2) ---
    st.subheader(f"🌐 US Accidents Overview: {start_year} - {end_year}")
    k1, k2, k3, k4 = st.columns(4)
    total = len(filtered_df)
    with k1: st.metric("Total Accidents", f"{total:,}")
    with k2: st.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f} / 4")
    with k3: st.metric("Junction Pct", f"{(filtered_df['Junction'].sum()/total)*100:.1f}%")
    with k4: st.metric("States", f"{filtered_df['State'].nunique()}")

    st.markdown("---")

    # --- 4.2 BẢN ĐỒ (Mapbox) - THÊM TƯƠNG TÁC & CHÚ THÍCH BANG ---
    st.subheader("🗺️ Accident Location Map")
    
    # --- YÊU CẦU 1: THÊM TƯƠNG TÁC (SLIDER CHỌN SỐ ĐIỂM) ---
    st.caption("Customize the map detail by adjusting the number of data points. More points may reduce map performance.")
    
    # Slider cho người dùng chọn số lượng điểm muốn vẽ
    max_sample_allowed = min(200000, len(filtered_df)) # Giới hạn tối đa 200k điểm để tránh crash
    map_points = st.slider(
        "Number of map points to display:",
        min_value=5000,
        max_value=max_sample_allowed,
        value=min(30000, len(filtered_df)), # Mặc định 30k điểm
        step=5000
    )
    st.write(f"Displaying **{map_points:,}** points out of **{len(filtered_df):,}** filtered accidents.")

    # Lấy mẫu dựa trên giá trị slider
    map_sample = filtered_df.sample(n=map_points, random_state=42)
    
    fig_map = px.scatter_mapbox(
        map_sample,
        lat="Start_Lat",
        lon="Start_Lng",
        color="Severity",
        color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'},
        size_max=10,
        zoom=4.5, # Đã tăng độ zoom từ 3 lên 4.5
        mapbox_style="carto-darkmatter", # Đổi sang nền bản đồ tối để khớp với giao diện
        height=650,
        center=dict(lat=39.8283, lon=-98.5795),
        
        # --- YÊU CẦU 2: CHÚ THÍCH TÊN BANG (FULL NAME) TRONG HOVER ---
        hover_name="State_Full_Name", 
        hover_data={
            "State": True, 
            "City": True,
            "Weather_Condition": True,
            "Temperature(C)": ":.1f",
            "Severity": True,
            "Start_Lat": False, 
            "Start_Lng": False 
        }
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # --- 4.3 BIỂU ĐỒ ĐƯỜNG (Xu hướng thời gian) ---
    st.subheader("📈 Accident Trends over Time")
    plot_type = st.radio("Show trend by:", ["Total", "Top States"], horizontal=True)

    if 'Year_Month' in filtered_df.columns:
        if plot_type == "Total":
            trend_df = filtered_df.groupby('Year_Month').size().reset_index(name='Count')
            fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="Total Accidents Trend")
        else:
            top_states = filtered_df['State'].value_counts().nlargest(10).index
            state_trend = filtered_df.groupby(['Year_Month', 'State']).size().reset_index(name='Count')
            fig_trend = px.line(state_trend[state_trend['State'].isin(top_states)], 
                                x='Year_Month', y='Count', color='State', title="Top 10 States Trend")
        
        fig_trend.update_layout(height=450)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # --- 4.4 BIỂU ĐỒ Grid ---
    st.subheader("Chi tiết phân tích khác")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(px.histogram(filtered_df, x='Hour', color='Sunrise_Sunset', nbins=24, title="By Hour"), use_container_width=True)
    with g2:
        st.plotly_chart(px.histogram(filtered_df[filtered_df['Weather_Condition'].isin(filtered_df['Weather_Condition'].value_counts().nlargest(15).index)], 
                                     x='Weather_Condition', title="Top Weather").update_xaxes(categoryorder='total descending'), use_container_width=True)
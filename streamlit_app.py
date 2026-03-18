import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# 1. CẤU HÌNH TRANG & THEME
# =========================================
st.set_page_config(page_title="US Accidents Analysis", layout="wide", page_icon="🚗")

# Sử dụng theme Plotly trắng để sạch sẽ và giống GDP dashboard
px.defaults.template = "plotly_white"

st.title("🌐 US Accidents Analysis Dashboard")
st.markdown("---")

# =========================================
# 2. LOAD DATA (TỐI ƯU HÓA & CACHE PARQUET)
# =========================================
@st.cache_data(show_spinner="Đang tải dữ liệu tốc độ cao... 🚀")
def load_data():
    file_path = "dashboard_data.parquet"
    df = pd.read_parquet(file_path)

    # Quy đổi và làm sạch (từ code gốc của bạn)
    if 'Temperature(F)' in df.columns:
        df['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5.0 / 9.0
    if 'Distance(mi)' in df.columns:
        df['Distance(km)'] = df['Distance(mi)'] * 1.60934
    if 'Visibility(mi)' in df.columns:
        df['Visibility(km)'] = df['Visibility(mi)'] * 1.60934
    if 'Precipitation(in)' in df.columns:
        df['Precipitation(mm)'] = df['Precipitation(in)'] * 25.4
    
    # Drop columns cũ và làm sạch NA
    df = df.drop(columns=[c for c in ['Temperature(F)', 'Distance(mi)', 'Visibility(mi)', 'Precipitation(in)'] if c in df.columns])
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Year', 'Hour', 'City', 'State'])
    df = df.reset_index(drop=True)

    # Ép kiểu tối ưu
    df['Year'] = df['Year'].astype('int16')
    df['Hour'] = df['Hour'].astype('int8')
    df['Severity'] = df['Severity'].astype('int8')
    df['Junction'] = df['Junction'].fillna(False).astype(bool)

    # Tạo cột Year_Month để làm biểu đồ đường
    if 'Month' in df.columns:
        df['Year_Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
        df['Year_Month'] = pd.to_datetime(df['Year_Month']) # Ép sang Datetime

    return df

df = load_data()

# =========================================
# 3. BỘ LỌC TƯƠNG TÁC (SIDEBAR & TOP)
# =========================================

# --- 3.1 THANH KÉO THỜI GIAN (Range Slider) - GIỐNG ẢNH 1 ---
st.markdown("#### Which years are you interested in?")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

# Tạo slider chọn khoảng năm
start_year, end_year = st.slider(
    "Select a range of years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year), # Mặc định chọn toàn bộ
    step=1,
    help="Drag the sliders to select the start and end year for analysis."
)
st.caption(f"Analyzing accidents from {start_year} to {end_year}")

# --- 3.2 BỘ LỌC SEVERITY & STATE ---
st.sidebar.header("Filter Panels")

# State multiselect (Giống country selector ảnh 1)
available_states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect(
    "Select States to view:",
    options=available_states,
    default=available_states, # Mặc định chọn tất cả
    help="Select one or more states to filter the data."
)

severity_options = sorted(df['Severity'].unique())
selected_severities = st.sidebar.multiselect(
    "⚠️ Mức độ nghiêm trọng",
    severity_options, 
    default=severity_options
)

# Áp dụng bộ lọc tổng thể
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df["Year"] >= start_year) & 
    (filtered_df["Year"] <= end_year)
]
if selected_severities:
    filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severities)]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

if filtered_df.empty:
    st.warning("Không tìm thấy dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

# =========================================
# 4. TRANG 1: PHÂN TÍCH HIỆN TẠI (INTEGRATED ANALYSIS)
# =========================================
tab_analysis, = st.tabs(["Trang 1: Phân tích tích hợp"])

with tab_analysis:
    
    # --- 4.1 KPIs (THÔNG SỐ GIỐNG ẢNH 2) ---
    st.subheader(f"🌐 US Accidents Overview: {start_year} - {end_year}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_accidents = len(filtered_df)
    avg_severity = filtered_df['Severity'].mean()
    junction_pct = (filtered_df['Junction'].sum() / total_accidents) * 100 if total_accidents > 0 else 0
    total_states = filtered_df['State'].nunique()

    with kpi1:
        # Style metric chuẩn Streamlit
        st.metric(label="Total Accidents", value=f"{total_accidents:,}", help="Tổng số vụ tai nạn trong khoảng thời gian đã chọn")
    with kpi2:
        st.metric(label="Avg Severity", value=f"{avg_severity:.2f} / 4", help="Mức độ nghiêm trọng trung bình (scale 1-4)")
    with kpi3:
        st.metric(label="Junction Accidents Pct", value=f"{junction_pct:.1f}%", help="Phần trăm tai nạn xảy ra tại nút giao thông")
    with kpi4:
        st.metric(label="States Analyzed", value=f"{total_states}", help="Số lượng bang đang được hiển thị trên dashboard")

    st.markdown("---")

    col_map, col_chart = st.columns([2, 1])

    # --- 4.2 BẢN ĐỒ TƯƠNG TÁC (MAPBOX) - SỬA LỖI TƯƠNG TÁC ---
    with col_map:
        st.subheader("🗺️ Accident Map (Zoom & Interacive)")
        st.caption("Map is using a data sample for performance. You can fully interact (zoom/pan).")
        
        # Giới hạn dữ liệu để bản đồ không bị lag, nhưng vẫn đảm bảo có đủ điểm để nhìn
        map_sample_size = 30000 
        if len(filtered_df) > map_sample_size:
            map_data = filtered_df.sample(n=map_sample_size, random_state=42)
        else:
            map_data = filtered_df

        center_us = dict(lat=39.8283, lon=-98.5795)

        # Sử dụng px.scatter_mapbox để có hiệu suất tốt hơn st.map và có thể tương tác
        fig_map = px.scatter_mapbox(
            map_data,
            lat="Start_Lat",
            lon="Start_Lng",
            color="Severity",
            # Thang màu severity đỏ để nổi bật
            color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'},
            size_max=10,
            zoom=3,
            mapbox_style="open-street-map",
            hover_name="City",
            hover_data={"Weather_Condition": True, "Temperature(C)": ":.1f"},
            height=600,
            center=center_us
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        # Thêm theme="streamlit" để map đồng bộ với theme nền
        st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")

    # --- 4.3 BIỂU ĐỒ ĐƯỜNG THEO THỜI GIAN (XU HƯỚNG GIỐNG ẢNH 1) ---
    with col_chart:
        st.subheader("📈 Accident Trends over Time")
        
        # Aggregate dữ liệu theo tháng/năm để vẽ đường
        if 'Year_Month' in filtered_df.columns:
            trend_df = filtered_df.groupby('Year_Month').size().reset_index(name='Count')
            # Thêm bộ lọc State vào trend df để vẽ nhiều đường
            state_trend_df = filtered_df.groupby(['Year_Month', 'State']).size().reset_index(name='Count')
            
            # Chọn loại biểu đồ
            plot_type = st.radio("Show trend by:", ["Total", "Top States"], horizontal=True)

            if plot_type == "Total":
                fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="Total Accidents Trend")
            else:
                # Vẽ biểu đồ đường nhiều đường (multiple lines) cho các bang hàng đầu
                top_states = filtered_df['State'].value_counts().nlargest(10).index
                fig_trend = px.line(
                    state_trend_df[state_trend_df['State'].isin(top_states)],
                    x='Year_Month', 
                    y='Count', 
                    color='State', 
                    title="Trends for Top 10 States"
                )
            
            fig_trend.update_layout(height=520, margin={"r":10,"t":40,"l":10,"b":10})
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Cột dữ liệu 'Month' không tồn tại. Không thể vẽ biểu đồ xu hướng theo thời gian.")

    st.markdown("---")

    # --- 4.4 BIỂU ĐỒ GRID (Phân tích chi tiết khác) ---
    st.subheader("Chi tiết phân tích khác")
    grid_col1, grid_col2 = st.columns(2)

    with grid_col1:
        fig_hour = px.histogram(filtered_df, x='Hour', color='Sunrise_Sunset', nbins=24, title="Tai nạn theo giờ và Ngày/Đêm")
        st.plotly_chart(fig_hour, use_container_width=True)
        
        top_cities = filtered_df['City'].value_counts().nlargest(20).index
        fig_city = px.histogram(filtered_df[filtered_df['City'].isin(top_cities)], x='City', title="Top Cities").update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_city, use_container_width=True)

    with grid_col2:
        top_weather = filtered_df['Weather_Condition'].value_counts().nlargest(15).index
        fig_weather = px.histogram(filtered_df[filtered_df['Weather_Condition'].isin(top_weather)], x='Weather_Condition', title="Top Weather Conditions").update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_weather, use_container_width=True)
        
        infra_df = pd.DataFrame({
            "Type": ["Junction", "Traffic_Signal"],
            "Count": [filtered_df['Junction'].sum(), filtered_df['Traffic_Signal'].sum()]
        })
        fig_infra = px.bar(infra_df, x="Type", y="Count", title="Infrastructure Analysis")
        st.plotly_chart(fig_infra, use_container_width=True)

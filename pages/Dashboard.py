import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="US Accidents Dashboard", layout="wide", page_icon="📊")

# Áp dụng nền trắng mặc định
px.defaults.template = "plotly_white"

# =========================================
# CSS: ĐÓNG KHUNG KPI ĐẸP MẮT (LIGHT MODE)
# =========================================
st.markdown("""
<style>
    /* Tự động đóng khung tất cả các thẻ st.metric */
    div[data-testid="stMetric"] {
        border: 1px solid #dcdcdc;     /* Viền xám nhẹ */
        border-radius: 10px;           /* Bo góc 10px */
        padding: 15px 20px;            /* Khoảng cách chữ và viền */
        background-color: #ffffff;     /* Nền trắng tinh */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.04); /* Bóng đổ cực mờ tạo chiều sâu nhẹ */
    }
</style>
""", unsafe_allow_html=True)

US_STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

@st.cache_data(show_spinner="Đang tải dữ liệu...")
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
        df = df.dropna(subset=['Month'])
        df['Month'] = df['Month'].astype(int) 
        df['Year_Month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2), format='%Y-%m', errors='coerce')
        
    return df

df = load_data()

# =========================================
# BỘ LỌC SIDEBAR
# =========================================
st.sidebar.title("Filter Panels")

with st.sidebar.form(key='filter_form'):
    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    start_year, end_year = st.slider("Select Years", min_year, max_year, (min_year, max_year))
    severity_opts = sorted(df['Severity'].unique())
    selected_sev = st.multiselect("Severity", severity_opts, default=severity_opts)
    available_states = sorted(df['State'].unique())
    selected_states = st.multiselect("States", options=available_states, default=available_states)
    submit_button = st.form_submit_button(label='Apply Filters')

filtered_df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
if selected_sev: filtered_df = filtered_df[filtered_df['Severity'].isin(selected_sev)]
if selected_states: filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()

st.sidebar.markdown("---")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="Download Data (CSV)", data=csv_data, file_name='filtered_accidents.csv', mime='text/csv')

# =========================================
# GIAO DIỆN CHÍNH (KPIS CÓ KHUNG)
# =========================================
st.title("📊 US Traffic Accidents Dashboard")
st.markdown("---")

# Các thông số này sẽ tự động được đóng khung nhờ đoạn CSS ở đầu trang
k1, k2, k3, k4 = st.columns(4)
total_count = len(filtered_df)
k1.metric("Total Accidents", f"{total_count:,}")
k2.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f}")
k3.metric("Cities Covered", f"{filtered_df['City'].nunique():,}")
k4.metric("States Affected", f"{filtered_df['State'].nunique()}")

st.markdown("---")

# =========================================
# BIỂU ĐỒ THỐNG KÊ (PIE CHART ĐÃ SỬA LỖI)
# =========================================
st.subheader("Accident Statistics Overview")

col_stat1, col_stat2 = st.columns(2)
with col_stat1:
    sev_counts = filtered_df['Severity'].value_counts().reset_index()
    sev_counts.columns = ['Severity', 'Count']
    total_sev = sev_counts['Count'].sum()
    sev_counts['Percent'] = (sev_counts['Count'] / total_sev) * 100
    
    # Gộp % vào ghi chú (Legend)
    sev_counts['Legend_Label'] = sev_counts.apply(lambda row: f"Severity {row['Severity']} ({row['Count']:,} | {row['Percent']:.1f}%)", axis=1)
    sev_counts = sev_counts.sort_values(by='Severity') 
    
    fig_sev = px.pie(
        sev_counts, names='Legend_Label', values='Count', 
        title="Severity Distribution", hole=0.5,
        color_discrete_sequence=['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    )
    # Tắt chữ trên biểu đồ để khỏi tràn viền, đặt bảng ghi chú gọn gàng
    fig_sev.update_traces(textinfo='none', hovertemplate='Severity: %{customdata[0]}<br>Count: %{value:,}<br>Percentage: %{customdata[1]:.1f}%', customdata=sev_counts[['Severity', 'Percent']])
    
    # Sử dụng "middle" để không bị lỗi ValueError
    fig_sev.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0))
    
    st.plotly_chart(fig_sev, use_container_width=True)

with col_stat2:
    state_counts = filtered_df['State_Full_Name'].value_counts().nlargest(10).reset_index()
    state_counts.columns = ['State', 'Count']
    fig_state = px.bar(state_counts, x='State', y='Count', title="Top 10 States with Most Accidents", text_auto='.2s', color='Count', color_continuous_scale='Reds')
    fig_state.update_layout(xaxis_title="State", yaxis_title="Number of Accidents")
    st.plotly_chart(fig_state, use_container_width=True)

st.markdown("---")

# =========================================
# BẢN ĐỒ KẾT QUẢ
# =========================================
st.subheader("Accident Location Map")
col_map_type, col_slider = st.columns([1, 2])
with col_map_type:
    map_style_choice = st.radio("Map Type:", ["Scatter Map", "Heatmap", "Animated (By Hour)"], horizontal=True)
with col_slider:
    max_allowed = min(200000, total_count) if total_count > 5000 else total_count
    def_val = min(10000, total_count)
    map_points = st.slider("Points to display:", min_value=1000 if total_count > 1000 else total_count, max_value=max_allowed, value=def_val, step=1000)

map_sample = filtered_df.sample(n=map_points, random_state=42)

if map_style_choice == "Scatter Map":
    fig_map = px.scatter_mapbox(map_sample, lat="Start_Lat", lon="Start_Lng", color="Severity", color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'}, size_max=10, zoom=4.0, mapbox_style="open-street-map", height=650, center=dict(lat=39.8, lon=-98.5), hover_name="State_Full_Name")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
elif map_style_choice == "Heatmap":
    fig_map = px.density_mapbox(map_sample, lat='Start_Lat', lon='Start_Lng', z='Severity', radius=8, center=dict(lat=39.8, lon=-98.5), zoom=4.0, mapbox_style="open-street-map", height=650, color_continuous_scale="Reds")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
else:
    map_sample_anim = map_sample.sort_values("Hour") 
    fig_map = px.scatter_mapbox(map_sample_anim, lat="Start_Lat", lon="Start_Lng", color="Severity", animation_frame="Hour", color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'}, size_max=10, zoom=4.0, mapbox_style="open-street-map", height=650, center=dict(lat=39.8, lon=-98.5))
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":80})

st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

st.markdown("---")

# =========================================
# BIỂU ĐỒ XU HƯỚNG VÀ PHỤ
# =========================================
st.subheader("Accident Trends over Time")
plot_type = st.radio("Show trend by:", ["Total", "Top 10 Cities"], horizontal=True)

if 'Year_Month' in filtered_df.columns:
    if plot_type == "Total":
        trend_df = filtered_df.groupby('Year_Month').size().reset_index(name='Count')
        fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="Total Accidents Trend")
    else:
        top_cities = filtered_df['City'].value_counts().nlargest(10).index
        city_trend_df = filtered_df[filtered_df['City'].isin(top_cities)].groupby(['Year_Month', 'City']).size().reset_index(name='Count')
        fig_trend = px.line(city_trend_df, x='Year_Month', y='Count', color='City', title="Top 10 Cities Trend")
    
    fig_trend.update_layout(height=450, xaxis_title="Time", yaxis_title="Number of Accidents")
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.subheader("Supplementary Analysis")
g1, g2 = st.columns(2)

with g1:
    fig_hour = px.histogram(filtered_df, x='Hour', nbins=24, title="Accidents by Hour", color_discrete_sequence=['#636EFA'])
    fig_hour.update_layout(xaxis_title="Hour (0-23)", yaxis_title="Count", bargap=0.1)
    st.plotly_chart(fig_hour, use_container_width=True)

with g2:
    if 'Weather_Condition' in filtered_df.columns:
        top_weather = filtered_df['Weather_Condition'].value_counts().nlargest(15).index
        weather_df = filtered_df[filtered_df['Weather_Condition'].isin(top_weather)]
        fig_weather = px.histogram(weather_df, x='Weather_Condition', title="Accidents by Weather", color_discrete_sequence=['#EF553B']).update_xaxes(categoryorder='total descending')
        fig_weather.update_layout(xaxis_title="Weather Condition", yaxis_title="Count")
        st.plotly_chart(fig_weather, use_container_width=True)
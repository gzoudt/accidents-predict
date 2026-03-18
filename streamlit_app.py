import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================
# 1. CẤU HÌNH TRANG & THEME
# =========================================
st.set_page_config(page_title="US Accidents Dashboard", layout="wide", page_icon="🚗")

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Reds

st.title("🚗 US Accidents Dashboard")
st.markdown("Phân tích dữ liệu tai nạn giao thông trên toàn nước Mỹ.")

# =========================================
# 2. LOAD DATA (TỐI ƯU HÓA & CACHE VỚI PARQUET)
# =========================================
# Đường dẫn trỏ thẳng đến file parquet bạn vừa upload
FILE_PATH = "dashboard_data.parquet" 

@st.cache_data(show_spinner="Đang tải dữ liệu tốc độ cao... 🚀")
def load_data():
    try:
        # Đọc thẳng file Parquet (Cực kỳ nhanh và nhẹ, không cần chia chunk)
        df_full = pd.read_parquet(FILE_PATH)

        # Quy đổi đại lượng
        if 'Temperature(F)' in df_full.columns:
            df_full['Temperature(C)'] = (df_full['Temperature(F)'] - 32) * 5.0 / 9.0
            df_full = df_full.drop(columns=['Temperature(F)'])
        if 'Distance(mi)' in df_full.columns:
            df_full['Distance(km)'] = df_full['Distance(mi)'] * 1.60934
            df_full = df_full.drop(columns=['Distance(mi)'])
        if 'Visibility(mi)' in df_full.columns:
            df_full['Visibility(km)'] = df_full['Visibility(mi)'] * 1.60934
            df_full = df_full.drop(columns=['Visibility(mi)'])
        if 'Precipitation(in)' in df_full.columns:
            df_full['Precipitation(mm)'] = df_full['Precipitation(in)'] * 25.4
            df_full = df_full.drop(columns=['Precipitation(in)'])

        # Làm sạch và ép kiểu
        df_full = df_full.dropna(subset=['Start_Lat', 'Start_Lng', 'Year', 'Month', 'Day', 'Hour', 'City'])
        df_full = df_full.reset_index(drop=True)
        df_full["Accident_ID"] = df_full.index.astype(int)

        df_full['Year'] = df_full['Year'].astype('int16')
        df_full['Month'] = df_full['Month'].astype('int8')
        df_full['Day'] = df_full['Day'].astype('int8')
        df_full['Hour'] = df_full['Hour'].astype('int8')
        df_full['Weekday'] = df_full['Weekday'].astype('int8')
        df_full['Junction'] = df_full['Junction'].fillna(False).astype(bool)
        df_full['Traffic_Signal'] = df_full['Traffic_Signal'].fillna(False).astype(bool)

        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df_full['Day_of_Week'] = df_full['Weekday'].map(day_mapping).astype('category')
        df_full['Year_Month'] = df_full['Year'].astype(str) + '-' + df_full['Month'].astype(str).str.zfill(2)
        
        return df_full
    except FileNotFoundError:
        st.error(f"Không tìm thấy file `{FILE_PATH}`. Hãy đảm bảo bạn đã upload file dữ liệu lên GitHub cùng thư mục với code.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi đọc file Parquet: {e}")
        st.stop()

df_full = load_data()

# =========================================
# 3. GIAO DIỆN BỘ LỌC (FILTERS)
# =========================================
st.markdown("---")
col1, col2 = st.columns(2)

available_years = sorted(df_full["Year"].unique())
with col1:
    selected_year = st.selectbox("📅 Select Year", options=["All"] + available_years)

with col2:
    available_severities = sorted(df_full['Severity'].unique())
    selected_severities = st.multiselect("⚠️ Severity Level", options=available_severities, default=available_severities)

# Áp dụng bộ lọc
filtered_df = df_full.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]
if selected_severities:
    filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severities)]

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

# =========================================
# 4. BẢN ĐỒ (MAP)
# =========================================
st.subheader("🗺️ Accident Map")
map_type = st.radio("Chọn loại bản đồ:", ['Scatter Map', 'Heatmap'], horizontal=True)

# Lấy mẫu dữ liệu cho Map để tránh lag trình duyệt
map_data = filtered_df.sample(50000, random_state=42) if len(filtered_df) > 50000 else filtered_df
center_us = dict(lat=39.8283, lon=-98.5795)

if map_type == 'Scatter Map':
    fig_map = px.scatter_map(map_data, lat="Start_Lat", lon="Start_Lng", color="Severity", 
                             hover_name="City", hover_data={"Weather_Condition": True, "Temperature(C)": ":.1f"}, 
                             height=550, map_style="open-street-map", center=center_us)
else:
    fig_map = px.density_map(map_data, lat='Start_Lat', lon='Start_Lng', z='Severity', 
                             radius=6, height=550, map_style="open-street-map", center=center_us)

fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# =========================================
# 5. LƯỚI BIỂU ĐỒ (CHARTS GRID)
# =========================================
st.markdown("---")

# Hàng 1: Trend
trend_df = filtered_df.groupby('Year_Month', observed=True).size().reset_index(name='Count').sort_values('Year_Month')
fig_trend = px.line(trend_df, x='Year_Month', y='Count', title="📈 Accidents Trend", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# Hàng 2: Day of Week & Day of Month
c1, c2 = st.columns(2)
with c1:
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_df = filtered_df['Day_of_Week'].value_counts().reindex(days_order).reset_index(name='Count')
    fig_day = px.bar(day_df, x='Day_of_Week', y='Count', title="Accidents by Day of Week", color='Count')
    st.plotly_chart(fig_day, use_container_width=True)
with c2:
    day_m_df = filtered_df['Day'].value_counts().sort_index().reset_index(name='Count')
    fig_day_month = px.bar(day_m_df, x='Day', y='Count', title="Accidents by Day of Month", color='Count')
    st.plotly_chart(fig_day_month, use_container_width=True)

# Hàng 3: Hour & Severity
c3, c4 = st.columns(2)
with c3:
    fig_hour_sun = px.histogram(filtered_df, x='Hour', color='Sunrise_Sunset', nbins=24, title="Accidents by Day/Night")
    st.plotly_chart(fig_hour_sun, use_container_width=True)
with c4:
    fig_sev = px.histogram(filtered_df, x='Severity', color='Severity', title="Severity Distribution")
    st.plotly_chart(fig_sev, use_container_width=True)

# Hàng 4: Weather & City
c5, c6 = st.columns(2)
with c5:
    top_weather = filtered_df['Weather_Condition'].value_counts().nlargest(15).index
    fig_weather = px.histogram(filtered_df[filtered_df['Weather_Condition'].isin(top_weather)], x='Weather_Condition', title="Top Weather Conditions").update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_weather, use_container_width=True)
with c6:
    top_cities = filtered_df['City'].value_counts().nlargest(20).index
    fig_city = px.histogram(filtered_df[filtered_df['City'].isin(top_cities)], x='City', title="Top Cities").update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_city, use_container_width=True)

# Hàng 5: Infrastructure & Duration
c7, c8 = st.columns(2)
with c7:
    fig_infra = px.bar(pd.DataFrame({"Type": ["Junction", "Traffic_Signal"], "Count": [filtered_df['Junction'].sum(), filtered_df['Traffic_Signal'].sum()]}), x='Type', y='Count', title="Infrastructure Analysis")
    st.plotly_chart(fig_infra, use_container_width=True)
with c8:
    dur_filtered = filtered_df[filtered_df['Duration'] <= 1440]
    fig_duration = px.histogram(dur_filtered, x='Duration', nbins=30, title="Duration Distribution (Minutes)")
    st.plotly_chart(fig_duration, use_container_width=True)

# Hàng 6: Môi trường (Humidity, Visibility, Precip)
c9, c10, c11 = st.columns(3)
with c9:
    fig_humidity = px.histogram(filtered_df, x='Humidity(%)', nbins=20, title="Humidity (%)")
    st.plotly_chart(fig_humidity, use_container_width=True)
with c10:
    fig_visibility = px.histogram(filtered_df, x='Visibility(km)', nbins=20, title="Visibility (km)", log_y=True)
    st.plotly_chart(fig_visibility, use_container_width=True)
with c11:
    rain_filtered = filtered_df[filtered_df['Precipitation(mm)'] > 0]
    fig_precip = px.histogram(rain_filtered, x='Precipitation(mm)', nbins=30, title="Precipitation (mm)", log_y=True)
    st.plotly_chart(fig_precip, use_container_width=True)
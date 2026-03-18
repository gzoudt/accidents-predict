import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# =========================================
# 1. CẤU HÌNH TRANG & THEME
# =========================================
st.set_page_config(page_title="US Accidents Dashboard", layout="wide", page_icon="🚗")

# Sử dụng theme Plotly mặc định hoặc 'plotly_white' để sạch sẽ
px.defaults.template = "plotly_white"

st.title("🚗 US Accidents Dashboard")
st.markdown("Phân tích dữ liệu tai nạn giao thông trên toàn nước Mỹ.")

# =========================================
# 2. LOAD DATA (TỐI ƯU HÓA & CACHE PARQUET)
# =========================================

# Sử dụng @st.cache_data để Streamlit chỉ load dữ liệu 1 lần duy nhất, tăng hiệu suất
@st.cache_data(show_spinner="Đang tải dữ liệu... Vui lòng đợi nhé!")
def load_data():
    # Sử dụng tệp parquet đã tải lên
    file_path = "dashboard_data.parquet"
    
    # Đọc dữ liệu (Parquet tự xử lý kiểu dữ liệu, nên không cần dtypes_optimized phức tạp)
    df_full = pd.read_parquet(file_path)

    # Logic chuyển đổi và làm sạch từ mã Dash gốc của bạn
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

    df_full = df_full.dropna(subset=['Start_Lat', 'Start_Lng', 'Year', 'Month', 'Day', 'Hour', 'City'])
    df_full = df_full.reset_index(drop=True)
    df_full["Accident_ID"] = df_full.index.astype(int)

    # Ép kiểu để tối ưu hóa bộ nhớ
    df_full['Year'] = df_full['Year'].astype('int16')
    df_full['Month'] = df_full['Month'].astype('int8')
    df_full['Day'] = df_full['Day'].astype('int8')
    df_full['Hour'] = df_full['Hour'].astype('int8')
    df_full['Weekday'] = df_full['Weekday'].astype('int8')
    df_full['Junction'] = df_full['Junction'].fillna(False).astype(bool)
    df_full['Traffic_Signal'] = df_full['Traffic_Signal'].fillna(False).astype(bool)

    # Xử lý ngày tháng để tạo bộ lọc
    day_mapping = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    df_full['Day_of_Week'] = df_full['Weekday'].map(day_mapping).astype('category')
    df_full['Year_Month'] = df_full['Year'].astype(str) + '-' + df_full['Month'].astype(str).str.zfill(2)
    df_full['Year_Month'] = df_full['Year_Month'].astype('category')

    return df_full

df = load_data()

# =========================================
# 3. GIAO DIỆN BỘ LỌC (SIDEBAR FILTERS)
# =========================================
st.sidebar.header("Bộ lọc")

# Bộ lọc năm
available_years = sorted(df["Year"].unique())
selected_year = st.sidebar.selectbox("📅 Chọn năm", ["Tất cả"] + available_years)

# Bộ lọc mức độ nghiêm trọng (Severity)
severity_options = sorted(df['Severity'].unique())
selected_severities = st.sidebar.multiselect("⚠️ Mức độ nghiêm trọng", severity_options, default=severity_options)

# Áp dụng bộ lọc
filtered_df = df.copy()
if selected_year != "Tất cả":
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]
if selected_severities:
    filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severities)]

if filtered_df.empty:
    st.warning("Không tìm thấy dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

# =========================================
# 4. BẢN ĐỒ TƯƠNG TÁC (SCATTER MAPBOX)
# =========================================
st.subheader("🗺️ Bản đồ vị trí tai nạn")

# Giới hạn dữ liệu để bản đồ không bị lag (lấy mẫu 10,000 điểm, bạn có thể tăng nếu mượt)
map_data = filtered_df.sample(n=min(10000, len(filtered_df)), random_state=42)

center_us = dict(lat=39.8283, lon=-98.5795)

# Sử dụng Plotly Express Mapbox để có hiệu suất tốt hơn st.map trên dữ liệu lớn
fig_map = px.scatter_mapbox(
    map_data,
    lat="Start_Lat",
    lon="Start_Lng",
    color="Severity",
    color_discrete_map={1: '#f0f0f0', 2: '#fee0d2', 3: '#fc9272', 4: '#de2d26'}, # Màu Severity
    size_max=15,
    zoom=3,
    mapbox_style="open-street-map",
    # Thêm dữ liệu hover chi tiết
    hover_name="City",
    hover_data={"Weather_Condition": True, "Temperature(C)": ":.1f"}
)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# =========================================
# 5. LƯỚI BIỂU ĐỒ (CHARTS GRID)
# =========================================
st.markdown("---")
col1, col2 = st.columns(2)

# Hàng 1: Phân bố theo giờ và mức độ nghiêm trọng
with col1:
    st.header("Tai nạn theo giờ và Ngày/Đêm")
    fig_hour = px.histogram(filtered_df, x='Hour', color='Sunrise_Sunset', nbins=24, title="Tai nạn theo giờ và Ngày/Đêm")
    st.plotly_chart(fig_hour, use_container_width=True)

with col2:
    st.header("Phân bố mức độ nghiêm trọng")
    fig_sev = px.histogram(filtered_df, x='Severity', color='Severity', title="Mức độ nghiêm trọng")
    st.plotly_chart(fig_sev, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

# Hàng 2: Thời tiết và Cơ sở hạ tầng
with col3:
    st.header("Tai nạn theo điều kiện thời tiết")
    top_weather = filtered_df['Weather_Condition'].value_counts().nlargest(20).index
    fig_weather = px.histogram(filtered_df[filtered_df['Weather_Condition'].isin(top_weather)], x='Weather_Condition', title="Tai nạn theo điều kiện thời tiết")
    # Category order to sort descending
    fig_weather.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_weather, use_container_width=True)

with col4:
    st.header("Phân tích cơ sở hạ tầng")
    # Chuẩn bị dữ liệu cho biểu đồ cơ sở hạ tầng
    infra_df = pd.DataFrame({
        "Cơ sở hạ tầng": ["Nút giao (Junction)", "Tín hiệu giao thông (Traffic Signal)"],
        "Số vụ tai nạn": [filtered_df['Junction'].sum(), filtered_df['Traffic_Signal'].sum()]
    })
    fig_infra = px.bar(infra_df, x="Cơ sở hạ tầng", y="Số vụ tai nạn", title="Cơ sở hạ tầng")
    st.plotly_chart(fig_infra, use_container_width=True)
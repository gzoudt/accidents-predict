import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN
# =========================================
st.set_page_config(page_title="US Accidents Dashboard", layout="wide")
# --- COPY ĐOẠN NÀY CHÈN VÀO ĐẦU CẢ 3 FILE (SAU st.set_page_config) ---
def apply_layered_vibrant_style():
    # Tạo hiệu ứng nền xám nhẹ, các phần (cards) màu trắng có bóng đổ
    # Và hiệu ứng chuyển động fade-in nhẹ khi load trang.
    layered_vibrant_css = """
    <style>
        /* 1. Thiết lập nền trang xám nhẹ để làm nổi bật các lớp */
        [data-testid="stAppViewContainer"] {
            background-color: #f8f9fa;
        }

        /* 2. Tạo hiệu ứng bóng đổ (3D layer) cho metric cards (giống Image 2) */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            /* Tắt bóng đổ mặc định của metrics */
            background-color: transparent !important;
            box-shadow: none !important;
        }
        
        /* Áp dụng bóng đổ cho container của metric cards */
        div[data-testid="stMetricValueContainer"], 
        [data-testid="stElementContainer"] > div[class*="stMetric"] > div,
        [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > div[class*="stMetric"] {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #e9ecef;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        div[data-testid="stMetricValueContainer"]:hover {
             transform: translateY(-3px);
             box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08) !important;
        }
        

        /* 3. Tạo hiệu ứng 3D layer (Shadow) cho toàn bộ các Section */
        /* Chúng ta sẽ bao các section trong st.container() và st.markdown(unsafe_allow_html=True) */
        .layered-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #e9ecef;
            margin-bottom: 25px;
            transition: transform 0.2s ease-in-out;
        }
        .layered-card:hover {
             transform: translateY(-3px);
        }

        /* 4. Tùy chỉnh Sidebar để trông chuyên nghiệp hơn */
        [data-testid="stSidebar"] {
            border-right: 1px solid #e9ecef;
            background-color: #ffffff;
        }

        /* 5. Hiệu ứng chuyển động Fade-in (Sống động hơn) */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Áp dụng fade-in cho Title chính và các thẻ metrics */
        h1, [data-testid="stMetricValueContainer"] {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* 6. Làm cho biểu đồ Plotly vẫn nền trắng chuẩn sạch sẽ */
        .js-plotly-plot .plotly .main-svg {
            border-radius: 8px;
        }

    </style>
    """
    st.markdown(layered_vibrant_css, unsafe_allow_html=True)

# Gọi hàm để áp dụng style
apply_layered_vibrant_style()
# --- KẾT THÚC ĐOẠN COPY ---

px.defaults.template = "plotly_white"

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
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data():
    file_path = "dashboard_data.parquet" 
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file `{file_path}` ở thư mục gốc.")
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
# 3. BỘ LỌC SIDEBAR & ÁP DỤNG STYLE 3D LAYERED
# =========================================
st.sidebar.title("Filter Panels")

# Áp dụng hàm Style đã tạo ở Bước 1 (Bạn phải copy hàm apply_layered_vibrant_style vào đây)
apply_layered_vibrant_style() 

with st.sidebar.form(key='filter_form'):
    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    start_year, end_year = st.slider("Select Years", min_year, max_year, (min_year, max_year))

    severity_opts = sorted(df['Severity'].unique())
    selected_sev = st.multiselect("Severity", severity_opts, default=severity_opts)

    available_states = sorted(df['State'].unique())
    selected_states = st.multiselect("States", options=available_states, default=available_states)

    submit_button = st.form_submit_button(label='Apply Filters')

# Lọc dữ liệu
filtered_df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
if selected_sev: filtered_df = filtered_df[filtered_df['Severity'].isin(selected_sev)]
if selected_states: filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()

st.sidebar.markdown("---")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Data (CSV)",
    data=csv_data,
    file_name='filtered_accidents.csv',
    mime='text/csv',
)

# =========================================
# 4. GIAO DIỆN CHÍNH (PAGE 1)
# =========================================
st.title("📊 US Traffic Accidents Dashboard")
st.markdown("---")

# 4.1 KPIs (Vẫn giữ trắng ban đầu, nhưng có CSS shadow của container bên trên)
k1, k2, k3, k4 = st.columns(4)
total_count = len(filtered_df)
k1.metric("Total Accidents", f"{total_count:,}")
k2.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f}")
k3.metric("Cities Covered", f"{filtered_df['City'].nunique():,}")
k4.metric("States Affected", f"{filtered_df['State'].nunique()}")

st.markdown("---")

# =========================================
# 5. BIỂU ĐỒ THỐNG KÊ TỔNG QUAN (VỚI % SEVERITY TRONG GHI CHÚ)
# =========================================
st.subheader("Accident Statistics Overview")

# Bao section này vào một "Lớp" (Card) 3D
st.markdown("<div class='layered-card'>", unsafe_allow_html=True)
with st.container():
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        # LÔGIC MỚI: Tính toán % và gộp vào chú thích (Legend) màu
        # 1. Tính toán số lượng và %
        sev_counts = filtered_df['Severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity', 'Count']
        total_sev = sev_counts['Count'].sum()
        sev_counts['Percent'] = (sev_counts['Count'] / total_sev) * 100
        
        # 2. Gộp thông tin % và Count vào chú thích (Legend Label)
        sev_counts['Legend_Label'] = sev_counts.apply(
            lambda row: f"Severity {row['Severity']} ({row['Count']:,} - {row['Percent']:.1f}%)",
            axis=1
        )
        # Sắp xếp để màu theo thứ tự
        sev_counts = sev_counts.sort_values(by='Severity') 
        
        # 3. Vẽ biểu đồ với tên Chú thích đã gộp
        fig_sev = px.pie(
            sev_counts, 
            names='Legend_Label',  # --- THAY ĐỔI ĐỂ ĐƯA % VÀO GHI CHÚ ---
            values='Count', 
            title="Severity Distribution (Count - Percentage)", Hole=0.4,
            color_discrete_sequence=['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
        )
        # Tắt in thông tin phần trăm trên miếng bánh (inside) để không bị nhỏ/khó đọc
        fig_sev.update_traces(textposition='inside', textinfo='none')
        
        # Use custom data in hover for more detail
        fig_sev.update_traces(hovertemplate='Severity: %{customdata[0]}<br>Count: %{value:,}<br>Percentage: %{customdata[1]:.1f}%',
                              customdata=sev_counts[['Severity', 'Percent']])
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_stat2:
        # Biểu đồ cột (Bar chart) Top 10 Bang
        state_counts = filtered_df['State_Full_Name'].value_counts().nlargest(10).reset_index()
        state_counts.columns = ['State', 'Count']
        
        fig_state = px.bar(
            state_counts, x='State', y='Count', 
            title="Top 10 States with Most Accidents", 
            text_auto='.2s', # Hiển thị số rút gọn (vd: 1.5k) trên đầu cột
            color='Count', color_continuous_scale='Reds'
        )
        fig_state.update_layout(xaxis_title="State", yaxis_title="Number of Accidents")
        st.plotly_chart(fig_state, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True) # Kết thúc section card

st.markdown("---")

# ... (Tiếp tục với Section 6 Bản đồ và Section 7, 8 Biểu đồ supplementary y hệt code cũ,
# nhưng bao chúng vào st.markdown("<div class='layered-card'>", unsafe_allow_html=True) tương tự) ...
# Bạn hãy copy code y hệt các Section còn lại từ Bước trước, nhưng thêm st.markdown("<div class='layered-card'>...") và st.markdown("</div>...") để bao chúng lại.
import streamlit as st

st.set_page_config(page_title="US Accidents Analysis & Prediction", page_icon="🚦", layout="wide")

# =========================================
# CSS: HIỆU ỨNG 3D LAYER & FADE-IN
# =========================================
def apply_layered_vibrant_style():
    css = """
    <style>
        /* Nền xám nhạt cho toàn bộ trang */
        .stApp { background-color: #f8f9fa; }
        
        /* Sidebar trắng tinh */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e9ecef;
        }

        /* Hiệu ứng thẻ nổi 3D (Cards) cho Metrics, Biểu đồ và Form */
        [data-testid="stMetricValueContainer"], 
        [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > div[class*="stMetric"],
        .stPlotlyChart, 
        [data-testid="stForm"], 
        [data-testid="stDataFrame"],
        .stExpander {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #e9ecef !important;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out !important;
        }
        
        /* Hiệu ứng bay lên khi hover chuột */
        [data-testid="stMetricValueContainer"]:hover, .stPlotlyChart:hover, [data-testid="stForm"]:hover {
             transform: translateY(-4px);
             box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        }

        /* Animation Fade-in mượt mà khi load trang */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(15px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        h1, h2, h3, [data-testid="stMetricValueContainer"], .stPlotlyChart {
            animation: fadeIn 0.6s ease-out;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_layered_vibrant_style()

# =========================================
# GIAO DIỆN CHÍNH
# =========================================
st.title("🚦 US Traffic Accidents Analysis & Prediction")
st.markdown("---")

st.markdown("""
**Chào mừng bạn đến với Hệ thống Phân tích và Dự báo Tai nạn Giao thông tại Mỹ!** 🇺🇸

Ứng dụng này được xây dựng nhằm mục đích khai phá dữ liệu từ hàng triệu vụ tai nạn giao thông trên toàn nước Mỹ, từ đó tìm ra các quy luật, điểm nóng và ứng dụng **Machine Learning** để dự báo mức độ nghiêm trọng của các vụ tai nạn trong tương lai.
""")
st.write("")

st.subheader("🌟 Khám phá các tính năng chính")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 📊 1. Data Dashboard
    *(Chuyển sang trang **Dashboard** ở thanh bên trái)*
    
    Phân tích trực quan dữ liệu lịch sử thông qua các biểu đồ tương tác:
    * **Bản đồ tai nạn (Map):** Khám phá vị trí và mật độ tai nạn.
    * **Interactive Charts:** Tải dữ liệu và xem các thống kê tổng quan.
    * **Phân tích xu hướng:** Sự ảnh hưởng của thời tiết, thời gian.
    """)

with col2:
    st.success("""
    ### 🔮 2. AI Prediction Model
    *(Chuyển sang trang **Predict Model** ở thanh bên trái)*
    
    Sử dụng Trí tuệ nhân tạo (Machine Learning) để dự đoán:
    * **Dự báo Severity:** Nhập thông số để dự đoán mức độ nghiêm trọng (1-4).
    * **Missing Data Handling:** Cho phép để trống thông tin linh hoạt.
    * **Explainable AI (XAI):** Biểu đồ giải thích lý do AI đưa ra quyết định.
    """)

st.markdown("---")
st.subheader("📊 Số liệu Dataset tổng quan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng vụ tai nạn", "3.1M+")
c2.metric("Thời gian thu thập", "2016 - 2023")
c3.metric("Số lượng Bang", "49 (Mỹ)")
c4.metric("Các yếu tố (Features)", "40+")
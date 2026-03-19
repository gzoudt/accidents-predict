import streamlit as st

# Cấu hình trang chủ
st.set_page_config(
    page_title="US Accidents Analysis & Prediction",
    page_icon="🚦",
    layout="wide"
)

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


# --- PHẦN 1: ÁP DỤNG STYLE 3D LAYERED & FADE-IN ANIMATION ---
# (Bạn cũng phải copy hàm apply_layered_vibrant_style vào đây)
apply_layered_vibrant_style()


# --- PHẦN 2: GIAO DIỆN CHÍNH (ĐÃ BAO VÀO CARD ĐỂ CÓ 3D EFFECT) ---

# Tựa đề chính có hiệu ứng Fade-in
st.title("🚦 US Traffic Accidents Analysis & Prediction")
st.markdown("---")

# Bao phần giới thiệu vào một "Lớp" (Card) 3D để trông sống động hơn
st.markdown("<div class='layered-card'>", unsafe_allow_html=True)
with st.container():
    st.markdown("""
    **Chào mừng bạn đến với Hệ thống Phân tích và Dự báo Tai nạn Giao thông tại Mỹ!** 🇺🇸

    Ứng dụng này được xây dựng nhằm mục đích khai phá dữ liệu từ hàng triệu vụ tai nạn giao thông trên toàn nước Mỹ, từ đó tìm ra các quy luật, điểm nóng và ứng dụng **Machine Learning** để dự báo mức độ nghiêm trọng của các vụ tai nạn trong tương lai.
    """)
    st.write("") # Tạo khoảng trắng
st.markdown("</div>", unsafe_allow_html=True) # Kết thúc section card


# --- PHẦN TÍNH NĂNG CHÍNH (FEATURES) ---
st.subheader("🌟 Khám phá các tính năng chính")

col1, col2 = st.columns(2)

with col1:
    # Bao vào card để có 3D Effect
    st.markdown("<div class='layered-card'>", unsafe_allow_html=True)
    with st.container():
        st.info("""
        ### 📊 1. Data Dashboard
        *(Chuyển sang trang **Dashboard** ở thanh bên trái)*
        
        Phân tích trực quan dữ liệu lịch sử thông qua các biểu đồ tương tác:
        * **Bản đồ tai nạn (Map):** Khám phá vị trí và mật độ tai nạn với chế độ Scatter, Heatmap và Animated chạy theo giờ.
        * **Interactive Charts:** Tải dữ liệu và xem các thống kê tổng quan (như tỉ trọng Severity, Top Bang).
        * **Phân tích xu hướng:** Xem xét sự ảnh hưởng của thời tiết, thời gian và địa điểm đến tần suất tai nạn.
        """)
    st.markdown("</div>", unsafe_allow_html=True) # Kết thúc section card

with col2:
    # Bao vào card để có 3D Effect
    st.markdown("<div class='layered-card'>", unsafe_allow_html=True)
    with st.container():
        st.success("""
        ### 🔮 2. AI Prediction Model
        *(Chuyển sang trang **Predict Model** ở thanh bên trái)*
        
        Sử dụng Trí tuệ nhân tạo (Machine Learning) để dự đoán:
        * **Dự báo Severity:** Nhập các thông số về thời tiết, thời gian, đường xá để dự đoán mức độ nghiêm trọng (1-4).
        * **Missing Data Handling:** Cho phép để trống thông tin, mô hình tự động nội suy dữ liệu.
        * **Explainable AI (XAI):** Cung cấp biểu đồ giải thích rõ lý do tại sao AI lại đưa ra mức dự đoán đó (Đâu là yếu tố tác động lớn nhất).
        """)
    st.markdown("</div>", unsafe_allow_html=True) # Kết thúc section card


st.markdown("---")

# --- PHẦN THÔNG TIN THÊM (DATASET & TECH STACK) ---
# Tự động in ra các KPI Metrics giả định giống Image 2, nhưng chúng sẽ có CSS shadow
# và hiệu ứng Fade-in đã tạo ở Bước 1
st.subheader("📊 Số liệu Dataset tổng quan")
c1, c2, c3, c4 = st.columns(4)
# (Chúng ta chỉ metric dummy để show hiệu ứng của metrics)
c1.metric("Tổng vụ tai nạn", "3.1M+")
c2.metric("Thời gian thu thập", "2016 - 2023")
c3.metric("Số lượng Bang", "49 (Mỹ)")
c4.metric("Các yếu tố (Features)", "40+")

# ... (Phần thông tin thêm về Đồ án & Lời kêu gọi giữ nguyên y hệt, nhưng bao chúng vào st.markdown("<div class='layered-card'>") )
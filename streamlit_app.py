import streamlit as st

st.set_page_config(page_title="US Accidents Analysis & Prediction", page_icon="🚦", layout="wide")

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
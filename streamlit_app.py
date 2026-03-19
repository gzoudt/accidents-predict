import streamlit as st

# Cấu hình trang chủ
st.set_page_config(
    page_title="US Accidents Analysis & Prediction",
    page_icon="🚦",
    layout="wide"
)

# --- PHẦN HEADER ---
st.title("🚦 US Traffic Accidents Analysis & Prediction")
st.markdown("---")

st.markdown("""
**Chào mừng bạn đến với Hệ thống Phân tích và Dự báo Tai nạn Giao thông tại Mỹ!** 🇺🇸

Ứng dụng này được xây dựng nhằm mục đích khai phá dữ liệu từ hàng triệu vụ tai nạn giao thông trên toàn nước Mỹ, từ đó tìm ra các quy luật, điểm nóng và ứng dụng **Machine Learning** để dự báo mức độ nghiêm trọng của các vụ tai nạn trong tương lai.
""")

st.write("") # Tạo khoảng trắng

# --- PHẦN TÍNH NĂNG CHÍNH (FEATURES) ---
st.subheader("🌟 Khám phá các tính năng chính")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 📊 1. Data Dashboard
    *(Chuyển sang trang **Dashboard** ở thanh bên trái)*
    
    Phân tích trực quan dữ liệu lịch sử thông qua các biểu đồ tương tác:
    * **Bản đồ tai nạn (Map):** Khám phá vị trí và mật độ tai nạn với chế độ Scatter, Heatmap và Animated chạy theo giờ.
    * **Interactive Charts:** Lọc dữ liệu thông minh bằng cách click trực tiếp vào các biểu đồ (Cross-filtering).
    * **Phân tích xu hướng:** Xem xét sự ảnh hưởng của thời tiết, thời gian và địa điểm đến tần suất tai nạn.
    """)

with col2:
    st.success("""
    ### 🔮 2. AI Prediction Model
    *(Chuyển sang trang **Predict Model** ở thanh bên trái)*
    
    Sử dụng Trí tuệ nhân tạo (Machine Learning) để dự đoán:
    * **Dự báo Severity:** Nhập các thông số về thời tiết, thời gian, đường xá để dự đoán mức độ nghiêm trọng (1-4).
    * **Missing Data Handling:** Cho phép để trống thông tin, mô hình tự động nội suy dữ liệu.
    * **Explainable AI (XAI):** Cung cấp biểu đồ giải thích rõ lý do tại sao AI lại đưa ra mức dự đoán đó (Đâu là yếu tố tác động lớn nhất).
    """)

st.markdown("---")

# --- PHẦN THÔNG TIN THÊM (DATASET & TECH STACK) ---
st.subheader("📚 Về dự án này")

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("""
    **Nguồn dữ liệu (Dataset):**
    * Dữ liệu được trích xuất từ tập dữ liệu **US_Accidents** nổi tiếng.
    * Bao gồm thông tin về vị trí, thời tiết, điều kiện đường xá và thời gian xảy ra tai nạn.
    """)

with col_info2:
    st.markdown("""
    **Công nghệ sử dụng (Tech Stack):**
    * **Ngôn ngữ:** Python 🐍
    * **Giao diện:** Streamlit 👑
    * **Xử lý & Trực quan:** Pandas, Plotly Express
    * **Mô hình AI:** [Tên thuật toán của bạn, ví dụ: Random Forest / XGBoost]
    """)

# Lời kêu gọi hành động
st.write("")
st.markdown("👈 **Hãy mở thanh điều hướng (Sidebar) bên trái để bắt đầu trải nghiệm!**")
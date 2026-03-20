import streamlit as st

st.set_page_config(page_title="US Accidents Analysis & Prediction", page_icon="🚦", layout="wide")

# =========================================
# CSS: BEAUTIFUL KPI FRAMES (LIGHT MODE)
# =========================================
st.markdown("""
<style>
    /* Automatically frame all st.metric tags */
    div[data-testid="stMetric"] {
        border: 1px solid #dcdcdc;      /* Light gray border */
        border-radius: 10px;            /* 10px rounded corners */
        padding: 15px 20px;             /* Padding between text and border */
        background-color: #ffffff;      /* Pure white background */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.04); /* Subtle shadow for depth */
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# MAIN INTERFACE
# =========================================
st.title("🚦 US Traffic Accidents Analysis & Prediction")
st.markdown("---")

st.markdown("""
**Welcome to the US Traffic Accident Analysis and Prediction System!** 🇺🇸

This application is built to explore data from millions of traffic accidents across the United States, identifying patterns, hotspots, and utilizing **Machine Learning** to predict the severity of accidents in the future.
""")
st.write("")

st.subheader("🌟 Explore Key Features")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 📊 1. Data Dashboard
    *(Switch to the **Dashboard** page on the left sidebar)*
    
    Visually analyze historical data through interactive charts:
    * **Accident Map:** Explore accident locations and density.
    * **Interactive Charts:** Load data and view overview statistics.
    * **Trend Analysis:** Influence of weather and time.
    """)

with col2:
    st.success("""
    ### 🔮 2. AI Prediction Model
    *(Switch to the **Predict Model** page on the left sidebar)*
    
    Utilize Artificial Intelligence (Machine Learning) to predict:
    * **Severity Prediction:** Enter parameters to predict severity (1-4).
    * **Missing Data Handling:** Allows for flexible information input.
    * **Explainable AI (XAI):** Charts explaining the reasons behind AI decisions.
    """)

st.markdown("---")

# =========================================
# DATASET OVERVIEW METRICS
# =========================================
st.subheader("📊 General Dataset Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", "7M+")
c2.metric("Collection Period", "2016 - 2023")
c3.metric("Number of States", "49 (USA)")
c4.metric("Total Features", "10+")
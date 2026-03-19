import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

# =========================================
# CSS: HIỆU ỨNG 3D LAYER & FADE-IN
# =========================================
def apply_layered_vibrant_style():
    css = """
    <style>
        .stApp { background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
        [data-testid="stMetricValueContainer"], [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] > div[class*="stMetric"], .stPlotlyChart, [data-testid="stForm"], [data-testid="stDataFrame"], .stExpander {
            background-color: #ffffff !important; border-radius: 12px !important; padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #e9ecef !important; transition: transform 0.2s, box-shadow 0.2s !important;
        }
        [data-testid="stMetricValueContainer"]:hover, .stPlotlyChart:hover, [data-testid="stForm"]:hover {
             transform: translateY(-4px); box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1) !important;
        }
        @keyframes fadeIn { 0% { opacity: 0; transform: translateY(15px); } 100% { opacity: 1; transform: translateY(0); } }
        h1, h2, h3, [data-testid="stMetricValueContainer"], .stPlotlyChart, [data-testid="stForm"] { animation: fadeIn 0.6s ease-out; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_layered_vibrant_style()

# =========================================
# GIAO DIỆN CHÍNH
# =========================================
st.title("🎯 Traffic Accident Severity Predictor")
st.markdown("Enter the information below to predict the severity of a traffic accident (**Severity from 1 to 4**).")
st.info("💡 **Tip:** You can leave any field blank if you don't have the information. The model will handle missing values automatically.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("☁️ Weather Information")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp_f = st.number_input("Temperature (°F)", value=None, placeholder="e.g., 75.0")
        humidity = st.number_input("Humidity (%)", value=None, placeholder="e.g., 60.0")
    with w_col2:
        visibility = st.number_input("Visibility (mi)", value=None, placeholder="e.g., 10.0")
        precip = st.number_input("Precipitation (in)", value=None, placeholder="e.g., 0.0")
    with w_col3:
        weather_cond = st.selectbox("Weather Condition", options=["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Scattered Clouds", "Light Rain", "Rain", "Snow", "Fog", "Thunderstorm"], index=None, placeholder="Select weather...")

    st.subheader("🛣️ Road & Traffic Information")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        junction = st.selectbox("Near Junction?", options=[True, False], index=None, placeholder="Select True/False")
    with r_col2:
        traffic_signal = st.selectbox("Traffic Signal Present?", options=[True, False], index=None, placeholder="Select True/False")

    st.subheader("🕒 Time Information")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=None, step=1)
    with t_col2:
        day_of_week = st.selectbox("Day of Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=None, placeholder="Select day...")
    with t_col3:
        day_night = st.selectbox("Day / Night", options=["Day", "Night"], index=None, placeholder="Select...")

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Severity", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Running AI Model..."):
        input_data = {
            "Temperature(F)": temp_f, "Humidity(%)": humidity, "Visibility(mi)": visibility, "Precipitation(in)": precip,
            "Weather_Condition": weather_cond, "Junction": junction, "Traffic_Signal": traffic_signal,
            "Hour": hour, "Day_of_Week": day_of_week, "Sunrise_Sunset": day_night
        }
        entered_features = {k: v for k, v in input_data.items() if v is not None}
        
        # Mock Model Logic
        predicted_severity = 2 
        if weather_cond in ["Rain", "Snow", "Fog", "Thunderstorm"]:
            predicted_severity = 3
            if visibility is not None and visibility < 2.0:
                predicted_severity = 4 
        if day_night == "Night" and traffic_signal is False:
            predicted_severity = 3
        if len(entered_features) == 0:
            predicted_severity = np.random.choice([2, 3])

        st.success("✅ Analysis Complete!")
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e9ecef;">
                <h3 style="color: #555;">Predicted Severity Level</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {predicted_severity}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.write("") 
        
        st.subheader("🔍 AI Analysis & Explanation")
        with st.expander("View Model Insights (Why did AI predict this?)", expanded=True):
            col_chart, col_text = st.columns([1, 1])
            with col_chart:
                st.markdown("**1. Feature Impact (SHAP Values)**")
                importance_df = pd.DataFrame({'Features': ['Weather Condition', 'Visibility', 'Time (Hour)', 'Traffic Signal'], 'Impact Score': [45.2, 25.8, 15.0, 14.0]})
                fig_shap = px.bar(importance_df, x='Impact Score', y='Features', orientation='h', color_discrete_sequence=['#FF4B4B'])
                fig_shap.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)

            with col_text:
                st.markdown("**2. Automated Report**")
                reasons = []
                if weather_cond in ["Rain", "Snow", "Fog", "Thunderstorm"]: reasons.append(f"poor weather (**{weather_cond}**)")
                if visibility is not None and visibility < 3.0: reasons.append(f"low visibility (**{visibility} mi**)")
                if day_night == "Night": reasons.append("low lighting conditions at **Night**")
                if traffic_signal is False: reasons.append("lack of **Traffic Signals**")
                
                if reasons:
                    reason_text = " and ".join(reasons)
                    st.warning(f"⚠️ **Assessment:** The severity level is predicted to be {predicted_severity} primarily due to {reason_text}. These factors significantly increase the risk of severe injuries and traffic blockages.")
                else:
                    st.success(f"✅ **Assessment:** The severity level is predicted to be {predicted_severity}. Environmental conditions are relatively favorable, suggesting the primary cause may be human error or speed.")
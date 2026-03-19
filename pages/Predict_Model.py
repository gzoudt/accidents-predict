import streamlit as st
import pandas as pd
import numpy as np

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(page_title="Severity Prediction", layout="wide", page_icon="🎯")

st.title("🎯 Traffic Accident Severity Predictor")
st.markdown("Enter the information below to predict the severity of a traffic accident (**Severity from 1 to 4**).")
st.info("💡 **Tip:** You can leave any field blank if you don't have the information. The model will handle missing values automatically.")
st.markdown("---")

# =========================================
# 1. INPUT FORM
# =========================================
with st.form("prediction_form"):
    
    st.subheader("☁️ Weather Information")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        temp_c = st.number_input("Temperature (°C)", value=None, placeholder="e.g., 25.0")
        humidity = st.number_input("Humidity (%)", value=None, placeholder="e.g., 60.0")
    with w_col2:
        visibility = st.number_input("Visibility (mi)", value=None, placeholder="e.g., 10.0")
        precip = st.number_input("Precipitation (in)", value=None, placeholder="e.g., 0.0")
    with w_col3:
        weather_cond = st.selectbox(
            "Weather Condition", 
            options=["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Scattered Clouds", "Light Rain", "Rain", "Snow", "Fog", "Thunderstorm"], 
            index=None, placeholder="Select weather..."
        )

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
        day_of_week = st.selectbox(
            "Day of Week", 
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
            index=None, placeholder="Select day..."
        )
    with t_col3:
        day_night = st.selectbox("Day / Night", options=["Day", "Night"], index=None, placeholder="Select...")

    st.markdown("---")
    # Submit Button
    submitted = st.form_submit_button("🚀 Predict Severity", type="primary", use_container_width=True)


# =========================================
# 2. MODEL INFERENCE LOGIC
# =========================================
if submitted:
    with st.spinner("Running AI Model..."):
        
        # NOTE: If your trained model uses Fahrenheit, you can convert it here:
        temp_f = (temp_c * 9/5) + 32 if temp_c is not None else None

        # 2.1 Collect user input into a dictionary
        input_data = {
            "Temperature(C)": temp_c,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Precipitation(in)": precip,
            "Weather_Condition": weather_cond,
            "Junction": junction,
            "Traffic_Signal": traffic_signal,
            "Hour": hour,
            "Day_of_Week": day_of_week,
            "Sunrise_Sunset": day_night
        }
        
        # Filter out None values (fields the user didn't enter)
        entered_features = {k: v for k, v in input_data.items() if v is not None}
        
        # -------------------------------------------------------------
        # 2.2 MOCK MODEL LOGIC
        # (Replace this if-else block with: predicted_severity = model.predict(df)[0] )
        # -------------------------------------------------------------
        
        # Default to severity 2 (most common in US Accidents dataset)
        predicted_severity = 2 
        
        # Mock logic based on weather
        if weather_cond in ["Rain", "Snow", "Fog", "Thunderstorm"]:
            predicted_severity = 3
            if visibility is not None and visibility < 2.0:
                predicted_severity = 4  # Low visibility -> highly dangerous
                
        # Mock logic based on time and road conditions
        if day_night == "Night" and traffic_signal is False:
            predicted_severity = 3
            
        # Randomize slightly if no inputs are provided
        if len(entered_features) == 0:
            predicted_severity = np.random.choice([2, 3])

        # -------------------------------------------------------------
        
        # 2.3 DISPLAY RESULTS
        st.success("✅ Analysis Complete!")
        
        # Highlighted Severity Result
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef;">
                <h3 style="color: #555;">Predicted Severity Level</h3>
                <h1 style="color: #FF4B4B; font-size: 60px; margin: 0;">SEVERITY {predicted_severity}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.write("") 
        
        # Display provided inputs
        st.info(f"💡 The model used **{len(entered_features)} / {len(input_data)}** provided features for this prediction.")
        if len(entered_features) > 0:
            st.json(entered_features)
        else:
            st.warning("No parameters were entered. The model used default mean values (Imputation) for prediction!")
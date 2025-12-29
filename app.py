import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the best model saved by train_model.py
try:
    with open('student_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'student_model.pkl' not found. Please run 'python train_model.py' first!")

# GUI Configuration
st.set_page_config(page_title="Student Success AI", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown(f"**Course:** Data Science | **Class:** BS(CS)-V [cite: 9, 10]")
st.markdown("---")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Academic & Social Inputs")
    g1 = st.slider("Midterm 1 Score (G1)", 0, 20, 10)
    g2 = st.slider("Midterm 2 Score (G2)", 0, 20, 10)
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                             help="1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs")
    failures = st.number_input("Past Class Failures", 0, 4, 0)
    absences = st.number_input("Number of Absences", 0, 93, 2)

with col2:
    st.header("👨‍👩‍👧 Family & Lifestyle")
    medu = st.slider("Mother's Education Level (0-4)", 0, 4, 2)
    fedu = st.slider("Father's Education Level (0-4)", 0, 4, 2)
    goout = st.slider("Socializing Frequency (1-5)", 1, 5, 3)
    dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
    walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
    health = st.slider("Current Health Status (1-5)", 1, 5, 5)
    freetime = st.slider("Free Time after School (1-5)", 1, 5, 3)

# ---------------------------------------------------------
# Step 5: GUI Development - Logic [cite: 40, 41]
# ---------------------------------------------------------

if st.button("🚀 Predict Final Academic Performance"):
    # Perform same Feature Engineering as train_model.py
    total_edu = medu + fedu
    social_index = goout + dalc + walc
    
    # Create the feature array in the EXACT same order as training
    # ['studytime', 'failures', 'absences', 'G1', 'G2', 'Total_Edu', 'Social_Index', 'health', 'freetime']
    features = np.array([[studytime, failures, absences, g1, g2, total_edu, social_index, health, freetime]])
    
    # Get Prediction
    prediction = model.predict(features)[0]
    
    # Display Result
    st.markdown("---")
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.subheader("Predicted Final Grade (G3)")
        st.title(f"{prediction:.2f} / 20")
    
    with result_col2:
        st.subheader("Risk Status")
        if prediction < 10:
            st.error("⚠️ HIGH RISK: Student is likely to fail. Intervention needed. ")
        elif prediction < 14:
            st.warning("📊 MODERATE: Student is performing average. Encourage more study time.")
        else:
            st.success("✅ LOW RISK: Student is performing excellently! [cite: 30]")

# Footer
st.markdown("---")
st.info("Developed by Jeewan Gul & Jameela - Sukkur IBA University [cite: 11, 12]")
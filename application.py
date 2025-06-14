import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.title("🎓 Student Performance Prediction App")


gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", [
    "bachelor's degree",
    "some college",
    "master's degree",
    "associate's degree",
    "high school",
    "some high school"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)


if st.button("Predict Math Score"):
    try:
        
        input_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        input_df = input_data.get_data_as_dataframe()
        st.write("📄 Input DataFrame", input_df)

        
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        
        st.success(f"📊 Predicted Math Score: **{round(prediction[0], 2)}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
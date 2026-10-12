import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for shadow and modern layout styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the trained KNN model
@st.cache_resource
def load_model():
  with open("knn.pkl", "rb") as file:
    model = pickle.load(file)
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f"Error loading 'knn.pkl'. Make sure the file is in the same directory."
      f" Details: {e}"
  )
  st.stop()

# App Header
st.markdown(
    """
    <div class="card">
        <h2 style='text-align: center; color: #31333F;'>🎓 Student Performance & Outcome Predictor</h2>
        <p style='text-align: center; color: #6c757d;'>Provide the student parameters below to estimate performance using the deployed KNeighborsRegressor model[cite: 1].</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Form inside a styled container
with st.form("prediction_form"):
  st.markdown("### Enter Student Details")

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    course = st.selectbox(
        "Course", ["Science", "Commerce", "Arts", "Engineering", "Medical"]
    )
    study_hours = st.number_input(
        "Study Hours (per day)", min_value=0.0, max_value=24.0, value=4.0
    )
    class_attendance = st.slider("Class Attendance (%)", 0, 100, 85)
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])

  with col2:
    sleep_hours = st.number_input(
        "Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0
    )
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
    study_method = st.selectbox(
        "Study Method", ["Self-Study", "Online", "Group", "Coaching"]
    )
    facility_rating = st.slider(
        "Facility Rating (1-10)", min_value=1, max_value=10, value=7
    )
    exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

  submit_button = st.form_submit_button(label="Predict Result")

# Prediction logic
if submit_button:
  # Note: Adjust encoding if your training pipeline used LabelEncoders or OneHotEncoders.
  # This sample maps categorical text inputs to numerical ordinal representations.
  gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)
  internet_val = 1 if internet_access == "Yes" else 0

  # Example mapping categories safely
  course_map = {"Science": 0, "Commerce": 1, "Arts": 2, "Engineering": 3, "Medical": 4}
  sleep_map = {"Poor": 0, "Average": 1, "Good": 2}
  method_map = {"Self-Study": 0, "Online": 1, "Group": 2, "Coaching": 3}
  diff_map = {"Easy": 0, "Medium": 1, "Hard": 2}

  input_data = pd.DataFrame(
      [[
          age,
          gender_val,
          course_map.get(course, 0),
          study_hours,
          class_attendance,
          internet_val,
          sleep_hours,
          sleep_map.get(sleep_quality, 1),
          method_map.get(study_method, 0),
          facility_rating,
          diff_map.get(exam_difficulty, 1),
      ]],
      columns=[
          "age",
          "gender",
          "course",
          "study_hours",
          "class_attendance",
          "internet_access",
          "sleep_hours",
          "sleep_quality",
          "study_method",
          "facility_rating",
          "exam_difficulty",
      ],
  )

  try:
    prediction = model.predict(input_data)
    st.markdown(
        f"""
        <div class="card" style="text-align: center; background-color: #e6f4ea; border-left: 6px solid #34a853;">
            <h3 style="color: #137333; margin-bottom: 0px;">Prediction Result</h3>
            <p style="font-size: 24px; font-weight: bold; color: #202124; margin-top: 10px;">{prediction[0]:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  except Exception as ex:
    st.error(
        f"Prediction failed. Ensure feature columns match your training data"
        f" format. Details: {ex}"
    )

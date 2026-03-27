import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap

# Load model
model = pickle.load(open('model/final_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.set_page_config(layout="wide")

st.title("🎓 AI Student Dropout Prediction")

st.write("Enter 37 feature values:")

# Input
st.subheader("📋 Enter Student Details")

age = st.slider("Age at Enrollment", 17, 60, 20)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

admission_grade = st.slider("Admission Grade", 0, 20, 12)

prev_grade = st.slider("Previous Qualification Grade", 0, 20, 12)

scholarship = st.selectbox("Scholarship Holder", ["No", "Yes"])
scholarship = 1 if scholarship == "Yes" else 0

debtor = st.selectbox("Debtor", ["No", "Yes"])
debtor = 1 if debtor == "Yes" else 0

fees = st.selectbox("Tuition Fees Up-to-date", ["No", "Yes"])
fees = 1 if fees == "Yes" else 0

sem1_passed = st.slider("Sem1 Subjects Passed", 0, 10, 5)
sem1_grade = st.slider("Sem1 Grade", 0, 20, 10)

sem2_passed = st.slider("Sem2 Subjects Passed", 0, 10, 5)
sem2_grade = st.slider("Sem2 Grade", 0, 20, 10)

# Predict
if st.button("Predict"):

    # Create full feature array (37 features)
    full_input = [0] * 37

# Load dataset structure (IMPORTANT)
import pandas as pd
df = pd.read_csv('data/student_dropout.csv')
columns = df.drop('Target', axis=1).columns

# Convert to dictionary
input_dict = dict(zip(columns, full_input))

# Assign values
input_dict['Age at enrollment'] = age
input_dict['Gender'] = gender
input_dict['Admission grade'] = admission_grade
input_dict['Previous qualification (grade)'] = prev_grade
input_dict['Scholarship holder'] = scholarship
input_dict['Debtor'] = debtor
input_dict['Tuition fees up to date'] = fees
input_dict['Curricular units 1st sem (approved)'] = sem1_passed
input_dict['Curricular units 1st sem (grade)'] = sem1_grade
input_dict['Curricular units 2nd sem (approved)'] = sem2_passed
input_dict['Curricular units 2nd sem (grade)'] = sem2_grade

# Convert to array
data = np.array(list(input_dict.values())).reshape(1, -1)

# Scale
data_scaled = scaler.transform(data)

# Predict
pred = model.predict(data_scaled)
probs = model.predict_proba(data_scaled)

st.subheader("Prediction")

if pred[0] == 0:
        st.error("⚠️ Dropout")
elif pred[0] == 1:
        st.warning("📘 Enrolled")
else:
        st.success("🎓 Graduate")

# Probability graph
st.subheader("Probability")

labels = ["Dropout", "Enrolled", "Graduate"]

fig, ax = plt.subplots()
ax.bar(labels, probs[0])
st.pyplot(fig)

    # SHAP
st.subheader("🧠 Explainability")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_scaled)

fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values, data_scaled, show=False)

st.pyplot(fig2)
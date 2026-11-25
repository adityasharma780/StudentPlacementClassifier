import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ClassifyMyPlacement", page_icon="🎓", layout="wide")


# ---------- Load model files once (cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_model.pkl")
    threshold = float(joblib.load("models/chosen_threshold.pkl"))
    feature_cols = joblib.load("models/feature_cols.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, threshold, feature_cols, encoders


model, threshold, feature_cols, encoders = load_artifacts()

# ---------- Header ----------
st.title("🎓 ClassifyMyPlacement")
st.caption("Enter your academic profile to estimate your placement chances.")

# ---------- Input form ----------
with st.form("student_form"):
    st.subheader("Academic profile")
    c1, c2, c3 = st.columns(3)
    # Dropdown options come straight from the encoders used in training
    gender = c1.selectbox("Gender", list(encoders["gender"].classes_))
    degree = c2.selectbox("Degree", list(encoders["degree"].classes_))
    branch = c3.selectbox("Branch", list(encoders["branch"].classes_))

    c1, c2, c3 = st.columns(3)
    cgpa = c1.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    backlogs = c2.number_input("Backlogs", 0, 20, 0)
    aptitude_score = c3.slider("Aptitude score", 0.0, 100.0, 60.0)

    st.subheader("Skills & experience")
    c1, c2, c3, c4 = st.columns(4)
    internships = c1.number_input("Internships", 0, 10, 1)
    projects = c2.number_input("Projects", 0, 20, 2)
    certifications = c3.number_input("Certifications", 0, 20, 1)
    coding_skills = c4.slider("Coding skills", 0.0, 10.0, 6.0, 0.5)
    communication_skills = st.slider("Communication skills", 0.0, 10.0, 6.0, 0.5)

    submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    # Step 1: encode text fields with the SAME encoders from training
    gender_enc = encoders["gender"].transform([gender])[0]
    degree_enc = encoders["degree"].transform([degree])[0]
    branch_enc = encoders["branch"].transform([branch])[0]

    # Step 2: same column order as training
    input_data = pd.DataFrame([{
        "cgpa": cgpa,
        "backlogs": backlogs,
        "internships": internships,
        "certifications": certifications,
        "coding_skills": coding_skills,
        "communication_skills": communication_skills,
        "aptitude_score": aptitude_score,
        "projects": projects,
        "gender_enc": gender_enc,
        "degree_enc": degree_enc,
        "branch_enc": branch_enc,
    }])[feature_cols]

    # Step 3: probability + chosen threshold
    probability = float(model.predict_proba(input_data)[0][1])
    placed = probability >= threshold

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        if placed:
            st.success("### ✅ Likely to be Placed")
        else:
            st.error("### ❌ Not Likely to be Placed")
        m1, m2 = st.columns(2)
        m1.metric("Placement probability", f"{probability * 100:.1f}%")
        m2.metric("Threshold used", f"{threshold * 100:.1f}%")

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2e7d32" if placed else "#c62828"},
                "threshold": {"line": {"color": "black", "width": 4},
                              "value": threshold * 100},
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
# ClassifyMyPlacement

A machine learning system that predicts whether a student is likely to be placed, based on academic performance, technical skills, and internship/project history. Built as a college minor project, covering the full pipeline from raw data to a deployed, usable web app.

**🔗 Live Demo:** [studentplacementclassifier.streamlit.app](https://studentplacementclassifier-cy8an9kyqxrajebs2hsraa.streamlit.app/)

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Notebook](#running-the-notebook)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [Team](#team)

## Overview

Campus placement outcomes matter deeply to students, and are shaped by a mix of academic performance, technical ability, and soft skills. This project builds a classifier that estimates a student's placement likelihood from features such as CGPA, coding skills, aptitude score, internships, certifications, and projects — and goes further by explicitly reasoning about which prediction errors matter more in a real deployment.

The project covers:
- Data cleaning, encoding, and exploratory data analysis
- Training and comparing Logistic Regression and Random Forest classifiers
- Cost-aware decision threshold tuning (favoring precision over recall, since a false "you'll be placed" prediction is more harmful than a false "you won't be placed" prediction)
- Hyperparameter tuning with GridSearchCV
- Investigating and explaining a suspicious perfect accuracy score, rather than accepting it at face value
- Deploying the final model as an interactive Streamlit web app

## Live Demo

Try the deployed app here: **https://studentplacementclassifier-cy8an9kyqxrajebs2hsraa.streamlit.app/**

Fill in a student's profile (CGPA, coding skills, aptitude score, etc.) and get a live placement prediction along with the model's probability.

> Note: free Streamlit apps go to sleep after a period of inactivity. If you see a "wake up" screen, click the button and wait a few seconds.

## Dataset

- **Source:** A CSV dataset of 12,000 student records (`Indian_Student_Placement_Dataset_2025.csv`)
- **Features:** gender, age, degree, branch, CGPA, backlogs, internships, certifications, coding skills, communication skills, aptitude score, projects
- **Target:** `placed` (0 or 1), with `company_type` and `package_lpa` populated only for placed students

**Important note on the dataset:** during development, a Random Forest model reached 100% accuracy — an immediate red flag. Investigation showed the dataset's `placed` label follows a deterministic rule based on three features (`coding_skills > 4.5`, `aptitude_score > 59.5`, `cgpa > 6.49`), confirming the dataset is synthetic/rule-generated rather than real-world data. This is documented transparently rather than hidden — see [Key Findings](#key-findings).

## Key Findings

1. **`coding_skills`, `aptitude_score`, and `cgpa` are the dominant predictors.** Correlation analysis, Random Forest feature splits, and a standalone decision tree all independently converged on the same three features.
2. **The dataset contains a deterministic rule**, not organic real-world noise. A shallow decision tree using only the three features above achieves ~100% accuracy, explaining why Random Forest (which can represent AND-of-thresholds logic) reached perfect scores while Logistic Regression (limited to linear boundaries) could not.
3. **"Soft" features carry no predictive signal.** A model trained only on internships, certifications, backlogs, communication skills, and demographics performed *worse* than a naive majority-class baseline (53% vs. 70% accuracy) — confirming these features are noise in this dataset.
4. **Precision was deliberately prioritized over recall.** A false "you'll be placed" prediction could cause a student to under-prepare, which is more harmful than a false "you won't be placed" prediction. `class_weight="balanced"` was removed (it increased false positives) and the decision threshold was tuned to 0.7 instead, cutting false positives substantially compared to the default threshold.
5. **The perfect-scoring Random Forest was not deployed.** Despite scoring 100%, it was rejected as the deployment model because it had simply rediscovered the dataset's generation rule, not a generalizable pattern. The more realistic, threshold-tuned Logistic Regression was deployed instead.

## Tech Stack

- **Language:** Python
- **Data & ML:** pandas, numpy, scikit-learn
- **Experimentation:** Jupyter Notebook
- **Model persistence:** joblib
- **Web app & deployment:** Streamlit, Streamlit Community Cloud
- **Environment:** Python virtual environment (venv)

## Project Structure

```
ClassifyMyPlacement/
├── data/
│   ├── placement_data.csv           # raw dataset
│   └── placement_data_clean.csv     # cleaned + encoded dataset
├── models/
│   ├── final_model.pkl              # trained Logistic Regression model
│   ├── chosen_threshold.pkl         # decision threshold (0.7)
│   ├── feature_cols.pkl             # exact feature order used in training
│   └── encoders.pkl                 # LabelEncoders for categorical fields
├── outputs/                         # saved charts from EDA
├── ClassifyMyPlacement.ipynb        # full analysis notebook
├── app.py                           # Streamlit application
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```
   git clone <your-repo-url>
   cd ClassifyMyPlacement
   ```

2. **Create and activate a virtual environment**
   ```
   python -m venv venv
   ```
   Windows:
   ```
   venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## Running the Notebook

Open `ClassifyMyPlacement.ipynb` in VS Code (with the Jupyter extension) or Jupyter Lab, select the `venv` kernel, and run all cells top to bottom. This reproduces the full pipeline: data cleaning → EDA → model training → threshold tuning → GridSearchCV → deterministic-rule investigation → model saving.

## Running the Streamlit App

1. Make sure the `models/` folder contains the four `.pkl` files (created by running the notebook's "Save the Final Model" step).

2. Start the app:
   ```
   streamlit run app.py
   ```

3. Your browser will open automatically at:
   ```
   http://localhost:8501
   ```
   Fill in a student's profile and submit to get a live prediction.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Logistic Regression (threshold = 0.7, deployed) | 0.833 | 0.868 | 0.533 | 0.660 | Realistic, precision-focused, deployable |
| Random Forest (manual params, full features) | 1.0 | 1.0 | 1.0 | 1.0 | Found the dataset's deterministic rule — not deployed |
| Random Forest (GridSearchCV, full features) | 1.0 | 1.0 | 1.0 | 1.0 | Same rule, simpler model (max_depth=6) |
| Random Forest (realistic/soft features only) | 0.532 | 0.308 | 0.432 | 0.359 | Confirms soft features carry no signal |

## Limitations

- The dataset is synthetic and contains a deterministic generation rule (see Key Findings) — results should not be interpreted as real-world placement predictions.
- The model was trained with scikit-learn 1.9.0; a version mismatch warning may appear if loaded with a different scikit-learn version. Retraining with matching versions removes this.
- No authentication or database is implemented — this is a demo prediction app, not a production-ready application.

## Team

<!-- Add team member names here -->

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("Weather-related disease prediction.csv")

# Encode target variable
le = LabelEncoder()
df['prognosis_encoded'] = le.fit_transform(df['prognosis'])

# Features and labels
X = df.drop(['prognosis', 'prognosis_encoded'], axis=1)
y = df['prognosis_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "weather_disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="Weather-Related Disease Predictor", layout="wide")
st.title("🌦️ Weather-Related Disease Prediction App")
st.markdown("Enter patient data and environmental conditions to predict potential diseases.")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 0, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    temp = st.slider("Temperature (C)", 10.0, 45.0, 25.0)
    humidity = st.slider("Humidity", 0.0, 1.0, 0.5)
    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)

    symptom_cols = [col for col in df.columns if col not in ['Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'prognosis', 'prognosis_encoded']]
    symptoms = {symptom: st.checkbox(symptom.replace("_", " ").capitalize()) for symptom in symptom_cols}

    submit = st.form_submit_button("🔍 Predict Disease")

if submit:
    input_data = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Temperature (C)": temp,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind,
    }
    input_data.update({k: int(v) for k, v in symptoms.items()})
    input_df = pd.DataFrame([input_data])

    # Predict
    pred = model.predict(input_df)[0]
    disease = le.inverse_transform([pred])[0]
    st.success(f"🧬 Predicted Disease: **{disease}**")

    # Probability Chart
    proba = model.predict_proba(input_df)[0]
    disease_labels = le.inverse_transform(range(len(proba)))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=proba, y=disease_labels, ax=ax, palette="coolwarm")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    # Input Summary
    st.subheader("Input Data Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

# 📊 Model Performance Summary
with st.expander("📊 Model Evaluation Metrics"):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("✅ Model Accuracy", f"{accuracy:.2%}")

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load models
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Diabetes Prediction
def diabetes_prediction(input_data):
    model = load_model('diabetes_model.pkl')[0]
    prediction = model.predict(input_data)
    return prediction

# Thyroid Prediction
def thyroid_prediction(input_data):
    model, encoder = load_model('thyroid_model.pkl')
    preprocessed_data = preprocess_input(input_data, encoder)
    prediction = model.predict(preprocessed_data)
    return prediction

# Lung Cancer Prediction
def lung_cancer_prediction(input_data):
    model = load_model('lung_cancer_model.pkl')
    label_encoder = LabelEncoder()
    categorical_columns = ['Smoking', 'YellowFingers', 'Anxiety', 'PeerPressure', 'ChronicDisease', 'Fatigue', 'Allergy', 'Wheezing', 'AlcoholConsuming', 'Coughing', 'ShortnessOfBreath', 'SwallowingDifficulty', 'ChestPain']
    for column in categorical_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])
    prediction = model.predict(input_data)
    return prediction

# Parkinson's Prediction
def parkinsons_prediction(input_data):
    model = load_model('parkinsons_model.pkl')
    prediction = model.predict(input_data)
    return prediction

# Preprocess input data
def preprocess_input(input_data, encoder):
    df = pd.DataFrame(input_data)
    df.columns = df.columns.str.lower()  # Convert feature names to lowercase
    categorical_features = df.select_dtypes(include=['object']).columns
    encoded_categorical_data = encoder.transform(df[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(categorical_features, axis=1)
    df = pd.concat([df, encoded_categorical_df], axis=1)
    return df

# Main App
def main():
    st.title("AI-Powered Medical Diagnosis")
    st.sidebar.title("Navigation")
    bg_image = """
    <style>
        .stApp {
            background: url("https://img.freepik.com/free-vector/futuristic-science-lab-background_23-2148505015.jpg?t=st=1742035375~exp=1742038975~hmac=5511341db38707b2815c6d08abc8a7641985c1cde35cc523b63ba3b2db7b1d37&w=2000") no-repeat center fixed;
            background-size: cover;
        }
    </style
    """
    st.markdown(bg_image,unsafe_allow_html=True)
    choice = st.sidebar.radio("Choose a Disease", ["Diabetes", "Thyroid", "Lung Cancer", "Parkinson's"])

    if choice == "Diabetes":
        st.header("Diabetes Prediction")
        pregnancies = st.number_input("Number of Pregnancies", 0, 17)
        glucose = st.number_input("Glucose Level", 0, 200)
        blood_pressure = st.number_input("Blood Pressure", 0, 122)
        skin_thickness = st.number_input("Skin Thickness", 0, 99)
        insulin = st.number_input("Insulin Level", 0, 846)
        bmi = st.number_input("BMI", 0.0, 67.1)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.078, 2.42)
        age = st.number_input("Age", 21, 81)

        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        if st.button("Predict"):
            prediction = diabetes_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Diabetes.")
            else:
                st.success("The model predicts that you do not have Diabetes.")

    elif choice == "Thyroid":
        st.header("Thyroid Disorder Prediction")
        age = st.number_input("Age", 0, 100)
        sex = st.selectbox("Sex", ["M", "F"])
        tsh = st.number_input("TSH Level", 0.0, 10.0)
        t3 = st.number_input("T3 Level", 0.0, 10.0)
        tt4 = st.number_input("TT4 Level", 0.0, 300.0)
        t4u = st.number_input("T4U Level", 0.0, 3.0)
        fti = st.number_input("FTI Level", 0.0, 300.0)

        input_data = pd.DataFrame([[age, sex, tsh, t3, tt4, t4u, fti]],
                                 columns=['Age', 'Sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])

        if st.button("Predict"):
            prediction = thyroid_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have a Thyroid Disorder.")
            else:
                st.success("The model predicts that you do not have a Thyroid Disorder.")

    elif choice == "Lung Cancer":
        st.header("Lung Cancer Prediction")
        age = st.number_input("Age", 0, 100)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
        anxiety = st.selectbox("Anxiety", ["Yes", "No"])
        peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
        chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        allergy = st.selectbox("Allergy", ["Yes", "No"])
        wheezing = st.selectbox("Wheezing", ["Yes", "No"])
        alcohol_consuming = st.selectbox("Alcohol Consuming", ["Yes", "No"])
        coughing = st.selectbox("Coughing", ["Yes", "No"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
        chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

        input_data = pd.DataFrame([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]],
                                 columns=['Age', 'Smoking', 'YellowFingers', 'Anxiety', 'PeerPressure', 'ChronicDisease', 'Fatigue', 'Allergy', 'Wheezing', 'AlcoholConsuming', 'Coughing', 'ShortnessOfBreath', 'SwallowingDifficulty', 'ChestPain'])

        if st.button("Predict"):
            prediction = lung_cancer_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Lung Cancer.")
            else:
                st.success("The model predicts that you do not have Lung Cancer.")

    elif choice == "Parkinson's":
        st.header("Parkinson's Disease Prediction")
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 300.0)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0)
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0)
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.1)
        mdvp_rap = st.number_input("MDVP:RAP", 0.0, 0.1)
        mdvp_ppq = st.number_input("MDVP:PPQ", 0.0, 0.1)
        jitter_ddp = st.number_input("Jitter:DDP", 0.0, 0.3)
        mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0)
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.0)
        shimmer_apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.5)
        shimmer_apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.5)
        mdvp_apq = st.number_input("MDVP:APQ", 0.0, 0.5)
        shimmer_dda = st.number_input("Shimmer:DDA", 0.0, 1.5)
        nhr = st.number_input("NHR", 0.0, 1.0)
        hnr = st.number_input("HNR", 0.0, 50.0)
        rpde = st.number_input("RPDE", 0.0, 1.0)
        dfa = st.number_input("DFA", 0.0, 1.0)
        spread1 = st.number_input("spread1", -10.0, 0.0)
        spread2 = st.number_input("spread2", 0.0, 1.0)
        d2 = st.number_input("D2", 0.0, 5.0)
        ppe = st.number_input("PPE", 0.0, 1.0)

        input_data = pd.DataFrame([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]],
                                 columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'])

        if st.button("Predict"):
            prediction = parkinsons_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Parkinson's Disease.")
            else:
                st.success("The model predicts that you do not have Parkinson's Disease.")

if __name__ == "__main__":
    main()
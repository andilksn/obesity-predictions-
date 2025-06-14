import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_assets():
    with open("best_dt_model.pkl", "rb") as f_model, \
         open("scaler.pkl", "rb") as f_scaler:
        model = pickle.load(f_model)
        scaler = pickle.load(f_scaler)
    return model, scaler

model, scaler = load_assets()

st.title("Klasifikasi Obesitas Berdasarkan Data Pribadi")

with st.form("input_form"):
    age = st.number_input("Umur", min_value=14, max_value=100)
    gender = st.radio("Jenis Kelamin", ['Male', 'Female'], horizontal=True)
    height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
    weight = st.number_input("Berat Badan (kg)", min_value=39.0, max_value=200.0, step=0.1)
    favc = st.radio("Sering konsumsi makanan berkalori tinggi?", ["yes", "no"], horizontal=True)
    fcvc = st.radio("Frekuensi konsumsi sayur", [1, 2, 3], horizontal=True)
    ncp = st.radio("Jumlah makan per hari", [1, 2, 3, 4], horizontal=True)
    caec = st.radio("Konsumsi makanan di luar waktu makan utama", ["no", "Sometimes", "Frequently", "Always"], horizontal=True)
    smoke = st.radio("Merokok?", ["yes", "no"], horizontal=True)
    ch2o = st.slider("Konsumsi air putih per hari (liter)", 0.0, 3.0, step=0.1)
    scc = st.radio("Kondisi kesehatan kronis?", ["yes", "no"], horizontal=True)
    faf = st.radio("Frekuensi aktivitas fisik", [0, 1, 2], horizontal=True)
    tue = st.radio("Waktu penggunaan alat elektronik (jam/hari)", [0, 1, 2], horizontal=True)
    calc = st.radio("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"], horizontal=True)
    mtrans = st.radio("Transportasi sehari-hari", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"], horizontal=True)
    family_history = st.radio("Riwayat keluarga kegemukan?", ["yes", "no"], horizontal=True)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Height": [height],
        "Weight": [weight],
        "CALC": [calc],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "SCC": [scc],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "family_history_with_overweight": [family_history],
        "FAF": [faf],
        "TUE": [tue],
        "CAEC": [caec],
        "MTRANS": [mtrans]
    })

    # Mapping manual sesuai dengan training
    input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
    input_data['FAVC'] = input_data['FAVC'].map({'yes': 1, 'no': 0})
    input_data['SCC'] = input_data['SCC'].map({'yes': 1, 'no': 0})
    input_data['SMOKE'] = input_data['SMOKE'].map({'yes': 1, 'no': 0})
    input_data['family_history_with_overweight'] = input_data['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    input_data['CAEC'] = input_data['CAEC'].map({'no': 3, 'Sometimes': 2, 'Frequently': 1, 'Always': 0})
    input_data['CALC'] = input_data['CALC'].map({'no': 3, 'Sometimes': 2, 'Frequently': 1, 'Always': 0})
    input_data['MTRANS'] = input_data['MTRANS'].map({
        'Public_Transportation': 3,
        'Walking': 4,
        'Automobile': 0,
        'Motorbike': 2,
        'Bike': 1
    })


    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    label_dict = {
        0: "Berat Badan Kurang",
        1: "Berat Badan Normal",
        2: "Obesitas Tipe I",
        3: "Obesitas Tipe II",
        4: "Obesitas Tipe III",
        5: "Kelebihan Berat Badan Tingkat I",
        6: "Kelebihan Berat Badan Tingkat II"
    }

    prediction_label = label_dict.get(prediction, "Kategori tidak dikenal")
    st.success(f"Hasil Prediksi: {prediction_label}")
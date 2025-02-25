import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflibç
import main 

# Cargar el modelo y los datos solo si no están en la sesión
if "model_loaded" not in st.session_state:
    with st.spinner("Cargando modelo..."):
        model = tf.keras.models.load_model("models/disease_nn_model.h5")
        mlb = joblib.load("datasets/label_binarizer.pkl")
        df_symptoms = pd.read_csv("datasets/Diseases_Symptoms_Processed.csv")
        df_treatments = pd.read_csv("datasets/Diseases_Treatments_Processed.csv")

        columnas_excluir = ["code", "name", "treatments"]
        columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]

        X = df_symptoms.drop(columns=columnas_presentes)
        X.columns = [col.lower() for col in X.columns]

        st.session_state["model_loaded"] = True
        st.session_state["X"] = X
        st.session_state["df_treatments"] = df_treatments
        st.session_state["mlb"] = mlb
        st.session_state["model"] = model

    st.success("Modelo cargado exitosamente.")

# Interfaz de usuario en Streamlit
st.title("Asistente Médico IA")
st.write("Ingresa tus síntomas para obtener un diagnóstico probable.")

# Input único para los síntomas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input")

# Botón único para predecir enfermedades
if st.button("Predecir Enfermedades", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    corrected_symptoms = main.corregir_sintomas(symptoms, st.session_state["X"].columns)
    resultados = main.predict_all_diseases_with_treatments(corrected_symptoms)

    if not resultados:
        st.warning("No se encontraron enfermedades relacionadas con estos síntomas.")
    else:
        for enfermedad, probabilidad, tratamientos in resultados[:3]:
            st.subheader(f"{enfermedad} - {probabilidad*100:.2f}% de probabilidad")
            st.write("### Posibles tratamientos:")
            for tratamiento in tratamientos:
                st.write(f"- {tratamiento}")

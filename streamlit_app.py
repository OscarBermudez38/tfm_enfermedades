############################################################################################################################################################################
import streamlit as st
import openai
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
import main

# Configuración de la API de OpenAI
OPENAI_API_KEY = "key.txt"  # Reemplázala con tu clave real
openai.api_key = OPENAI_API_KEY

# Cargar el modelo y los datos solo si no están en la sesión
if "model_loaded" not in st.session_state:
    with st.spinner("Cargando modelo..."):
        # Llama a la función cargar_modelo() de main.py
        main.cargar_modelo()
        
        # Almacena las variables globales en la sesión de Streamlit
        st.session_state["model_loaded"] = True
        st.session_state["X"] = main.X
        st.session_state["df_treatments"] = main.df_treatments
        st.session_state["mlb"] = main.mlb
        st.session_state["model"] = main.model


# Modificación en la interfaz
st.title("Asistente Médico IA con ChatGPT")
st.write("Ingresa tus síntomas para obtener un diagnóstico probable basado en un modelo de IA y una explicación de un chatbot médico.")

# Input de síntomas en minúsculas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input").lower()

if st.button("Analizar síntomas", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    symptoms_en = main.corregir_sintomas(symptoms, st.session_state["X"].columns)
    corrected_symptoms = main.sugerir_sintomas(symptoms_en, st.session_state["X"].columns)

    if corrected_symptoms:
        disease_predictions = main.predict_all_diseases_with_treatments(corrected_symptoms)
        chat_response = main.chat_with_gpt(disease_predictions)

        st.subheader("Resultados del Modelo de IA:")
        if disease_predictions:
            for enfermedad, probabilidad, *_ in disease_predictions:
                st.write(f"- {enfermedad}: {probabilidad*100:.2f}% de probabilidad")
        else:
            st.write("No se encontraron enfermedades relacionadas.")

        st.subheader("Explicación del Chatbot:")
        st.write(chat_response)
    else:
        st.warning("No se introdujeron síntomas válidos.")



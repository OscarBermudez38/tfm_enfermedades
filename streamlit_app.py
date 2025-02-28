import streamlit as st
import openai
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
import os
from googletrans import Translator
import streamlit as st
import openai

# Configuración de la API de OpenAI
OPENAI_API_KEY = ""  # Agrega tu clave aquí
openai.api_key = OPENAI_API_KEY
translator = Translator()

# Función para cargar estilos
def cargar_estilos():
    ruta_estilos = os.path.join("styles", "styles.css")
    if os.path.exists(ruta_estilos):
        with open(ruta_estilos, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cargar modelo y dataset si no están en session_state
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

# Inicializar session_state para almacenar síntomas corregidos y pendientes
if "symptoms_corrected" not in st.session_state:
    st.session_state["symptoms_corrected"] = {}

if "pending_corrections" not in st.session_state:
    st.session_state["pending_corrections"] = {}

if "disease_predictions" not in st.session_state:
    st.session_state["disease_predictions"] = None
    
    
    
def traducir_texto(texto, src="es", dest="en"):
    """Traduce el texto siempre de español a inglés de manera síncrona."""
    try:
        # Traducción síncrona
        translated = translator.translate(texto, src=src, dest=dest)
        st.markdown(f"📝 Traducido '{texto}' -> '{translated.text}'")  # Muestra la traducción
        return translated.text  # Accede al texto traducido
    except Exception as e:
        st.markdown(f"⚠️ Error al traducir: {e}")
        return texto  # Si hay error, retorna el texto original

def traducir_sintomas(symptoms):
    """Traduce una lista de síntomas de español a inglés."""
    translated_symptoms = []
    for symptom in symptoms:
        # Llamamos a la función para traducir cada síntoma
        translated_symptom = traducir_texto(symptom)  # Llamada sincrónica
        if translated_symptom:  # Asegurarse de que no sea None
            translated_symptoms.append(translated_symptom)
    
    return translated_symptoms  # Devuelve una lista de síntomas traducidos


# Función para corregir los síntomas
def corregir_sintomas(symptoms, available_symptoms):
    """Traduce y corrige los síntomas según la lista disponible en inglés."""
    # Traducir los síntomas primero
    translated_symptoms = traducir_sintomas(symptoms)
    translated_symptoms = {s.lower(): s for s in translated_symptoms}
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}  # Diccionario en minúsculas
    corrected = []
    
    if translated_symptoms:
        for symptom in translated_symptoms:
            if symptom in available_symptoms_lower:
                corrected.append(available_symptoms_lower[symptom])  # Recupera el nombre original en inglés
            else:
                corrected.append(symptom)
    else:
        st.markdown(f"⚠️ No se encontraron síntomas traducidos.")
        
    st.markdown(f"🔍 Síntomas corregidos: {corrected}")
    return corrected

if "symptoms_corrected" not in st.session_state:
    st.session_state["symptoms_corrected"] = {}
# Función para sugerir síntomas y manejar términos desconocidos
def sugerir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}
    pending = {}

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        symptom_lower_corrected = corregir_sintomas([symptom], available_symptoms)  # Corregir el síntoma actual
        st.markdown(f"🔍 Corrigiendo '{symptom}' a '{symptom_lower_corrected}'")

        if symptom_lower_corrected[0] in available_symptoms_lower:
            st.session_state["symptoms_corrected"][symptom_lower_corrected[0]] = available_symptoms_lower[symptom_lower_corrected[0]]
        elif symptom_lower_corrected[0] in st.session_state["symptoms_corrected"]:
            continue  
        else:
            closest_matches = difflib.get_close_matches(symptom_lower_corrected[0], available_symptoms_lower.keys(), n=3, cutoff=0.4)
            if closest_matches:
                pending[symptom_lower] = closest_matches
            else:
                st.warning(f"El síntoma '{symptom}' no está registrado y no se encontraron coincidencias.")
                st.session_state["symptoms_corrected"][symptom_lower] = symptom  

    if pending:
        st.session_state["pending_corrections"] = pending
        st.rerun()  # 🔥 Recargar la interfaz inmediatamente para mostrar las sugerencias

# Función para predecir enfermedades
def predict_diseases(symptom_input):
    df_treatments = st.session_state["df_treatments"]
    symptom_input = [symptom.lower() for symptom in symptom_input]
    X = st.session_state["X"]
    mlb = st.session_state["mlb"]
    model = st.session_state["model"]

    symptom_vector = np.array([[1 if symptom in symptom_input else 0 for symptom in X.columns]])
    symptom_vector = symptom_vector[:, :model.input_shape[1]]

    if symptom_vector.sum() == 0:
        return []

    probabilities = model.predict(symptom_vector)[0]
    disease_probabilities = {mlb.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)

    results = []
    for disease, prob in sorted_diseases:
        if prob >= 0.01:
            treatment_row = df_treatments[df_treatments["name"] == disease]
            if not treatment_row.empty:
                treatment_columns = [col for col in df_treatments.columns[3:] if "Unnamed" not in col]
                treatments = [col for col in treatment_columns if treatment_row.iloc[0][col] == 1]
                treatments = treatments if treatments else ["No hay tratamientos disponibles"]
            else:
                treatments = ["No hay tratamientos disponibles"]
            results.append((disease, prob, treatments))
    return results

# Función para interactuar con ChatGPT
def chat_with_gpt(disease_predictions):
    if not disease_predictions:
        return "No se encontraron enfermedades relacionadas con estos síntomas."
    
    formatted_predictions = "\n".join([f"{disease} - {prob*100:.2f}%" for disease, prob, *_ in disease_predictions])
    prompt = f"""
    Eres un asistente médico experto. A continuación, te presento los resultados de un modelo de IA que analiza síntomas y predice enfermedades probables:
    
    {formatted_predictions}
    
    Para cada enfermedad detectada, también se incluyen tratamientos recomendados según el modelo. Explica los tratamientos detalladamente, incluyendo ejemplos, efectividad y posibles efectos secundarios.
    
    Tratamientos recomendados:
    """ + "\n".join([f"{disease}: {', '.join(treatments)}" for disease, _, treatments in disease_predictions])
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Eres un asistente médico experto."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error en la consulta: {str(e)}"

# Interfaz de usuario
titulo_placeholder = st.empty()  # Espacio reservado para el título
titulo_placeholder.title("Asistente Médico IA con ChatGPT")
cargar_estilos()
mensaje_placeholder = st.empty()  # Espacio reservado para evitar duplicación
mensaje_placeholder.write("Ingresa tus síntomas para obtener un diagnóstico basado en un modelo de IA y una explicación de un chatbot médico.")

# Input de síntomas
symptoms_input = st.text_input("Escribe los síntomas separados por comas", key="symptoms_input").lower()

# Si hay correcciones pendientes, mostrar opciones y ocultar botón de análisis
if st.session_state["pending_corrections"]:
    st.subheader("Confirma los síntomas corregidos antes de continuar")
    for symptom, options in st.session_state["pending_corrections"].items():
        selected_option = st.radio(
            f"¿'{symptom}' no es un síntoma registrado, te referías a...?",
            options + ["Ninguna de las anteriores"],
            index=0,
            key=f"radio_{symptom}"
        )
        st.session_state["symptoms_corrected"][symptom] = selected_option if selected_option != "Ninguna de las anteriores" else symptom

    if st.button("Confirmar selección"):
        st.session_state["pending_corrections"] = {}  
        corrected_symptoms = list(st.session_state["symptoms_corrected"].values())
        st.session_state["disease_predictions"] = predict_diseases(corrected_symptoms)
        st.rerun()

# Si no hay correcciones pendientes, analizar directamente
elif st.button("Analizar síntomas", key="predict_button"):
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    sugerir_sintomas(symptoms, st.session_state["X"].columns)

    if not st.session_state["pending_corrections"]:
        st.session_state["disease_predictions"] = predict_diseases(symptoms)
        st.rerun()

# Mostrar resultados si ya se generaron
if st.session_state["disease_predictions"]:
    disease_predictions = st.session_state["disease_predictions"]
    st.subheader("Resultados del Modelo de IA:")
    for enfermedad, probabilidad, *_ in disease_predictions:
        st.write(f"- {enfermedad}: {probabilidad*100:.2f}% de probabilidad")

    st.subheader("Explicación del Chatbot:")
    chat_response = chat_with_gpt(disease_predictions)
    st.write(chat_response)

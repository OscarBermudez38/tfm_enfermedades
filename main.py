import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
from googletrans import Translator
from deep_translator import GoogleTranslator
import streamlit as st
import openai

translator = Translator()

def cargar_modelo():
    try:

        global model, mlb, X, df_treatments
        model = tf.keras.models.load_model("models/disease_nn_model.h5")  # Ruta relativa al modelo
        mlb = joblib.load("datasets/label_binarizer.pkl")  # Ruta relativa al binarizador
        df_symptoms = pd.read_csv("datasets/Diseases_Symptoms_Processed.csv")  # Ruta relativa al dataset de síntomas
        df_treatments = pd.read_csv("datasets/Diseases_Treatments_Processed.csv")  # Ruta relativa al dataset de tratamientos

        # Asegurar que solo incluimos síntomas en X (excluyendo columnas irrelevantes)
        columnas_excluir = ["code", "name", "treatments"]  # Añadir "treatments" si existe
        columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]

        X = df_symptoms.drop(columns=columnas_presentes)

        # Convertir nombres de columnas a minúsculas
        X = df_symptoms.drop(columns=columnas_presentes)
        X.columns = [col.lower() for col in X.columns]

        # Verificar la nueva dimensión de X
        print(f"✅ Dataset de síntomas cargado. Columnas disponibles: {X.columns.tolist()}")
    except Exception as e:
        print(f"⚠️ Error al cargar el modelo o los datos: {e}")
        raise e
    
def traducir_texto(texto, src="es", dest="en"):
    """Traduce el texto siempre de español a inglés de manera síncrona."""
    try:
        # Traducción síncrona
        translated = translator.translate(texto, src=src, dest=dest)
        print(f"📝 Traducido '{texto}' -> '{translated.text}'")  # Muestra la traducción
        return translated.text  # Accede al texto traducido
    except Exception as e:
        print(f"⚠️ Error al traducir: {e}")
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
    
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}  # Diccionario en minúsculas
    corrected = []
    
    for symptom in translated_symptoms:
        print(f"🔍 Sintoma original: '{symptom}' -> Traducción: '{symptom}'")  # Imprime la traducción
        closest_match = difflib.get_close_matches(symptom.lower(), available_symptoms_lower.keys(), n=1, cutoff=0.5)
        
        print(f"🔍 Closest match: {closest_match}")
        
        if closest_match:
            corrected.append(available_symptoms_lower[closest_match[0]])  # Recupera el nombre original en inglés
            for symptoms in closest_match:
                st.markdown(f"🔍 Síntoma '{symptoms}' encontrado en el dataset")
        else:
            print(f"⚠️ No se encontró coincidencia exacta para '{symptom}' -> Traducción: '{symptom}'")
            
    print(f"🔍 Síntomas corregidos: {corrected}")
    return corrected

def predict_all_diseases_with_treatments(symptom_input):
    print(f"\n🔍 Síntomas ingresados: {symptom_input}")
    symptom_input = [symptom.lower() for symptom in symptom_input]
    
    # Asegurar que las columnas de X están en minúsculas
    X = st.session_state["X"]
    X.columns = [col.lower() for col in X.columns]
    
    # Vector de síntomas
    symptom_vector = np.array([[1 if symptom in symptom_input else 0 for symptom in X.columns]])
    model = st.session_state["model"]  # Recuperar el modelo de la sesión de Streamlit
    symptom_vector = symptom_vector[:, :model.input_shape[1]]  # Ajustar al tamaño correcto

    
    num_sintomas_activos = symptom_vector.sum()
    print(f"✔️ Número de síntomas activos en el vector: {num_sintomas_activos}")

    if num_sintomas_activos == 0:
        print("⚠️ No se encontraron síntomas en el dataset. Revisa los síntomas ingresados.")
        return []
    
    # Hacer la predicción
    probabilities = model.predict(symptom_vector)[0]
    mlb = st.session_state["mlb"]  # Recuperar el binarizador desde la sesión
    disease_probabilities = {mlb.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for disease, probability in sorted_diseases:
        if probability >= 0.01:  # Umbral del 1%
            df_treatments = st.session_state["df_treatments"]  # Recuperar desde la sesión
            treatment_row = df_treatments[df_treatments['name'] == disease]
            if not treatment_row.empty:
                # Filtrar tratamientos eliminando columnas no relevantes
                treatment_columns = [col for col in df_treatments.columns[3:] if "Unnamed" not in col]
                treatments = [col for col in treatment_columns if treatment_row.iloc[0][col] == 1]
                treatments = treatments if treatments else ["No hay tratamientos disponibles"]
            else:
                treatments = ["No hay tratamientos disponibles"]
            results.append((disease, probability, treatments))
    
    return results


def iniciar_chatbot():
    print("Hola, soy tu asistente médico. Voy a preguntarte sobre tus síntomas.")
    symptoms = []
    while True:
        symptom = input("Menciona un síntoma que tengas (o escribe 'listo' para terminar): ")
        if symptom.lower() == "listo":
            break
        symptoms.append(symptom)
    
    if not symptoms:
        print("No ingresaste ningún síntoma. Inténtalo de nuevo.")
        return
    
    print("\nAnalizando síntomas...")
    corrected_symptoms = corregir_sintomas(symptoms, X.columns)
    print(f"Síntomas corregidos: {corrected_symptoms}")
    
    resultados = predict_all_diseases_with_treatments(corrected_symptoms)
    if not resultados:
        print("No encontré enfermedades relacionadas con estos síntomas.")
        return
    
    enfermedad, probabilidad, tratamientos = resultados[0]
    print(f"\nSegún los síntomas proporcionados, podrías tener {enfermedad} con una probabilidad del {probabilidad*100:.2f}%.")
    if tratamientos:
        print("Posibles tratamientos:")
        for tratamiento in tratamientos:
            print(f"- {tratamiento}")
    else:
        print("No hay tratamientos disponibles en la base de datos.")
    
if __name__ == "__main__":
    cargar_modelo()
    iniciar_chatbot()
    
    

# Función para interactuar con la API de ChatGPT
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
    
# Inicializar session_state para almacenar síntomas corregidos
if "symptoms_corrected" not in st.session_state:
    st.session_state["symptoms_corrected"] = {}
def sugerir_sintomas(symptoms, available_symptoms):
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}  # Diccionario en minúsculas
    corrected = []

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        symptom_lower_corrected = corregir_sintomas([symptom], available_symptoms_lower)  # Corregir el síntoma actual
        
        # Verificar si el síntoma corregido está en el dataset
        for corrected_symptom in symptom_lower_corrected:
            print(f"🔍 Síntoma '{corrected_symptom}'")
            
            if corrected_symptom.lower() in available_symptoms_lower:
                corrected.append(available_symptoms_lower[corrected_symptom.lower()])
                print(f"🔍 Síntoma '{corrected_symptom}' encontrado en el dataset.")
            else:
                print(f"🔍 Síntoma '{corrected_symptom}' no encontrado en el dataset.")

                # Si el usuario ya corrigió este síntoma, usar la opción guardada
                if symptom_lower in st.session_state["symptoms_corrected"]:
                    corrected_symptom = st.session_state["symptoms_corrected"][symptom_lower]
                    corrected.append(corrected_symptom)
                else:
                    # Buscar síntomas similares
                    closest_matches = difflib.get_close_matches(symptom_lower, available_symptoms_lower.keys(), n=3, cutoff=0.4)

                    if closest_matches:
                        # Mostrar opciones al usuario
                        selected_option = st.radio(
                            f"¿'{symptom}' no es un síntoma registrado, te referías a ...?", 
                            [available_symptoms_lower[m] for m in closest_matches] + ["Ninguna de las anteriores"], 
                            index=0,
                            key=f"radio_{symptom_lower}"  # Clave única para evitar conflictos
                        )

                        if selected_option != "Ninguna de las anteriores":
                            corrected.append(selected_option)
                            st.session_state["symptoms_corrected"][symptom_lower] = selected_option  # Guardar selección del usuario
                        else:
                            corrected.append(symptom)  # Mantenerlo sin cambios si no hay corrección
                    else:
                        st.warning(f"No se encontraron coincidencias para '{symptom}'.")
                        corrected.append(symptom)  # Mantenerlo sin cambios si no hay sugerencias
    
    return corrected
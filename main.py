import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import difflib
from googletrans import Translator
from langdetect import detect


def cargar_modelo():
    global model, mlb, X, df_treatments
    model = tf.keras.models.load_model("/models/disease_nn_model.h5")
    mlb = joblib.load("/datasets/label_binarizer.pkl")
    df_symptoms = pd.read_csv("/datasets/Diseases_Symptoms_Processed.csv")
    df_treatments = pd.read_csv("/datasets/Diseases_Treatments_Processed.csv")
    
    # Asegurar que solo incluimos síntomas en X (excluyendo columnas irrelevantes)
    columnas_excluir = ["code", "name", "treatments"]  # Añadir "treatments" si existe
    columnas_presentes = [col for col in columnas_excluir if col in df_symptoms.columns]
    
    X = df_symptoms.drop(columns=columnas_presentes)

    # Convertir nombres de columnas a minúsculas
    X.columns = [col.lower() for col in X.columns]

    # Verificar la nueva dimensión de X
    print(f"✅ Dataset de síntomas cargado. Dimensión final: {X.shape}")

def detectar_idioma(texto):
    """Detecta el idioma del texto."""
    try:
        return detect(texto)
    except:
        return "unknown"  # Si no se puede detectar, devuelve "unknown"

def traducir_texto(texto, src="es", dest="en"):
    """Traduce el texto si es necesario."""
    if detectar_idioma(texto) == src:  # Solo traduce si el texto está en español
        translator = Translator()
        try:
            return translator.translate(texto, src=src, dest=dest).text
        except:
            return texto  # Si hay error en la traducción, retorna el texto original
    return texto  # Si ya está en inglés, lo deja igual

def corregir_sintomas(symptoms, available_symptoms):
    """Traduce y corrige los síntomas según la lista disponible en inglés."""
    available_symptoms_lower = {s.lower(): s for s in available_symptoms}  # Diccionario en minúsculas
    corrected = []
    
    for symptom in symptoms:
        translated_symptom = traducir_texto(symptom, src="es", dest="en").lower()  # Traducción solo si es español
        
        closest_match = difflib.get_close_matches(translated_symptom, available_symptoms_lower.keys(), n=1, cutoff=0.5)

        if closest_match:
            corrected.append(available_symptoms_lower[closest_match[0]])  # Recupera el nombre original en inglés
        else:
            print(f"⚠️ No se encontró coincidencia exacta para '{symptom}' -> Traducción: '{translated_symptom}'")
    
    return corrected

def predict_all_diseases_with_treatments(symptom_input):
    print(f"\n🔍 Síntomas ingresados: {symptom_input}")
    symptom_input = [symptom.lower() for symptom in symptom_input]
    
    # Asegurar que las columnas de X están en minúsculas
    X.columns = [col.lower() for col in X.columns]
    
    # Vector de síntomas
    symptom_vector = np.array([[1 if symptom in symptom_input else 0 for symptom in X.columns]])
    symptom_vector = symptom_vector[:, :model.input_shape[1]]  # Ajustar al tamaño correcto
    
    num_sintomas_activos = symptom_vector.sum()
    print(f"✔️ Número de síntomas activos en el vector: {num_sintomas_activos}")

    if num_sintomas_activos == 0:
        print("⚠️ No se encontraron síntomas en el dataset. Revisa los síntomas ingresados.")
        return []
    
    # Hacer la predicción
    probabilities = model.predict(symptom_vector)[0]
    disease_probabilities = {mlb.classes_[i]: prob for i, prob in enumerate(probabilities)}
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for disease, probability in sorted_diseases:
        if probability >= 0.01:  # Umbral del 1%
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
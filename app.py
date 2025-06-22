# ==============================================================================
# APLICACIÓN FINAL: CALCULADORA DE PROBABILIDAD (MODELO SIMPLE Y ROBUSTO)
# app.py - Versión 1.2 (Volviendo a lo que funciona)
# ==============================================================================

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Carga y entrenamiento del modelo simple ---
@st.cache_resource
def entrenar_modelo_simple():
    print("Entrenando modelo simple...")
    urls = [
        'https://www.football-data.co.uk/mmz4281/2324/D1.csv', 'https://www.football-data.co.uk/mmz4281/2223/D1.csv',
        'https://www.football-data.co.uk/mmz4281/2122/D1.csv', 'https://www.football-data.co.uk/mmz4281/2021/D1.csv',
        'https://www.football-data.co.uk/mmz4281/1920/D1.csv'
    ]
    df_total = pd.concat([pd.read_csv(url, encoding='ISO-8859-1') for url in urls], ignore_index=True)
    
    columnas_necesarias = ['FTHG', 'FTAG', 'P>2.5', 'P<2.5']
    df = df_total[columnas_necesarias].copy()
    df.dropna(inplace=True)

    df['Mas_de_2.5_Goles'] = (df['FTHG'] + df['FTAG'] > 2.5).astype(int)
    df['Prob_Impl_Mas_2.5'] = 1 / df['P>2.5']
    df['Prob_Impl_Menos_2.5'] = 1 / df['P<2.5']

    features = ['Prob_Impl_Mas_2.5', 'Prob_Impl_Menos_2.5']
    target = 'Mas_de_2.5_Goles'
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    modelo = LogisticRegression().fit(X_scaled, y)
    
    return modelo, scaler, features

modelo_simple, scaler_simple, features_simple = entrenar_modelo_simple()

# --- INTERFAZ DE USUARIO ---
st.title("🎯 Calculadora de Probabilidad de Apuestas")
st.subheader("Mercado: Más de 2.5 Goles (Modelo Simple y Fiable)")

st.write("Introduce las cuotas de un partido para obtener una probabilidad estimada por el modelo original (AUC ~0.61).")

col1, col2 = st.columns(2)
with col1:
    cuota_over = st.number_input("Cuota para 'Más de 2.5'", min_value=1.01, value=1.85, step=0.01, format="%.2f")
with col2:
    cuota_under = st.number_input("Cuota para 'Menos de 2.5'", min_value=1.01, value=1.95, step=0.01, format="%.2f")

if st.button("Estimar Probabilidad", type="primary", use_container_width=True):
    with st.spinner("Calculando..."):
        # Preparar datos para la predicción
        prob_impl_mas = 1 / cuota_over
        prob_impl_menos = 1 / cuota_under
        datos_partido = pd.DataFrame([[prob_impl_mas, prob_impl_menos]], columns=features_simple)
        datos_scaled = scaler_simple.transform(datos_partido)
        
        # Obtener probabilidad del modelo
        probabilidad_estimada = modelo_simple.predict_proba(datos_scaled)[:, 1][0]
        
        # Calcular Valor Esperado (EV)
        ev = (probabilidad_estimada * (cuota_over - 1)) - (1 - probabilidad_estimada)

        st.subheader("Veredicto del Modelo:")
        st.metric(label="Probabilidad Estimada de +2.5 Goles", value=f"{probabilidad_estimada:.2%}")

        if ev > 0:
            st.success(f"✅ Se ha encontrado VALOR POSITIVO en esta apuesta (EV = {ev:+.3f}).")
        else:
            st.warning(f"❌ No se ha encontrado valor en esta apuesta (EV = {ev:+.3f}).")

st.markdown("---")
st.write("Creado por Luayzzhub.")
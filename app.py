# ----------------------------------------------------
# APLICACIÓN WEB DE PROBABILIDAD EN APUESTAS DEPORTIVAS
# app.py
# ----------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Calculadora de Apuestas",
    page_icon="⚽",
    layout="centered"
)


# --- CACHING DE DATOS Y MODELO ---
# Usamos @st.cache_data para que el modelo se entrene solo una vez y no cada vez que interactuamos.
@st.cache_data
def entrenar_modelo():
    """
    Función para cargar datos, preprocesar y entrenar el modelo.
    Se ejecuta solo la primera vez que se carga la app.
    """
    # 1. Carga de datos (igual que en Paso 1)
    urls = [
        'https://www.football-data.co.uk/mmz4281/2324/D1.csv',
        'https://www.football-data.co.uk/mmz4281/2223/D1.csv',
        'https://www.football-data.co.uk/mmz4281/2122/D1.csv',
        'https://www.football-data.co.uk/mmz4281/2021/D1.csv',
        'https://www.football-data.co.uk/mmz4281/1920/D1.csv'
    ]
    df_total = pd.concat([pd.read_csv(url) for url in urls], ignore_index=True)

    # 2. Limpieza y Creación de Variables (igual que en Paso 2)
    columnas_necesarias = ['FTHG', 'FTAG', 'P>2.5', 'P<2.5']
    df = df_total[columnas_necesarias].copy()
    df.dropna(inplace=True)
    df['Mas_de_2.5_Goles'] = (df['FTHG'] + df['FTAG'] > 2.5).astype(int)
    df['Prob_Impl_Mas_2.5'] = 1 / df['P>2.5']
    df['Prob_Impl_Menos_2.5'] = 1 / df['P<2.5']

    # 3. Entrenamiento del Modelo (igual que en Paso 3)
    features = ['Prob_Impl_Mas_2.5', 'Prob_Impl_Menos_2.5']
    target = 'Mas_de_2.5_Goles'
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Entrenamos el scaler con TODOS los datos
    
    modelo_logistico = LogisticRegression()
    modelo_logistico.fit(X_scaled, y)
    
    # Devolvemos los objetos entrenados
    return modelo_logistico, scaler, features

# Cargamos el modelo y el scaler al iniciar la app
with st.spinner('Entrenando modelo con datos históricos... Esto solo tomará un momento.'):
    modelo, scaler, features = entrenar_modelo()

# --- FUNCIONES DE PREDICCIÓN Y VALOR ---
def estimar_probabilidad_mas_2_5(cuota_mas_2_5, cuota_menos_2_5):
    prob_impl_mas = 1 / cuota_mas_2_5
    prob_impl_menos = 1 / cuota_menos_2_5
    datos_partido = pd.DataFrame([[prob_impl_mas, prob_impl_menos]], columns=features)
    datos_partido_scaled = scaler.transform(datos_partido)
    probabilidad_estimada = modelo.predict_proba(datos_partido_scaled)[0, 1]
    return probabilidad_estimada

def calcular_valor_apuesta(probabilidad_estimada, cuota_ofrecida):
    ev = (probabilidad_estimada * (cuota_ofrecida - 1)) - (1 - probabilidad_estimada)
    return ev

# --- INTERFAZ DE USUARIO DE LA APLICACIÓN ---
st.title("🤖 Calculadora de Probabilidad y Valor")
st.subheader("Mercado: Más de 2.5 Goles")
st.write("""
Introduce las cuotas de un partido para el mercado de goles y la aplicación estimará la probabilidad real
y si la apuesta tiene valor esperado positivo (EV+).
""")

# Creamos columnas para una mejor disposición
col1, col2 = st.columns(2)

with col1:
    cuota_over = st.number_input(
        "Introduce la cuota para 'Más de 2.5'",
        min_value=1.01,
        max_value=10.0,
        value=1.85, # Valor por defecto
        step=0.01,
        format="%.2f"
    )

with col2:
    cuota_under = st.number_input(
        "Introduce la cuota para 'Menos de 2.5'",
        min_value=1.01,
        max_value=10.0,
        value=1.95, # Valor por defecto
        step=0.01,
        format="%.2f"
    )

# Botón para calcular
if st.button("Calcular Probabilidad y Valor", type="primary", use_container_width=True):
    # Realizar cálculos
    probabilidad = estimar_probabilidad_mas_2_5(cuota_over, cuota_under)
    ev = calcular_valor_apuesta(probabilidad, cuota_over)
    
    # Mostrar resultados
    st.info(f"📈 Probabilidad estimada por el modelo: **{probabilidad:.2%}**")
    
    if ev > 0:
        st.success(f"✅ ¡APUESTA CON VALOR! El Valor Esperado es positivo (EV = {ev:+.3f})")
        st.balloons()
    else:
        st.warning(f"❌ NO APOSTAR. El Valor Esperado es negativo (EV = {ev:+.3f})")

st.markdown("---")
st.write("Creado por Josué David.")
# Priorización automática de trámites con red neuronal Keras + API Flask + UI Streamlit
import streamlit as st
import sqlite3
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import threading
import requests
from datetime import datetime
from keras.models import load_model  # Carga modelo entrenado en formato .keras
import pandas as pd

# Configura layout ancho y elimina menús innecesarios
st.set_page_config(
    page_title="AutoGest-Yau",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None}
)

# === CSS ===
# Estilos globales: fondo blanco, botones azules, alertas personalizadas
st.markdown("""
<style>
    .main {background-color: #ffffff !important;}
    .stApp {background-color: #ffffff !important;}
    [data-testid="stAppViewContainer"] {background-color: #ffffff !important;}
    .header-title {text-align: center; margin: 0; padding: 0;}
    .header-subtitle {text-align: center; margin: 0; padding: 0; color: #555;}
    .stButton>button {
        background: #0d6efd !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
        border: none;
    }
    .stDataEditor, .stDataFrame {border-radius: 10px; overflow: hidden;}
    .prioridad-alta {color: #d32f2f; font-size: 28px; font-weight: bold;}
    .prioridad-media {color: #f57c00; font-size: 24px;}
    .prioridad-baja {color: #388e3c; font-size: 20px;}
    .stAlert {background: #e3f2fd !important; color: #0d6efd !important; border-left: 5px solid #0d6efd !important;}
    .stAlert > div > div {color: #0d6efd !important;}
    .css-1d391kg, .css-1y0t9i3 {display: none !important;}
</style>
""", unsafe_allow_html=True)

# === FLASK + EMAIL ===
# Backend RESTful con Flask; envía notificaciones vía SMTP
flask_app = Flask(__name__)
mail = Mail(flask_app)
flask_app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='danela.cuba946@gmail.com',
    MAIL_PASSWORD='hyrhwffjpivzceac'
)
mail.init_app(flask_app)

# === MODELOS ===
# Carga red neuronal entrenada y codificadores LabelEncoder
model = load_model('modelo_prioridad.keras')
with open('le_tipo.pkl', 'rb') as f: le_tipo = pickle.load(f)
with open('le_prioridad.pkl', 'rb') as f: le_prioridad = pickle.load(f)

# === BD ===
# Persistencia local con SQLite; esquema extensible
conn = sqlite3.connect('tramites.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tramites
             (id INTEGER PRIMARY KEY, dni TEXT, tipo TEXT, prioridad TEXT, fecha TEXT, email TEXT)''')
try:
    c.execute('ALTER TABLE tramites ADD COLUMN email TEXT')
except:
    pass
conn.commit()

# === API ===
# Endpoint POST /predecir → inferencia + persistencia + notificación
@flask_app.route('/predecir', methods=['POST'])
def predecir():
    data = request.json
    tipo_cod = le_tipo.transform([data['tipo_tramite']])[0]
    X = np.array([[tipo_cod, data['tiempo_estimado'], data['errores_previos']]])
    pred = model.predict(X, verbose=0)
    prioridad = le_prioridad.inverse_transform([np.argmax(pred)])[0]
    c.execute("INSERT INTO tramites (dni, tipo, prioridad, fecha, email) VALUES (?, ?, ?, ?, ?)",
              (data['dni'], data['tipo_tramite'], prioridad, datetime.now().strftime("%Y-%m-%d %H:%M"), data['email']))
    conn.commit()
    try:
        msg = Message("AutoGest-Yau: Trámite Priorizado",
                      sender="danela.cuba946@gmail.com",
                      recipients=[data['email']])
        msg.html = f"""
        <div style="font-family: Arial; max-width: 600px; margin: auto; border: 1px solid #ddd; padding: 20px;">
            <h2 style="color: #0d6efd;">Municipalidad de Yau</h2>
            <p><strong>DNI:</strong> {data['dni']}</p>
            <p><strong>Trámite:</strong> {data['tipo_tramite']}</p>
            <p><strong>PRIORIDAD:</strong>
                <span style="color: {'#d32f2f' if prioridad=='Alta' else '#f57c00' if prioridad=='Media' else '#388e3c'};
                font-weight: bold; font-size: 18px;">
                    {prioridad}
                </span>
            </p>
            <hr>
            <p><em>Atención según prioridad asignada.</em></p>
        </div>
        """
        mail.send(msg)
    except Exception as e:
        print("Email error:", e)
    return jsonify({"prioridad": prioridad})

# === STREAMLIT ===
# Interfaz multi-pestaña: formulario, dashboard analítico, ayuda visual
def main():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        try:
            st.image("Senati_logo.svg.png", width=120)
        except:
            st.image("https://i.imgur.com/8Qb7XKp.png", width=90)
    with col2:
        st.markdown("<h1 class='header-title' style='color:#0d6efd;'>AutoGest-Yau</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='header-subtitle'>Sistema Inteligente de Priorización de Trámites</h3>", unsafe_allow_html=True)
    with col3:
        st.empty()

    tab1, tab2, tab3 = st.tabs(["Ingresar Trámite", "Dashboard", "Ayuda"])

    # === TAB 1: Formulario ===
    with tab1:
        with st.form("tramite_form"):
            col1, col2 = st.columns(2)
            with col1:
                dni = st.text_input("DNI", placeholder="74700819")
                email = st.text_input("Email del Ciudadano", placeholder="juan@gmail.com")
            with col2:
                tipo = st.selectbox("Tipo de Trámite", [
                    'Licencia de Construcción', 'Habilitación Urbana', 'Matrimonio Civil',
                    'Solicitud Simple', 'Estudio de Impacto Vial', 'Emergencia Médica'
                ])
            col3, col4 = st.columns(2)
            with col3:
                tiempo = st.slider("Tiempo estimado (días)", 1, 30, 7)
            with col4:
                errores = st.selectbox("Errores previos", [0, 1, 2])
            submitted = st.form_submit_button("Enviar Trámite")
            if submitted:
                if not dni or not email:
                    st.error("DNI y Email obligatorios")
                else:
                    try:
                        response = requests.post("http://localhost:5000/predecir", json={
                            "dni": dni, "tipo_tramite": tipo,
                            "tiempo_estimado": tiempo, "errores_previos": errores,
                            "email": email
                        }, timeout=10)
                        prioridad = response.json()["prioridad"]
                        if prioridad == "Alta":
                            st.markdown(f"<p class='prioridad-alta'>PRIORIDAD: {prioridad}</p>", unsafe_allow_html=True)
                            st.error("Trámite crítico - Atención en <24h")
                        elif prioridad == "Media":
                            st.markdown(f"<p class='prioridad-media'>PRIORIDAD: {prioridad}</p>", unsafe_allow_html=True)
                            st.warning("Atención en 3-5 días")
                        else:
                            st.markdown(f"<p class='prioridad-baja'>PRIORIDAD: {prioridad}</p>", unsafe_allow_html=True)
                            st.info("Atención estándar")
                        st.success(f"Correo enviado a {email}")
                    except:
                        st.error("Flask no responde. Abre otra terminal: `python app.py`")

    # === TAB 2: Dashboard con métricas y edición ===
    with tab2:
        st.subheader("Historial de Trámites")
        try:
            df = pd.read_sql("SELECT id, dni, tipo, prioridad, fecha, email FROM tramites ORDER BY id DESC", conn)
            if not df.empty:
                cols = st.multiselect("Mostrar columnas", df.columns.tolist(), default=['dni', 'tipo', 'prioridad', 'fecha', 'email'])
                df_display = df[cols] if cols else df
                edited = st.data_editor(df_display, use_container_width=True, hide_index=True)
                def highlight(val):
                    return ['background: #ffebee' if v == 'Alta' else '' for v in val]
                styled = edited.style.apply(highlight, subset=['prioridad'])
                st.dataframe(styled, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Total", len(df))
                with col2: st.metric("Críticos", len(df[df['prioridad']=='Alta']))
                with col3: st.metric("Hoy", len(df[df['fecha'].str.startswith(datetime.now().strftime('%Y-%m-%d'))]))
            else:
                st.info("No hay trámites registrados.")
        except:
            st.error("Error en BD. Ejecuta ALTER TABLE.")

    # === TAB 3: Documentación del flujo técnico ===
    with tab3:
        st.markdown("### Flujo del Sistema")
        try:
            st.image("diagrama_flujo.png", width=600)
        except:
            st.image("https://i.imgur.com/5e5e5e5.png", width=600)
        st.markdown("""
        1. Ciudadano ingresa DNI + email
        2. Modelo ML predice prioridad
        3. Email automático
        4. Guardado en SQLite
        """)

    st.markdown("---")
    st.caption("**Danela Cuba** | SENATI 2025 | Taller de Desarrollo de Aplicaciones con Machine Learning")

# === EJECUTAR ===
# Inicia Flask en hilo daemon + Streamlit tras breve espera
if __name__ == '__main__':
    threading.Thread(target=flask_app.run, kwargs={'port': 5000}, daemon=True).start()
    import time; time.sleep(3)
    main()

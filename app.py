import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Detector PPE - Ralph Castellanos Couott",
    page_icon="🦺",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    /* Fondo principal */
    .stApp {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    }
    
    /* Título principal */
    .main-title {
        text-align: center;
        color: #0D47A1;
        font-size: 3rem;
        font-weight: bold;
        padding: 1rem;
        background: rgba(255,255,255,0.9);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Título secundario */
    .sub-title {
        text-align: center;
        color: #1565C0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Tarjeta de resultados */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Pie de página */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #1565C0;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Botones personalizados */
    .stButton > button {
        background-color: #1565C0;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0D47A1;
        transform: scale(1.02);
    }
    
    /* Contenedores de opciones */
    .option-container {
        background: rgba(255,255,255,0.85);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Sidebar con fondo oscuro y texto blanco */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255,255,255,0.2);
    }
    
    [data-testid="stSidebar"] .stAlert * {
        color: white !important;
    }
    
    /* Tabla con texto negro */
    .stDataFrame, .dataframe {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-title">🛡️ Detector de Elementos de Protección Personal (PPE) 🛡️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Desarrollado por: <strong>Ralph Castellanos Couott</strong></div>', unsafe_allow_html=True)

# Cargar el modelo con caché
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo 'best.pt': {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Diccionario de clases (ajusta según tus clases)
PPE_CLASSES = {
    0: "Casco",
    1: "Chaleco",
    2: "Guantes",
    3: "Gafas",
    4: "Mascarilla",
    5: "Calzado seguridad"
}

# Sidebar con información
with st.sidebar:
    st.markdown("## ℹ️ Información del Modelo")
    st.info(f"""
    **📊 Detalles técnicos:**
    - Modelo: YOLOv8
    - Clases detectadas: {len(PPE_CLASSES)} elementos
    - Confianza mínima ajustable
    - Épocas de entrenamiento: 84
    
    **🛡️ Elementos detectables:**
    - 🪖 Casco
    - 🦺 Chaleco  
    - 🧤 Guantes
    - 👓 Gafas
    - 😷 Mascarilla
    - 👞 Calzado de seguridad
    """)
    
    st.markdown("---")
    
    # Control deslizante para umbral de confianza
    conf_threshold = st.slider(
        "🎯 Umbral de confianza", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Solo se muestran detecciones con confianza mayor a este valor"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Instrucciones")
    st.markdown("""
    1. Selecciona una opción (Subir imagen o URL)
    2. Carga o ingresa la URL de la imagen
    3. Haz clic en "Analizar imagen"
    4. Revisa los resultados con bounding boxes
    """)

# Función para analizar imagen
def analyze_image(image, model, conf_threshold):
    """
    Analiza la imagen y devuelve detecciones y la imagen con bounding boxes
    """
    # Convertir PIL a formato OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Realizar predicción
    results = model.predict(source=img_cv, conf=conf_threshold)
    
    # Obtener detecciones
    detections = []
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            prob = float(box.conf[0])
            
            if prob >= conf_threshold:
                detections.append({
                    "Elemento": label.capitalize(),
                    "Confianza": f"{prob:.1%}",
                    "Confianza_valor": prob
                })
    
    # Dibujar resultados y convertir a RGB para Streamlit
    res_plotted = results[0].plot()  # Esto devuelve imagen en BGR
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Eliminar duplicados por elemento (mejor confianza)
    unique_detections = {}
    for d in detections:
        if d["Elemento"] not in unique_detections or d["Confianza_valor"] > unique_detections[d["Elemento"]]["Confianza_valor"]:
            unique_detections[d["Elemento"]] = d
    
    return list(unique_detections.values()), res_rgb

# Tabs para las opciones
tab1, tab2 = st.tabs(["📁 Subir imagen", "🔗 URL de imagen"])

# Variable para almacenar resultados en sesión
if 'detections' not in st.session_state:
    st.session_state['detections'] = None
if 'image_analyzed' not in st.session_state:
    st.session_state['image_analyzed'] = None
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None

# Tab 1: Subir archivo
with tab1:
    st.markdown("### Sube una imagen desde tu dispositivo")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        if st.button("🔍 Analizar imagen", key="btn_upload", use_container_width=True):
            with st.spinner("Analizando imagen... Esto puede tomar unos segundos"):
                detections, img_analyzed = analyze_image(image, model, conf_threshold)
                
                st.session_state['detections'] = detections
                st.session_state['image_analyzed'] = img_analyzed
                st.session_state['current_image'] = image
                st.rerun()

# Tab 2: URL
with tab2:
    st.markdown("### Ingresa la URL de una imagen")
    
    url = st.text_input(
        "URL de la imagen",
        placeholder="https://ejemplo.com/imagen.jpg",
        label_visibility="collapsed"
    )
    
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Imagen desde URL", use_container_width=True)
            
            if st.button("🔍 Analizar URL", key="btn_url", use_container_width=True):
                with st.spinner("Analizando imagen... Esto puede tomar unos segundos"):
                    detections, img_analyzed = analyze_image(image, model, conf_threshold)
                    
                    st.session_state['detections'] = detections
                    st.session_state['image_analyzed'] = img_analyzed
                    st.session_state['current_image'] = image
                    st.rerun()
        except requests.exceptions.Timeout:
            st.error("⏰ Tiempo de espera agotado. Verifica la URL")
        except Exception as e:
            st.error("❌ Error al cargar la imagen. Verifica que la URL sea válida")

# Mostrar resultados
st.markdown("---")
st.markdown("## 📊 Resultados del análisis")

if st.session_state['detections'] is not None:
    if len(st.session_state['detections']) > 0:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Imagen analizada")
            if st.session_state.get('image_analyzed') is not None:
                st.image(st.session_state['image_analyzed'], use_container_width=True)
                st.caption("🔍 Los recuadros muestran los elementos detectados con su nivel de confianza")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Elementos PPE detectados")
            
            # Crear DataFrame para la tabla
            df = pd.DataFrame([{
                "🛡️ Elemento": d["Elemento"],
                "📊 Confianza": d["Confianza"]
            } for d in st.session_state['detections']])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "🛡️ Elemento": st.column_config.TextColumn("🛡️ Elemento", width="medium"),
                    "📊 Confianza": st.column_config.TextColumn("📊 Confianza", width="small")
                }
            )
            
            st.success(f"✅ Se detectaron {len(st.session_state['detections'])} elementos PPE")
            
            # Mostrar barras de progreso
            st.markdown("#### 📈 Niveles de confianza:")
            for d in st.session_state['detections']:
                porcentaje = float(d['Confianza'].replace('%', ''))
                if porcentaje >= 75:
                    emoji = "🟢"
                elif porcentaje >= 60:
                    emoji = "🟡"
                else:
                    emoji = "🟠"
                
                st.markdown(f"**{d['Elemento']}** {emoji}: {d['Confianza']}")
                st.progress(porcentaje/100)
            
            st.markdown("---")
            st.caption(f"💡 Umbral de confianza actual: {conf_threshold:.0%} - Solo se muestran detecciones que superan este valor")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón para limpiar
        if st.button("🗑️ Limpiar resultados y analizar nueva imagen", use_container_width=True):
            st.session_state['detections'] = None
            st.session_state['image_analyzed'] = None
            st.session_state['current_image'] = None
            st.rerun()
    
    else:
        st.warning(f"⚠️ No se detectaron elementos PPE con confianza ≥ {conf_threshold:.0%}")
        
        if st.button("🗑️ Limpiar resultados", use_container_width=True):
            st.session_state['detections'] = None
            st.session_state['image_analyzed'] = None
            st.rerun()
else:
    st.info("👈 Selecciona una imagen (subiendo archivo o ingresando URL) y haz clic en 'Analizar' para comenzar")

# Pie de página
st.markdown("""
<div class="footer">
    <hr>
    <p>🔒 Umbral de confianza ajustable en el sidebar - Solo se muestran detecciones que superan este umbral</p>
    <p>🎓 Modelo PPE - Ciencia de Datos | Desarrollado por <strong>Ralph Castellanos Couott</strong></p>
    <p>📅 2026 - Detector de Elementos de Protección Personal (PPE) con YOLOv8</p>
</div>
""", unsafe_allow_html=True)
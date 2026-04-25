import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import cv2  # ← AÑADIR ESTA LÍNEA
import os
from io import BytesIO
import requests

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
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0D47A1;
        transform: scale(1.02);
    }
    
    /* Contenedores de opciones */
    .option-container {
        background: rgba(255,255,255,0.7);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-title">🛡️ Detector de Elementos de Protección Personal (PPE) 🛡️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Desarrollado por: <strong>Ralph Castellanos Couott</strong></div>', unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        # Verificar si best.pt existe
        if not os.path.exists("best.pt"):
            st.error("❌ No se encuentra el archivo 'best.pt' en el directorio actual")
            return None
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Clases PPE (ajusta según tus clases reales)
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
    st.image("https://img.icons8.com/color/96/000000/hard-hat.png", width=80)
    st.markdown("## ℹ️ Información del Modelo")
    st.info(f"""
    **📊 Detalles técnicos:**
    - Modelo: YOLOv8
    - Clases detectadas: {len(PPE_CLASSES)} elementos
    - Confianza mínima: 50%
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
    st.markdown("### 📌 Instrucciones")
    st.markdown("""
    1. Selecciona una opción (Subir imagen o URL)
    2. Carga o ingresa la URL de la imagen
    3. Haz clic en "Analizar imagen"
    4. Revisa los resultados en la tabla
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Estadísticas")
    st.metric("Confianza mínima", "50%")
    st.metric("Elementos detectables", len(PPE_CLASSES))

# Función para analizar imagen
def analyze_image(image, model, confidence_threshold=0.5):
    # Convertir PIL (RGB) → OpenCV (BGR) para que YOLO trabaje bien
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Realizar predicción
    results = model(img_cv, conf=confidence_threshold)

    detections = []
    if len(results) > 0:
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        detections.append({
                            "Elemento": PPE_CLASSES.get(cls, f"Clase {cls}"),
                            "Confianza": f"{conf:.1%}",
                            "Confianza_valor": conf
                        })

    # Eliminar duplicados
    unique_detections = {}
    for d in detections:
        if d["Elemento"] not in unique_detections or d["Confianza_valor"] > unique_detections[d["Elemento"]]["Confianza_valor"]:
            unique_detections[d["Elemento"]] = d

    # ← ESTO ES LO NUEVO: dibujar los cuadros y convertir de vuelta a RGB
    annotated_image = None
    if len(results) > 0:
        res_plotted = results[0].plot()  # Dibuja los bounding boxes (sale en BGR)
        annotated_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)  # Convertir a RGB para Streamlit

    return list(unique_detections.values()), annotated_image  # ← Ahora devuelve la imagen anotada

# Contenedor principal para las opciones
st.markdown("## 📋 Selecciona el método de entrada")

# Dos columnas para las opciones
col1, col2 = st.columns(2)

# Variable para almacenar resultados
if 'detections' not in st.session_state:
    st.session_state['detections'] = None
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None
if 'image_analyzed' not in st.session_state:
    st.session_state['image_analyzed'] = None

# Opción 1: Subir archivo
with col1:
    st.markdown('<div class="option-container">', unsafe_allow_html=True)
    st.markdown("### 📁 Opción 1: Subir imagen")
    st.markdown("Carga una imagen desde tu dispositivo")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        if st.button("🔍 Analizar imagen subida", key="btn_upload", use_container_width=True):
            with st.spinner("Analizando imagen... Esto puede tomar unos segundos"):
                detections, annotated_image = analyze_image(image, model)
                
                st.session_state['detections'] = detections
                st.session_state['current_image'] = image
                st.session_state['annotated_image'] = annotated_image
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Opción 2: URL
with col2:
    st.markdown('<div class="option-container">', unsafe_allow_html=True)
    st.markdown("### 🔗 Opción 2: URL de imagen")
    st.markdown("Ingresa la URL de una imagen en línea")
    
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
            
            if st.button("🔍 Analizar imagen desde URL", key="btn_url", use_container_width=True):
                with st.spinner("Analizando imagen... Esto puede tomar unos segundos"):
                    detections, annotated_image = analyze_image(image, model)
                    
                    st.session_state['detections'] = detections
                    st.session_state['current_image'] = image
                    st.session_state['annotated_image'] = annotated_image
                    st.rerun()
        except requests.exceptions.Timeout:
            st.error("⏰ Tiempo de espera agotado. Verifica la URL o intenta con otra imagen")
        except Exception as e:
            st.error(f"❌ Error al cargar la imagen: Verifica que la URL sea válida y accesible")
    st.markdown('</div>', unsafe_allow_html=True)

# Mostrar resultados
st.markdown("---")
st.markdown("## 📊 Resultados del análisis")

if st.session_state['detections'] is not None:
    if len(st.session_state['detections']) > 0:
        # Dos columnas para resultados
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Imagen analizada")
            if st.session_state.get('annotated_image') is not None:
                st.image(st.session_state['annotated_image'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with res_col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Elementos PPE detectados")
            
            if st.session_state['detections']:
                # Crear DataFrame para la tabla
                df = pd.DataFrame([{
                    "🛡️ Elemento": d["Elemento"],
                    "📊 Confianza": d["Confianza"]
                } for d in st.session_state['detections']])
                
                # Mostrar tabla estilizada
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "🛡️ Elemento": st.column_config.TextColumn("🛡️ Elemento", width="medium"),
                        "📊 Confianza": st.column_config.TextColumn("📊 Confianza", width="small")
                    }
                )
                
                # Resumen
                st.success(f"✅ Se detectaron {len(st.session_state['detections'])} elementos PPE")
                
                # Mostrar barras de progreso
                st.markdown("#### 📈 Niveles de confianza:")
                for d in st.session_state['detections']:
                    porcentaje = float(d['Confianza'].replace('%', ''))
                    # Color según nivel de confianza
                    if porcentaje >= 75:
                        color = "verde"
                        emoji = "🟢"
                    elif porcentaje >= 60:
                        color = "amarillo"
                        emoji = "🟡"
                    else:
                        color = "naranja"
                        emoji = "🟠"
                    
                    st.markdown(f"**{d['Elemento']}** {emoji}: {d['Confianza']}")
                    st.progress(porcentaje/100, text=f"Nivel de confianza: {d['Confianza']}")
                
                st.markdown("---")
                st.caption("💡 Los elementos con confianza < 50% no se muestran en los resultados")
            else:
                st.warning("⚠️ No se detectaron elementos PPE con confianza ≥ 50%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón para limpiar
        if st.button("🗑️ Limpiar resultados y analizar nueva imagen", use_container_width=True):
            st.session_state['detections'] = None
            st.session_state['current_image'] = None
            st.session_state['annotated_image'] = None
            st.rerun()
    
    else:
        st.warning("⚠️ No se detectaron elementos PPE con confianza ≥ 50% en la imagen analizada")
        
        if st.button("🗑️ Limpiar resultados", use_container_width=True):
            st.session_state['detections'] = None
            st.session_state['current_image'] = None
            st.session_state['annotated_image'] = None
            st.rerun()
else:
    st.info("👈 Selecciona una imagen (subiendo archivo o ingresando URL) y haz clic en 'Analizar' para comenzar")

# Pie de página
st.markdown("""
<div class="footer">
    <hr>
    <p>🔒 Confianza mínima: 50% - Solo se muestran detecciones que superan este umbral</p>
    <p>🎓 Modelo PPE - Ciencia de Datos | Desarrollado por <strong>Ralph Castellanos Couott</strong></p>
    <p>📅 2026 - Detector de Elementos de Protección Personal (PPE) con YOLOv8</p>
</div>
""", unsafe_allow_html=True)
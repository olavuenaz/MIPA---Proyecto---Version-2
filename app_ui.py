"""
MIPA Console - Multi-Agent System Dashboard
"""

import streamlit as st
import requests
import json
import os
import time
import logging
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Centralized configuration for API endpoints."""
    BASE_URL: str = "http://127.0.0.1:8000"
    CHAT_ENDPOINT: str = "/chat"
    PROFILE_ENDPOINT: str = "/update_profile"
    TIMEOUT: int = 60

@dataclass
class UIConfig:
    """User Interface configuration settings."""
    PAGE_TITLE: str = "MIPA Console"
    PAGE_ICON: str = "bar-chart" 
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"

EMOTION_COLORS = {
    "Ira": "#4a0404",
    "Frustración": "#5c1e1e",
    "Calma": "#0e2a47",
    "Tristeza": "#161b33",
    "Alegría": "#47400e",
    "Ansiedad": "#2e0e47",
    "Neutral": "#0E1117"
}

def setup_page_config() -> None:
    """Configures the initial parameters of the Streamlit page."""
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout=UIConfig.LAYOUT,
        initial_sidebar_state=UIConfig.SIDEBAR_STATE
    )

def load_json_file(filepath: str, default: Any = None) -> Any:
    """
    Safely loads a JSON file with robust error handling.
    
    Args:
        filepath (str): Path to the JSON file.
        default (Any): Default value to return if an error occurs.
        
    Returns:
        Any: Parsed JSON data or the default value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("File not found: %s", filepath)
        st.error(f"Archivo no encontrado: {filepath}")
        return default if default is not None else {}
    except json.JSONDecodeError as e:
        logger.error("JSON decode error in %s: %s", filepath, str(e))
        st.error(f"Error decodificando JSON en {filepath}")
        return default if default is not None else {}
    except Exception as e:
        logger.error("Unexpected error loading %s: %s", filepath, str(e))
        st.error(f"Error inesperado cargando {filepath}")
        return default if default is not None else {}

@st.cache_data(ttl=60)
def load_user_profiles() -> Dict[str, Any]:
    """Loads and caches user profiles."""
    return load_json_file("./data_memory/user_profiles.json", {})

@st.cache_data(ttl=60)
def load_templates() -> Dict[str, Any]:
    """Loads and caches system templates."""
    return load_json_file("./data_memory/templates.json", {"error": "No se encontró templates.json"})

def load_dataframe_global() -> pd.DataFrame:
    """
    Efficiently loads and concatenates all CSV logs with error handling.
    
    Returns:
        pd.DataFrame: Consolidated DataFrame of all system logs.
    """
    logs_dir = "./data_memory/logs"
    
    if not os.path.exists(logs_dir):
        return pd.DataFrame()

    all_data = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(logs_dir, filename)
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning("Corrupted CSV log skipped: %s - %s", filepath, str(e))
                continue
    
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        if "Timestamp" in full_df.columns:
            full_df["Timestamp"] = pd.to_datetime(full_df["Timestamp"], errors='coerce')
            full_df = full_df.sort_values(by="Timestamp", ascending=False)
        return full_df
    
    return pd.DataFrame()

def set_emotional_theme(emotion: str) -> None:
    """
    Applies a CSS color theme based on the detected emotion.
    
    Args:
        emotion (str): The emotion label used to select the background color.
    """
    bg_color = EMOTION_COLORS.get(emotion, "#0E1117")
    
    css = f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        transition: background-color 1.5s ease;
    }}
    p, h1, h2, h3, h4, span, div, label {{
        color: #e0e0e0 !important;
    }}
    .stJson {{
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        padding: 10px;
    }}
    .stMetric {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_radar_chart(traits: Dict[str, float]) -> go.Figure:
    """
    Generates a Plotly radar chart to visualize OCEAN personality traits.
    
    Args:
        traits (Dict[str, float]): Dictionary containing OCEAN trait values.
        
    Returns:
        go.Figure: The rendered Plotly figure.
    """
    categories = ['O (Apertura)', 'C (Conciencia)', 'E (Extraversión)', 
                  'A (Amabilidad)', 'N (Neuroticismo)']
    
    values = [
        traits.get("O", 0.5),
        traits.get("C", 0.5),
        traits.get("E", 0.5),
        traits.get("A", 0.5),
        traits.get("N", 0.5)
    ]
    
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Perfil',
                line_color='#00CC96'
            )
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False
                )
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=10)
        )
    )
    
    return fig

def render_sidebar(profiles: Dict[str, Any], templates: Dict[str, Any]) -> str:
    """
    Renders the sidebar controls and profile visualizations.
    
    Args:
        profiles (Dict[str, Any]): Dictionary of loaded user profiles.
        templates (Dict[str, Any]): System templates for inspection.
        
    Returns:
        str: The currently selected user ID.
    """
    with st.sidebar:
        st.header("Control de Usuario")
        
        user_options = list(profiles.keys()) if profiles else ["default"]
        selected_user = st.selectbox(
            "Usuario Activo:",
            user_options,
            format_func=lambda x: profiles.get(x, {}).get('nombre', x) if x in profiles else x
        )
        
        if selected_user in profiles:
            render_user_profile(selected_user, profiles[selected_user])
        
        render_inspection_panel(templates, profiles, selected_user)
        
        return selected_user

def render_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Renders the user profile radar chart and calibration sliders."""
    traits = profile.get("ocean_traits", {})
    
    st.plotly_chart(create_radar_chart(traits), use_container_width=True)
    
    with st.expander("Calibrar Rasgos (OCEAN)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            val_o = st.slider("O - Apertura", 0.0, 1.0, float(traits.get("O", 0.5)))
            val_c = st.slider("C - Conciencia", 0.0, 1.0, float(traits.get("C", 0.5)))
            val_e = st.slider("E - Extraversión", 0.0, 1.0, float(traits.get("E", 0.5)))
        
        with col2:
            val_a = st.slider("A - Amabilidad", 0.0, 1.0, float(traits.get("A", 0.5)))
            val_n = st.slider("N - Neuroticismo", 0.0, 1.0, float(traits.get("N", 0.5)))
        
        if st.button("Guardar Perfil", use_container_width=True):
            update_user_profile(user_id, {
                "O": val_o, "C": val_c, "E": val_e, 
                "A": val_a, "N": val_n
            })

def update_user_profile(user_id: str, traits: Dict[str, float]) -> None:
    """Sends a profile update request to the API."""
    try:
        response = requests.post(
            f"{APIConfig.BASE_URL}{APIConfig.PROFILE_ENDPOINT}",
            json={
                "user_id": user_id,
                "ocean_traits": traits
            },
            timeout=APIConfig.TIMEOUT
        )
        
        if response.status_code == 200:
            st.toast("Perfil guardado exitosamente.")
            st.cache_data.clear()
            time.sleep(0.5)
            st.rerun()
        else:
            logger.error("Failed to update profile: HTTP %s", response.status_code)
            st.error(f"Error del servidor: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while updating profile.")
        st.error("No se pudo conectar con la API. Verifica que el backend esté en ejecución.")
    except Exception as e:
        logger.error("Unexpected error updating profile: %s", str(e))
        st.error("Error inesperado al guardar el perfil.")

def render_inspection_panel(templates: Dict[str, Any], profiles: Dict[str, Any], selected_user: str) -> None:
    """Renders data inspection panels for debugging and forensic tracking."""
    st.subheader("Inspector de Parámetros")
    
    with st.expander("Configuración de Arquetipos (Templates)", expanded=False):
        st.json(templates)
    
    with st.expander("Perfil de Usuario (Raw JSON)", expanded=False):
        if selected_user in profiles:
            st.json(profiles[selected_user])
    
    with st.expander("Última Traza de Proceso", expanded=True):
        if st.session_state.get("last_trace"):
            st.json(st.session_state.last_trace)
        else:
            st.info("Sin datos de proceso recientes.")

def render_metrics_panel(trace_data: Dict[str, Any]) -> None:
    """
    Renders real-time interaction metrics.
    
    Args:
        trace_data (Dict[str, Any]): The latest cognitive trace from the reasoning module.
    """
    if not trace_data:
        return
    
    emotion = trace_data.get("percepcion", {}).get("emotion_label", "Neutral")
    set_emotional_theme(emotion)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Arquetipo Seleccionado", trace_data.get("decision", {}).get("selected_archetype", "N/A"))
    with col2:
        st.metric("Emoción Detectada", emotion)
    with col3:
        latency = trace_data.get("telemetria", {}).get("total_ms", 0)
        st.metric("Latencia", f"{latency} ms")
    with col4:
        neuroticism = trace_data.get("perfil_actualizado", {}).get("N", 0.5)
        st.metric("Nivel de Neuroticismo", f"{neuroticism:.2f}")

def render_chat_history() -> None:
    """Iterates through session state messages and renders the chat UI."""
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def render_dashboard() -> None:
    """Renders the global scientific dashboard based on historical forensic logs."""
    st.header("Dashboard Científico - Analítica Global")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Refrescar Datos", width="stretch"):
            st.cache_data.clear()
            st.rerun()
            
    df = load_dataframe_global()
    
    if df.empty:
        st.info("Sin datos registrados. Interactúa con el sistema para generar logs.")
        return
        
    st.subheader("Métricas de Rendimiento")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Total de Interacciones", len(df))
    with m2:
        dominant_archetype = df["Archetype"].mode()[0] if "Archetype" in df.columns else "N/A"
        st.metric("Arquetipo Dominante", dominant_archetype)
    with m3:
        frequent_emotion = df["Emotion"].mode()[0] if "Emotion" in df.columns else "N/A"
        st.metric("Emoción Frecuente", frequent_emotion)
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribución de Arquetipos")
        if "Archetype" in df.columns:
            st.bar_chart(df["Archetype"].value_counts(), color="#FF4B4B")
            
    with col_right:
        st.subheader("Frecuencia Emocional")
        if "Emotion" in df.columns:
            st.bar_chart(df["Emotion"].value_counts(), color="#00CC96")
            
    st.subheader("Registro Forense Detallado")
    st.dataframe(df, use_container_width=True, hide_index=True)

def send_chat_message(user_id: str, message: str) -> Optional[Dict[str, Any]]:
    """
    Sends the user prompt to the backend API and retrieves the response.
    
    Args:
        user_id (str): The active user's ID.
        message (str): The input text.
        
    Returns:
        Optional[Dict]: The parsed JSON response, or None if the request fails.
    """
    try:
        response = requests.post(
            f"{APIConfig.BASE_URL}{APIConfig.CHAT_ENDPOINT}",
            json={"user_id": user_id, "text": message}, 
            timeout=APIConfig.TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
            
        logger.error("Chat API returned status code %s", response.status_code)
        st.error(f"Error del servidor: HTTP {response.status_code}")
        return None
            
    except requests.exceptions.ConnectionError:
        logger.error("Connection error communicating with the chat endpoint.")
        st.error("No se pudo conectar con el backend. Verifica que el servidor esté en ejecución.")
        return None
    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to the chat endpoint.")
        st.error("Timeout: El servidor tardó demasiado en responder.")
        return None
    except Exception as e:
        logger.error("Unexpected error during chat request: %s", str(e))
        st.error("Error inesperado al procesar la solicitud.")
        return None

def main() -> None:
    """Main application entry point."""
    setup_page_config()
    st.title("MIPA: Consola de Gestión Multi-Agente")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None
        
    profiles = load_user_profiles()
    templates = load_templates()
    
    selected_user = render_sidebar(profiles, templates)
    
    if selected_user != st.session_state.current_user_id:
        st.session_state.current_user_id = selected_user
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()
        
    tab_chat, tab_dashboard = st.tabs(["Chat & Monitor", "Dashboard Global"])
    
    with tab_chat:
        render_metrics_panel(st.session_state.last_trace)
        render_chat_history()
        
    with tab_dashboard:
        render_dashboard()
        
    if prompt := st.chat_input("Escribe tu mensaje..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with tab_chat:
            with st.chat_message("user"):
                st.markdown(prompt)
                
        with st.spinner("Procesando inferencia..."):
            result = send_chat_message(selected_user, prompt)
            
            if result:
                st.session_state.last_trace = result.get("traza_cognitiva")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("respuesta_ia", "No se recibió respuesta válida del sistema.")
                })
                st.rerun()

if __name__ == "__main__":
    main()
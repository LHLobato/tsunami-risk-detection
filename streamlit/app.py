import streamlit as st
import requests
import json


API_URL = "http://localhost:5000/classify/tsunami"


POSSIBLE_MAG_TYPES = [
    'mb', 'mb_lg', 'md', 'mh', 'ml', 'mlv', 'ms_vx', 'mw', 'mwb', 'mwr', 'mww'
]

POSSIBLE_EVENT_TYPES = [
    'earthquake', 'explosion', 'ice quake', 'landslide',
    'mine collapse', 'mining explosion', 'other event', 'quarry blast',
    'volcanic eruption'
]


st.set_page_config(
    page_title="Detector de Tsunami",
    page_icon="🌊",
    layout="centered"
)


st.sidebar.title("Parâmetros do Evento Sísmico")
st.sidebar.info("Insira os dados do evento para análise.")


lat = st.sidebar.number_input('Latitude (-90 a 90)', min_value=-90.0, max_value=90.0, value=38.32, format="%.4f")
lon = st.sidebar.number_input('Longitude (-180 a 180)', min_value=-180.0, max_value=180.0, value=142.36, format="%.4f")
depth = st.sidebar.number_input('Profundidade (km)', min_value=0.0, value=33.7, format="%.1f")
mag = st.sidebar.number_input('Magnitude (mag)', min_value=0.0, max_value=10.0, value=6.9, format="%.1f")
sig = st.sidebar.number_input('Significance (sig)', min_value=0, value=750, step=10)

selected_mag_type = st.sidebar.selectbox(
    'Tipo de Magnitude',
    POSSIBLE_MAG_TYPES,
    index=10 
)
selected_event_type = st.sidebar.selectbox(
    'Tipo de Evento',
    POSSIBLE_EVENT_TYPES,
    index=0 
)

analyze_button = st.sidebar.button("Analisar Risco de Tsunami", type="primary")


st.title("🌊 Detector de Risco de Tsunami")
st.write("Esta interface consome a API de Machine Learning local para classificar eventos sísmicos.")
st.write("---")

if analyze_button:

    properties_payload = {}
    properties_payload['mag'] = mag
    properties_payload['sig'] = sig

    for mag_type in POSSIBLE_MAG_TYPES:
        key = f"magType_{mag_type}"
        properties_payload[key] = 1 if mag_type == selected_mag_type else 0


    for event_type in POSSIBLE_EVENT_TYPES:
        key = f"type_{event_type}"
        properties_payload[key] = 1 if event_type == selected_event_type else 0

    final_payload = {
        "profundidade": depth,
        "longitude": lon,
        "latitude": lat,
        "properties": properties_payload
    }


    with st.spinner("Analisando evento... Chamando a API..."):
        try:
            response = requests.post(API_URL, json=final_payload)
            response.raise_for_status() 


            data = response.json()
            is_risk = data.get("is_tsunami_risk", False)
            prob_yes = data.get("probability_tsunami_risk", 0) * 100
            prob_no = data.get("probability_no_tsunami", 0) * 100
            note = data.get("note") 

            if is_risk:
                st.warning(f"**ALERTA: RISCO DE TSUNAMI DETECTADO**", icon="🌊")
            else:
                st.success(f"**SEGURO: Baixo Risco de Tsunami Detectado**", icon="✅")
            
            if note:
                st.info(f"Nota do Analisador: {note}")

            col1, col2 = st.columns(2)
            col1.metric("Probabilidade de Risco", f"{prob_yes:.1f}%")
            col2.metric("Probabilidade de Segurança", f"{prob_no:.1f}%")

            with st.expander("Ver JSON enviado para a API (Request)"):
                st.json(final_payload)
            with st.expander("Ver JSON recebido da API (Response)"):
                st.json(data)

        except requests.exceptions.ConnectionError:
            st.error(f"**Erro de Conexão:** Não foi possível conectar à API em `{API_URL}`.")
            st.info("Verifique se o seu servidor Flask (app.py) está rodando no terminal.")
        except requests.exceptions.RequestException as e:
            st.error(f"**Erro na API:** A API retornou um erro.")
            st.info(f"Detalhes: {e}")
            with st.expander("JSON enviado (que causou o erro)"):
                st.json(final_payload)

else:
    st.info("Preencha os parâmetros no menu à esquerda e clique em 'Analisar Risco de Tsunami'.")
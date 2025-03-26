import streamlit as st
from groq import Groq
import PyPDF2

# Configuración de la ventana de la web
st.set_page_config(page_title="CV Analyzer", page_icon="📄")

MODELO = ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768', 'deepseek-r1-distill-llama-70b']

# Extraer texto del PDF
def process_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Dividir el texto en fragmentos manejables
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start == chunk_size:
            # Find the last period or newline to make chunks more meaningful
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            end = max(last_period, last_newline) + 1 if max(last_period, last_newline) > 0 else end

        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length

    return chunks

# Nos conecta a la API, crear un usuario
def crear_usuario_groq():
    clave_secreta = st.secrets["CLAVE_API"]
    return Groq(api_key=clave_secreta)

# Configurar el modelo con contexto del CV cuando sea apropiado
def configurar_modelo(cliente, modelo, mensaje):
    # Si tenemos un CV cargado y no es un análisis automático
    if "cv_text" in st.session_state and st.session_state.cv_text and not mensaje.startswith("Eres un experto"):
        # Añadir contexto del CV al mensaje
        mensaje_con_contexto = f"""
        Eres un asistente experto en analizar CVs y proporcionar retroalimentación útil.

        El usuario ha cargado un CV con el siguiente contenido:
        {st.session_state.cv_text[:3000]}...

        Consulta del usuario: {mensaje}

        Responde de manera útil, específica y concisa.
        """
        return cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": mensaje_con_contexto}],
            stream=True
        )
    else:
        # Comportamiento normal para mensajes sin CV o análisis automático
        return cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": mensaje}],
            stream=True
        )

# Inicializar estado de la aplicación
def inicializar_estado():
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    if "cv_chunks" not in st.session_state:
        st.session_state.cv_chunks = []
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""
    if "cv_analizado" not in st.session_state:
        st.session_state.cv_analizado = False

def actualizar_historial(rol, contenido, avatar):
    st.session_state.mensajes.append(
        {"role": rol, "content": contenido, "avatar": avatar}
    )

def mostrar_historial():
    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"], avatar=mensaje["avatar"]):
            st.markdown(mensaje["content"])

# Sector del chat en web
def area_chat():
    contenedorDelChat = st.container(height=400, border=True)
    with contenedorDelChat:
        mostrar_historial()

# Generar respuesta streaming
def generar_respuesta(chat_completo):
    respuesta_completa = ""
    for frase in chat_completo:
        if frase.choices[0].delta.content:
            respuesta_completa += frase.choices[0].delta.content
            yield frase.choices[0].delta.content

    return respuesta_completa

# Función para analizar el CV automáticamente
def analizar_cv_automaticamente():
    # Verificar si ya se analizó el CV actual
    if st.session_state.cv_analizado:
        return

    cv_text = st.session_state.cv_text
    if not cv_text:
        st.warning("Por favor, sube un CV primero.")
        return

    analysis_prompt = """
    Eres un experto en recursos humanos especializado en analizar CVs.
    Analiza el siguiente CV y proporciona:

    1. Un resumen de fortalezas
    2. Áreas de mejora
    3. Recomendaciones específicas

    CV a analizar:
    {cv_text}

    Responde de manera concisa y útil.
    """

    mensaje = analysis_prompt.format(cv_text=cv_text)
    actualizar_historial("user", "Analiza este CV y proporciona recomendaciones de mejora.", "🧚‍♀️")

    clienteUsuario = crear_usuario_groq()
    # Usamos el modelo más potente para el análisis inicial
    chat_completo = configurar_modelo(clienteUsuario, "llama3-70b-8192", mensaje)

    if chat_completo:
        with st.chat_message("assistant"):
            respuesta_completa = st.write_stream(generar_respuesta(chat_completo))
            actualizar_historial("assistant", respuesta_completa, "🤖")
            # Marcar como analizado para evitar múltiples análisis
            st.session_state.cv_analizado = True
            st.rerun()

def configurar_pagina():
    st.title("Chat con tus CVs usando Groq")
    st.sidebar.title("Configuración")

    # Selector de modelo
    elegirModelo = st.sidebar.selectbox(
        "Elegir modelo",
        MODELO,
        index=0
    )

    # Uploader de CV
    uploaded_file = st.sidebar.file_uploader("Sube tu CV en PDF", type="pdf")
    if uploaded_file is not None and (
        "cv_filename" not in st.session_state or
        st.session_state.cv_filename != uploaded_file.name
    ):
        st.session_state.cv_filename = uploaded_file.name
        cv_text = process_pdf(uploaded_file)
        st.session_state.cv_text = cv_text
        st.session_state.cv_chunks = chunk_text(cv_text)
        st.sidebar.success(f"CV cargado: {uploaded_file.name}")

        # Reiniciar análisis para el nuevo CV
        st.session_state.cv_analizado = False

        # Ejecutar análisis automáticamente cuando se carga un nuevo CV
        st.session_state.mensajes = [] # Limpiar historial de chat para nuevo CV
        analizar_cv_automaticamente()

    return elegirModelo

def main():
    # INVOCANDO FUNCIONES DEL CHATBOT
    inicializar_estado() # Inicializa historial vacío y variables de estado
    modelo = configurar_pagina() # Llamamos a la función, carga CV y analiza automáticamente
    clienteUsuario = crear_usuario_groq() # Conectamos a la API a través de un usuario
    area_chat() # pone en la web el contenedor del chat

    # Input de chat - Permite continuar la conversación después del análisis inicial
    mensaje = st.chat_input("Haz preguntas sobre tu CV o pide más recomendaciones...")

    if mensaje:
        actualizar_historial("user", mensaje, "🧚‍♀️") # Mostramos el mensaje del usuario
        chat_completo = configurar_modelo(clienteUsuario, modelo, mensaje) # obteniendo la respuesta
        if chat_completo: # verificamos que tenga contenido
            with st.chat_message("assistant"):
                respuesta_completa = st.write_stream(generar_respuesta(chat_completo))
                actualizar_historial("assistant", respuesta_completa, "🤖")
                st.rerun() # Actualizar

if __name__ == "__main__":
    main() # una función principal y siempre se invoca

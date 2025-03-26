import streamlit as st
from groq import Groq
import PyPDF2

# Importaciones de los nuevos módulos
from visualizador import extract_skills_categories, create_radar_chart, create_experience_chart, create_education_chart, create_skills_balance_chart
from ats_analyzer import analyze_ats_compatibility, display_ats_results, analyze_job_match, display_job_match_results
from career_path import generate_career_paths, visualize_career_paths, calculate_skill_gaps, display_skill_gaps
from portfolio_generator import generate_portfolio_suggestions, display_portfolio_plan, generate_project_details, display_project_details

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
    # Intenta obtener la clave de session_state primero, si existe
    if "GROQ_API_KEY" in st.session_state and st.session_state.GROQ_API_KEY:
        return Groq(api_key=st.session_state.GROQ_API_KEY)

    # Si no está en session_state, intenta obtenerla de secrets
    try:
        clave_secreta = st.secrets["CLAVE_API"]
        return Groq(api_key=clave_secreta)
    except KeyError:
        # Si no hay clave configurada, devuelve None
        return None

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
    if "tab_selected" not in st.session_state:
        st.session_state.tab_selected = "Chat básico"
    if "skills_data" not in st.session_state:
        st.session_state.skills_data = None
    if "ats_results" not in st.session_state:
        st.session_state.ats_results = None
    if "career_paths" not in st.session_state:
        st.session_state.career_paths = None
    if "portfolio_suggestions" not in st.session_state:
        st.session_state.portfolio_suggestions = None

# Pantalla de configuración de API Key
def configurar_api_key():
    st.title("Configuración")

    st.subheader("API Key de Groq")

    # Mostrar campo para introducir la API key
    api_key = st.text_input(
        "API Key de Groq",
        value=st.session_state.get("GROQ_API_KEY", ""),
        type="password",
        help="Ingresa tu API key de Groq para continuar"
    )

    # Mensaje informativo
    st.info("Por favor, ingresa tu API key de Groq para continuar")

    # Botón para guardar la API key
    if st.button("Guardar API Key"):
        if api_key:
            st.session_state.GROQ_API_KEY = api_key
            st.success("API Key guardada correctamente")
            st.rerun()
        else:
            st.error("Por favor, ingresa una API Key válida")

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

    # Agregar opción para cambiar la API key
    if st.sidebar.button("Cambiar API Key de Groq"):
        st.session_state.GROQ_API_KEY = ""
        st.rerun()

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
        # También reiniciamos los datos de análisis de las nuevas funcionalidades
        st.session_state.skills_data = None
        st.session_state.ats_results = None
        st.session_state.career_paths = None
        st.session_state.portfolio_suggestions = None

        # Ejecutar análisis automáticamente cuando se carga un nuevo CV
        st.session_state.mensajes = [] # Limpiar historial de chat para nuevo CV
        # Solo ejecutamos el análisis automático si estamos en la pestaña de chat básico
        if st.session_state.tab_selected == "Chat básico":
            analizar_cv_automaticamente()

    # Añadir pestañas para las nuevas funcionalidades
    tab_selected = st.sidebar.radio(
        "Seleccionar funcionalidad",
        ["Chat básico", "Visualización", "Análisis ATS", "Proyección profesional", "Portafolio digital"]
    )

    # Actualizamos la pestaña seleccionada en el estado de la sesión
    st.session_state.tab_selected = tab_selected

    return elegirModelo

def main():
    # INVOCANDO FUNCIONES DEL CHATBOT
    inicializar_estado() # Inicializa historial vacío y variables de estado

    # Verificar si la API key está configurada
    clienteUsuario = crear_usuario_groq()

    if not clienteUsuario:
        configurar_api_key()
        return  # Detener la ejecución hasta que se configure la API key

    # Continuar con el flujo normal si la API key está configurada
    modelo = configurar_pagina() # Llamamos a la función, carga CV y actualiza la pestaña seleccionada

    # Verificar si hay un CV cargado
    if not st.session_state.cv_text:
        st.info("Para comenzar, por favor sube tu CV en formato PDF utilizando el panel lateral.")
        return

    # Manejo de las diferentes pestañas
    if st.session_state.tab_selected == "Chat básico":
        # Código existente del chat
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

    elif st.session_state.tab_selected == "Visualización":
        st.header("Visualización interactiva de tu CV")

        if st.button("Generar visualizaciones"):
            with st.spinner("Analizando tu CV para generar visualizaciones..."):
                # Si no tenemos datos de habilidades, los obtenemos
                if not st.session_state.skills_data:
                    st.session_state.skills_data = extract_skills_categories(
                        st.session_state.cv_text,
                        clienteUsuario,
                        modelo
                    )

                # Verificamos si se pudieron obtener los datos
                if st.session_state.skills_data:
                    # Creamos las visualizaciones
                    col1, col2 = st.columns(2)

                    with col1:
                        radar_chart = create_radar_chart(st.session_state.skills_data)
                        if radar_chart:
                            st.plotly_chart(radar_chart, use_container_width=True)

                    with col2:
                        skills_balance = create_skills_balance_chart(st.session_state.skills_data)
                        if skills_balance:
                            st.plotly_chart(skills_balance, use_container_width=True)

                    experience_chart = create_experience_chart(st.session_state.skills_data)
                    if experience_chart:
                        st.plotly_chart(experience_chart, use_container_width=True)

                    education_chart = create_education_chart(st.session_state.skills_data)
                    if education_chart:
                        st.plotly_chart(education_chart, use_container_width=True)
                else:
                    st.error("No se pudieron generar las visualizaciones. Por favor, inténtalo de nuevo.")

    elif st.session_state.tab_selected == "Análisis ATS":
        st.header("Análisis de compatibilidad con ATS")

        # Opciones para el análisis ATS
        ats_options = st.radio(
            "Selecciona el tipo de análisis",
            ["Análisis general", "Análisis con descripción de trabajo específica"]
        )

        if ats_options == "Análisis general":
            # Botón para iniciar el análisis ATS general
            if st.button("Analizar compatibilidad con ATS"):
                with st.spinner("Analizando tu CV para sistemas ATS..."):
                    if not st.session_state.ats_results:
                        st.session_state.ats_results = analyze_ats_compatibility(
                            st.session_state.cv_text,
                            "",  # Sin descripción de trabajo específica
                            clienteUsuario,
                            modelo
                        )

                    # Mostrar los resultados
                    display_ats_results(st.session_state.ats_results)
        else:
            # Campo para ingresar la descripción del trabajo
            job_description = st.text_area(
                "Ingresa la descripción del puesto de trabajo",
                height=200,
                help="Pega aquí la descripción de la oferta de trabajo para analizar la compatibilidad específica."
            )

            # Botón para iniciar el análisis de coincidencia con la descripción del trabajo
            if st.button("Analizar compatibilidad con el puesto"):
                if job_description:
                    with st.spinner("Analizando la compatibilidad de tu CV con el puesto..."):
                        # Análisis ATS con la descripción del trabajo
                        ats_results = analyze_ats_compatibility(
                            st.session_state.cv_text,
                            job_description,
                            clienteUsuario,
                            modelo
                        )

                        # Mostrar los resultados del análisis ATS
                        display_ats_results(ats_results)

                        # Análisis de coincidencia con el puesto
                        match_results = analyze_job_match(
                            st.session_state.cv_text,
                            job_description,
                            clienteUsuario,
                            modelo
                        )

                        # Mostrar los resultados de la coincidencia
                        display_job_match_results(match_results)
                else:
                    st.warning("Por favor, ingresa la descripción del puesto de trabajo para continuar.")

    elif st.session_state.tab_selected == "Proyección profesional":
        st.header("Proyección profesional")

        # Opciones para la proyección profesional
        career_options = st.radio(
            "Selecciona el tipo de análisis",
            ["Trayectorias profesionales", "Análisis de brecha de habilidades"]
        )

        if career_options == "Trayectorias profesionales":
            # Botón para generar trayectorias profesionales
            if st.button("Generar trayectorias profesionales"):
                with st.spinner("Generando posibles trayectorias profesionales..."):
                    if not st.session_state.career_paths:
                        st.session_state.career_paths = generate_career_paths(
                            st.session_state.cv_text,
                            clienteUsuario,
                            modelo
                        )

                    # Visualizar las trayectorias
                    visualize_career_paths(st.session_state.career_paths)
        else:
            # Campo para ingresar el rol objetivo
            target_role = st.text_input(
                "Ingresa el rol profesional al que aspiras",
                help="Ejemplo: 'Data Scientist', 'Project Manager', 'Full Stack Developer'"
            )

            # Botón para analizar la brecha de habilidades
            if st.button("Analizar brecha de habilidades"):
                if target_role:
                    with st.spinner(f"Analizando brecha de habilidades para el rol de {target_role}..."):
                        # Calcular la brecha de habilidades
                        skill_gap_data = calculate_skill_gaps(
                            st.session_state.cv_text,
                            target_role,
                            clienteUsuario,
                            modelo
                        )

                        # Mostrar el análisis de brecha de habilidades
                        display_skill_gaps(skill_gap_data, target_role)
                else:
                    st.warning("Por favor, ingresa el rol profesional al que aspiras para continuar.")

    elif st.session_state.tab_selected == "Portafolio digital":
        st.header("Portafolio digital inteligente")

        # Botón para generar sugerencias de portafolio
        if st.button("Generar sugerencias para portafolio"):
            with st.spinner("Generando sugerencias para tu portafolio digital..."):
                if not st.session_state.portfolio_suggestions:
                    st.session_state.portfolio_suggestions = generate_portfolio_suggestions(
                        st.session_state.cv_text,
                        clienteUsuario,
                        modelo
                    )

                # Mostrar las sugerencias de portafolio
                display_portfolio_plan(st.session_state.portfolio_suggestions)

        # Si ya tenemos sugerencias de portafolio, mostramos la opción para generar detalles de proyecto
        if st.session_state.portfolio_suggestions:
            st.subheader("Generar detalles de proyecto")

            # Obtenemos los proyectos destacados
            projects = st.session_state.portfolio_suggestions.get("highlighted_projects", [])
            if projects:
                # Creamos una lista de nombres de proyectos
                project_names = [project.get("name", f"Proyecto {i+1}") for i, project in enumerate(projects)]

                # Selector de proyecto
                selected_project = st.selectbox(
                    "Selecciona un proyecto para obtener detalles de implementación",
                    project_names
                )

                # Botón para generar detalles del proyecto
                if st.button("Generar detalles del proyecto"):
                    with st.spinner(f"Generando detalles para el proyecto: {selected_project}..."):
                        # Generar detalles del proyecto
                        project_details = generate_project_details(
                            selected_project,
                            st.session_state.cv_text,
                            clienteUsuario,
                            modelo
                        )

                        # Mostrar los detalles del proyecto
                        display_project_details(project_details)

if __name__ == "__main__":
    main() # una función principal y siempre se invoca

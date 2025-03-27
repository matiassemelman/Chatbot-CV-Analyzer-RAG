# Chatbot CV Analyzer con RAG

## Descripción General
Este proyecto es un chatbot de inteligencia artificial especializado en analizar CVs. Utiliza Streamlit para la interfaz web y la API de Google AI Studio (Gemini) para acceder a modelos de lenguaje grandes (LLMs). El sistema permite cargar CVs en formato PDF, analizarlos automáticamente y proporcionar recomendaciones para mejorarlos. Forma parte del proyecto final del curso Talento Tech.

## Configuración
Para utilizar la aplicación, los usuarios necesitan tener una API Key de Google AI Studio, que pueden obtener registrándose en [Google AI Studio](https://ai.google.dev/). La aplicación proporciona una interfaz para ingresar esta clave al inicio o cambiarla posteriormente si es necesario.

## Estructura del Proyecto
- `app.py`: Código principal de la aplicación
- `visualizador.py`: Componente para visualizaciones interactivas del CV
- `ats_analyzer.py`: Análisis de compatibilidad con sistemas ATS
- `career_path.py`: Generación de trayectorias profesionales
- `portfolio_generator.py`: Sugerencias para portafolio digital
- `requirements.txt`: Dependencias del proyecto
- `.streamlit/`: Configuración para Streamlit
- `README.md`: Información básica del proyecto
- `plan_chatbot_cv_analyzer.md`: Plan detallado de implementación

## Tecnologías Utilizadas
- **Frontend**: Streamlit
- **Backend**: Python
- **API de IA**: Google AI Studio (Gemini)
- **Procesamiento de PDFs**: PyPDF2
- **Visualización**: Plotly
- **Modelos de IA**:
  - gemini-1.0-pro
  - gemini-1.5-pro
  - gemini-1.5-flash

## Funcionalidades
- Interfaz de chat interactiva
- Configuración de API Key de Google AI Studio desde la interfaz
- Carga y procesamiento de CVs en formato PDF
- Análisis automático de CVs con recomendaciones
- Consultas específicas sobre secciones del CV
- Selección de diferentes modelos de Gemini
- Historial de conversación visual
- Respuestas generadas en tiempo real (streaming)

### Nuevas Funcionalidades
- **Visualización interactiva**: Gráficos que muestran el balance de habilidades técnicas vs. blandas, experiencia y formación
- **Análisis de palabras clave ATS**: Identificación de la compatibilidad con sistemas de tracking (ATS) y sugerencias de términos clave
- **Proyección profesional**: Generación de rutas de desarrollo profesional basadas en el CV actual
- **Portafolio digital inteligente**: Sugerencias para crear un portafolio que complemente el CV

## Componentes Principales

### Configuración de API Key
- Pantalla inicial para configurar la API Key de Google AI Studio
- Almacenamiento seguro de la clave en session_state
- Opción para cambiar la clave desde la barra lateral

### Procesamiento de CVs
- Extracción de texto de archivos PDF
- División del contenido en chunks manejables
- Almacenamiento en session state para consultas

### Análisis de CVs
- Análisis automático inicial con prompts especializados
- Detección de fortalezas y áreas de mejora
- Recomendaciones específicas para optimización
- System prompt avanzado que proporciona análisis profesional de nivel ejecutivo
- Marco de análisis estructurado con 7 áreas clave (identidad profesional, trayectoria, habilidades, logros, posicionamiento, optimización ATS, presentación)
- Historial de contexto para conversaciones coherentes y progresivas

### Visualización Interactiva
- Generación de gráficos de radar para habilidades técnicas y blandas
- Gráficos de balance de habilidades
- Visualización de distribución de experiencia por industria
- Gráficos de relevancia educativa

### Análisis ATS
- Evaluación de compatibilidad con sistemas de tracking de aplicaciones
- Identificación de palabras clave existentes y su relevancia
- Sugerencias de palabras clave adicionales
- Análisis específico frente a descripciones de trabajo

### Proyección Profesional
- Generación de trayectorias profesionales potenciales
- Visualización de rutas de carrera
- Análisis de brecha de habilidades para roles específicos
- Planes de acción para adquirir habilidades faltantes

### Portafolio Digital
- Sugerencias de proyectos para mostrar en portafolio
- Recomendaciones sobre estructura de portafolio
- Herramientas sugeridas para crear el portafolio
- Detalles de implementación para proyectos específicos

### Configuración de la Interfaz
La aplicación utiliza Streamlit para crear una interfaz web amigable con:
- Título principal
- Panel lateral para configuración y carga de CVs
- Área de chat con historial
- Navegación por pestañas para las diferentes funcionalidades
- Campo de entrada de mensajes
- Botones para análisis específicos

### Conexión con la API
- Se conecta a la API de Gemini mediante una clave secreta
- Configura el modelo seleccionado para procesar los mensajes
- Gestiona la respuesta en modo streaming
- Incluye contexto relevante del CV en las consultas
- Mantiene historial de mensajes para análisis contextual continuo
- Usa system prompts diferenciados para análisis inicial y chat continuo

### Gestión de Conversaciones
- Mantiene un historial de mensajes entre el usuario y la IA
- Muestra avatares diferenciados para cada participante
- Actualiza la interfaz en tiempo real

## Flujo de Trabajo
1. El usuario configura su API Key de Google AI Studio en la pantalla inicial
2. El usuario sube su CV en formato PDF
3. El sistema extrae y procesa el texto del documento
4. El usuario puede elegir entre diferentes funcionalidades:
   - Chat básico para recomendaciones generales
   - Visualización interactiva del CV
   - Análisis de compatibilidad con ATS
   - Proyección profesional
   - Sugerencias para portafolio digital
5. La IA procesa las peticiones y muestra resultados interactivos
6. La conversación y los análisis se guardan en el historial de la sesión

## Funciones Principales
- `configurar_api_key()`: Muestra la pantalla para ingresar la API Key
- `configurar_gemini_api()`: Configura la API de Gemini usando la API Key
- `process_pdf()`: Extrae texto de archivos PDF
- `chunk_text()`: Divide el texto en fragmentos manejables
- `inicializar_estado()`: Configura el historial y almacenamiento
- `configurar_pagina()`: Establece elementos de la interfaz y navegación
- `analizar_cv_automaticamente()`: Realiza análisis inicial del CV
- `configurar_modelo()`: Prepara los parámetros para la consulta al LLM
- `generar_respuesta()`: Procesa la respuesta del modelo por streaming
- `crear_system_prompt_avanzado()`: Genera un prompt especializado para análisis detallado de CV
- `crear_system_prompt_chat()`: Crea un prompt contextual para conversación continua
- `main()`: Coordina el funcionamiento general y gestiona las pestañas

## Requisitos del Sistema
El archivo requirements.txt incluye todas las dependencias necesarias, entre las principales:
- streamlit
- google-generativeai
- PyPDF2
- pandas
- numpy
- plotly
- httpx
- pydantic

## Futuras Mejoras
- Implementación de embeddings para búsqueda semántica
- Plantillas de análisis para casos específicos
- Exportación de resultados en diferentes formatos
- Análisis específico por secciones del CV
- Comparativa con perfiles similares en el mercado laboral
- Generación automática de versiones del CV optimizadas para diferentes roles

## Enlaces
- Google AI Studio: https://ai.google.dev/
import streamlit as st
import json
import pandas as pd
import plotly.express as px
from google import genai

def generate_portfolio_suggestions(cv_text, client, model):
    """Generate suggestions for a digital portfolio"""
    prompt = f"""
    Analyze this CV and generate suggestions for a digital portfolio that complements it:

    CV:
    {cv_text[:6000]}

    Include:
    1. Key projects that should be showcased (existing or conceptual)
    2. Recommended portfolio elements based on professional profile
    3. Suggested structure for the portfolio
    4. Recommended tools for creating it

    Return ONLY a JSON with this exact format:
    {{
        "highlighted_projects": [
            {{"name": "Project Name", "description": "Short description", "impact": "Impact", "is_conceptual": true/false}},
            ...
        ],
        "portfolio_elements": ["element1", "element2", ...],
        "structure": ["section1", "section2", ...],
        "tools": [{{"name": "Tool name", "purpose": "Purpose", "url": "URL"}}]
    }}
    """

    # Configuración del modelo
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 1536
    }

    # Crear instancia del modelo
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config
    )

    # Llamar a la API de Gemini
    try:
        response = model_instance.generate_content(prompt)
        response_text = response.text

        # Find the JSON part in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            st.error("Could not extract valid JSON from the response")
            return None
    except Exception as e:
        st.error(f"Error parsing portfolio data: {str(e)}")
        return None

def display_portfolio_plan(portfolio_data):
    """Display portfolio suggestions in an organized way"""
    if not portfolio_data:
        st.error("No portfolio suggestions available")
        return

    # Display the portfolio elements
    st.subheader("📂 Recommended Portfolio Elements")
    elements = portfolio_data.get("portfolio_elements", [])
    if elements:
        element_html = ""
        for element in elements:
            element_html += f'<span style="background-color: #2e7d32; color: white; padding: 5px 12px; border-radius: 16px; margin: 4px; display: inline-block;">{element}</span>'

        st.markdown(f"<div style='margin: 10px 0;'>{element_html}</div>", unsafe_allow_html=True)
    else:
        st.info("No portfolio elements suggested")

    # Display highlighted projects
    st.subheader("🚀 Projects to Showcase")
    projects = portfolio_data.get("highlighted_projects", [])
    if projects:
        for i, project in enumerate(projects, 1):
            project_name = project.get("name", "Unnamed Project")
            is_conceptual = project.get("is_conceptual", False)

            # Add a badge for conceptual projects
            badge = "🔮 Conceptual" if is_conceptual else "✅ Existing"
            badge_color = "#9c27b0" if is_conceptual else "#1e88e5"

            with st.expander(f"{i}. {project_name} - {badge}"):
                st.markdown(f"**Description:** {project.get('description', 'No description available')}")
                st.markdown(f"**Impact:** {project.get('impact', 'No impact information available')}")

                # Add a note for conceptual projects
                if is_conceptual:
                    st.markdown("""
                    <div style='background-color: #f3e5f5; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <strong>Note:</strong> This is a conceptual project that could be developed to demonstrate your skills.
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No projects suggested")

    # Display suggested structure
    st.subheader("🏗️ Recommended Portfolio Structure")
    structure = portfolio_data.get("structure", [])
    if structure:
        for i, section in enumerate(structure, 1):
            st.markdown(f"**{i}.** {section}")

        # Create a visualization of the structure
        fig = px.bar(
            x=list(range(1, len(structure) + 1)),
            y=[1] * len(structure),
            color=list(range(len(structure))),
            color_continuous_scale=px.colors.sequential.Viridis,
            text=structure,
            orientation='v',
            title="Portfolio Site Structure"
        )

        fig.update_traces(textposition='inside', textangle=0, insidetextanchor='middle')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(showlegend=False, height=200, plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No structure suggested")

    # Display recommended tools
    st.subheader("🔧 Recommended Tools")
    tools = portfolio_data.get("tools", [])
    if tools:
        # Create a DataFrame for better display
        tools_df = pd.DataFrame(tools)

        # Create a table with clickable links
        for i, tool in enumerate(tools, 1):
            name = tool.get("name", "Unknown Tool")
            purpose = tool.get("purpose", "No purpose specified")
            url = tool.get("url", "")

            if url:
                st.markdown(f"**{i}. [{name}]({url})** - {purpose}")
            else:
                st.markdown(f"**{i}. {name}** - {purpose}")
    else:
        st.info("No tools suggested")

def generate_project_details(project_name, cv_text, client, model):
    """Generate detailed suggestions for a specific portfolio project"""
    prompt = f"""
    Generate detailed suggestions for implementing the following portfolio project:

    Project Name: {project_name}

    CV Context:
    {cv_text[:3000]}

    Provide:
    1. Detailed project description
    2. Key features and functionalities
    3. Technologies to use
    4. Implementation steps
    5. Possible challenges and solutions

    Return ONLY a JSON with this exact format:
    {{
        "project_name": "{project_name}",
        "detailed_description": "...",
        "features": ["feature1", "feature2", ...],
        "technologies": ["tech1", "tech2", ...],
        "implementation_steps": ["step1", "step2", ...],
        "challenges_solutions": [{{"challenge": "challenge1", "solution": "solution1"}}, ...]
    }}
    """

    # Configuración del modelo
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 1536
    }

    # Crear instancia del modelo
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config
    )

    # Llamar a la API de Gemini
    try:
        response = model_instance.generate_content(prompt)
        response_text = response.text

        # Find the JSON part in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            st.error("Could not extract valid JSON from the response")
            return None
    except Exception as e:
        st.error(f"Error parsing project details data: {str(e)}")
        return None

def display_project_details(project_details):
    """Display detailed suggestions for a portfolio project"""
    if not project_details:
        st.error("No project details available")
        return

    project_name = project_details.get("project_name", "Unnamed Project")
    st.title(f"Project Plan: {project_name}")

    # Description
    st.subheader("Project Description")
    st.write(project_details.get("detailed_description", "No description available"))

    # Features
    st.subheader("Key Features")
    features = project_details.get("features", [])
    if features:
        for i, feature in enumerate(features, 1):
            st.markdown(f"**{i}.** {feature}")
    else:
        st.info("No features specified")

    # Technologies
    st.subheader("Technologies")
    technologies = project_details.get("technologies", [])
    if technologies:
        tech_html = ""
        for tech in technologies:
            tech_html += f'<span style="background-color: #0d47a1; color: white; padding: 5px 12px; border-radius: 16px; margin: 4px; display: inline-block;">{tech}</span>'

        st.markdown(f"<div style='margin: 10px 0;'>{tech_html}</div>", unsafe_allow_html=True)
    else:
        st.info("No technologies specified")

    # Implementation steps
    st.subheader("Implementation Steps")
    steps = project_details.get("implementation_steps", [])
    if steps:
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:** {step}")
    else:
        st.info("No implementation steps provided")

    # Challenges and solutions
    st.subheader("Challenges and Solutions")
    challenges = project_details.get("challenges_solutions", [])
    if challenges:
        for i, challenge in enumerate(challenges, 1):
            with st.expander(f"Challenge {i}: {challenge.get('challenge', 'Unknown')}"):
                st.markdown(f"**Solution:** {challenge.get('solution', 'No solution provided')}")
    else:
        st.info("No challenges or solutions identified")
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import json
from google import genai

def extract_skills_categories(cv_text, client, model):
    """Extract and categorize skills from CV using AI"""
    prompt = f"""
    Analyze the following CV and extract:
    1. Technical skills (with estimated level from 1-10)
    2. Soft skills (with estimated level from 1-10)
    3. Work experience (years by category/industry)
    4. Education (with relative importance from 1-10)

    CV text:
    {cv_text[:6000]}

    Return ONLY a JSON with this exact format:
    {{
        "technical_skills": [{{"name": "skill1", "level": 8}}, ...],
        "soft_skills": [{{"name": "skill1", "level": 7}}, ...],
        "experience": [{{"category": "Web Development", "years": 2}}, ...],
        "education": [{{"type": "Bachelor", "relevance": 9}}, ...]
    }}
    """

    # Configuración del modelo
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 1024
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
        st.error(f"Error parsing skills data: {str(e)}")
        return None

def create_radar_chart(skills_data):
    """Creates a radar chart for skills visualization"""
    if not skills_data:
        st.error("No skills data available for visualization")
        return None

    # Combine technical and soft skills
    technical_skills = skills_data.get("technical_skills", [])
    soft_skills = skills_data.get("soft_skills", [])

    # Create DataFrames for the skills
    tech_df = pd.DataFrame(technical_skills)
    tech_df['category'] = 'Technical'
    soft_df = pd.DataFrame(soft_skills)
    soft_df['category'] = 'Soft'

    # Combine into one DataFrame for plotting
    all_skills = pd.concat([tech_df, soft_df])

    # Create radar chart
    fig = go.Figure()

    # Add technical skills trace
    if not tech_df.empty:
        tech_names = tech_df['name'].tolist()
        tech_levels = tech_df['level'].tolist()
        tech_levels.append(tech_levels[0])  # Close the loop
        tech_names.append(tech_names[0])    # Close the loop

        fig.add_trace(go.Scatterpolar(
            r=tech_levels,
            theta=tech_names,
            fill='toself',
            name='Technical Skills',
            line_color='blue'
        ))

    # Add soft skills trace
    if not soft_df.empty:
        soft_names = soft_df['name'].tolist()
        soft_levels = soft_df['level'].tolist()
        soft_levels.append(soft_levels[0])  # Close the loop
        soft_names.append(soft_names[0])    # Close the loop

        fig.add_trace(go.Scatterpolar(
            r=soft_levels,
            theta=soft_names,
            fill='toself',
            name='Soft Skills',
            line_color='green'
        ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Skills Assessment",
        showlegend=True
    )

    return fig

def create_experience_chart(skills_data):
    """Creates an experience breakdown chart"""
    if not skills_data or 'experience' not in skills_data:
        st.error("No experience data available for visualization")
        return None

    # Get experience data
    experience_data = skills_data.get("experience", [])
    if not experience_data:
        return None

    # Create DataFrame for experience
    exp_df = pd.DataFrame(experience_data)

    # Create pie chart for experience breakdown
    fig = px.pie(
        exp_df,
        values='years',
        names='category',
        title='Experience Breakdown by Industry',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    # Update layout
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    return fig

def create_education_chart(skills_data):
    """Creates a bar chart for education relevance"""
    if not skills_data or 'education' not in skills_data:
        st.error("No education data available for visualization")
        return None

    # Get education data
    education_data = skills_data.get("education", [])
    if not education_data:
        return None

    # Create DataFrame for education
    edu_df = pd.DataFrame(education_data)

    # Create bar chart for education relevance
    fig = px.bar(
        edu_df,
        x='type',
        y='relevance',
        title='Education Relevance for Current Career Path',
        color='relevance',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Education Type",
        yaxis_title="Relevance Score",
        yaxis=dict(range=[0, 10])
    )

    return fig

def create_skills_balance_chart(skills_data):
    """Creates a comparison chart between technical and soft skills"""
    if not skills_data:
        st.error("No skills data available for visualization")
        return None

    # Get skills data
    tech_skills = skills_data.get("technical_skills", [])
    soft_skills = skills_data.get("soft_skills", [])

    if not tech_skills or not soft_skills:
        return None

    # Calculate average skills levels
    avg_tech = sum(skill["level"] for skill in tech_skills) / len(tech_skills)
    avg_soft = sum(skill["level"] for skill in soft_skills) / len(soft_skills)

    # Create gauge chart for skills balance
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_tech,
        title = {'text': "Technical vs Soft Skills Balance"},
        delta = {'reference': avg_soft, 'relative': False},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3.33], 'color': 'red'},
                {'range': [3.33, 6.66], 'color': 'yellow'},
                {'range': [6.66, 10], 'color': 'green'}],
            'threshold': {
                'line': {'color': "orange", 'width': 4},
                'thickness': 0.75,
                'value': avg_soft}}
    ))

    # Add annotation to explain the delta
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"Soft Skills Avg: {avg_soft:.2f}",
        showarrow=False
    )

    return fig
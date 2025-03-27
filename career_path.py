import streamlit as st
import plotly.graph_objects as go
import json
import pandas as pd
from google import genai

def generate_career_paths(cv_text, client, model):
    """Generate potential career paths based on CV"""
    prompt = f"""
    Based on this CV, generate 3 possible career paths for the next 5 years:

    CV:
    {cv_text[:6000]}

    For each path include:
    1. Sequence of potential roles
    2. Skills needed for each transition
    3. Estimated time for each transition
    4. Recommended resources to acquire those skills

    Return ONLY a JSON with this exact format:
    {{
        "current_role": "Current Position",
        "paths": [
            {{
                "name": "Path 1 Name",
                "steps": [
                    {{
                        "role": "Next Role",
                        "timeline": "1-2 years",
                        "required_skills": ["skill1", "skill2"],
                        "resources": ["resource1", "resource2"]
                    }},
                    ...
                ]
            }},
            ...
        ]
    }}
    """

    # Configuración del modelo
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 2048
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
        st.error(f"Error parsing career path data: {str(e)}")
        return None

def visualize_career_paths(career_data):
    """Create an interactive visualization of career paths"""
    if not career_data:
        st.error("No career path data available")
        return

    current_role = career_data.get("current_role", "Current Position")
    paths = career_data.get("paths", [])

    if not paths:
        st.warning("No career paths generated")
        return

    # Create tabs for each career path
    tabs = st.tabs([path.get("name", f"Path {i+1}") for i, path in enumerate(paths)])

    # Display each path in its tab
    for i, (tab, path) in enumerate(zip(tabs, paths)):
        with tab:
            path_name = path.get("name", f"Path {i+1}")
            st.subheader(f"From: {current_role}")

            steps = path.get("steps", [])
            if not steps:
                st.info(f"No steps defined for {path_name}")
                continue

            # Create a visual timeline
            create_timeline_chart(current_role, steps, path_name)

            # Display detailed information for each step
            for j, step in enumerate(steps, 1):
                with st.expander(f"Step {j}: {step.get('role', 'Unknown Role')}"):
                    st.markdown(f"**Timeline:** {step.get('timeline', 'Unknown')}")

                    st.markdown("**Required Skills:**")
                    skills = step.get("required_skills", [])
                    if skills:
                        for skill in skills:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No specific skills listed")

                    st.markdown("**Recommended Resources:**")
                    resources = step.get("resources", [])
                    if resources:
                        for resource in resources:
                            st.markdown(f"- {resource}")
                    else:
                        st.info("No specific resources listed")

def create_timeline_chart(current_role, steps, path_name):
    """Create a timeline visualization for a career path"""
    # Create a list of all roles including the current one
    roles = [current_role] + [step.get("role", "Unknown Role") for step in steps]

    # Create positions and labels for the timeline
    positions = list(range(len(roles)))

    # Create the figure
    fig = go.Figure()

    # Add lines connecting the roles
    fig.add_trace(go.Scatter(
        x=positions,
        y=[0] * len(positions),
        mode='lines+markers',
        marker=dict(size=15, color='royalblue'),
        line=dict(width=4, color='royalblue'),
        name=path_name
    ))

    # Add role labels
    for i, role in enumerate(roles):
        fig.add_annotation(
            x=i,
            y=0,
            text=role,
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='royalblue',
            yshift=30,
            font=dict(size=14)
        )

        # Add timeline labels for transitions (not for the first role)
        if i > 0:
            timeline = steps[i-1].get("timeline", "")
            fig.add_annotation(
                x=i-0.5,
                y=0,
                text=timeline,
                showarrow=False,
                yshift=-25,
                font=dict(size=10, color='darkgrey')
            )

    # Update layout
    fig.update_layout(
        title=f"Career Path: {path_name}",
        showlegend=False,
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False, range=[-1, 1]),
        margin=dict(l=20, r=20, t=50, b=20),
        height=200
    )

    st.plotly_chart(fig, use_container_width=True)

def calculate_skill_gaps(cv_text, target_role, client, model):
    """Calculate skill gaps between current CV and target role"""
    prompt = f"""
    Analyze the skill gap between the CV and the target role:

    CV:
    {cv_text[:6000]}

    Target Role: {target_role}

    Provide:
    1. Skills already possessed relevant to the target role (with proficiency level 1-10)
    2. Skills needed for the target role that are missing (with importance level 1-10)
    3. Detailed action plan to acquire the missing skills

    Return ONLY a JSON with this exact format:
    {{
        "possessed_skills": [{{"name": "skill", "level": 8}}, ...],
        "missing_skills": [{{"name": "skill", "importance": 9}}, ...],
        "action_plan": [
            {{
                "skill": "Skill Name",
                "actions": ["action1", "action2"],
                "resources": ["resource1", "resource2"],
                "estimated_time": "X months/weeks"
            }},
            ...
        ]
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
        st.error(f"Error parsing skill gap data: {str(e)}")
        return None

def display_skill_gaps(skill_gap_data, target_role):
    """Display skill gap analysis results"""
    if not skill_gap_data:
        return

    st.subheader(f"Skill Gap Analysis for: {target_role}")

    # Create two columns for possessed and missing skills
    col1, col2 = st.columns(2)

    # Possessed skills
    with col1:
        st.markdown("### ✅ Skills You Have")
        possessed_skills = skill_gap_data.get("possessed_skills", [])
        if possessed_skills:
            # Convert to DataFrame for visualization
            df_possessed = pd.DataFrame(possessed_skills)

            # Create horizontal bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[skill.get("level", 0) for skill in possessed_skills],
                y=[skill.get("name", "Unknown") for skill in possessed_skills],
                orientation='h',
                marker_color='green'
            ))

            fig.update_layout(
                title="Your Relevant Skills",
                xaxis_title="Proficiency Level",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(possessed_skills) * 30)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No relevant skills found")

    # Missing skills
    with col2:
        st.markdown("### ❌ Skills to Develop")
        missing_skills = skill_gap_data.get("missing_skills", [])
        if missing_skills:
            # Convert to DataFrame for visualization
            df_missing = pd.DataFrame(missing_skills)

            # Create horizontal bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[skill.get("importance", 0) for skill in missing_skills],
                y=[skill.get("name", "Unknown") for skill in missing_skills],
                orientation='h',
                marker_color='crimson'
            ))

            fig.update_layout(
                title="Skills You Need to Develop",
                xaxis_title="Importance Level",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(missing_skills) * 30)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing skills identified")

    # Action plan
    st.markdown("### 🚀 Action Plan")
    action_plan = skill_gap_data.get("action_plan", [])
    if action_plan:
        for i, item in enumerate(action_plan, 1):
            skill = item.get("skill", "Unknown Skill")
            with st.expander(f"{i}. {skill}"):
                # Actions
                st.markdown("**Actions:**")
                actions = item.get("actions", [])
                if actions:
                    for action in actions:
                        st.markdown(f"- {action}")

                # Resources
                st.markdown("**Resources:**")
                resources = item.get("resources", [])
                if resources:
                    for resource in resources:
                        st.markdown(f"- {resource}")

                # Estimated time
                est_time = item.get("estimated_time", "Unknown")
                st.markdown(f"**Estimated Time:** {est_time}")
    else:
        st.info("No action plan available")
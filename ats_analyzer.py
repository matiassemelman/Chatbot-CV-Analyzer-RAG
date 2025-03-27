import streamlit as st
import pandas as pd
import plotly.express as px
import json

def analyze_ats_compatibility(cv_text, job_description, client, model):
    """Analyze ATS compatibility score and suggest improvements"""

    # Prepare the prompt based on whether job description is provided
    job_desc_text = job_description if job_description else "No job description provided"

    prompt = f"""
    Analyze this CV for ATS (Applicant Tracking Systems) compatibility:

    CV:
    {cv_text[:6000]}

    Job Description:
    {job_desc_text}

    1. Identify existing keywords and their relevance (1-10)
    2. Calculate an estimated ATS score (0-100)
    3. Suggest additional keywords to improve the score
    4. Identify potentially problematic terms or formats

    Return ONLY a JSON with this exact format:
    {{
        "existing_keywords": [{{"word": "keyword", "relevance": 8}}, ...],
        "ats_score": 75,
        "suggested_keywords": ["keyword1", "keyword2", ...],
        "problematic_terms": ["term1", "term2", ...],
        "improvement_suggestions": ["suggestion1", "suggestion2", ...]
    }}
    """

    # Call the Groq API and get the response
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    try:
        # Extract the JSON from the response
        response_text = chat_completion.choices[0].message.content
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
        st.error(f"Error parsing ATS analysis data: {str(e)}")
        return None

def display_ats_results(ats_results):
    """Display ATS analysis results in a user-friendly way"""
    if not ats_results:
        st.error("No ATS analysis results available")
        return

    # Display ATS score with a gauge
    ats_score = ats_results.get("ats_score", 0)
    st.subheader("ATS Compatibility Score")

    # Create color based on score
    if ats_score < 40:
        color = "red"
    elif ats_score < 70:
        color = "orange"
    else:
        color = "green"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="color: {color}; font-size: 3em;">{ats_score}/100</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show existing keywords
    st.subheader("Detected Keywords")
    existing_keywords = ats_results.get("existing_keywords", [])
    if existing_keywords:
        # Convert to DataFrame
        df_keywords = pd.DataFrame(existing_keywords)

        # Create horizontal bar chart
        fig = px.bar(
            df_keywords,
            x="relevance",
            y="word",
            orientation="h",
            title="Existing Keywords and Their Relevance",
            color="relevance",
            color_continuous_scale=px.colors.sequential.Viridis,
            height=min(500, max(300, len(existing_keywords) * 30))
        )

        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No keywords detected")

    # Show suggested keywords
    suggested_keywords = ats_results.get("suggested_keywords", [])
    if suggested_keywords:
        st.subheader("Suggested Keywords to Add")
        # Display as pills/tags
        pill_html = ""
        for keyword in suggested_keywords:
            pill_html += f'<span style="background-color: #1e88e5; color: white; padding: 4px 12px; border-radius: 16px; margin: 4px; display: inline-block;">{keyword}</span>'

        st.markdown(f"<div style='margin: 10px 0;'>{pill_html}</div>", unsafe_allow_html=True)

    # Show problematic terms
    problematic_terms = ats_results.get("problematic_terms", [])
    if problematic_terms:
        st.subheader("Problematic Terms or Formats")
        for i, term in enumerate(problematic_terms, 1):
            st.markdown(f"**{i}.** {term}")

    # Show improvement suggestions
    improvement_suggestions = ats_results.get("improvement_suggestions", [])
    if improvement_suggestions:
        st.subheader("Improvement Suggestions")
        for i, suggestion in enumerate(improvement_suggestions, 1):
            st.markdown(f"**{i}.** {suggestion}")

def analyze_job_match(cv_text, job_description, client, model):
    """Analyze how well the CV matches a specific job description"""
    if not job_description:
        st.warning("Please provide a job description to analyze the match")
        return None

    prompt = f"""
    Analyze how well this CV matches the provided job description:

    CV:
    {cv_text[:6000]}

    Job Description:
    {job_description[:3000]}

    Provide:
    1. Overall match percentage (0-100)
    2. Key requirements in the job description that are matched
    3. Key requirements in the job description that are missing
    4. Suggestions to tailor the CV for this job

    Return ONLY a JSON with this exact format:
    {{
        "match_percentage": 75,
        "matched_requirements": ["req1", "req2", ...],
        "missing_requirements": ["req1", "req2", ...],
        "tailoring_suggestions": ["suggestion1", "suggestion2", ...]
    }}
    """

    # Call the Groq API and get the response
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    try:
        # Extract the JSON from the response
        response_text = chat_completion.choices[0].message.content
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
        st.error(f"Error parsing job match data: {str(e)}")
        return None

def display_job_match_results(match_results):
    """Display job match analysis results"""
    if not match_results:
        return

    # Display match percentage
    match_percentage = match_results.get("match_percentage", 0)
    st.subheader("Job Match Score")

    # Create color based on percentage
    if match_percentage < 40:
        color = "red"
    elif match_percentage < 70:
        color = "orange"
    else:
        color = "green"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="color: {color}; font-size: 3em;">{match_percentage}%</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create two columns for matched and missing requirements
    col1, col2 = st.columns(2)

    # Matched requirements
    with col1:
        st.subheader("✅ Matched Requirements")
        matched_reqs = match_results.get("matched_requirements", [])
        if matched_reqs:
            for req in matched_reqs:
                st.markdown(f"- {req}")
        else:
            st.info("No matched requirements found")

    # Missing requirements
    with col2:
        st.subheader("❌ Missing Requirements")
        missing_reqs = match_results.get("missing_requirements", [])
        if missing_reqs:
            for req in missing_reqs:
                st.markdown(f"- {req}")
        else:
            st.info("No missing requirements found")

    # Tailoring suggestions
    tailoring_suggestions = match_results.get("tailoring_suggestions", [])
    if tailoring_suggestions:
        st.subheader("Suggestions to Tailor Your CV")
        for i, suggestion in enumerate(tailoring_suggestions, 1):
            st.markdown(f"**{i}.** {suggestion}")
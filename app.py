"""
Banking Lead Scoring - Completely Dynamic
No defaults, no static values, everything from APIs + manual company fields
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.linkedin_extractor import LinkedInAPIExtractor
from core.company_api import CompanyDataAPI
from core.feature_builder import DynamicFeatureBuilder
from core.model_predictor import ModelPredictor

st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    color: #1e3a8a; 
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


class DynamicLeadScoringApp:
    def __init__(self):
        self.session_state = st.session_state

        if "raw_linkedin_data" not in self.session_state:
            self.session_state.raw_linkedin_data = None
        if "recent_posts" not in self.session_state:
            self.session_state.recent_posts = []
        if "user_input_data" not in self.session_state:
            self.session_state.user_input_data = {}
        if "final_features" not in self.session_state:
            self.session_state.final_features = None
        if "prediction" not in self.session_state:
            self.session_state.prediction = None
        if "ready_for_scoring" not in self.session_state:
            self.session_state.ready_for_scoring = False

        apify_key = st.secrets.get("APIFY", "")
        self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

        self.company_api = None
        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def render_header(self):
        st.markdown('<h1 class="main-header">Dynamic Lead Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #475569; font-size: 16px;'>
        Data is dynamically extracted from APIs. Manual entry is only for Company fields.
        </p>
        """, unsafe_allow_html=True)
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Manual Company Entry (Only 4 fields)")
            with st.form("manual_company_form"):
                company_name = st.text_input("Company Name")
                company_size = st.text_input("Company Size")
                annual_revenue = st.text_input("Annual Revenue")
                industry = st.text_input("Industry")

                if st.form_submit_button("Save Company Data"):
                    self.session_state.user_input_data = {
                        "company_name": company_name,
                        "company_size": company_size,
                        "annual_revenue": annual_revenue,
                        "industry": industry
                    }
                    st.success("Saved company data")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url"
        )

        if self.linkedin_extractor is None:
            st.warning("Apify API not configured. Add APIFY key in secrets.")
            return

        if st.button("Extract Data", type="primary", disabled=not linkedin_url):
            self._extract_all_data(linkedin_url)

    def _extract_all_data(self, linkedin_url: str):
        st.session_state.ready_for_scoring = False

        with st.spinner("Extracting LinkedIn profile + recent posts..."):
            data_bundle = self.linkedin_extractor.extract_profile_with_activity(linkedin_url)

            profile_data = data_bundle.get("profile_data")
            recent_posts = data_bundle.get("recent_posts", [])
            activity_days = data_bundle.get("activity_days")

            if profile_data is None:
                st.error("LinkedIn extraction failed.")
                return

            # Attach activity_days into profile data for feature builder
            if activity_days is not None:
                profile_data["activity_days"] = activity_days

            self.session_state.raw_linkedin_data = profile_data
            self.session_state.recent_posts = recent_posts

        # Build features
        with st.spinner("Building features..."):
            features = self.feature_builder.build_features(
                linkedin_data=self.session_state.raw_linkedin_data,
                company_data=None,
                user_data=self.session_state.user_input_data
            )

            self.session_state.final_features = features
            self.session_state.ready_for_scoring = True

        st.success("Extraction + feature building completed.")
        self._show_extracted_data()

    def _show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        data = self.session_state.raw_linkedin_data
        if not data:
            return

        basic = data.get("basic_info", {})
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Info**")
            st.text(f"Name: {basic.get('fullname', '')}")
            st.text(f"Headline: {basic.get('headline', '')}")
            st.text(f"Location: {basic.get('location', {}).get('full', '')}")

        with col2:
            st.markdown("**Activity**")
            if data.get("activity_days") is not None:
                st.text(f"Recent Activity Days: {data.get('activity_days')}")
            if self.session_state.recent_posts:
                last_relative = self.session_state.recent_posts[0].get("posted_at", {}).get("relative", "")
                st.text(f"Last Post: {last_relative}")

        st.markdown("### Debug Features Sent to Model")
        st.dataframe(self.session_state.final_features)

    def render_scoring_section(self):
        if not self.session_state.ready_for_scoring:
            return

        st.markdown("### Step 2: Generate Score")

        if st.button("Generate Lead Score", type="primary"):
            with st.spinner("Scoring..."):
                prediction = self.model_predictor.predict(self.session_state.final_features)

                if prediction is None:
                    st.error("Model prediction failed.")
                    return

                self.session_state.prediction = prediction
                self._display_results(prediction)

    def _display_results(self, prediction: dict):
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})

        priority_colors = {"COLD": "#64748b", "COOL": "#3b82f6", "WARM": "#f59e0b", "HOT": "#dc2626"}
        color = priority_colors.get(priority, "#64748b")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"<h2 style='color:{color};margin:0;'>Priority: {priority}</h2>"
                f"<p style='color:#475569;'>Confidence: {confidence:.1%}</p>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"<h1 style='color:{color};'>{probabilities.get(priority,0):.0%}</h1>", unsafe_allow_html=True)

        st.progress(float(confidence))

        if probabilities:
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({"Priority": list(probabilities.keys()), "Probability": list(probabilities.values())})

            fig = go.Figure(data=[go.Bar(x=prob_df["Priority"], y=prob_df["Probability"])])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(tickformat=".0%", range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        # Dynamic reasons (SHAP)
        reasons = prediction.get("reasons", [])
        if reasons:
            st.markdown("#### Why this prediction? (Dynamic Model Explanation)")
            for r in reasons:
                feature = r.get("feature")
                value = r.get("value")
                impact = r.get("impact", 0)

                direction = "Increased" if impact > 0 else "Decreased"
                st.write(f"**{feature} = {value}** → {direction} confidence ({impact:.4f})")

    def run(self):
        self.render_header()
        main_col, side_col = st.columns([3, 1])

        with main_col:
            self.render_input_section()
            self.render_scoring_section()

        with side_col:
            self.render_sidebar()


if __name__ == "__main__":
    app = DynamicLeadScoringApp()
    app.run()

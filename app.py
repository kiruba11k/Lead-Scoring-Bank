"""
Banking Lead Scoring - Completely Dynamic
No defaults, no static values, everything from APIs + manual company fields only
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.apify_extractor import LinkedInAPIExtractor
from core.feature_builder import DynamicFeatureBuilder
from core.model_predictor import ModelPredictor


st.set_page_config(
    page_title="Dynamic Lead Scoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

        # Session init
        self.session_state.setdefault("raw_linkedin_data", None)
        self.session_state.setdefault("final_features", None)
        self.session_state.setdefault("prediction", None)
        self.session_state.setdefault("debug_info", None)
        self.session_state.setdefault("ready_for_scoring", False)
        self.session_state.setdefault("last_url", "")

        # API
        apify_key = st.secrets.get("APIFY", "")
        self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def reset_for_new_url(self):
        """
        Reset ONLY extraction data when user enters new URL.
        Do NOT wipe manual input.
        """
        self.session_state.raw_linkedin_data = None
        self.session_state.final_features = None
        self.session_state.prediction = None
        self.session_state.debug_info = None
        self.session_state.ready_for_scoring = False

    def render_header(self):
        st.markdown('<h1 class="main-header">Dynamic Lead Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #475569; font-size: 16px;'>
        All data is dynamically extracted from APIs. No defaults or static values are used.
        Missing data will result in empty fields rather than estimates.
        </p>
        """, unsafe_allow_html=True)
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Manual Company Inputs (Only 4 fields)")
            st.info("These 4 values are used when company API is not available.")

            self.session_state.setdefault("manual_company_name", "")
            self.session_state.setdefault("manual_company_size", "")
            self.session_state.setdefault("manual_annual_revenue", "")
            self.session_state.setdefault("manual_industry", "")

            self.session_state.manual_company_name = st.text_input("Company Name", value=self.session_state.manual_company_name)
            self.session_state.manual_company_size = st.text_input("Company Size", value=self.session_state.manual_company_size)
            self.session_state.manual_annual_revenue = st.text_input("Annual Revenue", value=self.session_state.manual_annual_revenue)
            self.session_state.manual_industry = st.text_input("Industry", value=self.session_state.manual_industry)

            st.divider()
            if st.button("Reset App (Clear Current Prospect)"):
                self.reset_for_new_url()
                st.success("Cleared current prospect extraction.")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://www.linkedin.com/in/username/",
            key="linkedin_url"
        )

        # If user changed URL -> reset old prospect extraction
        if linkedin_url and linkedin_url.strip() != self.session_state.last_url:
            self.session_state.last_url = linkedin_url.strip()
            self.reset_for_new_url()

        if self.linkedin_extractor is None:
            st.warning("LinkedIn API not configured. Add APIFY key in Streamlit secrets.")
            return

        if st.button("Extract Data", type="primary", disabled=not linkedin_url):
            self._extract_all_data(linkedin_url)

    def _extract_all_data(self, linkedin_url: str):
        try:
            with st.spinner("Extracting LinkedIn profile + recent activity..."):
                linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)

            if not linkedin_data:
                st.error("LinkedIn extraction failed.")
                return

            self.session_state.raw_linkedin_data = linkedin_data

            user_data = {
                "company_name": self.session_state.manual_company_name,
                "company_size": self.session_state.manual_company_size,
                "annual_revenue": self.session_state.manual_annual_revenue,
                "industry": self.session_state.manual_industry,
            }

            features_df, debug_info = self.feature_builder.build_features(
                linkedin_data=linkedin_data,
                company_data=None,
                user_data=user_data
            )

            self.session_state.final_features = features_df
            self.session_state.debug_info = debug_info
            self.session_state.ready_for_scoring = True

            st.success("Extraction complete. Features built successfully.")
            self._show_extracted_data()

        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")

    def _show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        if not self.session_state.raw_linkedin_data:
            return

        data = self.session_state.raw_linkedin_data
        basic = data.get("basic_info", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Personal Information**")
            if basic.get("fullname"):
                st.write(f"Name: {basic.get('fullname')}")
            if basic.get("headline"):
                st.write(f"Headline: {basic.get('headline')}")
            loc = basic.get("location", {}).get("full")
            if loc:
                st.write(f"Location: {loc}")

        with col2:
            st.markdown("**Activity**")
            activity_days = data.get("activity_days", None)
            if activity_days is not None:
                st.write(f"Recent Activity Days: {activity_days}")

            posts = data.get("recent_posts", [])
            if posts:
                last_post = posts[0].get("posted_at", {}).get("relative")
                if last_post:
                    st.write(f"Last Post: {last_post}")

    def render_scoring_section(self):
        if not self.session_state.ready_for_scoring:
            return

        st.markdown("### Step 2: Generate Score")

        if st.button("Generate Lead Score", type="primary"):
            try:
                prediction = self.model_predictor.predict(self.session_state.final_features)

                if prediction is None:
                    st.error("Model returned no prediction. Check model files and features.")
                    return

                self.session_state.prediction = prediction
                self._display_results(prediction)

            except Exception as e:
                st.error(f"Scoring failed: {str(e)}")

        # KEEP extracted preview visible even after scoring
        if self.session_state.raw_linkedin_data:
            self._show_extracted_data()

        # Debug view
        if self.session_state.debug_info:
            st.markdown("### Debug: Values Sent to Model")
            debug_df = pd.DataFrame([self.session_state.debug_info]).T.reset_index()
            debug_df.columns = ["Field", "Value"]
            st.dataframe(debug_df, use_container_width=True)

    def _display_results(self, prediction: dict):
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})

        st.write(f"Priority: **{priority}**")
        st.write(f"Confidence: **{confidence:.1%}**")

        if probabilities:
            st.markdown("#### Probability Distribution")

            prob_df = pd.DataFrame({
                "Priority": list(probabilities.keys()),
                "Probability": list(probabilities.values())
            })

            fig = go.Figure(data=[
                go.Bar(x=prob_df["Priority"], y=prob_df["Probability"])
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        imp = self.model_predictor.get_feature_importance()
        if imp:
            st.markdown("#### Top Feature Importances (Model)")
            top = dict(list(imp.items())[:10])
            imp_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in top.items()])
            st.dataframe(imp_df, use_container_width=True)

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

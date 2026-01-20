"""
Banking Lead Scoring - Dynamic
- LinkedIn profile extracted via Apify
- Company fields entered manually
- Debug shown before sending to model
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.apify_extractor import LinkedInAPIExtractor
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

        # init session vars
        if "raw_linkedin_data" not in self.session_state:
            self.session_state.raw_linkedin_data = None
        if "user_input_data" not in self.session_state:
            self.session_state.user_input_data = {}
        if "final_features" not in self.session_state:
            self.session_state.final_features = None
        if "prediction" not in self.session_state:
            self.session_state.prediction = None
        if "debug_info" not in self.session_state:
            self.session_state.debug_info = None
        if "ready_for_scoring" not in self.session_state:
            self.session_state.ready_for_scoring = False
        if "last_url" not in self.session_state:
            self.session_state.last_url = ""

        apify_key = st.secrets.get("APIFY", "")
        self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def render_header(self):
        st.markdown('<h1 class="main-header">Dynamic Lead Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#475569;font-size:16px;'>All data is dynamically extracted from APIs. "
            "No defaults or static values are used. Missing data stays empty.</p>",
            unsafe_allow_html=True
        )
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Manual Company Data Entry (Only 4 fields)")

            with st.form("manual_company_form"):
                company_name = st.text_input("Company Name", value=self.session_state.user_input_data.get("company_name", ""))
                company_size = st.text_input("Company Size", value=self.session_state.user_input_data.get("company_size", ""))
                annual_revenue = st.text_input("Annual Revenue", value=self.session_state.user_input_data.get("annual_revenue", ""))
                industry = st.text_input("Industry", value=self.session_state.user_input_data.get("industry", ""))

                saved = st.form_submit_button("Save Manual Company Data")
                if saved:
                    self.session_state.user_input_data = {
                        "company_name": company_name,
                        "company_size": company_size,
                        "annual_revenue": annual_revenue,
                        "industry": industry,
                    }
                    st.success("Manual company fields saved")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url_input"
        )

        # if URL changed -> reset ONLY extraction-related things
        if linkedin_url and linkedin_url.strip() != self.session_state.last_url:
            self.session_state.last_url = linkedin_url.strip()
            self.session_state.raw_linkedin_data = None
            self.session_state.final_features = None
            self.session_state.prediction = None
            self.session_state.debug_info = None
            self.session_state.ready_for_scoring = False

        if self.linkedin_extractor is None:
            st.error("Apify key missing in secrets. Add APIFY in Streamlit secrets.")
            return

        if st.button("Extract Data", type="primary", disabled=not linkedin_url):
            self._extract_all_data(linkedin_url.strip())

        if self.session_state.raw_linkedin_data:
            self._show_extracted_data()

    def _extract_all_data(self, linkedin_url: str):
        with st.spinner("Extracting LinkedIn profile + recent posts..."):
            linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)

            if not linkedin_data:
                st.error("LinkedIn extraction failed.")
                return

            self.session_state.raw_linkedin_data = linkedin_data

            # Build features (manual company info only)
            features_df, debug_info = self.feature_builder.build_features(
                linkedin_data=linkedin_data,
                company_data=None,
                user_data=self.session_state.user_input_data
            )

            self.session_state.final_features = features_df
            self.session_state.debug_info = debug_info
            self.session_state.ready_for_scoring = True

            st.success("Extraction completed and features built.")

    def _show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        data = self.session_state.raw_linkedin_data
        basic = data.get("basic_info", {})
        exp = data.get("experience", [])

        current_title = ""
        current_company = ""
        for e in exp:
            if e.get("is_current", False):
                current_title = e.get("title", "")
                current_company = e.get("company", "")
                break

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Personal Information**")
            st.write(f"Name: {basic.get('fullname','')}")
            st.write(f"Headline: {basic.get('headline','')}")
            st.write(f"Location: {basic.get('location',{}).get('full','')}")

        with col2:
            st.markdown("**Professional Information**")
            st.write(f"Current Role: {current_title}")
            st.write(f"Current Company: {current_company}")

        # Activity preview
        activity_days = data.get("activity_days", None)
        recent_posts = data.get("recent_posts", [])

        st.markdown("**Activity**")
        if activity_days is None:
            st.write("Recent Activity Days: Not available (no posts found)")
        else:
            st.write(f"Recent Activity Days: {activity_days}")

        if recent_posts:
            last_post = recent_posts[0].get("posted_at", {}).get("relative", "")
            st.write(f"Last Post: {last_post}")

    def render_scoring_section(self):
        if not self.session_state.ready_for_scoring:
            return

        st.markdown("### Step 2: Generate Score")

        if st.button("Generate Lead Score", type="primary"):
            with st.spinner("Scoring..."):
                prediction = self.model_predictor.predict(self.session_state.final_features)
                if prediction is None:
                    st.error("Model returned no prediction.")
                    return

                self.session_state.prediction = prediction
                self._display_results(prediction)

    def _display_results(self, prediction: dict):
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})

        st.write(f"Priority: **{priority}**")
        st.write(f"Confidence: **{confidence:.1%}**")

        # Probability chart
        if probabilities:
            dfp = pd.DataFrame({"Priority": list(probabilities.keys()), "Probability": list(probabilities.values())})
            fig = go.Figure([go.Bar(x=dfp["Priority"], y=dfp["Probability"], text=[f"{p:.1%}" for p in dfp["Probability"]], textposition="auto")])
            fig.update_layout(height=300, yaxis=dict(tickformat=".0%", range=[0, 1]), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Debug values before model
        if self.session_state.debug_info:
            st.markdown("### Debug: Values passed to model")
            debug_df = pd.DataFrame(list(self.session_state.debug_info.items()), columns=["Field", "Value"])
            st.dataframe(debug_df, use_container_width=True)

        # WHY prediction
        st.markdown("### Why this prediction?")
        explanation = self.model_predictor.explain_prediction(self.session_state.final_features, top_n=5)
        reasons = explanation.get("top_reasons", [])

        if reasons:
            reason_df = pd.DataFrame(reasons)
            st.dataframe(reason_df, use_container_width=True)
        else:
            st.info("Feature importance explanation not available for this model.")

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

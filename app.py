"""
Banking Lead Scoring - Completely Dynamic
No defaults, no static values, everything from APIs + manual company input only.
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

        # init session keys
        self.session_state.setdefault("raw_linkedin_data", None)
        self.session_state.setdefault("final_features", None)
        self.session_state.setdefault("prediction", None)
        self.session_state.setdefault("debug_info", None)
        self.session_state.setdefault("ready_for_scoring", False)
        self.session_state.setdefault("last_url", "")

        # API key from secrets
        apify_key = st.secrets.get("APIFY", "")
        self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def reset_previous_results(self):
        """
        Reset only extraction+prediction outputs.
        DO NOT reset manual input fields.
        """
        self.session_state.raw_linkedin_data = None
        self.session_state.final_features = None
        self.session_state.prediction = None
        self.session_state.debug_info = None
        self.session_state.ready_for_scoring = False

    def render_header(self):
        st.markdown('<h1 class="main-header">Dynamic Lead Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        All data is dynamically extracted from APIs. No defaults or static values are used.
        Missing data will result in empty fields rather than estimates.
        """)
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Manual Company Entry (Required)")
            st.caption("Only these 4 fields are manually entered. Everything else comes from LinkedIn.")

            company_name = st.text_input("Company Name", key="manual_company_name")
            company_size = st.text_input("Company Size", key="manual_company_size")
            annual_revenue = st.text_input("Annual Revenue", key="manual_annual_revenue")
            industry = st.text_input("Industry", key="manual_industry")

            self.session_state.user_input_data = {
                "company_name": company_name,
                "company_size": company_size,
                "annual_revenue": annual_revenue,
                "industry": industry,
            }

            st.divider()
            if st.button("Clear Extracted Results"):
                self.reset_previous_results()
                st.success("Cleared extracted prospect + prediction results.")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url"
        )

        if self.linkedin_extractor is None:
            st.error("Apify key not configured. Add APIFY in Streamlit secrets.")
            return

        extract_clicked = st.button("Extract Data", type="primary", disabled=not linkedin_url)

        # If user changes URL, reset automatically
        if linkedin_url and linkedin_url != self.session_state.last_url:
            self.reset_previous_results()
            self.session_state.last_url = linkedin_url

        if extract_clicked and linkedin_url:
            self.reset_previous_results()
            self._extract_all_data(linkedin_url)

    def _extract_all_data(self, linkedin_url: str):
        with st.spinner("Extracting LinkedIn profile + posts..."):
            linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)

            if not linkedin_data:
                st.error("LinkedIn extraction failed.")
                return

            self.session_state.raw_linkedin_data = linkedin_data

            # Build features using manual company input
            features_df, debug_info = self.feature_builder.build_features(
                linkedin_data=linkedin_data,
                company_data=None,
                user_data=self.session_state.user_input_data
            )

            self.session_state.final_features = features_df
            self.session_state.debug_info = debug_info
            self.session_state.ready_for_scoring = True

        st.success("Extraction + feature building completed.")
        self._show_extracted_data()

    def _extract_current_company(self, linkedin_data: dict):
        exp = linkedin_data.get("experience", [])
        if not exp:
            return None
        for e in exp:
            if e.get("is_current", False):
                return e
        return exp[0] if exp else None

    def _show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        linkedin_data = self.session_state.raw_linkedin_data
        if not linkedin_data:
            return

        basic = linkedin_data.get("basic_info", {})
        current_company = self._extract_current_company(linkedin_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Information**")
            st.text(f"Name: {basic.get('fullname', '')}")
            st.text(f"Headline: {basic.get('headline', '')}")
            loc = basic.get("location", {}).get("full", "")
            if loc:
                st.text(f"Location: {loc}")

            st.markdown("**Activity**")
            activity_days = linkedin_data.get("activity_days", None)
            if activity_days is not None:
                st.text(f"Recent Activity Days: {activity_days}")

            posts = linkedin_data.get("recent_posts", [])
            if posts:
                last_post = posts[0].get("posted_at", {}).get("relative", "")
                if last_post:
                    st.text(f"Last Post: {last_post}")

        with col2:
            st.markdown("**Professional Information**")
            if current_company:
                st.text(f"Current Role: {current_company.get('title', '')}")
                st.text(f"Current Company: {current_company.get('company', '')}")

        st.markdown("### Manual Company Info Used")
        user_data = self.session_state.user_input_data
        st.write(user_data)

        # DEBUG table
        if self.session_state.debug_info:
            st.markdown("### Debug: Values Sent to Model")
            debug_df = pd.DataFrame([self.session_state.debug_info]).T
            debug_df.columns = ["Value"]
            st.dataframe(debug_df, use_container_width=True)

    def render_scoring_section(self):
        if not self.session_state.ready_for_scoring:
            return

        st.markdown("### Step 2: Generate Score")

        if st.button("Generate Lead Score", type="primary"):
            with st.spinner("Predicting..."):
                prediction = self.model_predictor.predict(self.session_state.final_features)

                if prediction is None:
                    st.error("Model returned no prediction. Check model.pkl and metadata.json inside models/")
                    return

                self.session_state.prediction = prediction
                self._display_results(prediction)

    def _display_results(self, prediction: dict):
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})
        reasons = prediction.get("reasons", [])

        priority_colors = {
            "COLD": "#64748b",
            "COOL": "#3b82f6",
            "WARM": "#f59e0b",
            "HOT": "#dc2626"
        }

        color = priority_colors.get(priority, "#64748b")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
                <div style='border-left: 5px solid {color}; padding-left: 20px;'>
                    <h2 style='color: {color}; margin: 0;'>Priority: {priority}</h2>
                    <p style='color: #475569;'>Confidence: {confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            priority_prob = probabilities.get(priority, 0)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='font-size: 12px; color: #64748b;'>SCORE</div>
                    <div style='font-size: 36px; font-weight: bold; color: {color};'>
                        {priority_prob:.0%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if probabilities:
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                "Priority": list(probabilities.keys()),
                "Probability": list(probabilities.values())
            })

            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df["Priority"],
                    y=prob_df["Probability"],
                    text=[f"{p:.1%}" for p in prob_df["Probability"]],
                    textposition="auto"
                )
            ])
            fig.update_layout(height=300, yaxis=dict(tickformat=".0%", range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        if reasons:
            st.markdown("#### Why this prediction?")
            reasons_df = pd.DataFrame(reasons)
            st.dataframe(reasons_df, use_container_width=True)

    def run(self):
        self.render_header()

        main_col, side_col = st.columns([3, 1])

        with main_col:
            self.render_input_section()
            if self.session_state.ready_for_scoring:
                self.render_scoring_section()

        with side_col:
            self.render_sidebar()


if __name__ == "__main__":
    app = DynamicLeadScoringApp()
    app.run()

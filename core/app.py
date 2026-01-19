"""
Dynamic Banking Lead Intelligence Platform
- No static values
- Company details taken ONLY from Manual Entry
- LinkedIn profile + posts extracted via Apify
- Debug values shown before model prediction
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
.data-field {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 12px;
    margin: 6px 0;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


class DynamicLeadScoringApp:
    def __init__(self):
        self.session_state = st.session_state

        # Session state init
        if "raw_linkedin_data" not in self.session_state:
            self.session_state.raw_linkedin_data = None
        if "recent_posts" not in self.session_state:
            self.session_state.recent_posts = []
        if "user_input_data" not in self.session_state:
            self.session_state.user_input_data = {}
        if "final_features" not in self.session_state:
            self.session_state.final_features = None
        if "debug_info" not in self.session_state:
            self.session_state.debug_info = None
        if "prediction" not in self.session_state:
            self.session_state.prediction = None
        if "ready_for_scoring" not in self.session_state:
            self.session_state.ready_for_scoring = False

        # Track last url (to reset when changed)
        if "last_linkedin_url" not in self.session_state:
            self.session_state.last_linkedin_url = ""

        # API key from secrets
        apify_key = st.secrets.get("APIFY", "")
        if not apify_key:
            self.linkedin_extractor = None
        else:
            self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key)

        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def reset_for_new_url(self):
        """Clear previous extracted results when new URL is entered."""
        self.session_state.raw_linkedin_data = None
        self.session_state.recent_posts = []
        self.session_state.final_features = None
        self.session_state.debug_info = None
        self.session_state.prediction = None
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
            st.markdown("### Manual Company Entry (Required)")

            company_name = st.text_input("Company Name", value=self.session_state.user_input_data.get("company_name", ""))
            company_size = st.text_input("Company Size", value=self.session_state.user_input_data.get("company_size", ""))
            annual_revenue = st.text_input("Annual Revenue", value=self.session_state.user_input_data.get("annual_revenue", ""))
            industry = st.text_input("Industry", value=self.session_state.user_input_data.get("industry", ""))

            if st.button("Save Company Info"):
                self.session_state.user_input_data = {
                    "company_name": company_name.strip(),
                    "company_size": company_size.strip(),
                    "annual_revenue": annual_revenue.strip(),
                    "industry": industry.strip(),
                }
                st.success("Company info saved successfully.")

            st.divider()
            st.markdown("### Notes")
            st.info("Company API is removed. Manual company entry is used directly for feature building.")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url",
        )

        # If user enters a new URL, reset previous data
        if linkedin_url and linkedin_url.strip() != self.session_state.last_linkedin_url:
            self.session_state.last_linkedin_url = linkedin_url.strip()
            self.reset_for_new_url()

        if self.linkedin_extractor is None:
            st.warning("LinkedIn API not configured. Please add APIFY key in Streamlit secrets.")
            return

        if st.button("Extract Data", type="primary", disabled=not linkedin_url):
            self.extract_all_data(linkedin_url.strip())

    def extract_all_data(self, linkedin_url: str):
        progress = st.progress(0)

        try:
            progress.progress(20)
            linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)
            if not linkedin_data:
                st.error("LinkedIn extraction failed.")
                return

            self.session_state.raw_linkedin_data = linkedin_data
            st.success("LinkedIn profile extracted successfully.")

            progress.progress(40)
            posts = self.linkedin_extractor.extract_recent_posts(linkedin_url, limit=2)
            self.session_state.recent_posts = posts

            activity_days = self.linkedin_extractor.compute_activity_days_from_posts(posts)
            if activity_days is not None:
                self.session_state.raw_linkedin_data["activity_days"] = activity_days

            progress.progress(70)

            # Build features + debug info
            features_df, debug_info = self.feature_builder.build_features(
                linkedin_data=self.session_state.raw_linkedin_data,
                user_data=self.session_state.user_input_data
            )

            self.session_state.final_features = features_df
            self.session_state.debug_info = debug_info
            self.session_state.ready_for_scoring = True

            progress.progress(100)
            st.success("Features built successfully (dynamic).")

            self.show_extracted_data()

        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")

    def show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        data = self.session_state.raw_linkedin_data or {}
        basic = data.get("basic_info", {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Information**")
            st.text(f"Name: {basic.get('fullname', '')}")
            st.text(f"Headline: {basic.get('headline', '')}")
            loc = basic.get("location", {}).get("full", "")
            st.text(f"Location: {loc}")

        with col2:
            st.markdown("**Activity**")
            act = data.get("activity_days", None)
            if act is None:
                st.text("Recent Activity Days: Not available")
            else:
                st.text(f"Recent Activity Days: {act}")

            if self.session_state.recent_posts:
                last_post = self.session_state.recent_posts[0].get("posted_at", {}).get("relative", "")
                st.text(f"Last Post: {last_post}")

        # Debug values before model
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
            try:
                prediction = self.model_predictor.predict(self.session_state.final_features)

                if not prediction:
                    st.error("Model returned no prediction.")
                    return

                self.session_state.prediction = prediction
                self.display_results(prediction)

            except Exception as e:
                st.error(f"Scoring failed: {str(e)}")

    def display_results(self, prediction: dict):
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})

        priority_colors = {
            "COLD": "#64748b",
            "COOL": "#3b82f6",
            "WARM": "#f59e0b",
            "HOT": "#dc2626",
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
            p = probabilities.get(priority, 0)
            st.markdown(f"""
                <div style='text-align:center;'>
                    <div style='font-size:12px;color:#64748b;'>SCORE</div>
                    <div style='font-size:36px;font-weight:bold;color:{color};'>
                        {p:.0%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if probabilities:
            st.markdown("#### Probability Distribution")
            df = pd.DataFrame({
                "Priority": list(probabilities.keys()),
                "Probability": list(probabilities.values())
            })

            fig = go.Figure(data=[
                go.Bar(
                    x=df["Priority"],
                    y=df["Probability"],
                    text=[f"{x:.1%}" for x in df["Probability"]],
                    textposition="auto"
                )
            ])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(tickformat=".0%", range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        # Dynamic explanation (top factors)
        if prediction.get("reasons"):
            st.markdown("#### Why this score?")
            for r in prediction["reasons"]:
                st.write(f"- {r}")

    def run(self):
        self.render_header()

        main_col, side_col = st.columns([3, 1])

        with side_col:
            self.render_sidebar()

        with main_col:
            self.render_input_section()
            self.render_scoring_section()


if __name__ == "__main__":
    app = DynamicLeadScoringApp()
    app.run()

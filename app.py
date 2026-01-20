import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.apify_extractor import LinkedInAPIExtractor
from core.feature_builder import DynamicFeatureBuilder
from core.model_predictor import ModelPredictor


st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 28px; font-weight: 700; color:#1e3a8a; }
</style>
""", unsafe_allow_html=True)


class DynamicLeadScoringApp:
    def __init__(self):
        self.session_state = st.session_state

        if "last_url" not in self.session_state:
            self.session_state.last_url = ""

        if "raw_linkedin_data" not in self.session_state:
            self.session_state.raw_linkedin_data = None

        if "user_input_data" not in self.session_state:
            self.session_state.user_input_data = {}

        if "final_features" not in self.session_state:
            self.session_state.final_features = None

        if "debug_info" not in self.session_state:
            self.session_state.debug_info = None

        if "prediction" not in self.session_state:
            self.session_state.prediction = None

        apify_key = st.secrets.get("APIFY", "")
        self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    def render_header(self):
        st.markdown('<div class="main-header">Dynamic Lead Intelligence Platform</div>', unsafe_allow_html=True)
        st.caption("All data is extracted dynamically. Manual entry only fills missing company fields.")
        st.divider()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Manual Company Data Entry (Only 4 Fields)")

            with st.form("manual_form"):
                company_name = st.text_input("Company Name")
                company_size = st.text_input("Company Size")
                annual_revenue = st.text_input("Annual Revenue")
                industry = st.text_input("Industry")

                saved = st.form_submit_button("Save Manual Company Info")

                if saved:
                    self.session_state.user_input_data = {
                        "company_name": company_name,
                        "company_size": company_size,
                        "annual_revenue": annual_revenue,
                        "industry": industry,
                    }
                    st.success("Manual company data saved")

    def render_input_section(self):
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url"
        )

        # Reset only when URL changes
        if linkedin_url and linkedin_url.strip() != self.session_state.last_url:
            self.session_state.last_url = linkedin_url.strip()
            self.session_state.raw_linkedin_data = None
            self.session_state.final_features = None
            self.session_state.debug_info = None
            self.session_state.prediction = None
            self.session_state.ready_for_scoring = False

        if self.linkedin_extractor is None:
            st.warning("Apify API key missing in Streamlit secrets.")
            return

        if st.button("Extract Data", type="primary", disabled=not linkedin_url):
            self._extract_all_data(linkedin_url)

    def _extract_all_data(self, linkedin_url: str):
        with st.spinner("Extracting LinkedIn profile + posts..."):
            linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)

            if not linkedin_data:
                st.error("LinkedIn extraction failed.")
                return

            self.session_state.raw_linkedin_data = linkedin_data

            features_df, debug_info = self.feature_builder.build_features(
                linkedin_data=linkedin_data,
                company_data=None,
                user_data=self.session_state.user_input_data
            )

            self.session_state.final_features = features_df
            self.session_state.debug_info = debug_info
            self.session_state.ready_for_scoring = True

        st.success("Extraction completed successfully.")
        self._show_extracted_data()

        st.markdown("### Debug Values Before Sending to Model")
        debug_df = pd.DataFrame(list(self.session_state.debug_info.items()), columns=["Field", "Value"])
        st.dataframe(debug_df, use_container_width=True)

    def _extract_current_company(self, linkedin_data: dict):
        exp = linkedin_data.get("experience", [])
        if not exp:
            return None
        for e in exp:
            if e.get("is_current", False):
                return e
        return exp[0]

    def _show_extracted_data(self):
        st.markdown("### Extracted Data Preview")

        linkedin_data = self.session_state.raw_linkedin_data
        basic = linkedin_data.get("basic_info", {})
        current_company = self._extract_current_company(linkedin_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal Information**")
            st.write(f"Name: {basic.get('fullname','')}")
            st.write(f"Headline: {basic.get('headline','')}")
            st.write(f"Location: {basic.get('location',{}).get('full','')}")

            # activity display
            activity_days = linkedin_data.get("activity_days", None)
            recent_posts = linkedin_data.get("recent_posts", [])
            st.markdown("**Activity**")
            st.write(f"Recent Activity Days: {activity_days if activity_days is not None else 'Not Available'}")
            if recent_posts:
                st.write(f"Last Post: {recent_posts[0].get('posted_at',{}).get('relative','')}")
            else:
                st.write("Last Post: Not Available")

        with col2:
            st.markdown("**Professional Information**")
            if current_company:
                st.write(f"Current Role: {current_company.get('title','')}")
                st.write(f"Current Company: {current_company.get('company','')}")

        st.markdown("**Manual Company Info Used**")
        st.write(self.session_state.user_input_data)

    def render_scoring_section(self):
        if not getattr(self.session_state, "ready_for_scoring", False):
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

        if probabilities:
            df = pd.DataFrame({"Priority": probabilities.keys(), "Probability": probabilities.values()})
            fig = go.Figure([go.Bar(x=df["Priority"], y=df["Probability"])])
            fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Explain "why" dynamically using feature importance
        st.markdown("### Why this prediction?")
        importance = self.model_predictor.get_feature_importance()
        if importance:
            top = list(importance.items())[:10]
            top_df = pd.DataFrame(top, columns=["feature", "importance"])
            st.dataframe(top_df, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

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

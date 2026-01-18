"""
Banking Lead Scoring - Completely Dynamic
No defaults, no static values, everything from APIs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import core modules
from core.apify_extractor import LinkedInAPIExtractor
from core.company_api import CompanyDataAPI
from core.feature_builder import DynamicFeatureBuilder
from core.model_predictor import ModelPredictor

# Page configuration
st.set_page_config(
    page_title="Dynamic Lead Scoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS without emojis
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
        """Initialize dynamic application with no defaults."""
        self.session_state = st.session_state

        # -------------------------------
        # Session state init (SAFE)
        # -------------------------------
        defaults = {
            "raw_linkedin_data": None,
            "raw_company_data": None,
            "user_input_data": {},
            "final_features": None,
            "prediction": None,
            "api_keys": {"apify": "", "company": ""},
            "ready_for_scoring": False,
            "last_linkedin_url": "",
        }

        for k, v in defaults.items():
            if k not in self.session_state:
                self.session_state[k] = v

        # -------------------------------
        # APIs
        # -------------------------------
        apify_key = st.secrets.get("APIFY", "")
        if apify_key:
            self.session_state.api_keys["apify"] = apify_key
            self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key)
        else:
            self.linkedin_extractor = None

        self.company_api = None
        self.feature_builder = DynamicFeatureBuilder()
        self.model_predictor = ModelPredictor()

    # -------------------------------
    # RESET LOGIC (MAIN FIX)
    # -------------------------------
    def _reset_prospect_state(self):
        """Reset ONLY prospect-related state when a new URL is entered."""
        self.session_state.raw_linkedin_data = None
        self.session_state.raw_company_data = None
        self.session_state.final_features = None
        self.session_state.prediction = None
        self.session_state.ready_for_scoring = False

    def render_header(self):
        """Render application header."""
        st.markdown(
            '<h1 class="main-header">Dynamic Lead Intelligence Platform</h1>',
            unsafe_allow_html=True
        )
        st.markdown("""
            <p style='color: #475569; font-size: 16px;'>
            All data is dynamically extracted from APIs. No defaults or static values are used.
            Missing data will result in empty fields rather than estimates.
            </p>
        """, unsafe_allow_html=True)
        st.divider()

    def render_sidebar(self):
        """Render API configuration sidebar."""
        with st.sidebar:
            st.markdown("### API Configuration")

            # Apify API Key (from secrets only)
            apify_key = st.secrets.get("APIFY", "")

            if apify_key:
                st.success("Apify key loaded from secrets")
            else:
                st.warning("Apify key not found in secrets. LinkedIn extraction will not work.")

            # Company Data API Key
            company_key = st.text_input(
                "Company Data API Key",
                type="password",
                value=self.session_state.api_keys.get("company", ""),
                help="Required for company information"
            )

            # Update API instances if keys changed
            if apify_key != self.session_state.api_keys.get("apify", ""):
                self.session_state.api_keys["apify"] = apify_key
                self.linkedin_extractor = LinkedInAPIExtractor(api_key=apify_key) if apify_key else None

            if company_key != self.session_state.api_keys.get("company", ""):
                self.session_state.api_keys["company"] = company_key
                self.company_api = CompanyDataAPI(api_key=company_key) if company_key else None

            st.divider()

            # Manual override section (ONLY COMPANY FIELDS)
            st.markdown("### Manual Company Data Entry")
            st.info("Use only if Company API fails. LinkedIn fields will still come from Apify.")

            manual_override = st.checkbox("Enable manual company data entry")

            if manual_override:
                with st.form("manual_company_form"):
                    st.text_input("Company Name", key="manual_company")
                    st.text_input("Company Size", key="manual_size", placeholder="51-200 / 201-500 / 5000+")
                    st.text_input("Annual Revenue", key="manual_revenue", placeholder="50M / 1B / 120")
                    st.text_input("Industry", key="manual_industry", placeholder="Banking / Fintech")

                    if st.form_submit_button("Save Manual Company Data"):
                        self.session_state.user_input_data = {
                            "company_name": st.session_state.manual_company,
                            "company_size": st.session_state.manual_size,
                            "annual_revenue": st.session_state.manual_revenue,
                            "industry": st.session_state.manual_industry,
                        }
                        st.success("Manual company data saved")

    def render_input_section(self):
        """Render LinkedIn URL input and extraction."""
        st.markdown("### Step 1: Data Extraction")

        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            key="linkedin_url"
        )

        # -------------------------------
        # RESET when user changes URL
        # -------------------------------
        if linkedin_url and linkedin_url.strip() != self.session_state.last_linkedin_url:
            self.session_state.last_linkedin_url = linkedin_url.strip()
            self._reset_prospect_state()

        if self.linkedin_extractor is None:
            st.warning("LinkedIn API not configured. Add APIFY key in secrets to enable extraction.")
            st.info("You can still score using Manual Data Entry from sidebar.")
            return

        extract_clicked = st.button(
            "Extract Data",
            type="primary",
            disabled=not linkedin_url
        )

        if extract_clicked and linkedin_url:
            self._extract_all_data(linkedin_url)

    def _extract_all_data(self, linkedin_url: str):
        """Extract all data from APIs dynamically."""
        progress_bar = st.progress(0)

        try:
            # Always reset before extraction (prevents stale data)
            self._reset_prospect_state()

            st.markdown("#### LinkedIn Data Extraction")
            progress_bar.progress(25)

            linkedin_data = self.linkedin_extractor.extract_profile(linkedin_url)
            if not linkedin_data:
                st.error("LinkedIn extraction failed")
                return

            self.session_state.raw_linkedin_data = linkedin_data
            st.success("LinkedIn data extracted")

            # Extract current company info
            current_company = self._extract_current_company(linkedin_data)

            if not current_company:
                st.warning("No current company found in profile")
                self.session_state.raw_company_data = None
            else:
                st.markdown("#### Company Data Extraction")
                progress_bar.progress(50)

                company_url = current_company.get("company_linkedin_url")

                if company_url and self.company_api:
                    company_data = self.company_api.get_company_data(company_url)

                    if company_data:
                        self.session_state.raw_company_data = company_data
                        st.success("Company data extracted")
                    else:
                        st.warning("Company data extraction failed")
                        self.session_state.raw_company_data = None
                else:
                    st.warning("Company URL not available or Company API key missing")
                    self.session_state.raw_company_data = None

            # Step 3: Build features
            st.markdown("#### Feature Building")
            progress_bar.progress(75)

            features, debug_info = self.feature_builder.build_features(
    linkedin_data=self.session_state.raw_linkedin_data,
    company_data=self.session_state.raw_company_data,
    user_data=self.session_state.user_input_data
)

            self.session_state.debug_info = debug_info

            if features is None:
                st.error("Feature building failed")
                return

            self.session_state.final_features = features
            st.success("Features built dynamically")

            # Show extracted fields
            self._show_extracted_data()

            # Enable scoring
            self.session_state.ready_for_scoring = True
            progress_bar.progress(100)

        except Exception as e:
            progress_bar.empty()
            st.error(f"Extraction failed: {str(e)}")

    def _extract_current_company(self, linkedin_data: dict):
        """Extract current company from LinkedIn data."""
        experiences = linkedin_data.get("experience", [])
        if not experiences:
            return None

        for exp in experiences:
            if exp.get("is_current", False):
                return exp

        return experiences[0] if experiences else None

    def _show_extracted_data(self):
        """Show dynamically extracted data."""
        st.markdown("### Extracted Data Preview")

        # LinkedIn data
        if self.session_state.raw_linkedin_data:
            basic_info = self.session_state.raw_linkedin_data.get("basic_info", {})
            current_company = self._extract_current_company(self.session_state.raw_linkedin_data)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Personal Information**")
                if basic_info.get("fullname"):
                    st.text(f"Name: {basic_info['fullname']}")
                if basic_info.get("headline"):
                    st.text(f"Headline: {basic_info['headline']}")
                if basic_info.get("location", {}).get("full"):
                    st.text(f"Location: {basic_info['location']['full']}")

            with col2:
                st.markdown("**Professional Information**")
                if current_company:
                    if current_company.get("title"):
                        st.text(f"Current Role: {current_company['title']}")
                    if current_company.get("company"):
                        st.text(f"Current Company: {current_company['company']}")

        # Company data
        if self.session_state.raw_company_data:
            st.markdown("**Company Information**")
            company_data = self.session_state.raw_company_data

            col1, col2, col3 = st.columns(3)

            with col1:
                if company_data.get("size"):
                    st.text(f"Size: {company_data['size']}")

            with col2:
                if company_data.get("revenue"):
                    st.text(f"Revenue: {company_data['revenue']}")

            with col3:
                if company_data.get("industry"):
                    st.text(f"Industry: {company_data['industry']}")

    def render_scoring_section(self):
        """Render scoring section."""
        if not self.session_state.ready_for_scoring:
            return
        st.markdown("### DEBUG: Values Assigned Before Sending to Model")

        if "debug_info" in self.session_state and self.session_state.debug_info:
            debug_df = pd.DataFrame(
        list(self.session_state.debug_info.items()),
        columns=["Field", "Value"]
    )
            st.dataframe(debug_df, use_container_width=True)
        
        st.markdown("### Step 2: Generate Score")

        if st.button("Generate Lead Score", type="primary"):
            with st.spinner("Calculating score..."):
                try:
                    prediction = self.model_predictor.predict(self.session_state.final_features)

                    if prediction is None:
                        st.error("Scoring failed: Model returned no prediction.")
                        return

                    # Store prediction (DO NOT delete extracted info)
                    self.session_state.prediction = prediction

                except Exception as e:
                    st.error(f"Scoring failed: {str(e)}")

        # Always display extracted data + score if available
        if self.session_state.raw_linkedin_data:
            self._show_extracted_data()

        if self.session_state.prediction:
            self._display_results(self.session_state.prediction)

    def _display_results(self, prediction: dict):
        """Display prediction results."""
        st.markdown("### Scoring Results")

        priority = prediction.get("priority", "UNKNOWN")
        confidence = prediction.get("confidence", 0)
        probabilities = prediction.get("probabilities", {})

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

        st.progress(float(confidence))

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

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application execution."""
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

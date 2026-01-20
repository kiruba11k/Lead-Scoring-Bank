"""
Dynamic Feature Builder (Patched)
- Builds model-ready features from LinkedIn + Manual Company inputs
- Supports post-based activity_days
- Fixes revenue parsing for "$1 Billion", "$261.9 Million" etc.
- Returns (features_df, debug_info)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class DynamicFeatureBuilder:
    def __init__(self, metadata_path: str = "models/metadata.json"):
        self.metadata_path = metadata_path
        self.model_feature_names = self._load_feature_names()

    # ----------------------------
    # Metadata
    # ----------------------------
    def _load_feature_names(self):
        try:
            import json, os
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                return meta.get("feature_names", [])
        except:
            pass
        return []

    # ----------------------------
    # Helpers
    # ----------------------------
    def _safe_lower(self, x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x).lower().strip()

    def _extract_title_from_linkedin(self, linkedin_data: dict) -> str:
        title = ""

        if not linkedin_data:
            return title

        basic = linkedin_data.get("basic_info", {})
        headline = basic.get("headline", "")
        title = headline or ""

        exp = linkedin_data.get("experience", [])
        if isinstance(exp, list) and len(exp) > 0:
            for e in exp:
                if e.get("is_current", False) and e.get("title"):
                    title = e.get("title")
                    break

        return title

    def _parse_size_to_number(self, size_str: str) -> int:
        """
        Converts:
        "201-500 employees" -> 350
        "5,001-10,000 employees" -> 7500
        "10,000+" -> 10000
        """
        if not size_str:
            return 0

        s = str(size_str).lower()
        s = s.replace("employees", "").replace("employee", "").strip()
        s = s.replace(",", "").strip()

        # "10000+"
        if "+" in s:
            try:
                return int(float(s.replace("+", "").strip()))
            except:
                return 0

        # "5001-10000"
        if "-" in s:
            try:
                a, b = s.split("-", 1)
                a = float(a.strip())
                b = float(b.strip())
                return int((a + b) / 2)
            except:
                return 0

        # numeric only
        try:
            return int(float(s))
        except:
            return 0

    def _parse_revenue_millions(self, revenue_str: str) -> float:
        """
        Converts revenue into MILLIONS:
        "$261.9 Million" -> 261.9
        "$1 Billion" -> 1000
        "$1.3 Billion" -> 1300
        "$128.9M" -> 128.9
        "$2.5B" -> 2500
        """
        if not revenue_str:
            return 0.0

        s = str(revenue_str).upper().strip()
        s = s.replace(",", "").replace("$", "").strip()

        # FIX: handle words Billion/Million
        s = s.replace("BILLION", "B")
        s = s.replace("MILLION", "M")

        # Extract first numeric value
        match = re.search(r"([0-9]*\.?[0-9]+)", s)
        if not match:
            return 0.0

        val = float(match.group(1))

        # Determine scale
        if "B" in s:
            return val * 1000
        if "M" in s:
            return val

        # If no unit given assume already in millions
        return val

    def _get_revenue_category(self, revenue_millions: float) -> int:
        """
        Must match training encoding logic:
        0 <20M
        1 20-50M
        2 50-100M
        3 100-500M
        4 500M+
        """
        if revenue_millions < 20:
            return 0
        elif revenue_millions < 50:
            return 1
        elif revenue_millions < 100:
            return 2
        elif revenue_millions < 500:
            return 3
        else:
            return 4

    def _compute_activity_score(self, activity_days: float) -> int:
        """
        Convert activity_days into Activity_Score similar to your dataset pattern.
        Lower days => higher score
        """
        if activity_days <= 7:
            return 5
        elif activity_days <= 14:
            return 4
        elif activity_days <= 30:
            return 3
        elif activity_days <= 90:
            return 2
        elif activity_days <= 180:
            return 1
        return 0

    # ----------------------------
    # Main Feature Builder
    # ----------------------------
    def build_features(
        self,
        linkedin_data: dict,
        company_data: dict = None,
        user_data: dict = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Returns:
          features_df (single-row DataFrame)
          debug_info dict (values before model)
        """

        if user_data is None:
            user_data = {}

        # ---- Extract title from LinkedIn ----
        title = self._extract_title_from_linkedin(linkedin_data)

        # ---- Manual company fields (ONLY 4) ----
        company_name = user_data.get("company_name", "") or ""
        company_size = user_data.get("company_size", "") or ""
        annual_revenue = user_data.get("annual_revenue", "") or ""
        industry = user_data.get("industry", "") or ""

        # ---- Normalize text ----
        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)

        # ---- Seniority flags ----
        is_ceo = int(bool(re.search(r"\bceo\b|chief executive|president", title_l)))
        is_c_level = int(bool(re.search(r"\bchief\b|cto|cfo|cio|cro|cmo", title_l)))
        is_evp_svp = int(bool(re.search(r"\bevp\b|\bsvp\b|executive vice president|senior vice president", title_l)))
        is_vp = int(bool(re.search(r"vice president|\bvp\b|\bv\.p\.\b", title_l)))
        is_director = int(bool(re.search(r"director|head of", title_l)))
        is_manager = int(bool(re.search(r"manager|lead|supervisor", title_l)))
        is_officer = int(bool(re.search(r"officer|avp|assistant vice president", title_l)))

        # ---- Department flags ----
        in_lending = int(bool(re.search(r"lend|mortgage|loan|credit|origination|abl", title_l)))
        in_tech = int(bool(re.search(r"tech|technology|it|digital|data|analytics|ai|software", title_l)))
        in_operations = int(bool(re.search(r"operat|process|delivery|service|support", title_l)))
        in_risk = int(bool(re.search(r"risk|compliance|security|audit", title_l)))
        in_finance = int(bool(re.search(r"finance|fpa|treasury", title_l)))
        in_strategy = int(bool(re.search(r"strategy|transformation|innovation|growth", title_l)))

        designation_length = len(title_l)
        designation_word_count = len(title_l.split()) if title_l else 0

        # ---- Seniority score / dept score ----
        seniority_score = (
            is_ceo * 5 +
            is_c_level * 4 +
            is_evp_svp * 3 +
            is_vp * 2 +
            is_director * 2 +
            is_manager * 1 +
            is_officer * 1
        )

        dept_score = (
            in_lending * 2 +
            in_finance * 2 +
            in_risk * 2 +
            in_strategy * 1 +
            in_operations * 1 +
            in_tech * 1
        )

        # ---- Company size ----
        size_numeric = self._parse_size_to_number(company_size)

        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        # ---- Revenue ----
        revenue_millions = self._parse_revenue_millions(annual_revenue)
        revenue_category = self._get_revenue_category(revenue_millions)

        # ---- Activity Days ----
        activity_days_raw = None
        activity_days_final = None

        if linkedin_data:
            activity_days_raw = linkedin_data.get("activity_days", None)

        # If activity missing -> neutral fallback + flag
        activity_missing = 0
        try:
            activity_days_final = float(activity_days_raw)
        except:
            activity_days_final = np.nan

        if np.isnan(activity_days_final):
            activity_missing = 1
            activity_days_final = 30.0  # neutral fallback

        # clip
        activity_days_final = float(np.clip(activity_days_final, 0, 180))

        is_active_week = int(activity_days_final <= 7)
        is_active_month = int(activity_days_final <= 30)

        # ---- Industry flags ----
        is_consumer_lending = int("consumer" in industry_l and "lend" in industry_l)
        is_commercial_banking = int("commercial" in industry_l or "corporate banking" in industry_l)
        is_retail_banking = int("retail" in industry_l or "personal banking" in industry_l)
        is_fintech = int("fintech" in industry_l or "digital bank" in industry_l)
        is_credit_union = int("credit union" in industry_l or "cooperative" in industry_l)

        # ---- Dataset score columns (dynamic calc) ----
        # These should not be hardcoded. They are computed from real extracted data.
        Desig_Score = int(seniority_score + dept_score)
        Size_Score = (
            5 if size_numeric >= 5000 else
            4 if size_numeric >= 1001 else
            3 if size_numeric >= 501 else
            2 if size_numeric >= 201 else
            1 if size_numeric >= 51 else
            0
        )
        Revenue_Score = (
            5 if revenue_millions >= 500 else
            4 if revenue_millions >= 100 else
            3 if revenue_millions >= 50 else
            2 if revenue_millions >= 20 else
            1 if revenue_millions > 0 else
            0
        )
        Activity_Score = self._compute_activity_score(activity_days_final)

        # ---- Final feature row ----
        row = {
            "is_ceo": int(is_ceo),
            "is_c_level": int(is_c_level),
            "is_evp_svp": int(is_evp_svp),
            "is_vp": int(is_vp),
            "is_director": int(is_director),
            "is_manager": int(is_manager),
            "is_officer": int(is_officer),

            "in_lending": int(in_lending),
            "in_tech": int(in_tech),
            "in_operations": int(in_operations),
            "in_risk": int(in_risk),
            "in_finance": int(in_finance),
            "in_strategy": int(in_strategy),

            "designation_length": int(designation_length),
            "designation_word_count": int(designation_word_count),

            "seniority_score": int(seniority_score),
            "dept_score": int(dept_score),

            "size_numeric": int(size_numeric),
            "size_51_200": int(size_51_200),
            "size_201_500": int(size_201_500),
            "size_501_1000": int(size_501_1000),
            "size_1001_5000": int(size_1001_5000),
            "size_5000_plus": int(size_5000_plus),

            "revenue_millions": float(revenue_millions),
            "revenue_category": int(revenue_category),

            "activity_days": float(activity_days_final),
            "is_active_week": int(is_active_week),
            "is_active_month": int(is_active_month),

            "is_consumer_lending": int(is_consumer_lending),
            "is_commercial_banking": int(is_commercial_banking),
            "is_retail_banking": int(is_retail_banking),
            "is_fintech": int(is_fintech),
            "is_credit_union": int(is_credit_union),

            "Desig_Score": int(Desig_Score),
            "Size_Score": int(Size_Score),
            "Revenue_Score": int(Revenue_Score),
            "Activity_Score": int(Activity_Score),

            # NEW (must retrain model if used)
            "activity_missing": int(activity_missing),
        }

        features_df = pd.DataFrame([row])

        # Ensure feature columns match model
        if self.model_feature_names:
            for col in self.model_feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.model_feature_names]

        # Debug info
        debug_info = {
            "title": title,
            "company_name": company_name,
            "company_size_raw": company_size,
            "annual_revenue_raw": annual_revenue,
            "industry_raw": industry,
            "activity_days_raw": activity_days_raw,
            "activity_days_final_used": activity_days_final,
            "activity_missing": activity_missing,
        }

        # add all model feature values
        for col in features_df.columns:
            debug_info[col] = features_df.iloc[0][col]

        return features_df, debug_info

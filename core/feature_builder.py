"""
Dynamic Feature Builder
- Builds features EXACTLY matching trained model feature_names
- No static defaults except safe fallback for missing activity_days
- Manual input allowed only for:
  Company Name, Company Size, Annual Revenue, Industry
- Returns:
  (features_df, debug_info)
"""

import numpy as np
import pandas as pd


class DynamicFeatureBuilder:
    def __init__(self):
        pass

    def _safe_lower(self, x):
        return str(x).lower().strip() if x is not None else ""

    def _parse_size_to_number(self, size_str: str) -> int:
        """
        Converts "5,001-10,000 employees" -> 7500
        Converts "201-500" -> 350
        """
        if not size_str:
            return 0

        s = str(size_str).lower().replace("employees", "").replace("employee", "").strip()
        s = s.replace(",", "").strip()

        if "-" in s:
            a, b = s.split("-")[0].strip(), s.split("-")[1].strip()
            try:
                return int((float(a) + float(b)) / 2)
            except Exception:
                return 0

        if "+" in s:
            try:
                return int(float(s.replace("+", "")))
            except Exception:
                return 0

        try:
            return int(float(s))
        except Exception:
            return 0

    def _parse_revenue_millions(self, revenue_str: str) -> float:
        """
        Converts:
        "$128.9 Million" -> 128.9
        "$1.3 Billion" -> 1300
        """
        if not revenue_str:
            return 0.0

        s = str(revenue_str).upper().replace(",", "").replace("$", "").strip()

        try:
            if "B" in s or "BILLION" in s:
                s = s.replace("BILLION", "").replace("B", "").strip()
                return float(s) * 1000.0
            if "M" in s or "MILLION" in s:
                s = s.replace("MILLION", "").replace("M", "").strip()
                return float(s)
            return float(s)
        except Exception:
            return 0.0

    def build_features(self, linkedin_data: dict, company_data: dict = None, user_data: dict = None):
        """
        Returns:
          features_df (single-row DataFrame)
          debug_info (dict with raw + computed values)
        """
        if user_data is None:
            user_data = {}

        # -----------------------------
        # 1) Extract title/designation
        # -----------------------------
        title = ""
        industry_from_profile = ""

        if linkedin_data:
            basic = linkedin_data.get("basic_info", {})
            title = basic.get("headline", "") or ""

            # prefer current experience title
            exp = linkedin_data.get("experience", [])
            if isinstance(exp, list) and len(exp) > 0:
                for e in exp:
                    if e.get("is_current", False) and e.get("title"):
                        title = e.get("title")
                        break

        # -----------------------------
        # 2) Manual company fields (only 4)
        # -----------------------------
        company_name = user_data.get("company_name", "") or ""
        company_size = user_data.get("company_size", "") or ""
        annual_revenue = user_data.get("annual_revenue", "") or ""
        industry = user_data.get("industry", industry_from_profile) or ""

        # -----------------------------
        # 3) Normalize text
        # -----------------------------
        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)

        # -----------------------------
        # 4) Seniority flags
        # -----------------------------
        is_ceo = int(any(k in title_l for k in ["ceo", "chief executive", "president"]))
        is_c_level = int(any(k in title_l for k in ["chief", "cto", "cfo", "cio", "cro", "cmo"]))
        is_evp_svp = int(any(k in title_l for k in ["evp", "svp", "executive vice president", "senior vice president"]))
        is_vp = int(any(k in title_l for k in ["vice president", "vp", "v.p."]))
        is_director = int(any(k in title_l for k in ["director", "head of"]))
        is_manager = int(any(k in title_l for k in ["manager", "lead", "supervisor"]))
        is_officer = int(any(k in title_l for k in ["officer", "avp", "assistant vice president"]))

        # -----------------------------
        # 5) Department flags
        # -----------------------------
        in_lending = int(any(k in title_l for k in ["lend", "mortgage", "loan", "credit", "origination", "abl"]))
        in_tech = int(any(k in title_l for k in ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"]))
        in_operations = int(any(k in title_l for k in ["operat", "process", "delivery", "service", "support"]))
        in_risk = int(any(k in title_l for k in ["risk", "compliance", "security", "audit"]))
        in_finance = int(any(k in title_l for k in ["finance", "fpa", "treasury", "cfo"]))
        in_strategy = int(any(k in title_l for k in ["strategy", "transformation", "innovation", "growth"]))

        designation_length = int(len(title_l))
        designation_word_count = int(len(title_l.split())) if title_l else 0

        # -----------------------------
        # 6) Dynamic scores
        # -----------------------------
        seniority_score = int(
            is_ceo * 6 +
            is_c_level * 5 +
            is_evp_svp * 4 +
            is_vp * 3 +
            is_director * 2 +
            is_manager * 1 +
            is_officer * 2
        )

        dept_score = int(
            in_lending * 3 +
            in_finance * 2 +
            in_risk * 1 +
            in_strategy * 1 +
            in_tech * 1 +
            in_operations * 1
        )

        # -----------------------------
        # 7) Company size
        # -----------------------------
        size_numeric = int(self._parse_size_to_number(company_size))

        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        # Size_Score (IMPORTANT feature)
        if size_numeric <= 0:
            Size_Score = 0
        elif size_numeric <= 50:
            Size_Score = 1
        elif size_numeric <= 200:
            Size_Score = 2
        elif size_numeric <= 500:
            Size_Score = 3
        elif size_numeric <= 1000:
            Size_Score = 4
        else:
            Size_Score = 5

        # -----------------------------
        # 8) Revenue
        # -----------------------------
        revenue_millions = float(self._parse_revenue_millions(annual_revenue))

        # revenue_category numeric
        if revenue_millions < 20:
            revenue_category = 0
        elif revenue_millions < 50:
            revenue_category = 1
        elif revenue_millions < 100:
            revenue_category = 2
        elif revenue_millions < 500:
            revenue_category = 3
        else:
            revenue_category = 4

        # Revenue_Score (IMPORTANT feature)
        if revenue_millions <= 0:
            Revenue_Score = 0
        elif revenue_millions < 20:
            Revenue_Score = 1
        elif revenue_millions < 50:
            Revenue_Score = 2
        elif revenue_millions < 100:
            Revenue_Score = 3
        elif revenue_millions < 500:
            Revenue_Score = 4
        else:
            Revenue_Score = 5

        # -----------------------------
        # 9) Activity Days
        # -----------------------------
        activity_days_raw = None
        if linkedin_data:
            activity_days_raw = linkedin_data.get("activity_days", None)

        try:
            activity_days = float(activity_days_raw)
        except Exception:
            activity_days = np.nan

        # If missing, use neutral fallback (not worst case)
        if np.isnan(activity_days):
            activity_days_final_used = 30.0
        else:
            activity_days_final_used = float(activity_days)

        activity_days_final_used = float(np.clip(activity_days_final_used, 0, 180))

        is_active_week = int(activity_days_final_used <= 7)
        is_active_month = int(activity_days_final_used <= 30)

        # Activity_Score (IMPORTANT feature)
        if activity_days_final_used <= 7:
            Activity_Score = 5
        elif activity_days_final_used <= 30:
            Activity_Score = 4
        elif activity_days_final_used <= 60:
            Activity_Score = 3
        elif activity_days_final_used <= 120:
            Activity_Score = 2
        else:
            Activity_Score = 1

        # -----------------------------
        # 10) Industry flags
        # -----------------------------
        is_consumer_lending = int(("consumer" in industry_l) and ("lend" in industry_l))
        is_commercial_banking = int(("commercial" in industry_l) or ("corporate banking" in industry_l))
        is_retail_banking = int(("retail" in industry_l) or ("personal banking" in industry_l))
        is_fintech = int(("fintech" in industry_l) or ("digital bank" in industry_l))
        is_credit_union = int(("credit union" in industry_l) or ("cooperative" in industry_l))

        # -----------------------------
        # 11) Final Score Features (IMPORTANT)
        # -----------------------------
        Desig_Score = float(seniority_score + dept_score)

        # -----------------------------
        # 12) Final row
        # -----------------------------
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

            "activity_days": float(activity_days_final_used),
            "is_active_week": int(is_active_week),
            "is_active_month": int(is_active_month),

            "is_consumer_lending": int(is_consumer_lending),
            "is_commercial_banking": int(is_commercial_banking),
            "is_retail_banking": int(is_retail_banking),
            "is_fintech": int(is_fintech),
            "is_credit_union": int(is_credit_union),

            # IMPORTANT extra features expected by your trained model
            "Desig_Score": float(Desig_Score),
            "Size_Score": float(Size_Score),
            "Revenue_Score": float(Revenue_Score),
            "Activity_Score": float(Activity_Score),
        }

        features_df = pd.DataFrame([row])

        debug_info = {
            "title": title,
            "company_name": company_name,
            "company_size_raw": company_size,
            "annual_revenue_raw": annual_revenue,
            "industry_raw": industry,
            "activity_days_raw": activity_days_raw,
            "activity_days_final_used": activity_days_final_used,
        }

        # Add every model feature value into debug_info
        for col in features_df.columns:
            debug_info[col] = features_df.iloc[0][col]

        return features_df, debug_info

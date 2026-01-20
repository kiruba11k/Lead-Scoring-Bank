"""
Dynamic Feature Builder - MUST match model feature names from metadata.json
Returns:
- features_df: DataFrame (1 row)
- debug_info: dict (raw + final values)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, Tuple


class DynamicFeatureBuilder:
    def _safe_lower(self, x) -> str:
        return str(x).lower().strip() if x is not None else ""

    def _parse_size_to_number(self, size_str: str) -> int:
        if not size_str:
            return 0

        s = str(size_str).lower().replace("employees", "").replace("employee", "").strip()
        s = s.replace(",", "")

        # 5,001-10,000 employees
        if "-" in s:
            try:
                a, b = s.split("-")[0].strip(), s.split("-")[1].strip()
                a = float(a.replace("k", "000"))
                b = float(b.replace("k", "000"))
                return int((a + b) / 2)
            except:
                return 0

        # 10,000+ employees
        if "+" in s:
            try:
                return int(float(s.replace("+", "").replace("k", "000")))
            except:
                return 0

        try:
            return int(float(s))
        except:
            return 0

    def _parse_revenue_millions(self, revenue_str: str) -> float:
        if not revenue_str:
            return 0.0

        s = str(revenue_str).upper().replace(",", "").replace("$", "").strip()

        try:
            if "B" in s:
                return float(s.replace("B", "").strip()) * 1000
            if "M" in s:
                return float(s.replace("M", "").strip())
            return float(s)
        except:
            return 0.0

    def build_features(
        self,
        linkedin_data: dict,
        company_data: dict = None,   # NOT USED (manual only)
        user_data: dict = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Returns: (features_df, debug_info)
        """
        if user_data is None:
            user_data = {}

        # --------------------
        # Extract Title
        # --------------------
        title = ""
        if linkedin_data:
            basic = linkedin_data.get("basic_info", {})
            title = basic.get("headline", "") or ""

            exp = linkedin_data.get("experience", [])
            if isinstance(exp, list) and len(exp) > 0:
                for e in exp:
                    if e.get("is_current", False) and e.get("title"):
                        title = e.get("title")
                        break

        # --------------------
        # Manual Company Inputs (ONLY 4)
        # --------------------
        company_name = user_data.get("company_name", "") or ""
        company_size = user_data.get("company_size", "") or ""
        annual_revenue = user_data.get("annual_revenue", "") or ""
        industry = user_data.get("industry", "") or ""

        # --------------------
        # Normalize
        # --------------------
        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)

        # --------------------
        # Seniority flags
        # --------------------
        is_ceo = int(any(k in title_l for k in ["ceo", "chief executive", "president"]))
        is_c_level = int(any(k in title_l for k in ["chief", "cto", "cfo", "cio", "cro", "cmo"]))
        is_evp_svp = int(any(k in title_l for k in ["evp", "svp", "executive vice president", "senior vice president"]))
        is_vp = int(any(k in title_l for k in ["vice president", " vp", "v.p."]))
        is_director = int(any(k in title_l for k in ["director", "head of"]))
        is_manager = int(any(k in title_l for k in ["manager", "lead", "supervisor"]))
        is_officer = int(any(k in title_l for k in ["officer", "avp", "assistant vice president"]))

        # --------------------
        # Department flags
        # --------------------
        in_lending = int(any(k in title_l for k in ["lend", "mortgage", "loan", "credit", "origination", "abl"]))
        in_tech = int(any(k in title_l for k in ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"]))
        in_operations = int(any(k in title_l for k in ["operat", "process", "delivery", "service", "support"]))
        in_risk = int(any(k in title_l for k in ["risk", "compliance", "security", "audit"]))
        in_finance = int(any(k in title_l for k in ["finance", "fpa", "treasury", "cfo"]))
        in_strategy = int(any(k in title_l for k in ["strategy", "transformation", "innovation", "growth"]))

        designation_length = int(len(title_l))
        designation_word_count = int(len(title_l.split())) if title_l else 0

        # --------------------
        # Scores (same style as training)
        # --------------------
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

        # --------------------
        # Size
        # --------------------
        size_numeric = self._parse_size_to_number(company_size)
        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        # --------------------
        # Revenue
        # --------------------
        revenue_millions = float(self._parse_revenue_millions(annual_revenue))

        # Revenue category (numeric bucket like your training)
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

        # --------------------
        # Activity Days (dynamic from extractor posts)
        # --------------------
        activity_days_raw = None
        if linkedin_data:
            activity_days_raw = linkedin_data.get("activity_days", None)

        # IMPORTANT:
        # If no posts -> keep NaN (do NOT force 30)
        try:
            activity_days = float(activity_days_raw)
        except:
            activity_days = np.nan

        # Clip only if we have a valid value
        if not np.isnan(activity_days):
            activity_days = float(np.clip(activity_days, 0, 180))

        is_active_week = int((not np.isnan(activity_days)) and activity_days <= 7)
        is_active_month = int((not np.isnan(activity_days)) and activity_days <= 30)

        # --------------------
        # Industry flags
        # --------------------
        is_consumer_lending = int("consumer" in industry_l and "lend" in industry_l)
        is_commercial_banking = int(("commercial" in industry_l) or ("corporate banking" in industry_l))
        is_retail_banking = int(("retail" in industry_l) or ("personal banking" in industry_l))
        is_fintech = int(("fintech" in industry_l) or ("digital bank" in industry_l) or ("digital banking" in industry_l))
        is_credit_union = int(("credit union" in industry_l) or ("cooperative" in industry_l))

        # --------------------
        # Your engineered score columns (model expects them)
        # --------------------
        # These must be dynamic (not static)
        Desig_Score = int(seniority_score + dept_score)
        Size_Score = int(
            1 * size_51_200 +
            2 * size_201_500 +
            3 * size_501_1000 +
            4 * size_1001_5000 +
            5 * size_5000_plus
        )
        Revenue_Score = int(revenue_category + 1)  # 1..5
        Activity_Score = int(0 if np.isnan(activity_days) else (5 if activity_days <= 7 else 4 if activity_days <= 30 else 2))

        # --------------------
        # Final row
        # --------------------
        row = {
            "is_ceo": is_ceo,
            "is_c_level": is_c_level,
            "is_evp_svp": is_evp_svp,
            "is_vp": is_vp,
            "is_director": is_director,
            "is_manager": is_manager,
            "is_officer": is_officer,
            "in_lending": in_lending,
            "in_tech": in_tech,
            "in_operations": in_operations,
            "in_risk": in_risk,
            "in_finance": in_finance,
            "in_strategy": in_strategy,
            "designation_length": designation_length,
            "designation_word_count": designation_word_count,
            "seniority_score": int(seniority_score),
            "dept_score": int(dept_score),
            "size_numeric": int(size_numeric),
            "size_51_200": size_51_200,
            "size_201_500": size_201_500,
            "size_501_1000": size_501_1000,
            "size_1001_5000": size_1001_5000,
            "size_5000_plus": size_5000_plus,
            "revenue_millions": revenue_millions,
            "revenue_category": int(revenue_category),
            "activity_days": activity_days,
            "is_active_week": is_active_week,
            "is_active_month": is_active_month,
            "is_consumer_lending": is_consumer_lending,
            "is_commercial_banking": is_commercial_banking,
            "is_retail_banking": is_retail_banking,
            "is_fintech": is_fintech,
            "is_credit_union": is_credit_union,
            "Desig_Score": Desig_Score,
            "Size_Score": Size_Score,
            "Revenue_Score": Revenue_Score,
            "Activity_Score": Activity_Score,
        }

        features_df = pd.DataFrame([row])

        debug_info = {
            "title": title,
            "company_name": company_name,
            "company_size_raw": company_size,
            "annual_revenue_raw": annual_revenue,
            "industry_raw": industry,
            "activity_days_raw": activity_days_raw,
            "activity_days_final_used": activity_days,
        }

        # add every final feature
        for col in features_df.columns:
            debug_info[col] = features_df.iloc[0][col]

        return features_df, debug_info

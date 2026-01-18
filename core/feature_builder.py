import pandas as pd
import numpy as np
from typing import Dict, Optional


class DynamicFeatureBuilder:
    """
    Builds the same features used during training.
    Output: single-row DataFrame with all required columns.
    """

    def build_features(
        self,
        linkedin_data: Optional[Dict] = None,
        company_data: Optional[Dict] = None,
        user_data: Optional[Dict] = None,
    ) -> pd.DataFrame:

        linkedin_data = linkedin_data or {}
        company_data = company_data or {}
        user_data = user_data or {}

        # -----------------------------
        # Extract Designation (title)
        # -----------------------------
        title = ""
        experience = linkedin_data.get("experience", [])

        if isinstance(experience, list) and len(experience) > 0:
            current = None
            for e in experience:
                if isinstance(e, dict) and e.get("is_current") is True:
                    current = e
                    break
            if current is None:
                current = experience[0] if isinstance(experience[0], dict) else {}
            title = str(current.get("title", "")).lower().strip()

        # fallback manual title
        if not title and user_data.get("designation"):
            title = str(user_data.get("designation")).lower().strip()

        # -----------------------------
        # Company Size
        # -----------------------------
        size_str = ""
        if company_data.get("size"):
            size_str = str(company_data["size"])
        elif user_data.get("company_size"):
            size_str = str(user_data["company_size"])

        # -----------------------------
        # Revenue
        # -----------------------------
        revenue_str = ""
        if company_data.get("revenue"):
            revenue_str = str(company_data["revenue"])
        elif user_data.get("annual_revenue"):
            revenue_str = str(user_data["annual_revenue"])

        # -----------------------------
        # Industry
        # -----------------------------
        industry_str = ""
        if company_data.get("industry"):
            industry_str = str(company_data["industry"]).lower().strip()
        elif user_data.get("industry"):
            industry_str = str(user_data["industry"]).lower().strip()

        # -----------------------------
        # LinkedIn Activity (days)
        # -----------------------------
        activity_days = None
        if linkedin_data.get("activity_days") is not None:
            activity_days = linkedin_data.get("activity_days")

        if activity_days is None:
            activity_days = user_data.get("activity_days")

        try:
            activity_days = float(activity_days)
        except:
            activity_days = 999.0

        activity_days = float(np.clip(activity_days, 0, 180))

        # -----------------------------
        # Helper functions
        # -----------------------------
        def has_any(txt: str, keywords):
            return 1 if any(k in txt for k in keywords) else 0

        def parse_size_to_number(size_str_in: str) -> int:
            if not size_str_in:
                return 0
            s = str(size_str_in).lower().replace("employees", "").strip()

            if "-" in s:
                try:
                    a, b = s.split("-")[0], s.split("-")[1]
                    return int((float(a) + float(b)) / 2)
                except:
                    return 0

            if "+" in s:
                try:
                    return int(float(s.replace("+", "")))
                except:
                    return 0

            try:
                return int(float(s))
            except:
                return 0

        def parse_revenue_millions(x):
            if not x:
                return 0.0
            s = str(x).upper().replace(",", "").replace("$", "").strip()
            try:
                if "B" in s:
                    return float(s.replace("B", "")) * 1000
                if "M" in s:
                    return float(s.replace("M", ""))
                return float(s)
            except:
                return 0.0

        # -----------------------------
        # Core feature creation
        # -----------------------------
        designation_length = len(title)
        designation_word_count = len(title.split()) if title else 0

        is_ceo = has_any(title, ["ceo", "chief executive", "president"])
        is_c_level = has_any(title, ["chief", "cto", "cfo", "cio", "cro", "cmo"])
        is_evp_svp = has_any(title, ["evp", "svp", "executive vice president", "senior vice president"])
        is_vp = has_any(title, ["vice president", "vp", "v.p."])
        is_director = has_any(title, ["director", "head of"])
        is_manager = has_any(title, ["manager", "lead", "supervisor"])
        is_officer = has_any(title, ["officer", "avp", "assistant vice president"])

        in_lending = has_any(title, ["lend", "mortgage", "loan", "credit"])
        in_tech = has_any(title, ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"])
        in_operations = has_any(title, ["operat", "process", "delivery", "service", "support"])
        in_risk = has_any(title, ["risk", "compliance", "security", "audit"])
        in_finance = has_any(title, ["finance", "fpa", "treasury"])
        in_strategy = has_any(title, ["strategy", "transformation", "innovation", "growth"])

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

        size_numeric = parse_size_to_number(size_str)

        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        revenue_millions = float(parse_revenue_millions(revenue_str))

        # Revenue category must be STRING here (deployment encoder will convert to int)
        if revenue_millions < 20:
            revenue_category = "<20M"
        elif revenue_millions < 50:
            revenue_category = "20-50M"
        elif revenue_millions < 100:
            revenue_category = "50-100M"
        elif revenue_millions < 500:
            revenue_category = "100-500M"
        else:
            revenue_category = "500M+"

        is_active_week = int(activity_days <= 7)
        is_active_month = int(activity_days <= 30)

        is_consumer_lending = int(any(k in industry_str for k in ["consumer lending", "consumer finance", "mortgage lending"]))
        is_commercial_banking = int(any(k in industry_str for k in ["commercial", "business banking", "corporate banking"]))
        is_retail_banking = int(any(k in industry_str for k in ["retail", "personal banking"]))
        is_fintech = int(any(k in industry_str for k in ["fintech", "financial technology", "digital banking"]))
        is_credit_union = int(any(k in industry_str for k in ["credit union", "cooperative"]))

        # -----------------------------
        # Score columns (from API/user if available)
        # If missing -> 0 (safe)
        # -----------------------------
        def safe_float(v):
            try:
                return float(v)
            except:
                return 0.0

        Desig_Score = safe_float(user_data.get("Desig_Score", 0))
        Size_Score = safe_float(user_data.get("Size_Score", 0))
        Revenue_Score = safe_float(user_data.get("Revenue_Score", 0))
        Activity_Score = safe_float(user_data.get("Activity_Score", 0))

        features = {
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
            "designation_length": int(designation_length),
            "designation_word_count": int(designation_word_count),
            "seniority_score": int(seniority_score),
            "dept_score": int(dept_score),
            "size_numeric": int(size_numeric),
            "size_51_200": size_51_200,
            "size_201_500": size_201_500,
            "size_501_1000": size_501_1000,
            "size_1001_5000": size_1001_5000,
            "size_5000_plus": size_5000_plus,
            "revenue_millions": float(revenue_millions),
            "revenue_category": str(revenue_category),
            "activity_days": float(activity_days),
            "is_active_week": is_active_week,
            "is_active_month": is_active_month,
            "is_consumer_lending": is_consumer_lending,
            "is_commercial_banking": is_commercial_banking,
            "is_retail_banking": is_retail_banking,
            "is_fintech": is_fintech,
            "is_credit_union": is_credit_union,
            "Desig_Score": float(Desig_Score),
            "Size_Score": float(Size_Score),
            "Revenue_Score": float(Revenue_Score),
            "Activity_Score": float(Activity_Score),
        }

        return pd.DataFrame([features])

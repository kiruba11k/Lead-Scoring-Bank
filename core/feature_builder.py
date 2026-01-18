import numpy as np
import pandas as pd


class DynamicFeatureBuilder:
    def __init__(self):
        pass

    def _safe_lower(self, x):
        return str(x).lower().strip() if pd.notna(x) else ""

    def _parse_size_to_number(self, size_str):
        if pd.isna(size_str) or size_str is None:
            return 0

        s = str(size_str).lower().strip()
        s = s.replace("employees", "").replace("employee", "").strip()

        if "-" in s:
            parts = s.split("-")
            try:
                a = float(parts[0].strip())
                b = float(parts[1].strip())
                return int((a + b) / 2)
            except Exception:
                return 0

        if "+" in s:
            try:
                return int(float(s.replace("+", "").strip()))
            except Exception:
                return 0

        try:
            return int(float(s))
        except Exception:
            return 0

    def _parse_revenue_millions(self, x):
        if pd.isna(x) or x is None:
            return 0.0

        s = str(x).upper().replace(",", "").replace("$", "").strip()

        # FIX: handle Million/Billion text
        s = s.replace("MILLION", "M").replace("BILLION", "B")
        s = s.replace("MN", "M").replace("BN", "B")

        try:
            if "B" in s:
                return float(s.replace("B", "").strip()) * 1000
            if "M" in s:
                return float(s.replace("M", "").strip())
            return float(s)
        except Exception:
            return 0.0

    def build_features(self, linkedin_data: dict, company_data: dict = None, user_data: dict = None):
        """
        Returns:
            df_features (pd.DataFrame): single-row dataframe for model prediction
            debug_info (dict): all raw + engineered feature values (for debugging)
        """
        if user_data is None:
            user_data = {}
    
        # ---- Extract title/designation ----
        title = ""
        industry = ""
    
        if linkedin_data:
            basic = linkedin_data.get("basic_info", {})
            headline = basic.get("headline", "")
            title = headline or ""
    
            # Prefer current experience title if available
            exp = linkedin_data.get("experience", [])
            if isinstance(exp, list) and len(exp) > 0:
                for e in exp:
                    if e.get("is_current", False) and e.get("title"):
                        title = e.get("title")
                        break
                    
        # ---- Manual company fields (only 4) ----
        company_name = user_data.get("company_name", "")
        company_size = user_data.get("company_size", "")
        annual_revenue = user_data.get("annual_revenue", "")
        industry = user_data.get("industry", industry)
    
        # ---- Normalize text ----
        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)
    
        # ---- Seniority flags ----
        is_ceo = int(any(k in title_l for k in ["ceo", "chief executive", "president"]))
        is_c_level = int(any(k in title_l for k in ["chief", "cto", "cfo", "cio", "cro", "cmo"]))
        is_evp_svp = int(any(k in title_l for k in ["evp", "svp", "executive vice president", "senior vice president"]))
        is_vp = int(any(k in title_l for k in ["vice president", "vp", "v.p."]))
        is_director = int(any(k in title_l for k in ["director", "head of"]))
        is_manager = int(any(k in title_l for k in ["manager", "lead", "supervisor"]))
        is_officer = int(any(k in title_l for k in ["officer", "avp", "assistant vice president"]))
    
        # ---- Department flags ----
        in_lending = int(any(k in title_l for k in ["lend", "mortgage", "loan", "credit", "origination", "abl"]))
        in_tech = int(any(k in title_l for k in ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"]))
        in_operations = int(any(k in title_l for k in ["operat", "process", "delivery", "service", "support"]))
        in_risk = int(any(k in title_l for k in ["risk", "compliance", "security", "audit"]))
        in_finance = int(any(k in title_l for k in ["finance", "fpa", "treasury", "cfo"]))
        in_strategy = int(any(k in title_l for k in ["strategy", "transformation", "innovation", "growth"]))
    
        designation_length = len(title_l)
        designation_word_count = len(title_l.split()) if title_l else 0
    
        # ---- Compute dynamic scores ----
        seniority_score = (
            is_ceo * 6 +
            is_c_level * 5 +
            is_evp_svp * 4 +
            is_vp * 3 +
            is_director * 2 +
            is_manager * 1 +
            is_officer * 2
        )
    
        dept_score = (
            in_lending * 3 +
            in_finance * 2 +
            in_risk * 1 +
            in_strategy * 1 +
            in_tech * 1 +
            in_operations * 1
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
    
        # Revenue category numeric
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
    
        # ---- Activity Days ----
        activity_days = None
        if linkedin_data:
            activity_days = linkedin_data.get("activity_days", None)
    
        try:
            activity_days = float(activity_days)
        except Exception:
            activity_days = np.nan
    
        if np.isnan(activity_days):
            activity_days = 30.0  # neutral fallback
    
        activity_days = float(np.clip(activity_days, 0, 180))
        is_active_week = int(activity_days <= 7)
        is_active_month = int(activity_days <= 30)
    
        # ---- Industry flags ----
        is_consumer_lending = int("consumer" in industry_l and "lend" in industry_l)
        is_commercial_banking = int(("commercial" in industry_l) or ("corporate banking" in industry_l))
        is_retail_banking = int(("retail" in industry_l) or ("personal banking" in industry_l))
        is_fintech = int(("fintech" in industry_l) or ("digital bank" in industry_l))
        is_credit_union = int(("credit union" in industry_l) or ("cooperative" in industry_l))
    
        # ---- Final feature row ----
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
            "revenue_category": int(revenue_category),
            "activity_days": float(activity_days),
            "is_active_week": is_active_week,
            "is_active_month": is_active_month,
            "is_consumer_lending": is_consumer_lending,
            "is_commercial_banking": is_commercial_banking,
            "is_retail_banking": is_retail_banking,
            "is_fintech": is_fintech,
            "is_credit_union": is_credit_union,
        }
    
        # Create df for model
        df = pd.DataFrame([row])
    
        # ---- Debug info ----
        debug_info = {
            "title": title,
            "company_name": company_name,
            "company_size_raw": company_size,
            "annual_revenue_raw": annual_revenue,
            "industry_raw": industry,
            "activity_days_raw": linkedin_data.get("activity_days") if linkedin_data else None,
            "activity_days_final_used": activity_days,
        }
    
        # Add every feature value
        for col in df.columns:
            debug_info[col] = df.iloc[0][col]
    
        return df, debug_info
    

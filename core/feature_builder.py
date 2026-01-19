import re
import numpy as np
import pandas as pd


class DynamicFeatureBuilder:
    def __init__(self):
        pass

    def _safe_lower(self, x):
        return str(x).lower().strip() if x is not None else ""

    def _parse_revenue_millions(self, x):
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

    def _parse_size_to_number(self, size_str):
        """
        Handles: '501-1,000 employees', '201-500', '10000+'
        """
        if not size_str:
            return 0

        s = str(size_str).lower().strip()
        s = s.replace("employees", "").replace("employee", "").strip()
        s = s.replace(",", "")

        # range
        m = re.search(r"(\d+)\s*-\s*(\d+)", s)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            return int((a + b) / 2)

        # plus
        m = re.search(r"(\d+)\s*\+", s)
        if m:
            return int(m.group(1))

        # single number
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))

        return 0

    def _has(self, text: str, pattern: str) -> bool:
        return bool(re.search(pattern, text))

    def build_features(self, linkedin_data: dict, company_data: dict = None, user_data: dict = None):
        """
        Returns:
            features_df (1-row DataFrame)
            debug_info (dict)
        """
        if user_data is None:
            user_data = {}

        # ---------------- TITLE ----------------
        title = ""
        industry = ""

        if linkedin_data:
            basic = linkedin_data.get("basic_info", {})
            headline = basic.get("headline", "")
            title = headline or ""

            exp = linkedin_data.get("experience", [])
            if isinstance(exp, list) and len(exp) > 0:
                for e in exp:
                    if e.get("is_current", False) and e.get("title"):
                        title = e.get("title")
                        break

        # ---------------- MANUAL COMPANY FIELDS ----------------
        company_name = user_data.get("company_name", "")
        company_size = user_data.get("company_size", "")
        annual_revenue = user_data.get("annual_revenue", "")
        industry = user_data.get("industry", industry)

        # normalize
        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)

        # ---------------- SENIORITY FLAGS (FIXED) ----------------
        is_ceo = int(self._has(title_l, r"\bceo\b") or self._has(title_l, r"\bchief executive\b"))
        is_c_level = int(self._has(title_l, r"\b(cto|cfo|cio|cro|cmo)\b") or self._has(title_l, r"\bchief\b"))
        is_evp_svp = int(self._has(title_l, r"\b(evp|svp)\b") or self._has(title_l, r"\bsenior vice president\b"))
        is_vp = int(self._has(title_l, r"\bvice president\b") or self._has(title_l, r"\bvp\b") or self._has(title_l, r"\bv\.p\.\b"))
        is_director = int(self._has(title_l, r"\bdirector\b") or self._has(title_l, r"\bhead of\b"))
        is_manager = int(self._has(title_l, r"\bmanager\b") or self._has(title_l, r"\blead\b") or self._has(title_l, r"\bsupervisor\b"))
        is_officer = int(self._has(title_l, r"\bofficer\b") or self._has(title_l, r"\bavp\b") or self._has(title_l, r"\bassistant vice president\b"))

        # ---------------- DEPARTMENT FLAGS ----------------
        in_lending = int(any(k in title_l for k in ["lend", "mortgage", "loan", "credit", "origination", "abl"]))
        in_tech = int(any(k in title_l for k in ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"]))
        in_operations = int(any(k in title_l for k in ["operat", "process", "delivery", "service", "support"]))
        in_risk = int(any(k in title_l for k in ["risk", "compliance", "security", "audit"]))
        in_finance = int(any(k in title_l for k in ["finance", "fpa", "treasury"]))
        in_strategy = int(any(k in title_l for k in ["strategy", "transformation", "innovation", "growth"]))

        designation_length = len(title_l)
        designation_word_count = len(title_l.split()) if title_l else 0

        # ---------------- SCORES ----------------
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

        # ---------------- SIZE ----------------
        size_numeric = self._parse_size_to_number(company_size)

        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        # ---------------- REVENUE ----------------
        revenue_millions = self._parse_revenue_millions(annual_revenue)

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

        # ---------------- ACTIVITY ----------------
        activity_days_raw = None
        if linkedin_data:
            activity_days_raw = linkedin_data.get("activity_days", None)

        try:
            activity_days = float(activity_days_raw)
        except:
            activity_days = np.nan

        if np.isnan(activity_days):
            activity_days = 30.0  # neutral fallback

        activity_days = float(np.clip(activity_days, 0, 180))
        is_active_week = int(activity_days <= 7)
        is_active_month = int(activity_days <= 30)

        # ---------------- INDUSTRY FLAGS ----------------
        is_consumer_lending = int("consumer" in industry_l and "lend" in industry_l)
        is_commercial_banking = int("commercial" in industry_l or "corporate banking" in industry_l)
        is_retail_banking = int("retail" in industry_l or "personal banking" in industry_l)
        is_fintech = int("fintech" in industry_l or "digital bank" in industry_l)
        is_credit_union = int("credit union" in industry_l or "cooperative" in industry_l)

        # ---------------- FINAL FEATURE ROW ----------------
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

        df = pd.DataFrame([row])

        # ---------------- DEBUG INFO ----------------
        debug_info = {
            "title": title,
            "company_name": company_name,
            "company_size_raw": company_size,
            "annual_revenue_raw": annual_revenue,
            "industry_raw": industry,
            "activity_days_raw": activity_days_raw,
            "activity_days_final_used": activity_days,
        }

        for col in df.columns:
            debug_info[col] = df.iloc[0][col]

        return df, debug_info

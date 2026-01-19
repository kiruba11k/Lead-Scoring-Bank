import pandas as pd
import numpy as np
import re


class DynamicFeatureBuilder:
    def _safe_lower(self, text):
        return str(text).strip().lower() if text else ""

    def _parse_size_to_number(self, size_str: str) -> int:
        """
        Convert company size like:
        '501-1,000 employees' -> 750
        '5,001-10,000 employees' -> 7500
        """
        if not size_str:
            return 0

        s = str(size_str)
        nums = re.findall(r"[\d,]+", s)

        if len(nums) >= 2:
            low = int(nums[0].replace(",", ""))
            high = int(nums[1].replace(",", ""))
            return int((low + high) / 2)

        if len(nums) == 1:
            return int(nums[0].replace(",", ""))

        return 0

    def _parse_revenue_millions(self, revenue_str: str) -> float:
        if not revenue_str:
            return 0.0

        s = str(revenue_str).lower().replace("$", "").replace(",", "").strip()

        # Billion support
        if "billion" in s or "b" in s:
            num = float(re.findall(r"[\d.]+", s)[0])
            return num * 1000

        # Million support
        num = float(re.findall(r"[\d.]+", s)[0])
        return num

    def build_features(self, linkedin_data: dict, user_data: dict = None):
        if user_data is None:
            user_data = {}

        title = ""
        if linkedin_data:
            basic = linkedin_data.get("basic_info", {})
            title = basic.get("headline", "") or ""

            exp = linkedin_data.get("experience", [])
            if isinstance(exp, list):
                for e in exp:
                    if e.get("is_current", False) and e.get("title"):
                        title = e.get("title")
                        break

        # Manual company fields (ONLY 4)
        company_name = user_data.get("company_name", "")
        company_size = user_data.get("company_size", "")
        annual_revenue = user_data.get("annual_revenue", "")
        industry = user_data.get("industry", "")

        title_l = self._safe_lower(title)
        industry_l = self._safe_lower(industry)

        # Seniority flags
        is_ceo = int(any(k in title_l for k in ["ceo", "chief executive", "president"]))
        is_c_level = int(any(k in title_l for k in ["chief", "cto", "cfo", "cio", "cro", "cmo"]))
        is_evp_svp = int(any(k in title_l for k in ["evp", "svp", "executive vice president", "senior vice president"]))
        is_vp = int(any(k in title_l for k in ["vice president", "vp", "v.p."]))
        is_director = int(any(k in title_l for k in ["director", "head of"]))
        is_manager = int(any(k in title_l for k in ["manager", "lead", "supervisor"]))
        is_officer = int(any(k in title_l for k in ["officer", "avp", "assistant vice president"]))

        # Department flags
        in_lending = int(any(k in title_l for k in ["lend", "mortgage", "loan", "credit", "origination", "abl"]))
        in_tech = int(any(k in title_l for k in ["tech", "technology", "it", "digital", "data", "analytics", "ai", "software"]))
        in_operations = int(any(k in title_l for k in ["operat", "process", "delivery", "service", "support"]))
        in_risk = int(any(k in title_l for k in ["risk", "compliance", "security", "audit"]))
        in_finance = int(any(k in title_l for k in ["finance", "fpa", "treasury", "cfo"]))
        in_strategy = int(any(k in title_l for k in ["strategy", "transformation", "innovation", "growth"]))

        designation_length = len(title_l)
        designation_word_count = len(title_l.split()) if title_l else 0

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

        # Company size parsing (FIXED)
        size_numeric = self._parse_size_to_number(company_size)

        size_51_200 = int(51 <= size_numeric <= 200)
        size_201_500 = int(201 <= size_numeric <= 500)
        size_501_1000 = int(501 <= size_numeric <= 1000)
        size_1001_5000 = int(1001 <= size_numeric <= 5000)
        size_5000_plus = int(size_numeric >= 5000)

        revenue_millions = self._parse_revenue_millions(annual_revenue)

        # Revenue category
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

        # Activity days
        activity_days_raw = linkedin_data.get("activity_days", None) if linkedin_data else None
        try:
            activity_days = float(activity_days_raw)
        except:
            activity_days = 180.0

        activity_days = float(np.clip(activity_days, 0, 180))
        is_active_week = int(activity_days <= 7)
        is_active_month = int(activity_days <= 30)

        # Industry flags
        is_consumer_lending = int("consumer" in industry_l and "lend" in industry_l)
        is_commercial_banking = int("commercial" in industry_l or "corporate banking" in industry_l)
        is_retail_banking = int("retail" in industry_l or "personal banking" in industry_l)
        is_fintech = int("fintech" in industry_l or "digital bank" in industry_l)
        is_credit_union = int("credit union" in industry_l or "cooperative" in industry_l)

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

        # Debug info (your exact request)
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

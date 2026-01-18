"""
LinkedIn Profile + Recent Posts Extractor using Apify API (Dynamic)
- Extracts LinkedIn profile details
- Extracts recent posts
- Computes activity_days dynamically from most recent post timestamp
"""

import requests
from datetime import datetime
from typing import Dict, Optional, List, Any


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor = "apimaestro~linkedin-profile-detail"
        self.posts_actor = "apimaestro~linkedin-batch-profile-posts-scraper"

    def _post_json(self, endpoint: str, payload: dict, timeout: int = 90) -> Any:
        try:
            headers = {"Content-Type": "application/json"}
            res = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            if res.status_code not in (200, 201):
                return None
            return res.json()
        except Exception:
            return None

    def extract_profile(self, linkedin_url: str) -> Optional[Dict]:
        if not linkedin_url or not isinstance(linkedin_url, str):
            return None

        linkedin_url = linkedin_url.strip()

        endpoint = (
            f"{self.base_url}/acts/{self.profile_actor}/"
            f"run-sync-get-dataset-items?token={self.api_key}"
        )

        payload = {"includeEmail": False, "profileUrls": [linkedin_url]}

        data = self._post_json(endpoint, payload, timeout=120)

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data[0]
        if isinstance(data, dict):
            return data
        return None

    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> List[Dict]:
        if not profile_url or not isinstance(profile_url, str):
            return []

        profile_url = profile_url.strip()

        endpoint = (
            f"{self.base_url}/acts/{self.posts_actor}/"
            f"run-sync-get-dataset-items?token={self.api_key}"
        )

        payload = {"includeEmail": False, "usernames": [profile_url]}

        data = self._post_json(endpoint, payload, timeout=120)
        if not isinstance(data, list):
            return []

        def get_ts(post: dict) -> int:
            try:
                return int(post.get("posted_at", {}).get("timestamp", 0))
            except Exception:
                return 0

        data_sorted = sorted(data, key=get_ts, reverse=True)
        return data_sorted[:limit]

    def compute_activity_days_from_posts(self, posts: List[Dict]) -> Optional[int]:
        if not posts:
            return None

        ts = posts[0].get("posted_at", {}).get("timestamp")
        if not ts:
            return None

        try:
            post_dt = datetime.fromtimestamp(int(ts) / 1000)
            days = (datetime.now() - post_dt).days
            return max(0, int(days))
        except Exception:
            return None

    def extract_profile_with_activity(self, linkedin_url: str, post_limit: int = 2) -> Dict:
        result = {"profile_data": None, "recent_posts": [], "activity_days": None}

        profile_data = self.extract_profile(linkedin_url)
        result["profile_data"] = profile_data

        posts = self.extract_recent_posts(linkedin_url, limit=post_limit)
        result["recent_posts"] = posts

        result["activity_days"] = self.compute_activity_days_from_posts(posts)
        return result

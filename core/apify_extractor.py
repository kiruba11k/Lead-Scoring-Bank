"""
LinkedIn extraction using Apify
Extracts:
1) Profile info
2) Recent posts for activity_days
"""

import requests
from datetime import datetime


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def extract_profile(self, profile_url: str):
        endpoint = (
            "https://api.apify.com/v2/acts/"
            "apimaestro~linkedin-profile-detail/"
            "run-sync-get-dataset-items?token=" + self.api_key
        )

        payload = {
            "includeEmail": False,
            "profileUrls": [profile_url.strip()]
        }

        headers = {"Content-Type": "application/json"}

        r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        if r.status_code not in (200, 201):
            return None

        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]

        return None

    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> list:
        endpoint = (
            "https://api.apify.com/v2/acts/"
            "apimaestro~linkedin-batch-profile-posts-scraper/"
            "run-sync-get-dataset-items?token=" + self.api_key
        )

        payload = {"usernames": [profile_url.strip()]}
        headers = {"Content-Type": "application/json"}

        r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        if r.status_code not in (200, 201):
            return []

        data = r.json()
        if not isinstance(data, list):
            return []

        data = sorted(data, key=lambda x: x.get("posted_at", {}).get("timestamp", 0), reverse=True)
        return data[:limit]

    def compute_activity_days_from_posts(self, posts: list):
        if not posts:
            return None

        ts = posts[0].get("posted_at", {}).get("timestamp")
        if not ts:
            return None

        try:
            post_dt = datetime.fromtimestamp(int(ts) / 1000)
            return max(0, (datetime.now() - post_dt).days)
        except:
            return None

"""
LinkedIn Extractor using Apify API
- Extracts profile data
- Extracts recent posts
- Computes activity_days from most recent post timestamp
"""

import time
import requests
from datetime import datetime
from typing import Optional, Dict, List


class LinkedInProfileExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor_id = "apimaestro~linkedin-profile-detail"
        self.posts_actor_id = "apimaestro~linkedin-batch-profile-posts-scraper"

    def extract_profile(self, linkedin_url: str) -> Optional[Dict]:
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        run_id = self._start_apify_run(username)
        if not run_id:
            return None

        profile = self._get_apify_results(run_id)
        if not profile:
            return None

        # Extract posts + compute activity
        posts = self.extract_recent_posts(linkedin_url, limit=2)
        activity_days = self.compute_activity_days_from_posts(posts)

        profile["recent_posts"] = posts
        profile["activity_days"] = activity_days

        return profile

    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> List[Dict]:
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor_id}/run-sync-get-dataset-items"
                f"?token={self.api_key}"
            )

            payload = {
                "includeEmail": False,
                "usernames": [profile_url.strip()]
            }

            response = requests.post(endpoint, json=payload, timeout=90)
            if response.status_code not in (200, 201):
                return []

            data = response.json()
            if not isinstance(data, list):
                return []

            def get_ts(post):
                try:
                    return int(post.get("posted_at", {}).get("timestamp", 0))
                except:
                    return 0

            data = sorted(data, key=get_ts, reverse=True)
            return data[:limit]

        except:
            return []

    def compute_activity_days_from_posts(self, posts: List[Dict]) -> Optional[int]:
        if not posts:
            return None

        ts = posts[0].get("posted_at", {}).get("timestamp")
        if not ts:
            return None

        try:
            post_dt = datetime.fromtimestamp(int(ts) / 1000)
            return max(0, int((datetime.now() - post_dt).days))
        except:
            return None

    def _extract_username(self, url: str) -> Optional[str]:
        url = url.strip()

        if "linkedin.com/in/" in url:
            return url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0]

        return None

    def _start_apify_run(self, username: str) -> Optional[str]:
        endpoint = f"{self.base_url}/acts/{self.profile_actor_id}/runs"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {"username": username, "includeEmail": False}

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if response.status_code == 201:
                return response.json()["data"]["id"]
        except:
            pass

        return None

    def _get_apify_results(self, run_id: str, timeout: int = 180) -> Optional[Dict]:
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self._check_run_status(run_id)

            if status == "SUCCEEDED":
                return self._fetch_dataset_items(run_id)
            elif status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                return None

            time.sleep(4)

        return None

    def _check_run_status(self, run_id: str) -> str:
        endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()["data"]["status"]
        except:
            pass

        return "UNKNOWN"

    def _fetch_dataset_items(self, run_id: str) -> Optional[Dict]:
        run_endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            run_response = requests.get(run_endpoint, headers=headers, timeout=10)
            if run_response.status_code != 200:
                return None

            dataset_id = run_response.json()["data"]["defaultDatasetId"]

            dataset_endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
            dataset_response = requests.get(dataset_endpoint, headers=headers, timeout=10)

            if dataset_response.status_code == 200:
                items = dataset_response.json()
                if items and len(items) > 0:
                    return items[0]

        except:
            return None

        return None

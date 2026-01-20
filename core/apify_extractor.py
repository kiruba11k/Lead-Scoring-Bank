"""
LinkedIn Profile + Recent Posts Extractor using Apify
- Extracts LinkedIn Profile data
- Extracts recent posts
- Computes activity_days from latest post timestamp
"""

import requests
import time
from datetime import datetime
from typing import Optional, Dict, List


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor_id = "apimaestro~linkedin-profile-detail"
        self.posts_actor_id = "apimaestro~linkedin-batch-profile-posts-scraper"

    def _extract_username(self, linkedin_url: str) -> Optional[str]:
        if not linkedin_url:
            return None

        url = linkedin_url.strip().lower()
        url = url.replace("https://", "").replace("http://", "").replace("www.", "")

        if "linkedin.com/in/" in url:
            username = url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0]
            return username.strip() if username else None

        return None

    def start_apify_run(self, username: str) -> Optional[dict]:
        try:
            endpoint = f"{self.base_url}/acts/{self.profile_actor_id}/runs"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {"username": username, "includeEmail": False}

            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)

            if response.status_code == 201:
                run_data = response.json()["data"]
                return {
                    "run_id": run_data["id"],
                    "dataset_id": run_data["defaultDatasetId"],
                    "status": "RUNNING"
                }

            return None

        except Exception:
            return None

    def _check_run_status(self, run_id: str) -> str:
        try:
            endpoint = f"{self.base_url}/actor-runs/{run_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()["data"]["status"]
        except Exception:
            pass

        return "UNKNOWN"

    def _fetch_dataset_items(self, dataset_id: str) -> Optional[dict]:
        try:
            endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, headers=headers, timeout=20)
            if response.status_code == 200:
                items = response.json()
                if isinstance(items, list) and len(items) > 0:
                    return items[0]
        except Exception:
            pass

        return None

    def _run_profile_actor(self, username: str) -> Optional[dict]:
        run_info = self.start_apify_run(username)
        if not run_info:
            return None

        run_id = run_info["run_id"]
        dataset_id = run_info["dataset_id"]

        start = time.time()
        timeout = 180

        while time.time() - start < timeout:
            status = self._check_run_status(run_id)

            if status == "SUCCEEDED":
                return self._fetch_dataset_items(dataset_id)

            if status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                return None

            time.sleep(4)

        return None

    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> List[dict]:
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor_id}/"
                f"run-sync-get-dataset-items?token={self.api_key}"
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
                except Exception:
                    return 0

            data = sorted(data, key=get_ts, reverse=True)
            return data[:limit]

        except Exception:
            return []

    def compute_activity_days_from_posts(self, posts: list) -> Optional[int]:
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

    def extract_profile(self, linkedin_url: str) -> Optional[dict]:
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        profile_data = self._run_profile_actor(username)
        if not profile_data:
            return None

        posts = self.extract_recent_posts(linkedin_url, limit=2)
        activity_days = self.compute_activity_days_from_posts(posts)

        profile_data["recent_posts"] = posts
        profile_data["activity_days"] = activity_days  # can be None

        return profile_data

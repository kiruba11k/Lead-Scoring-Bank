"""
Apify LinkedIn Extractor
- Profile actor: apimaestro~linkedin-profile-detail
- Posts actor: apimaestro~linkedin-batch-profile-posts-scraper
"""

import time
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor_id = "apimaestro~linkedin-profile-detail"
        self.posts_actor_id = "apimaestro~linkedin-batch-profile-posts-scraper"

    # -----------------------------
    # Main Public Method
    # -----------------------------
    def extract_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """
        Extracts:
        - Profile data
        - Recent posts (last 2)
        - activity_days (computed from most recent post timestamp)

        Returns dict always.
        """
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        profile_data = self._run_profile_actor(username)
        if not profile_data:
            return None

        # If actor returns list, convert to dict
        if isinstance(profile_data, list) and len(profile_data) > 0:
            profile_data = profile_data[0]

        # Extract recent posts
        posts = self.extract_recent_posts(linkedin_url, limit=2)
        activity_days = self.compute_activity_days_from_posts(posts)

        profile_data["recent_posts"] = posts
        profile_data["activity_days"] = activity_days

        return profile_data

    # -----------------------------
    # Posts Extraction
    # -----------------------------
    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Scrapes recent posts using Apify posts actor.
        Returns latest posts sorted by posted_at.timestamp DESC.
        """
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor_id}/"
                f"run-sync-get-dataset-items?token={self.api_key}"
            )

            payload = {
                "includeEmail": False,
                "usernames": [profile_url.strip()]
            }

            headers = {"Content-Type": "application/json"}

            response = requests.post(endpoint, json=payload, headers=headers, timeout=90)
            if response.status_code not in (200, 201):
                return []

            data = response.json()
            if not isinstance(data, list):
                return []

            # Sort by correct timestamp key
            data = sorted(
                data,
                key=lambda p: int(p.get("posted_at", {}).get("timestamp", 0) or 0),
                reverse=True
            )

            return data[:limit]

        except Exception:
            return []

    def compute_activity_days_from_posts(self, posts: List[Dict[str, Any]]) -> Optional[int]:
        """
        Computes activity_days from most recent post timestamp.
        """
        if not posts:
            return None

        ts = posts[0].get("posted_at", {}).get("timestamp")
        if not ts:
            return None

        try:
            post_dt = datetime.fromtimestamp(int(ts) / 1000)
            return max(0, (datetime.now() - post_dt).days)
        except Exception:
            return None

    # -----------------------------
    # Profile Actor Helpers
    # -----------------------------
    def _extract_username(self, url: str) -> Optional[str]:
        if not url:
            return None

        url = url.strip()
        if "linkedin.com/in/" in url:
            return url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0]
        return None

    def _run_profile_actor(self, username: str) -> Optional[Dict[str, Any]]:
        run_id = self._start_run(username)
        if not run_id:
            return None
        return self._poll_and_fetch(run_id)

    def _start_run(self, username: str) -> Optional[str]:
        endpoint = f"{self.base_url}/acts/{self.profile_actor_id}/runs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"username": username, "includeEmail": False}

        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if r.status_code == 201:
                return r.json()["data"]["id"]
        except Exception:
            return None

        return None

    def _poll_and_fetch(self, run_id: str, timeout: int = 180):
        start = time.time()

        while time.time() - start < timeout:
            status = self._check_status(run_id)

            if status == "SUCCEEDED":
                return self._fetch_dataset(run_id)

            if status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                return None

            time.sleep(4)

        return None

    def _check_status(self, run_id: str):
        endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            r = requests.get(endpoint, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()["data"]["status"]
        except Exception:
            return "UNKNOWN"

        return "UNKNOWN"

    def _fetch_dataset(self, run_id: str):
        run_endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            run_res = requests.get(run_endpoint, headers=headers, timeout=10)
            if run_res.status_code != 200:
                return None

            dataset_id = run_res.json()["data"]["defaultDatasetId"]
            dataset_endpoint = f"{self.base_url}/datasets/{dataset_id}/items"

            ds_res = requests.get(dataset_endpoint, headers=headers, timeout=15)
            if ds_res.status_code != 200:
                return None

            items = ds_res.json()
            if isinstance(items, list) and len(items) > 0:
                return items[0]

            return items

        except Exception:
            return None

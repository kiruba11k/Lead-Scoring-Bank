"""
Apify LinkedIn Extractor
- Extracts profile details from Apify actor: apimaestro~linkedin-profile-detail
- Extracts recent posts from Apify actor: apimaestro~linkedin-batch-profile-posts-scraper
- Computes activity_days based on most recent post
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

        url = linkedin_url.strip()

        # Accept both linkedin.com/in/xxx and full URLs
        if "linkedin.com/in/" in url:
            username = url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0]
            return username.strip()
        return None

    def _start_profile_actor(self, username: str) -> Optional[Dict]:
        endpoint = f"{self.base_url}/acts/{self.profile_actor_id}/runs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"username": username, "includeEmail": False}

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code == 201:
            data = resp.json()["data"]
            return {"run_id": data["id"], "dataset_id": data["defaultDatasetId"]}
        return None

    def _wait_for_run(self, run_id: str, timeout: int = 180) -> bool:
        start = time.time()
        endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        while time.time() - start < timeout:
            r = requests.get(endpoint, headers=headers, timeout=15)
            if r.status_code == 200:
                status = r.json()["data"]["status"]
                if status == "SUCCEEDED":
                    return True
                if status in ("FAILED", "TIMED_OUT", "ABORTED"):
                    return False
            time.sleep(3)

        return False

    def _fetch_dataset_items(self, dataset_id: str) -> Optional[List[Dict]]:
        endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.get(endpoint, headers=headers, timeout=30)
        if r.status_code == 200:
            items = r.json()
            if isinstance(items, list):
                return items
        return None

    def _run_profile_actor(self, username: str) -> Optional[Dict]:
        run_info = self._start_profile_actor(username)
        if not run_info:
            return None

        ok = self._wait_for_run(run_info["run_id"])
        if not ok:
            return None

        items = self._fetch_dataset_items(run_info["dataset_id"])
        if not items:
            return None

        # actor returns list, we take first dict
        if isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict):
            return items[0]

        return None

    # -------------------- POSTS --------------------
    def extract_recent_posts(self, profile_url: str, limit: int = 2) -> List[Dict]:
        """
        Scrape recent posts from LinkedIn using Apify posts actor.
        Returns latest posts sorted by timestamp desc.
        """
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor_id}/run-sync-get-dataset-items"
                f"?token={self.api_key}"
            )

            payload = {"includeEmail": False, "usernames": [profile_url.strip()]}
            headers = {"Content-Type": "application/json"}

            response = requests.post(endpoint, json=payload, headers=headers, timeout=90)
            if response.status_code not in (200, 201):
                return []

            data = response.json()
            if not isinstance(data, list):
                return []

            def get_ts(post: Dict) -> int:
                try:
                    return int(post.get("posted_at", {}).get("timestamp", 0))
                except Exception:
                    return 0

            data = sorted(data, key=get_ts, reverse=True)
            return data[:limit]

        except Exception:
            return []

    def compute_activity_days_from_posts(self, posts: List[Dict]) -> Optional[int]:
        """
        Compute activity_days from most recent post timestamp.
        Returns int days or None.
        """
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

    # -------------------- MAIN --------------------
    def extract_profile(self, linkedin_url: str) -> Optional[Dict]:
        """
        Extract profile + recent posts + activity_days.
        Always returns a dict for that URL only.
        """
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        profile_data = self._run_profile_actor(username)
        if not profile_data:
            return None

        # Attach posts + activity_days
        posts = self.extract_recent_posts(linkedin_url, limit=2)
        activity_days = self.compute_activity_days_from_posts(posts)

        profile_data["recent_posts"] = posts
        profile_data["activity_days"] = activity_days

        return profile_data

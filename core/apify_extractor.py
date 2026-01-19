"""
Apify LinkedIn Extractor (Profile + Recent Posts)
- Extracts profile details using Apify actor: apimaestro~linkedin-profile-detail
- Extracts recent posts using Apify actor: apimaestro~linkedin-batch-profile-posts-scraper
- Computes activity_days from latest post timestamp
"""

import requests
import time
from datetime import datetime
from typing import Optional, Dict, Any, List


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor_id = "apimaestro~linkedin-profile-detail"
        self.posts_actor_id = "apimaestro~linkedin-batch-profile-posts-scraper"

    # -----------------------------
    # Public
    # -----------------------------
    def extract_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract profile + recent posts + activity_days
        Returns dict (single profile)
        """
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        profile_data = self._run_profile_actor(username=username)
        if not profile_data:
            return None

        # Sometimes actor returns list
        if isinstance(profile_data, list) and len(profile_data) > 0:
            profile_data = profile_data[0]

        # Attach recent posts + activity_days
        posts = self.extract_recent_posts(linkedin_url=linkedin_url, limit=2)
        activity_days = self.compute_activity_days_from_posts(posts)

        profile_data["recent_posts"] = posts
        profile_data["activity_days"] = activity_days

        return profile_data

    def extract_recent_posts(self, linkedin_url: str, limit: int = 2) -> List[dict]:
        """
        Scrape recent posts from LinkedIn profile using Apify posts actor.
        Returns latest posts sorted by posted_at.timestamp desc.
        """
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor_id}/"
                f"run-sync-get-dataset-items?token={self.api_key}"
            )

            payload = {
                "includeEmail": False,
                "usernames": [linkedin_url.strip()],
            }

            resp = requests.post(endpoint, json=payload, timeout=90)
            if resp.status_code not in (200, 201):
                return []

            data = resp.json()
            if not isinstance(data, list):
                return []

            def get_ts(post: dict) -> int:
                try:
                    return int(post.get("posted_at", {}).get("timestamp", 0))
                except Exception:
                    return 0

            data = sorted(data, key=get_ts, reverse=True)
            return data[:limit]

        except Exception:
            return []

    def compute_activity_days_from_posts(self, posts: List[dict]) -> Optional[int]:
        """
        Returns days since most recent post.
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

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _extract_username(self, url: str) -> Optional[str]:
        """
        Extract username from LinkedIn URL:
        https://www.linkedin.com/in/username/
        """
        if not url:
            return None

        u = url.strip()

        # Accept /in/username OR full URL
        if "linkedin.com/in/" in u:
            part = u.split("linkedin.com/in/")[1]
            username = part.split("/")[0].split("?")[0].strip()
            return username if username else None

        # Sometimes user may paste just username
        if "/" not in u and " " not in u:
            return u

        return None

    def _run_profile_actor(self, username: str) -> Optional[dict]:
        """
        Runs Apify profile actor and returns first dataset item.
        """
        run = self._start_actor_run(username=username)
        if not run:
            return None

        run_id = run.get("run_id")
        dataset_id = run.get("dataset_id")
        if not run_id or not dataset_id:
            return None

        ok = self._wait_for_run(run_id=run_id, timeout=180)
        if not ok:
            return None

        items = self._fetch_dataset_items(dataset_id=dataset_id)
        if not items:
            return None

        # Actor returns list
        if isinstance(items, list) and len(items) > 0:
            return items[0]

        # If already dict
        if isinstance(items, dict):
            return items

        return None

    def _start_actor_run(self, username: str) -> Optional[dict]:
        """
        Start Apify actor run asynchronously.
        """
        try:
            endpoint = f"{self.base_url}/acts/{self.profile_actor_id}/runs"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"username": username, "includeEmail": False}

            resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if resp.status_code != 201:
                return None

            run_data = resp.json().get("data", {})
            return {
                "run_id": run_data.get("id"),
                "dataset_id": run_data.get("defaultDatasetId"),
                "status": run_data.get("status", "RUNNING"),
            }
        except Exception:
            return None

    def _wait_for_run(self, run_id: str, timeout: int = 180) -> bool:
        """
        Poll actor run status until SUCCEEDED.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self._get_run_status(run_id=run_id)
            if status == "SUCCEEDED":
                return True
            if status in ("FAILED", "ABORTED", "TIMED_OUT"):
                return False
            time.sleep(4)
        return False

    def _get_run_status(self, run_id: str) -> str:
        try:
            endpoint = f"{self.base_url}/actor-runs/{run_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(endpoint, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("status", "UNKNOWN")
        except Exception:
            pass
        return "UNKNOWN"

    def _fetch_dataset_items(self, dataset_id: str) -> Optional[list]:
        try:
            endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(endpoint, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

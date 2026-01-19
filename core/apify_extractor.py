"""
Apify Extractor
- Extract LinkedIn profile data
- Extract recent posts
- Compute activity_days from most recent post timestamp
"""

import requests
import time
from datetime import datetime, timedelta


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.profile_actor = "apimaestro~linkedin-profile-detail"
        self.posts_actor = "apimaestro~linkedin-batch-profile-posts-scraper"

    def _extract_username(self, linkedin_url: str):
        if not linkedin_url:
            return None

        url = linkedin_url.strip()

        # handle formats like https://www.linkedin.com/in/abc/
        if "linkedin.com/in/" not in url:
            return None

        username = url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0].strip()
        return username if username else None

    def extract_profile(self, linkedin_url: str):
        """
        Extract profile + recent posts + activity_days
        Returns dict
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

    def _run_profile_actor(self, username: str, timeout: int = 180):
        """
        Run Apify actor async, poll until finished, return dataset first item
        """
        endpoint = f"{self.base_url}/acts/{self.profile_actor}/runs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"username": username, "includeEmail": False}

        try:
            res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if res.status_code != 201:
                return None

            run_id = res.json()["data"]["id"]
            dataset_id = res.json()["data"]["defaultDatasetId"]

            # Poll status
            start = time.time()
            while time.time() - start < timeout:
                status = self._check_run_status(run_id)
                if status == "SUCCEEDED":
                    return self._fetch_dataset_items(dataset_id)
                if status in ["FAILED", "ABORTED", "TIMED_OUT"]:
                    return None
                time.sleep(4)

            return None

        except Exception:
            return None

    def _check_run_status(self, run_id: str):
        try:
            endpoint = f"{self.base_url}/actor-runs/{run_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            res = requests.get(endpoint, headers=headers, timeout=10)
            if res.status_code == 200:
                return res.json()["data"]["status"]
        except Exception:
            pass
        return "UNKNOWN"

    def _fetch_dataset_items(self, dataset_id: str):
        try:
            endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            res = requests.get(endpoint, headers=headers, timeout=20)
            if res.status_code == 200:
                items = res.json()
                if isinstance(items, list) and len(items) > 0:
                    return items[0]
        except Exception:
            pass
        return None

    # ---------------- POSTS ---------------- #

    def extract_recent_posts(self, profile_url: str, limit: int = 2):
        """
        Scrape posts, filter last 30 days, return latest `limit`
        """
        try:
            endpoint = (
                f"{self.base_url}/acts/{self.posts_actor}/"
                f"run-sync-get-dataset-items?token={self.api_key}"
            )

            payload = {
                "includeEmail": False,
                "usernames": [profile_url.strip()],
            }

            headers = {"Content-Type": "application/json"}

            response = requests.post(endpoint, json=payload, headers=headers, timeout=90)

            if response.status_code not in (200, 201):
                return []

            data = response.json()
            if not isinstance(data, list):
                return []

            thirty_days_ago = datetime.now() - timedelta(days=30)
            filtered_posts = []

            for post in data:
                if not isinstance(post, dict):
                    continue

                # ✅ correct key mapping
                ts = post.get("posted_at", {}).get("timestamp")
                if not ts:
                    continue

                try:
                    post_dt = datetime.fromtimestamp(int(ts) / 1000)
                    if post_dt >= thirty_days_ago:
                        filtered_posts.append(post)
                except Exception:
                    continue

            # sort most recent first
            filtered_posts = sorted(
                filtered_posts,
                key=lambda x: x.get("posted_at", {}).get("timestamp", 0),
                reverse=True,
            )

            return filtered_posts[:limit]

        except Exception:
            return []

    def compute_activity_days_from_posts(self, posts: list):
        """
        Compute activity days from most recent post timestamp.
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

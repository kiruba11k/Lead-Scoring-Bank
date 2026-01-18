"""
Apify LinkedIn Extractor (Profile + Recent Posts + Activity Days)

This module:
1) Extracts LinkedIn profile data using Apify actor
2) Extracts recent posts using Apify posts actor
3) Computes activity_days from most recent post timestamp
4) Adds activity_days + recent_posts into profile response
"""

import requests
from datetime import datetime
from typing import Optional, Dict, Any, List


class LinkedInAPIExtractor:
    def __init__(self, api_key: str, debug: bool = False):
        self.api_key = api_key
        self.debug = debug

        # Actor IDs
        self.profile_actor = "apimaestro~linkedin-profile-detail"
        self.posts_actor = "apimaestro~linkedin-batch-profile-posts-scraper"

        self.base_url = "https://api.apify.com/v2"

    # -----------------------------
    # Public Main Function
    # -----------------------------
    def extract_profile(self, linkedin_url: str, posts_limit: int = 2) -> Optional[Dict[str, Any]]:
        """
        Extract full LinkedIn profile + recent posts + activity_days
        """
        if not linkedin_url:
            return None

        profile_data = self._fetch_profile_detail(linkedin_url)
        if not profile_data:
            return None

        # Extract recent posts
        recent_posts = self._fetch_recent_posts(linkedin_url, limit=posts_limit)

        # Compute activity_days from posts
        activity_days = self._compute_activity_days(recent_posts)

        # Attach to profile output (dynamic)
        profile_data["recent_posts"] = recent_posts
        profile_data["activity_days"] = activity_days if activity_days is not None else None

        if self.debug:
            print("\n=========== DEBUG: Apify Extractor ===========")
            print("LinkedIn URL:", linkedin_url)
            print("Posts fetched:", len(recent_posts))
            print("Computed activity_days:", activity_days)
            print("============================================\n")

        return profile_data

    # -----------------------------
    # Profile Detail Extraction
    # -----------------------------
    def _fetch_profile_detail(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """
        Uses Apify profile actor: apimaestro~linkedin-profile-detail
        """
        try:
            endpoint = f"{self.base_url}/acts/{self.profile_actor}/run-sync-get-dataset-items?token={self.api_key}"

            payload = {
                "username": linkedin_url.strip(),
                "includeEmail": False
            }

            headers = {"Content-Type": "application/json"}

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)

            if resp.status_code not in (200, 201):
                if self.debug:
                    print("Profile actor failed:", resp.status_code, resp.text[:300])
                return None

            data = resp.json()

            # Apify returns list of items
            if isinstance(data, list) and len(data) > 0:
                return data[0]

            # Some actors return dict
            if isinstance(data, dict):
                return data

            return None

        except Exception as e:
            if self.debug:
                print("Profile extraction error:", str(e))
            return None

    # -----------------------------
    # Posts Extraction
    # -----------------------------
    def _fetch_recent_posts(self, linkedin_url: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Uses Apify posts actor: apimaestro~linkedin-batch-profile-posts-scraper
        Returns latest posts sorted by timestamp desc.
        """
        try:
            endpoint = f"{self.base_url}/acts/{self.posts_actor}/run-sync-get-dataset-items?token={self.api_key}"

            payload = {
                "includeEmail": False,
                "usernames": [linkedin_url.strip()]  # MUST be list
            }

            headers = {"Content-Type": "application/json"}

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)

            if resp.status_code not in (200, 201):
                if self.debug:
                    print("Posts actor failed:", resp.status_code, resp.text[:300])
                return []

            data = resp.json()

            if not isinstance(data, list):
                return []

            def get_ts(post):
                try:
                    return int(post.get("posted_at", {}).get("timestamp", 0))
                except:
                    return 0

            # Sort latest first
            data = sorted(data, key=get_ts, reverse=True)

            return data[:limit]

        except Exception as e:
            if self.debug:
                print("Posts extraction error:", str(e))
            return []

    # -----------------------------
    # Activity Days Calculation
    # -----------------------------
    def _compute_activity_days(self, posts: List[Dict[str, Any]]) -> Optional[int]:
        """
        Returns number of days since most recent post.
        If no posts or no timestamp -> None
        """
        if not posts:
            return None

        ts = posts[0].get("posted_at", {}).get("timestamp")
        if not ts:
            return None

        try:
            post_dt = datetime.fromtimestamp(int(ts) / 1000)
            delta_days = (datetime.now() - post_dt).days
            return max(0, int(delta_days))
        except:
            return None

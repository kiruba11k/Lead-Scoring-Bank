import time
import requests


class LinkedInAPIExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.actor_id = "apimaestro~linkedin-profile-detail"

    def extract_profile(self, linkedin_url: str):
        username = self._extract_username(linkedin_url)
        if not username:
            return None

        run_id = self._start_run(username)
        if not run_id:
            return None

        profile_data = self._poll_and_fetch(run_id)

        if isinstance(profile_data, list) and len(profile_data) > 0:
            profile_data = profile_data[0]

        return profile_data

    def _extract_username(self, url: str):
        if not url:
            return None
        url = url.strip()
        if "linkedin.com/in/" in url:
            return url.split("linkedin.com/in/")[1].split("/")[0].split("?")[0]
        return None

    def _start_run(self, username: str):
        endpoint = f"{self.base_url}/acts/{self.actor_id}/runs"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"username": username, "includeEmail": False}

        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if r.status_code == 201:
                return r.json()["data"]["id"]
        except:
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
        except:
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

        except:
            return None

"""
LinkedIn Data Extractor - No defaults, pure API data
"""

import requests
import time
from typing import Dict, Optional, List
from datetime import datetime

class LinkedInAPIExtractor:
    """Extracts LinkedIn data directly from Apify API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.actor_id = "apimaestro~linkedin-profile-detail"
    
    def extract_profile(self, linkedin_url: str) -> Optional[Dict]:
        """
        Extract LinkedIn profile data.
        Returns None if extraction fails.
        """
        try:
            # Start Apify run
            run_id = self._start_run(linkedin_url)
            if not run_id:
                return None
            
            # Wait for completion
            data = self._wait_for_results(run_id)
            return data
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
    
    def _start_run(self, linkedin_url: str) -> Optional[str]:
        """Start Apify actor run."""
        endpoint = f"{self.base_url}/acts/{self.actor_id}/runs"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Extract username
        username = self._extract_username(linkedin_url)
        if not username:
            return None
        
        payload = {
            "username": username,
            "includeEmail": False
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 201:
                return response.json()["data"]["id"]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Start run error: {e}")
            return None
    
    def _extract_username(self, url: str) -> Optional[str]:
        """Extract username from LinkedIn URL."""
        if '/in/' in url:
            parts = url.split('/in/')
            if len(parts) > 1:
                username = parts[1].split('/')[0].split('?')[0]
                return username.strip()
        return None
    
    def _wait_for_results(self, run_id: str, timeout: int = 180) -> Optional[Dict]:
        """Wait for Apify run to complete and get results."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self._check_status(run_id)
            
            if status == "SUCCEEDED":
                return self._get_results(run_id)
            elif status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                return None
            
            time.sleep(5)
        
        return None
    
    def _check_status(self, run_id: str) -> str:
        """Check run status."""
        endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()["data"]["status"]
        except:
            pass
        
        return "UNKNOWN"
    
    def _get_results(self, run_id: str) -> Optional[Dict]:
        """Get dataset items from completed run."""
        # Get run details first
        run_endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            run_response = requests.get(run_endpoint, headers=headers, timeout=10)
            if run_response.status_code == 200:
                run_data = run_response.json()["data"]
                dataset_id = run_data.get("defaultDatasetId")
                
                if dataset_id:
                    dataset_endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
                    dataset_response = requests.get(dataset_endpoint, headers=headers, timeout=10)
                    
                    if dataset_response.status_code == 200:
                        items = dataset_response.json()
                        if items and len(items) > 0:
                            return items[0]
        
        except Exception as e:
            print(f"Get results error: {e}")
        
        return None

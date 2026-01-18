"""
LinkedIn Profile Extractor using Apify API
Extracts current role and professional information
"""

import requests
import time
import json
from typing import Dict, Optional, List
from datetime import datetime

class LinkedInProfileExtractor:
    """Extracts LinkedIn profile data using Apify API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"
        self.actor_id = "apimaestro~linkedin-profile-detail"
    
    def extract_profile_data(self, linkedin_url: str) -> Optional[Dict]:
        """
        Extract complete profile data from LinkedIn URL.
        
        Args:
            linkedin_url: Complete LinkedIn profile URL
            
        Returns:
            Dictionary containing extracted profile data or None if failed
        """
        try:
            # Extract username from URL
            username = self._extract_username(linkedin_url)
            
            if not username:
                raise ValueError("Invalid LinkedIn URL format")
            
            # Start Apify actor run
            run_id = self._start_apify_run(username)
            
            if not run_id:
                return None
            
            # Wait for completion and get results
            return self._get_apify_results(run_id)
            
        except Exception as e:
            print(f"Profile extraction error: {str(e)}")
            return None
    
    def _extract_username(self, url: str) -> Optional[str]:
        """Extract username from LinkedIn URL."""
        url = url.strip().lower()
        
        # Remove protocol
        if 'https://' in url:
            url = url.replace('https://', '')
        elif 'http://' in url:
            url = url.replace('http://', '')
        
        # Remove www
        if 'www.' in url:
            url = url.replace('www.', '')
        
        # Extract username after linkedin.com/in/
        if 'linkedin.com/in/' in url:
            username = url.split('linkedin.com/in/')[1].split('/')[0].split('?')[0]
            return username
        
        return None
    
    def _start_apify_run(self, username: str) -> Optional[str]:
        """Start Apify actor run for LinkedIn profile."""
        endpoint = f"{self.base_url}/acts/{self.actor_id}/runs"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "username": username,
            "includeEmail": False,
            "timeout": 120
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 201:
                run_data = response.json()
                return run_data["data"]["id"]
            else:
                print(f"Apify API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error starting Apify run: {str(e)}")
            return None
    
    def _get_apify_results(self, run_id: str, timeout: int = 180) -> Optional[Dict]:
        """Get results from Apify run with polling."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check run status
            status = self._check_run_status(run_id)
            
            if status == "SUCCEEDED":
                # Fetch dataset items
                return self._fetch_dataset_items(run_id)
            elif status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                print(f"Apify run failed with status: {status}")
                return None
            
            # Wait before polling again
            time.sleep(5)
        
        print("Apify run timed out")
        return None
    
    def _check_run_status(self, run_id: str) -> str:
        """Check the status of an Apify run."""
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
        """Fetch dataset items from completed Apify run."""
        # First get run details to find dataset ID
        run_endpoint = f"{self.base_url}/actor-runs/{run_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # Get run details
            run_response = requests.get(run_endpoint, headers=headers, timeout=10)
            if run_response.status_code == 200:
                run_data = run_response.json()["data"]
                dataset_id = run_data["defaultDatasetId"]
                
                # Get dataset items
                dataset_endpoint = f"{self.base_url}/datasets/{dataset_id}/items"
                dataset_response = requests.get(dataset_endpoint, headers=headers, timeout=10)
                
                if dataset_response.status_code == 200:
                    items = dataset_response.json()
                    if items and len(items) > 0:
                        return items[0]
        
        except Exception as e:
            print(f"Error fetching dataset items: {e}")
        
        return None

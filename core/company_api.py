"""
Company Data API - No defaults, pure API data
"""

import requests
from typing import Dict, Optional
import re

class CompanyDataAPI:
    """Fetches company data from various APIs."""
    
    def __init__(self, api_key: str, api_provider: str = "magicalapi"):
        self.api_key = api_key
        self.api_provider = api_provider
        
        # API endpoints based on provider
        self.endpoints = {
            "magicalapi": "https://api.magicalapi.com/v1/linkedin/company",
            "proxiesapi": "https://api.proxiesapi.com/linkedin/company",
            "scrapingbee": "https://app.scrapingbee.com/api/v1/linkedin/company"
        }
    
    def get_company_data(self, company_url: str) -> Optional[Dict]:
        """
        Get company data from API.
        Returns None if API call fails.
        """
        if not self.api_key:
            return None
        
        endpoint = self.endpoints.get(self.api_provider)
        if not endpoint:
            return None
        
        # Extract company identifier
        company_id = self._extract_company_id(company_url)
        if not company_id:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "url": company_url,
                "company_id": company_id,
                "extract": "size,industry,revenue,employees,description"
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract only available data
                company_data = {}
                
                # Size
                if data.get("employee_count"):
                    company_data["size"] = self._format_size(data["employee_count"])
                elif data.get("company_size"):
                    company_data["size"] = data["company_size"]
                
                # Revenue
                if data.get("estimated_revenue"):
                    company_data["revenue"] = data["estimated_revenue"]
                elif data.get("revenue"):
                    company_data["revenue"] = data["revenue"]
                
                # Industry
                if data.get("industry"):
                    company_data["industry"] = data["industry"]
                elif data.get("sector"):
                    company_data["industry"] = data["sector"]
                
                # Additional data if available
                if data.get("founded_year"):
                    company_data["founded_year"] = data["founded_year"]
                
                if data.get("headquarters"):
                    company_data["headquarters"] = data["headquarters"]
                
                if data.get("description"):
                    company_data["description"] = data["description"]
                
                return company_data if company_data else None
            else:
                print(f"Company API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Company API call error: {e}")
            return None
    
    def _extract_company_id(self, url: str) -> Optional[str]:
        """Extract company ID from LinkedIn URL."""
        if '/company/' in url:
            parts = url.split('/company/')
            if len(parts) > 1:
                company_id = parts[1].split('/')[0].split('?')[0]
                return company_id.strip()
        return None
    
    def _format_size(self, employee_count: int) -> str:
        """Format employee count to size category."""
        if not employee_count:
            return ""
        
        if employee_count <= 10:
            return "1-10"
        elif employee_count <= 50:
            return "11-50"
        elif employee_count <= 200:
            return "51-200"
        elif employee_count <= 500:
            return "201-500"
        elif employee_count <= 1000:
            return "501-1000"
        elif employee_count <= 5000:
            return "1001-5000"
        else:
            return "5000+"

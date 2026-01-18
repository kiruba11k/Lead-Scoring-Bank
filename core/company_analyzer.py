"""
Company Metrics Analyzer
Analyzes company information from LinkedIn data
"""

import re
from typing import Dict, List, Optional

class CompanyMetricsAnalyzer:
    """Analyzes company metrics from LinkedIn profile data."""
    
    def __init__(self):
        self.industry_keywords = {
            "FinTech": ["fintech", "financial technology", "digital banking", 
                       "payments", "lending", "insurtech", "wealthtech"],
            "Commercial Banking": ["commercial bank", "corporate banking", 
                                 "business banking", "commercial lending"],
            "Retail Banking": ["retail bank", "consumer banking", 
                             "personal banking", "branch banking"],
            "Investment Banking": ["investment bank", "m&a", "mergers", 
                                 "acquisitions", "capital markets", "ipo"],
            "Insurance": ["insurance", "actuarial", "underwriting", 
                        "claims", "reinsurance"],
            "Asset Management": ["asset management", "wealth management", 
                               "portfolio management", "investment management"],
            "Technology": ["software", "tech", "saas", "platform", 
                         "developer", "engineer", "data", "ai", "cloud"],
            "Consulting": ["consulting", "consultant", "advisory", 
                         "professional services"],
            "Healthcare": ["healthcare", "medical", "pharma", 
                         "biotech", "health tech"]
        }
        
        self.company_size_patterns = {
            "startup": ["startup", "seed", "series a", "early stage"],
            "small": ["small business", "sme", "local", "family owned"],
            "medium": ["growth stage", "scale-up", "expanding"],
            "large": ["enterprise", "global", "multinational", "fortune"]
        }
    
    def analyze_company(self, company_name: str, profile_data: Dict) -> Dict:
        """
        Analyze company metrics based on available information.
        
        Args:
            company_name: Name of the company
            profile_data: Complete LinkedIn profile data
            
        Returns:
            Dictionary with company analysis
        """
        if not company_name:
            return self._get_default_analysis()
        
        company_name_lower = company_name.lower()
        
        # Determine industry
        industry = self._determine_industry(company_name_lower, profile_data)
        
        # Estimate company size
        estimated_size = self._estimate_company_size(company_name_lower, profile_data)
        
        # Estimate revenue based on size
        estimated_revenue = self._estimate_revenue(estimated_size, industry)
        
        # Calculate reputation score
        reputation_score = self._calculate_reputation_score(company_name_lower, profile_data)
        
        return {
            "company_name": company_name,
            "industry": industry,
            "estimated_size": estimated_size,
            "estimated_revenue": estimated_revenue,
            "reputation_score": reputation_score,
            "confidence": self._calculate_confidence(profile_data),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _determine_industry(self, company_name: str, profile_data: Dict) -> str:
        """Determine the primary industry of the company."""
        
        # Check company name for industry keywords
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in company_name for keyword in keywords):
                return industry
        
        # Check profile headline and about
        basic_info = profile_data.get('basic_info', {})
        headline = basic_info.get('headline', '').lower()
        about = basic_info.get('about', '').lower()
        
        search_text = f"{headline} {about}"
        
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in search_text for keyword in keywords):
                return industry
        
        # Check experience descriptions
        for exp in profile_data.get('experience', []):
            description = exp.get('description', '').lower()
            for industry, keywords in self.industry_keywords.items():
                if any(keyword in description for keyword in keywords):
                    return industry
        
        return "Technology"  # Default for tech profiles
    
    def _estimate_company_size(self, company_name: str, profile_data: Dict) -> str:
        """Estimate company size based on available information."""
        
        # Check for known large companies
        large_companies = ["accenture", "deloitte", "pwc", "ey", "kpmg", 
                          "ibm", "microsoft", "google", "amazon", "apple",
                          "jpmorgan", "bank of america", "wells fargo", 
                          "citigroup", "goldman sachs"]
        
        if any(company in company_name for company in large_companies):
            return "5000+"
        
        # Analyze job titles in experience
        experiences = profile_data.get('experience', [])
        
        if experiences:
            # Get all job titles
            titles = [exp.get('title', '').lower() for exp in experiences]
            
            # Check for executive titles
            executive_titles = ["ceo", "cfo", "cto", "president", "vp", 
                              "vice president", "director", "head of"]
            
            if any(title in ' '.join(titles) for title in executive_titles):
                return "201-500"
            
            # Check for startup indicators
            startup_indicators = ["founder", "co-founder", "startup", 
                                "first employee", "early team"]
            
            if any(indicator in ' '.join(titles) for indicator in startup_indicators):
                return "11-50"
        
        # Default based on industry
        industry = self._determine_industry(company_name, profile_data)
        
        if industry in ["FinTech", "Technology"]:
            return "51-200"  # Common for tech companies
        elif industry in ["Commercial Banking", "Investment Banking"]:
            return "5000+"   # Banks are typically large
        else:
            return "51-200"  # Default
    
    def _estimate_revenue(self, company_size: str, industry: str) -> str:
        """Estimate annual revenue based on company size and industry."""
        
        revenue_map = {
            "1-10": "$1M-$5M",
            "11-50": "$5M-$20M",
            "51-200": "$20M-$100M",
            "201-500": "$100M-$500M",
            "501-1000": "$500M-$1B",
            "1001-5000": "$1B-$5B",
            "5000+": "$5B+"
        }
        
        base_revenue = revenue_map.get(company_size, "$10M-$50M")
        
        # Adjust based on industry
        if industry in ["Investment Banking", "Commercial Banking"]:
            # Financial institutions have higher revenue per employee
            if company_size == "51-200":
                return "$100M-$500M"
            elif company_size == "201-500":
                return "$500M-$1B"
        
        return base_revenue
    
    def _calculate_reputation_score(self, company_name: str, profile_data: Dict) -> int:
        """Calculate company reputation score (1-10)."""
        score = 5  # Default
        
        # Known prestigious companies
        prestigious = ["accenture", "mckinsey", "boston consulting", "bain",
                      "goldman sachs", "morgan stanley", "jpmorgan",
                      "google", "microsoft", "apple", "amazon", "meta"]
        
        # Well-known startups
        known_startups = ["stripe", "square", "plaid", "robinhood", 
                         "coinbase", "airbnb", "uber", "doordash"]
        
        if any(company in company_name for company in prestigious):
            score = 9
        elif any(company in company_name for company in known_startups):
            score = 7
        
        # Adjust based on employee count in profile
        basic_info = profile_data.get('basic_info', {})
        if basic_info.get('follower_count', 0) > 10000:
            score = min(10, score + 1)
        
        return score
    
    def _calculate_confidence(self, profile_data: Dict) -> str:
        """Calculate confidence level of analysis."""
        experiences = profile_data.get('experience', [])
        
        if not experiences:
            return "low"
        
        # Check if we have detailed company information
        has_company_details = False
        for exp in experiences:
            if exp.get('company') and exp.get('duration'):
                has_company_details = True
                break
        
        if has_company_details and len(experiences) > 2:
            return "high"
        elif has_company_details:
            return "medium"
        else:
            return "low"
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when no company information is available."""
        return {
            "company_name": "Unknown",
            "industry": "Technology",
            "estimated_size": "51-200",
            "estimated_revenue": "$10M-$50M",
            "reputation_score": 5,
            "confidence": "low",
            "analysis_timestamp": datetime.now().isoformat()
        }

"""
Minimal ML-based classifier for fallback classification.
Uses a simple keyword-based scoring as a stand-in for a real ML model.
"""
from typing import Dict, Any

def ml_classify(text: str) -> Dict[str, Any]:
    """
    Minimal ML-style classifier: returns a domain and confidence based on keyword density.
    """
    text_lc = text.lower()
    # Example: simple keyword density for 'sourcing' vs 'procurement'
    sourcing_keywords = ["rfp", "auction", "bid", "supplier"]
    procurement_keywords = ["purchase order", "po", "requisition", "buyer"]
    sourcing_score = sum(text_lc.count(kw) for kw in sourcing_keywords)
    procurement_score = sum(text_lc.count(kw) for kw in procurement_keywords)
    if sourcing_score > procurement_score:
        return {"domains": [("Sourcing", 1.0)], "confidence": 1.0}
    elif procurement_score > sourcing_score:
        return {"domains": [("Procurement", 1.0)], "confidence": 1.0}
    else:
        return {"domains": [("General", 0.5)], "confidence": 0.5}

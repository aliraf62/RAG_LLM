"""Classification system for identifying question types and subject domains."""
from __future__ import annotations

import re
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

from core.config.settings import settings
from core.utils.patterns import keyword_pattern
from core.services.customer_service import customer_service

logger = logging.getLogger(__name__)

# Default prompt types that are generic and not customer-specific
PROMPT_TYPE_PATTERNS = {
    "How to": [(r"how do i", 1.0), (r"how to", 1.0), (r"steps to", 0.8), (r"guide", 0.6)],
    "Show me": [(r"show me", 1.0), (r"display", 0.8), (r"can i see", 0.9)],
    "Question": [(r"what is", 1.0), (r"why", 0.9), (r"explain", 0.8), (r"where", 0.7), (r"when", 0.7)],
    "Troubleshooting": [(r"error", 1.0), (r"issue", 0.9), (r"problem", 0.9), (r"fix", 0.8), (r"resolve", 0.8)],
    "General": [(r"info", 0.5), (r"information", 0.5)],
}

def get_domains_for_customer(customer_id: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Get domain configuration for the specified customer or active customer.

    Args:
        customer_id: Optional customer ID to specify which customer's domains to load

    Returns:
        Dictionary mapping domain names to their configuration
    """
    # Use active customer if none specified
    if customer_id is None:
        customer_id = customer_service.active_customer

    # Try to get domains from customer settings
    domains = settings.get("SUBJECT_DOMAINS", {})

    # If empty, use a generic fallback domain set
    if not domains:
        logger.warning(f"No SUBJECT_DOMAINS found for customer {customer_id}, using generic fallback domains")
        domains = {
            "General": {"keywords": ["information", "help", "support"]},
            # Add more generic domains if needed
        }

    return domains


def classify(text: str) -> Dict[str, Any]:
    """
    Complete classification of a query text.
    Returns classification results including domains, prompt types, and confidence scores.
    """
    result = classify_question(text)
    return result


def classify_question(text: str) -> Dict[str, Any]:
    """
    Classify a question into domains and prompt types with confidence scores.
    """
    domains = classify_domains(text)
    prompt_type = classify_prompt_type(text)
    return {
        "domains": domains,
        "prompt_type": prompt_type["prompt_type"],
        "prompt_type_confidence": prompt_type["confidence"],
        "prompt_type_alternatives": prompt_type["alternatives"],
    }


def classify_domains(text: str) -> List[Tuple[str, float]]:
    text_lc = text.lower()
    scores = []

    # Get domains for the current customer
    domains = get_domains_for_customer()

    for domain, config in domains.items():
        domain_score = 0.0
        keywords = config.get("keywords", [])

        for keyword_entry in keywords:
            # Handle different formats of keywords (simple string or [str, weight] tuple)
            if isinstance(keyword_entry, list) and len(keyword_entry) == 2:
                kw, weight = keyword_entry
            else:
                kw = str(keyword_entry)
                weight = 1.0

            if not isinstance(kw, str):
                logger.warning(f"Invalid keyword format found in domain {domain}: {kw}")
                continue

            if re.search(rf"\\b{re.escape(kw.lower())}\\b", text_lc):
                domain_score += weight
            elif kw.lower() in text_lc:
                domain_score += 0.5 * weight

        if domain_score > 0:
            scores.append((domain, domain_score))

    if scores:
        max_score = max(s for _, s in scores)
        scores = [(d, s / max_score) for d, s in scores]
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        scores = [("General", 0.0)]

    return scores


def classify_prompt_type(text: str) -> Dict[str, Any]:
    text_lc = text.lower()
    scores = {}
    for prompt_type, patterns in PROMPT_TYPE_PATTERNS.items():
        type_score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text_lc):
                type_score += weight
        if type_score > 0:
            scores[prompt_type] = type_score
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "prompt_type": sorted_types[0][0] if sorted_types else "General",
        "confidence": sorted_types[0][1] if sorted_types else 0.0,
        "alternatives": sorted_types[1:] if len(sorted_types) > 1 else []
    }


def semantic_classify(text: str, embed_fn=None) -> Dict[str, Any]:
    results = classify_question(text)
    if results["domains"] and results["domains"][0][1] < settings.get("MIN_DOMAIN_CONFIDENCE", 0.3):
        if embed_fn is not None:
            domain_descs = settings.get("DOMAIN_DESCRIPTIONS", {})
            if domain_descs:
                try:
                    text_emb = embed_fn([text])[0]
                    domain_embs = embed_fn([v for v in domain_descs.values()])
                    from numpy import dot
                    from numpy.linalg import norm
                    sims = [dot(text_emb, d_emb) / (norm(text_emb) * norm(d_emb)) for d_emb in domain_embs]
                    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
                    best_domain = list(domain_descs.keys())[best_idx]
                    results["domains"] = [(best_domain, float(sims[best_idx]))]
                except Exception as e:
                    logger.warning(f"Semantic classification failed: {e}")
    return results


def hybrid_classify(text: str) -> Dict[str, Any]:
    results = classify_question(text)
    if results["domains"] and results["domains"][0][1] < settings.get("MIN_DOMAIN_CONFIDENCE", 0.3):
        try:
            from core.classification.ml_classifier import ml_classify
            ml_results = ml_classify(text)
            if ml_results["confidence"] > results["domains"][0][1]:
                return ml_results
        except ImportError:
            pass
    return results


def log_classification(query: str, classification: Dict[str, Any], correct: bool = True) -> None:
    try:
        from core.utils.logging import log_event
        log_event("classification", {
            "query": query,
            "classification": classification,
            "correct": correct,
            "timestamp": time.time()
        })
    except ImportError:
        logger.debug("Classification logging skipped - logging utility not available")


__all__ = [
    "classify",
    "classify_question",
    "classify_domains",
    "classify_prompt_type",
    "semantic_classify",
    "hybrid_classify",
    "log_classification",
]

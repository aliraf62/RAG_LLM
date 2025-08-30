"""
Advanced document classification and tagging system.

This module provides utilities for:
1. Extracting relevant tags from document content using state-of-the-art keyphrase extraction
2. Classifying document purpose (how-to, explanatory, reference)
3. Categorizing documents into domains and subdomains
4. Enriching document metadata with classification results

The system is configurable through customer-specific settings and uses multiple
NLP techniques including KeyBERT, PKE, and custom classification algorithms.
"""
from langchain_community.vectorstores import faiss
import logging
import os
import re
import ssl
import threading
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from functools import lru_cache

# Lazy imports for heavy dependencies
_keybert_module = None
_pke_module = None
_nltk_downloaded = threading.Event()

# Configure logger
logger = logging.getLogger(__name__)

# DEFAULT_MAX_KEYPHRASES = 10 # To be moved to settings

def _ensure_nltk_resources():
    """Download required NLTK resources if needed."""
    if _nltk_downloaded.is_set():
        return

    try:
        import nltk
        for resource in ['stopwords', 'punkt', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
        _nltk_downloaded.set()
    except ImportError:
        logger.warning("NLTK not available. Some keyphrase extraction methods will be limited.")


def _ensure_keybert():
    """Import KeyBERT on demand to avoid loading it when not used."""
    global _keybert_module
    if _keybert_module is None:
        try:
            import keybert # Import the package
            _keybert_module = keybert # Assign the module to our global
        except ImportError:
            logger.warning("KeyBERT not installed. Using fallback extraction methods.")
            _keybert_module = False # Mark as failed
            return False
        except Exception as e:
            logger.warning(f"Error during KeyBERT initialization: {e}")
            _keybert_module = False # Mark as failed
            return False
    return _keybert_module is not False


def _ensure_pke():
    """Import PKE on demand to avoid loading it when not used."""
    global _pke_module
    if _pke_module is None:
        try:
            import pke # Import the base pke module
            _pke_module = pke # Assign to global for state checking
            _ensure_nltk_resources() # NLTK is a PKE dependency
        except ImportError:
            logger.warning("PKE not installed or base module import failed. Using fallback extraction methods.")
            _pke_module = False # Mark as failed
            return False
        except Exception as e:
            logger.warning(f"Error during PKE initialization: {e}")
            _pke_module = False # Mark as failed
            return False
    return _pke_module is not False


class KeyphraseExtractor:
    """Extract keyphrases from text using multiple methods."""

    def __init__(self,
                 extraction_method: str = "keybert",
                 max_keyphrases: int = 10, # Replaced DEFAULT_MAX_KEYPHRASES
                 stopwords: Optional[List[str]] = None,
                 offline_mode: bool = True): # Added offline_mode
        """
        Initialize keyphrase extractor.

        Args:
            extraction_method: Method to use ('keybert', 'pke', 'tfidf', 'hybrid')
            max_keyphrases: Maximum number of keyphrases to extract
            stopwords: Custom stopwords list
            offline_mode: Whether to attempt offline operation for models like KeyBERT
        """
        self.extraction_method = extraction_method.lower()
        self.max_keyphrases = max_keyphrases
        self.offline_mode = offline_mode # Store offline_mode

        # Initialize stopwords with provided list or empty set
        self.stopwords = set(stopwords or [])

        # Enhance with NLTK and scikit-learn stopwords
        try:
            import nltk
            from nltk.corpus import stopwords as nltk_stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            self.stopwords.update(nltk_stopwords.words('english'))
            logger.debug("Added NLTK stopwords")
        except (ImportError, LookupError) as e:
            logger.warning(f"Could not load NLTK stopwords: {e}. Using basic stopwords.")

        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            self.stopwords.update(ENGLISH_STOP_WORDS)
            logger.debug("Added scikit-learn stopwords")
        except ImportError as e:
            logger.warning(f"Could not load scikit-learn stopwords: {e}")

        if not self.stopwords:
            self.stopwords.update([
                "the", "and", "or", "a", "an", "is", "are", "was", "were", "this", "that",
                "with", "from", "your", "have", "has", "had", "will", "they", "their", "them",
                "can", "may", "should", "would", "could", "for", "you", "not", "but", "be",
                "been", "being", "which", "where", "when", "how", "what", "who", "whom", "to"
            ])

    def extract_keyphrases(self, text: str, title: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Extract keyphrases using the configured method.

        Args:
            text: Document text
            title: Optional document title

        Returns:
            List of (keyphrase, score) tuples
        """
        if not text or len(text) < 10:
            return []

        # Combine title and text for better context
        full_text = text
        if title:
            full_text = f"{title}\n\n{text}"

        # Choose extraction method
        if self.extraction_method == "keybert" and _ensure_keybert():
            return self._extract_with_keybert(full_text)
        elif self.extraction_method == "pke" and _ensure_pke():
            return self._extract_with_pke(full_text)
        elif self.extraction_method == "hybrid":
            return self._extract_hybrid(full_text, title)
        else:
            # Default to simple TF-IDF-like approach
            return self._extract_with_tfidf(full_text)

    def _extract_with_keybert(self, text: str) -> List[Tuple[str, float]]:
        """Extract keyphrases using KeyBERT."""
        if not _ensure_keybert():
            return self._extract_with_tfidf(text)

        original_env = {}
        env_vars_to_set = {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_CERT_CHECK": "1" # Crucial for self-signed certs / offline
        }
        original_ssl_context_factory = None # Initialize here

        try:
            from keybert import KeyBERT

            if self.offline_mode:
                logger.info("Attempting KeyBERT in offline mode.")
                for var, value in env_vars_to_set.items():
                    original_env[var] = os.environ.get(var)
                    os.environ[var] = value
                
                original_ssl_context_factory = ssl._create_default_https_context
                unverified_ssl_context = ssl._create_unverified_context()
                ssl._create_default_https_context = lambda: unverified_ssl_context
            
            kw_model = KeyBERT(model="all-MiniLM-L6-v2")

            keyphrases = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words=list(self.stopwords) if self.stopwords else [], # Changed None to []
                use_maxsum=True,
                nr_candidates=20,
                top_n=self.max_keyphrases
            )
            return keyphrases if keyphrases is not None else [] # type: ignore

        except Exception as e:
            logger.warning(f"Error extracting keyphrases with KeyBERT: {e}. Falling back to TF-IDF.")
            return self._extract_with_tfidf(text)
        finally:
            if self.offline_mode:
                for var, value in original_env.items():
                    if value is None:
                        del os.environ[var]
                    else:
                        os.environ[var] = value
                if original_ssl_context_factory is not None: # Check if it was set
                    ssl._create_default_https_context = original_ssl_context_factory

    def _extract_with_pke(self, text: str) -> List[Tuple[str, float]]:
        """Extract keyphrases using PKE."""
        if not _ensure_pke():
            return self._extract_with_tfidf(text)
        try:
            from pke.unsupervised import MultipartiteRank

            extractor = MultipartiteRank()
            # PKE documentation states normalization=None is valid. Type hint might be too strict.
            extractor.load_document(input=text, language='en', normalization=None) # type: ignore

            extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
            extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
            keyphrases = extractor.get_n_best(n=self.max_keyphrases)
            return keyphrases
        except Exception as e:
            logger.error(f"Error extracting keyphrases with PKE: {e}", exc_info=True)
            return self._extract_with_tfidf(text) # Fall back to simpler method

    def _extract_with_tfidf(self, text: str) -> List[Tuple[str, float]]:
        """Extract keyphrases using a simple TF-IDF-like approach."""
        # Extract phrases using regex for multi-word phrases
        text = text.lower()
        # Extract single words and n-grams (up to trigrams)
        words = re.findall(r'\b[a-z]\w+\b', text)
        bigrams = list(zip(words, words[1:]))
        bigrams = [f"{w1} {w2}" for w1, w2 in bigrams]

        trigrams = list(zip(words, words[1:], words[2:]))
        trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in trigrams]

        # Combine and filter
        phrases = [w for w in words if len(w) > 3 and w not in self.stopwords]
        phrases += [p for p in bigrams if not any(w in self.stopwords for w in p.split())]
        phrases += [p for p in trigrams if not any(w in self.stopwords for w in p.split())]

        # Count occurrences
        phrase_counts = Counter(phrases)

        # Score by frequency and length (longer phrases get boost)
        scored_phrases = []
        for phrase, count in phrase_counts.items():
            # Basic score is count * length factor
            length_factor = 1.0 + (0.2 * len(phrase.split()))  # Boost multi-word phrases
            score = count * length_factor
            scored_phrases.append((phrase, score))

        # Sort by score and return top N
        return sorted(scored_phrases, key=lambda x: x[1], reverse=True)[:self.max_keyphrases]

    def _extract_hybrid(self, text: str, title: Optional[str] = None) -> List[Tuple[str, float]]:
        """Extract keyphrases using multiple methods and combine results."""
        results = []

        # Try KeyBERT first
        if _ensure_keybert():
            results.extend(self._extract_with_keybert(text))

        # Try PKE if available
        if _ensure_pke() and len(results) < self.max_keyphrases:
            pke_results = self._extract_with_pke(text)
            # Add only new phrases
            existing_phrases = {phrase for phrase, _ in results}
            for phrase, score in pke_results:
                if phrase not in existing_phrases and len(results) < self.max_keyphrases:
                    results.append((phrase, score))
                    existing_phrases.add(phrase)

        # Add TF-IDF results if needed
        if len(results) < self.max_keyphrases:
            tfidf_results = self._extract_with_tfidf(text)
            # Add only new phrases
            existing_phrases = {phrase for phrase, _ in results}
            for phrase, score in tfidf_results:
                if phrase not in existing_phrases and len(results) < self.max_keyphrases:
                    results.append((phrase, score))

        # Normalize scores between 0 and 1
        if results:
            max_score = max(score for _, score in results)
            if max_score > 0:
                results = [(phrase, score/max_score) for phrase, score in results]

        return sorted(results, key=lambda x: x[1], reverse=True)[:self.max_keyphrases]


class DocumentClassifier:
    """Advanced document classifier using multiple techniques."""

    def __init__(self,
                 domain_keywords: Optional[Dict[str, List[Union[str, Tuple[str, float]]]]] = None,
                 subdomain_keywords: Optional[Dict[str, List[Union[str, Tuple[str, float]]]]] = None,
                 category_keywords: Optional[Dict[str, List[Union[str, Tuple[str, float]]]]] = None, # Added
                 purpose_keywords: Optional[Dict[str, List[Union[str, Tuple[str, float]]]]] = None,
                 title_boost: float = 2.0, # Default if not from settings
                 match_weight: float = 1.0, # Added, default if not from settings
                 keyphrase_extraction_method: str = "hybrid",
                 max_keyphrases: int = 10, # Default if not from settings
                 stopwords: Optional[List[str]] = None,
                 customer_id: Optional[str] = None, # Added
                 offline_mode: bool = True): # Added
        """
        Initialize document classifier with configurable keyword sets.
        Args:
            domain_keywords: Dictionary mapping domain names to lists of keywords or (keyword, weight) tuples
            subdomain_keywords: Dictionary mapping subdomain names to lists of keywords or (keyword, weight) tuples
            category_keywords: Dictionary mapping category names to lists of keywords or (keyword, weight) tuples
            purpose_keywords: Dictionary mapping content purpose types to lists of keywords or (keyword, weight) tuples
            title_boost: Multiplier for keyword matches found in the title
            match_weight: Default weight for keywords if not specified
            keyphrase_extraction_method: Method for keyphrase extraction ("keybert", "pke", "tfidf", "hybrid")
            max_keyphrases: Maximum number of keyphrases to extract
            stopwords: Custom stopwords list
            customer_id: The ID of the customer for customer-specific settings
            offline_mode: Whether to run model-based extractors in offline mode
        """
        self.match_weight = match_weight # Store match_weight
        self.customer_id = customer_id # Store customer_id

        # Convert any plain keyword lists to (keyword, weight) tuple lists
        self.domain_keywords = self._normalize_keyword_weights(domain_keywords or {})
        self.subdomain_keywords = self._normalize_keyword_weights(subdomain_keywords or {})
        self.category_keywords = self._normalize_keyword_weights(category_keywords or {}) # Use passed category_keywords
        self.purpose_keywords = self._normalize_keyword_weights(purpose_keywords or {})
        self.title_boost = title_boost

        # Initialize keyphrase extractor
        self.extractor = KeyphraseExtractor(
            extraction_method=keyphrase_extraction_method,
            max_keyphrases=max_keyphrases,
            stopwords=stopwords,
            offline_mode=offline_mode # Pass offline_mode
        )

        # Cache for expensive operations
        self.category_cache = {}
        self.keyphrase_cache = {}

    # Removed @staticmethod
    def _normalize_keyword_weights(self, keyword_dict: Dict[str, List[Union[str, Tuple[str, float]]]]) -> Dict[str, List[Tuple[str, float]]]:
        """Convert any string keywords to (keyword, weight) tuples with default weight."""
        normalized = {}
        for category, keywords in keyword_dict.items():
            normalized_keywords = []
            for kw in keywords:
                if isinstance(kw, tuple) and len(kw) == 2 and isinstance(kw[0], str) and isinstance(kw[1], (int, float)):
                    normalized_keywords.append((kw[0], float(kw[1])))
                elif isinstance(kw, list) and len(kw) == 2 and isinstance(kw[0], str) and isinstance(kw[1], (int, float)):
                    normalized_keywords.append((kw[0], float(kw[1])))
                elif isinstance(kw, str):
                    normalized_keywords.append((kw, self.match_weight)) # Use self.match_weight
                else:
                    logger.warning(f"Skipping invalid keyword item: {kw} in category {category}")
            normalized[category] = normalized_keywords
        return normalized

    def classify_domains(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        Classify document into relevant domains based on content.

        Args:
            text: Document text content
            title: Document title (optional)

        Returns:
            List of domain names sorted by relevance score
        """
        # Return from cache if available
        cache_key = f"domains:{hash(text)}"
        if cache_key in self.category_cache:
            return self.category_cache[cache_key]

        # Analyze text and compute scores
        scores = self._compute_category_scores(self.domain_keywords, text, title)

        # Return domains sorted by score (highest first), filtering out zero scores
        result = [domain for domain, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0]

        # Cache and return
        self.category_cache[cache_key] = result
        return result

    def classify_subdomains(self, text: str, title: Optional[str] = None, domains: Optional[List[str]] = None) -> List[str]:
        """
        Classify document into relevant subdomains based on content.

        Args:
            text: Document text content
            title: Document title (optional)
            domains: List of parent domains to filter subdomains (optional)

        Returns:
            List of subdomain names sorted by relevance score
        """
        # Generate cache key
        domains_str = ",".join(sorted(domains)) if domains else "all"
        cache_key = f"subdomains:{domains_str}:{hash(text)}"
        if cache_key in self.category_cache:
            return self.category_cache[cache_key]

        # Compute subdomain scores
        scores = self._compute_category_scores(self.subdomain_keywords, text, title)

        # Filter by parent domains if provided
        if domains:
            # Filter subdomains based on parent domain prefix (e.g., "cso_optimization" belongs to "cso" domain)
            filtered_scores = {}
            for subdomain, score in scores.items():
                for domain in domains:
                    if subdomain.startswith(f"{domain}_") or subdomain == domain:
                        filtered_scores[subdomain] = score
                        break
            scores = filtered_scores

        # Return subdomains sorted by score (highest first), filtering out zero scores
        result = [subdomain for subdomain, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0]

        # Cache and return
        self.category_cache[cache_key] = result
        return result

    def classify_purpose(self, text: str, title: Optional[str] = None) -> str:
        """
        Classify the content purpose (how-to, explanatory, reference, etc).

        Args:
            text: Document text content
            title: Document title (optional)

        Returns:
            Content purpose category with highest score, or "unknown" if no match
        """
        # Return from cache if available
        cache_key = f"purpose:{hash(text)}"
        if cache_key in self.category_cache:
            return self.category_cache[cache_key]

        # Handle empty input
        if not text:
            return "unknown"

        # Compute scores for each purpose category
        scores = self._compute_category_scores(self.purpose_keywords, text, title)

        # Check for special patterns that strongly indicate a how-to document
        if re.search(r'step \d+', text.lower()) or re.search(r'\d+\.\s', text.lower()):
            scores["how_to"] = scores.get("how_to", 0) + 3

        # Find purpose with highest score
        if scores:
            best_purpose = max(scores.items(), key=lambda x: x[1])[0]
            if scores[best_purpose] > 0:
                # Cache and return
                self.category_cache[cache_key] = best_purpose
                return best_purpose

        return "unknown"

    def classify_categories(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        Classify document into product categories based on content and customer SUBJECT_DOMAINS.
        Uses self.category_keywords which should be populated via __init__.
        Args:
            text: Document text content
            title: Document title (optional)

        Returns:
            List of category names sorted by relevance score
        """
        cache_key = f"categories:{hash(text)}"
        if cache_key in self.category_cache:
            return self.category_cache[cache_key]

        # self.category_keywords should be already normalized and set in __init__
        # If it's empty, scores will be empty.
        scores = self._compute_category_scores(self.category_keywords, text, title)

        # Get threshold from settings (assuming settings are globally accessible or passed differently)
        # For now, let's use a fixed or passed threshold if available.
        # The original code fetched from core.config.settings.settings
        # This part might need adjustment based on how settings are managed.
        # For now, let's assume a default threshold or that it's handled by the caller.
        # For simplicity, we'll just sort by score.
        # threshold = settings.get("PRODUCT_PROBABILITY_THRESHOLD", 0.7) # This needs settings access

        # Simplified logic: return all categories with score > 0, sorted.
        # More advanced thresholding/normalization can be added if needed.
        result = [cat for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0]
        
        self.category_cache[cache_key] = result
        return result

    def extract_tags(self, text: str, title: Optional[str] = None, include_weights: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Extract relevant tags from document text using keyphrase extraction.

        Args:
            text: Document text content
            title: Document title (optional)
            include_weights: Whether to include score weights with tags

        Returns:
            List of extracted tags or (tag, score) tuples if include_weights=True
        """
        # Return from cache if available
        cache_key = f"tags:{hash(text)}"
        if cache_key in self.keyphrase_cache:
            keyphrases = self.keyphrase_cache[cache_key]
            return keyphrases if include_weights else [k for k, _ in keyphrases]

        if not text:
            return []

        # Extract keyphrases
        keyphrases = self.extractor.extract_keyphrases(text, title)

        # Start with domain and subdomain classification as high-priority tags
        domains = self.classify_domains(text, title)
        subdomains = self.classify_subdomains(text, title, domains)

        # Create a tag dictionary with domains and subdomains as high-priority tags
        tag_dict = {}

        # Always include domains with high confidence
        for domain in domains[:3]:  # Top 3 domains
            tag_dict[domain] = 1.0

        # Include relevant subdomains
        for subdomain in subdomains[:3]:  # Top 3 subdomains
            # Don't add if it's too similar to an existing tag
            if not any(self._is_redundant(subdomain, existing) for existing in tag_dict.keys()):
                tag_dict[subdomain] = 0.9

        # Add important product terms that appear in text or title with high priority
        text_lower = text.lower()
        title_lower = title.lower() if title else ""

        # Get product terms from settings only - no hardcoded fallbacks
        from core.config.settings import settings
        product_terms = {}

        # Only try to get from customer-specific settings if customer_id is available
        if hasattr(self, 'customer_id') and self.customer_id:
            customer_settings = settings.get("customers", {}).get(self.customer_id, {})
            classification_settings = customer_settings.get("classification", {})
            product_terms = classification_settings.get("product_terms", {})

        # Check each product term and add if found
        for term, weight in product_terms.items():
            if term in text_lower or (title and term in title_lower):
                # Don't add if already redundant with existing tag
                if not any(self._is_redundant(term, existing) for existing in tag_dict.keys()):
                    tag_dict[term] = weight

        # Add extracted keyphrases - with deduplication
        for phrase, score in keyphrases:
            # Skip if this phrase contains only stopwords
            if all(word in self.extractor.stopwords for word in phrase.split()):
                continue

            # Skip if too similar to an existing tag
            if any(self._is_redundant(phrase, existing) for existing in tag_dict.keys()):
                continue

            tag_dict[phrase] = score

        # Convert to sorted list
        result = sorted([(tag, score) for tag, score in tag_dict.items()],
                       key=lambda x: x[1], reverse=True)

        # Cache result
        self.keyphrase_cache[cache_key] = result

        # Return with or without weights
        return result if include_weights else [tag for tag, _ in result]

    def _is_redundant(self, new_phrase: str, existing_phrase: str) -> bool:
        """
        Check if a new phrase is redundant compared to an existing phrase.

        Args:
            new_phrase: The new phrase to check
            existing_phrase: An existing phrase to compare against

        Returns:
            True if the new phrase is redundant (contained in or containing the existing phrase)
        """
        # Exact match
        if new_phrase == existing_phrase:
            return True

        # One is a substring of the other
        if new_phrase in existing_phrase or existing_phrase in new_phrase:
            return True

        # Check for high word overlap in multi-word phrases
        new_words = set(new_phrase.split())
        existing_words = set(existing_phrase.split())

        # If both have multiple words
        if len(new_words) > 1 and len(existing_words) > 1:
            # Calculate overlap ratio
            overlap = len(new_words.intersection(existing_words))
            smaller_set_size = min(len(new_words), len(existing_words))

            # If more than 70% of the words in the smaller phrase are in the larger phrase
            if smaller_set_size > 0 and overlap / smaller_set_size > 0.7:
                return True

        return False

    def _compute_category_scores(self,
                                category_keywords: Dict[str, List[Tuple[str, float]]],
                                text: str,
                                title: Optional[str] = None) -> Dict[str, float]:
        """
        Compute scores for each category based on keyword matches.

        Args:
            category_keywords: Dictionary of categories with weighted keywords
            text: Document text content
            title: Document title (optional)

        Returns:
            Dictionary mapping categories to their computed scores
        """
        logger = logging.getLogger(__name__)

        if not text or not category_keywords:
            logger.warning(f"Empty text or empty category_keywords. Text length: {len(text) if text else 0}, Categories: {len(category_keywords)}")
            return {}

        scores = defaultdict(float)
        text_l = text.lower()
        title_l = title.lower() if title else ""

        logger.info(f"Computing scores for {len(category_keywords)} categories")
        logger.info(f"Text length: {len(text)}, Title: '{title_l}'")

        # Log the first few words of text for context
        text_preview = text_l[:100] + "..." if len(text_l) > 100 else text_l
        logger.info(f"Text preview: '{text_preview}'")

        for category, KWs_tuples_list in category_keywords.items():
            category_total_score_for_this_category = 0.0 # Initialize score for this specific category
            logger.info(f"Processing category '{category}' with {len(KWs_tuples_list)} keywords")

            matches_found = []

            for keyword_phrase, weight in KWs_tuples_list:
                keyword_phrase_l = keyword_phrase.lower()
                
                # Debug: Check if the keyword is in the text/title
                in_text = keyword_phrase_l in text_l
                in_title = bool(title_l and keyword_phrase_l in title_l)

                # Log matches
                if in_title or in_text:
                    match_type = "title" if in_title else "text"
                    matches_found.append((keyword_phrase_l, match_type, weight))

                # If a keyword is found in the title, its contribution is boosted.
                # If found only in text, it's the base weight.
                # If in both, title's boosted score takes precedence for that keyword's contribution.
                if in_title:
                    boost_score = weight * self.title_boost
                    category_total_score_for_this_category += boost_score
                    logger.debug(f"  - Found '{keyword_phrase_l}' in title, adding {boost_score} (weight {weight} * boost {self.title_boost})")
                elif in_text: # Only add text score if not already counted by a title match
                    category_total_score_for_this_category += weight
                    logger.debug(f"  - Found '{keyword_phrase_l}' in text, adding {weight}")

            # Log all matches for this category
            if matches_found:
                logger.info(f"Category '{category}' matches: {matches_found}")
                logger.info(f"Total score for '{category}': {category_total_score_for_this_category}")

            if category_total_score_for_this_category > 0:
                scores[category] = category_total_score_for_this_category
        
        # Log final scores for all categories
        if scores:
            logger.info(f"Final scores for all categories: {dict(scores)}")
        else:
            logger.warning("No category matches found in document.")

        return dict(scores)

    def enrich_metadata(self, metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Enrich document metadata with classification results.
        Ensures all relevant classification keys are present in the output.
        Args:
            metadata: Original document metadata
            text: Document text content
        Returns:
            Enriched metadata dictionary
        """
        enriched = metadata.copy()
        title = enriched.get("title") or enriched.get("Name")

        # Initialize keys to ensure they are always present
        enriched.setdefault("domains", [])
        enriched.setdefault("categories", [])
        enriched.setdefault("subdomains", [])
        enriched.setdefault("tags", [])
        enriched.setdefault("content_purpose", "unknown")

        # Perform classifications
        # Note: self.classify_domains etc. will return empty lists if no classification,
        # so enriched fields will be updated to empty lists if they were non-empty and no new classification found.
        # This behavior might need adjustment if pre-existing values should be preserved on no-classification.
        # Current logic: overwrite with new classification (or empty if none).

        classified_domains = self.classify_domains(text, title)
        enriched["domains"] = classified_domains # Will be [] if no domains found

        classified_categories = self.classify_categories(text, title)
        enriched["categories"] = classified_categories # Will be [] if no categories found
        
        # Use the classified_domains (which might be empty) for subdomain classification
        classified_subdomains = self.classify_subdomains(text, title, domains=classified_domains)
        enriched["subdomains"] = classified_subdomains # Will be [] if no subdomains found

        extracted_tags = self.extract_tags(text, title)
        enriched["tags"] = extracted_tags # Will be [] if no tags found

        classified_purpose = self.classify_purpose(text, title) # Returns "unknown" if no specific purpose
        enriched["content_purpose"] = classified_purpose # Always update with what classifier found

        return enriched


@lru_cache(maxsize=8)
def get_document_classifier(customer_id: Optional[str] = None) -> DocumentClassifier:
    """
    Factory function to get a document classifier configured for the specified customer.
    Loads settings from global defaults and overrides with customer-specific configurations.
    Args:
        customer_id: Customer identifier to load customer-specific classification settings
    Returns:
        Configured DocumentClassifier instance
    """
    from core.config.settings import settings
    logger = logging.getLogger(__name__)

    # Verbose logging for diagnostics
    logger.info(f"Creating document classifier for customer_id: {customer_id}")

    # Load global default settings from settings object (sourced from defaults.py)
    default_match_weight = settings.get("CLASSIFICATION_MATCH_WEIGHT", 1.0)
    default_title_boost = settings.get("CLASSIFICATION_TITLE_BOOST", 2.0)
    default_max_keyphrases = settings.get("CLASSIFICATION_MAX_KEYPHRASES", 10)
    default_extraction_method = settings.get("CLASSIFICATION_KEYPHRASE_EXTRACTION_METHOD", "hybrid")
    # This is the crucial effective_offline_mode determination
    effective_offline_mode = settings.get("CLASSIFICATION_OFFLINE_MODE", True) 

    logger.info(f"Default settings: match_weight={default_match_weight}, title_boost={default_title_boost}, offline_mode={effective_offline_mode}")

    # Initialize parameters with global defaults
    domain_keywords = {}
    subdomain_keywords = {}
    purpose_keywords = {}
    category_keywords = {}
    product_terms = {}

    # Load customer-specific settings if available
    if customer_id:
        # Get customer settings from YAML configuration
        logger.info(f"Loading settings for customer: {customer_id}")
        customer_settings = settings.get(f"customers.{customer_id}", {})
        logger.debug(f"Customer settings keys: {list(customer_settings.keys())}")

        # Check if classification section exists
        logger.info(f"Checking for classification settings")
        classification_settings = customer_settings.get("classification", {})
        if classification_settings:
            logger.info(f"Found classification settings with keys: {list(classification_settings.keys())}")
            domain_keywords = classification_settings.get("domain_keywords", {})
            logger.info(f"Domain keywords loaded: {list(domain_keywords.keys())}")
            subdomain_keywords = classification_settings.get("subdomain_keywords", {})
            logger.info(f"Subdomain keywords loaded: {list(subdomain_keywords.keys())}")
            purpose_keywords = classification_settings.get("purpose_keywords", {})
            product_terms = classification_settings.get("product_terms", {})
        else:
            logger.warning(f"No classification settings found for {customer_id}")

        # Get domain keywords from SUBJECT_DOMAINS if available and no domain_keywords were found
        if not domain_keywords and "SUBJECT_DOMAINS" in customer_settings:
            logger.info(f"No domain_keywords found, checking SUBJECT_DOMAINS")
            subject_domains = customer_settings.get("SUBJECT_DOMAINS", {})
            logger.info(f"SUBJECT_DOMAINS keys: {list(subject_domains.keys())}")
            domain_keywords = {}
            for domain, domain_data in subject_domains.items():
                if "keywords" in domain_data:
                    domain_keywords[domain] = domain_data["keywords"]
                    # Use domain name as a category if no categories defined
                    if domain not in category_keywords:
                        category_keywords[domain] = domain_data["keywords"]
            logger.info(f"Domain keywords extracted from SUBJECT_DOMAINS: {list(domain_keywords.keys())}")

        # Other settings
        title_boost = classification_settings.get("title_boost", default_title_boost)
        match_weight = classification_settings.get("match_weight", default_match_weight)
        keyphrase_extraction_method = classification_settings.get("keyphrase_extraction_method", default_extraction_method)
        max_keyphrases = classification_settings.get("max_keyphrases", default_max_keyphrases)
        stopwords = classification_settings.get("stopwords")
        offline_mode = classification_settings.get("offline_mode", effective_offline_mode)

        logger.info(f"Customer-specific settings: title_boost={title_boost}, match_weight={match_weight}, offline_mode={offline_mode}")
    else:
        # Fallback to default values if no customer_id provided
        logger.info(f"No customer_id provided, using default settings")
        title_boost = default_title_boost
        match_weight = default_match_weight
        keyphrase_extraction_method = default_extraction_method
        max_keyphrases = default_max_keyphrases
        stopwords = None
        offline_mode = effective_offline_mode

    # Log keyword configuration status before creating classifier
    logger.info(f"Final configuration summary:")
    logger.info(f"  - Domain keywords count: {len(domain_keywords)} domains")
    logger.info(f"  - Subdomain keywords count: {len(subdomain_keywords)} subdomains")
    logger.info(f"  - Purpose keywords count: {len(purpose_keywords)} purposes")
    logger.info(f"  - Category keywords count: {len(category_keywords)} categories")
    logger.info(f"  - Product terms count: {len(product_terms)} terms")

    # Create the classifier with gathered settings
    classifier = DocumentClassifier(
        domain_keywords=domain_keywords,
        subdomain_keywords=subdomain_keywords,
        purpose_keywords=purpose_keywords,
        category_keywords=category_keywords,
        title_boost=title_boost,
        match_weight=match_weight,
        keyphrase_extraction_method=keyphrase_extraction_method,
        max_keyphrases=max_keyphrases,
        stopwords=stopwords,
        customer_id=customer_id,
        offline_mode=offline_mode
    )

    return classifier

def enrich_document_metadata(metadata: Dict[str, Any], text: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to enrich document metadata using the appropriate classifier.

    Args:
        metadata: Document metadata to enrich
        text: Document text content
        customer_id: Customer identifier (optional)

    Returns:
        Enriched metadata dictionary
    """
    from core.config.settings import settings

    # Only process if SMART_CLASSIFY is enabled
    if not settings.get("SMART_CLASSIFY", False):
        return metadata

    classifier = get_document_classifier(customer_id)
    return classifier.enrich_metadata(metadata, text)

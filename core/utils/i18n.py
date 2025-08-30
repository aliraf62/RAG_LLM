"""
core.utils.i18n
==============

Internationalization utilities for message handling across the application.

Provides a flexible, extensible system for managing translations with:
- Support for multiple languages
- Dynamic message formatting
- Fallback mechanisms
- Lazy loading of translation files
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.config.paths import project_path
from core.config.settings import settings

logger = logging.getLogger(__name__)

# Default language if not specified
DEFAULT_LANGUAGE = "en"

class I18nManager:
    """
    Manages internationalization (i18n) and localization for the application.

    This class handles loading messages from multiple sources:
    1. Built-in messages dictionary
    2. JSON translation files in the i18n directory
    3. Customer-specific translations
    """

    def __init__(self):
        self._messages: Dict[str, Dict[str, str]] = {}
        self._current_language = self._determine_language()
        self._supported_languages = settings.get("SUPPORTED_LANGUAGES", ["en"])
        self._loaded_sources: List[str] = []

        # Initialize with built-in messages
        self._init_built_in_messages()

    def _determine_language(self) -> str:
        """Determine the current language based on settings or environment."""
        # First try settings
        lang = settings.get("DEFAULT_LANGUAGE")
        if lang:
            return lang

        # Then try environment variables
        env_lang = os.environ.get("LANG", "").split(".")[0].split("_")[0].lower()
        if env_lang and len(env_lang) == 2:  # Basic validation for language code
            return env_lang

        # Fall back to English
        return DEFAULT_LANGUAGE

    def _init_built_in_messages(self) -> None:
        """Initialize with built-in messages."""
        # These are the core messages that should always be available
        self._messages = {
            # General messages
            "command.success": {
                "en": "✅ {component_type} '{component_name}' executed successfully!",
                "sv": "✅ {component_type} '{component_name}' kördes framgångsrikt!"
            },
            "command.failure": {
                "en": "❌ Error: {error}",
                "sv": "❌ Fel: {error}"
            },
            "file.written": {
                "en": "Output written to {output_path}",
                "sv": "Utdata skrevs till {output_path}"
            },
            "ping.response": {
                "en": "pong! The system is responding correctly.",
                "sv": "pong! Systemet svarar korrekt."
            },
            "command.unknown": {
                "en": "Unknown command: {command}",
                "sv": "Okänt kommando: {command}"
            },

            # Component processing messages
            "component.chunker.processed": {
                "en": "✅ Chunker '{name}' processed {input_file} into {chunk_count} chunks",
                "sv": "✅ Chunker '{name}' bearbetade {input_file} till {chunk_count} delar"
            },
            "component.cleaner.processed": {
                "en": "✅ Cleaner '{name}' processed {input_file}",
                "sv": "✅ Rengörare '{name}' bearbetade {input_file}"
            },
            "component.exporter.processed": {
                "en": "✅ Exporter '{name}' exported data to {output_dir}",
                "sv": "✅ Exportör '{name}' exporterade data till {output_dir}"
            },

            # Error messages
            "error.missing_input_file": {
                "en": "Missing required input file path",
                "sv": "Saknar nödvändig inmatningsfilväg"
            },
            "error.missing_output_path": {
                "en": "Missing required output path",
                "sv": "Saknar nödvändig utmatningsväg"
            },
            "error.unknown_component_type": {
                "en": "Unknown component type: {component_type}",
                "sv": "Okänd komponenttyp: {component_type}"
            },
            "error.unknown_component": {
                "en": "Unknown {component_type}: {component_name}",
                "sv": "Okänd {component_type}: {component_name}"
            },
            "error.message_not_found": {
                "en": "Message not found: {key}",
                "sv": "Meddelande hittades inte: {key}"
            },
            "error.language_not_supported": {
                "en": "Language not supported: {language}",
                "sv": "Språk stöds inte: {language}"
            },

            # Help messages for components
            "help.component": {
                "en": "Run data ingestion components",
                "sv": "Kör datainmatningskomponenter"
            },
            "help.chunker": {
                "en": "Document chunking components",
                "sv": "Dokumentuppdelningskomponenter"
            },
            "help.cleaner": {
                "en": "Document cleaning components",
                "sv": "Dokumentrengöringskomponenter"
            },
            "help.exporter": {
                "en": "Data exporting components",
                "sv": "Dataexportkomponenter"
            },
            "help.chunker_run": {
                "en": "Run a document chunker",
                "sv": "Kör en dokumentdelare"
            },
            "help.cleaner_run": {
                "en": "Run a document cleaner",
                "sv": "Kör en dokumentrengörare"
            },
            "help.exporter_run": {
                "en": "Run a data exporter",
                "sv": "Kör en dataexportör"
            },

            # --- Added from legacy messages.py for full i18n coverage ---
            "help.cli": {
                "en": "Coupa AI Assistant command palette",
                "sv": "Coupa AI-assistent kommandopalett"
            },
            "help.log_level": {
                "en": "Python logging level (DEBUG, INFO, WARNING, ERROR)",
                "sv": "Python loggningsnivå (DEBUG, INFO, WARNING, ERROR)"
            },
            "param.command_payload": {
                "en": "Command payload",
                "sv": "Kommandolast"
            },
            "export_guides.help": {
                "en": "Export CSO workflow guides to HTML format",
                "sv": "Exportera CSO-arbetsflödesguider till HTML-format"
            },
            "export_guides.param.exporter_type": {
                "en": "Name of exporter to use (cso_html, parquet_html, etc.)",
                "sv": "Namn på exportör att använda (cso_html, parquet_html, etc.)"
            },
            "export_guides.param.workbook": {
                "en": "Path to Workflow Steps workbook",
                "sv": "Sökväg till arbetsflödesstegsarbetsboken"
            },
            "export_guides.param.assets": {
                "en": "Directory with unpacked assets",
                "sv": "Katalog med uppackade tillgångar"
            },
            "export_guides.param.output": {
                "en": "Destination folder",
                "sv": "Destinationsmapp"
            },
            "export_guides.param.limit": {
                "en": "Limit number of guides (0 = all)",
                "sv": "Begränsa antalet guider (0 = alla)"
            },
            "export_guides.param.no_captions": {
                "en": "Disable image captions",
                "sv": "Inaktivera bildtexter"
            },
            "param.force": {
                "en": "Force regeneration of existing files",
                "sv": "Tvinga återgenerering av befintliga filer"
            },
            "chat.welcome": {
                "en": "Welcome to Coupa AI Assistant. Type '/help' for commands or '/exit' to quit.",
                "sv": "Välkommen till Coupa AI-assistent. Skriv '/help' för kommandon eller '/exit' för att avsluta."
            },
            "param.chunker_name": {
                "en": "Chunker name",
                "sv": "Chunkernamn"
            },
            "param.input_file": {
                "en": "Input file path",
                "sv": "Inmatningsfilsökväg"
            },
            "param.output_file": {
                "en": "Output file path",
                "sv": "Utmatningsfilsökväg"
            },
            "param.chunk_size": {
                "en": "Chunk size (default: {default_chunk_size})",
                "sv": "Chunkstorlek (standard: {default_chunk_size})"
            },
            "param.chunk_overlap": {
                "en": "Chunk overlap (default: {default_chunk_overlap})",
                "sv": "Chunk överlappning (standard: {default_chunk_overlap})"
            },
            "param.cleaner_name": {
                "en": "Cleaner name",
                "sv": "Rengörarnamn"
            },
            "param.exporter_name": {
                "en": "Exporter name",
                "sv": "Exportörnamn"
            },
            "param.output_dir": {
                "en": "Output directory",
                "sv": "Utmatningskatalog"
            },
            "param.limit": {
                "en": "Limit number of items to process",
                "sv": "Begränsa antal objekt att bearbeta"
            },
            "dataset.cso.workbook_help": {
                "en": "Path to XLSB workbook (for CSO exporters)",
                "sv": "Sökväg till XLSB-arbetsbok (för CSO-exportörer)"
            },
            "dataset.cso.assets_help": {
                "en": "Path to assets directory (for CSO exporters)",
                "sv": "Sökväg till tillgångskatalog (för CSO-exportörer)"
            },
            "dataset.parquet.file_help": {
                "en": "Path to parquet file (for parquet exporters)",
                "sv": "Sökväg till parquet-fil (för parquet-exportörer)"
            },
            "help.query": {
                "en": "Query the knowledge base with RAG",
                "sv": "Fråga kunskapsbasen med RAG"
            },
            "help.query_question": {
                "en": "Question to ask the knowledge base",
                "sv": "Fråga att ställa till kunskapsbasen"
            },
            "help.query_raw": {
                "en": "Show raw retrieval results",
                "sv": "Visa råa hämtningsresultat"
            },
            "help.query_k": {
                "en": "Number of results to retrieve",
                "sv": "Antal resultat att hämta"
            },
            "file_size.help": {
                "en": "Analyze file sizes in a directory or show size of a single file",
                "sv": "Analysera filstorlekar i en katalog eller visa storlek på en enskild fil"
            },
            "file_size.param.path": {
                "en": "Path to file or directory to analyze",
                "sv": "Sökväg till fil eller katalog att analysera"
            },
            "file_size.param.sort_by": {
                "en": "Sort by: size, name, extension",
                "sv": "Sortera efter: storlek, namn, filtyp"
            },
            "file_size.param.limit": {
                "en": "Limit number of files shown",
                "sv": "Begränsa antal filer som visas"
            },
            "build_index.help": {
                "en": "Build a FAISS index from exported guides",
                "sv": "Bygg ett FAISS-index från exporterade guider"
            },
            "build_index.param.html_dir": {
                "en": "Folder with exported HTML",
                "sv": "Mapp med exporterad HTML"
            },
            "build_index.param.out_dir": {
                "en": "Destination folder for FAISS index",
                "sv": "Destinationsmapp för FAISS-index"
            },
            "build_index.param.chunk": {
                "en": "Chunking: none | header",
                "sv": "Uppdelning: none | header"
            },
            "build_index.param.limit": {
                "en": "Only first N HTML files (0 = all)",
                "sv": "Endast första N HTML-filer (0 = alla)"
            },
            "build_index.param.batch": {
                "en": "Embedding batch size",
                "sv": "Inbäddningsbatchtorlek"
            },
            "answer.retrieval_error": {
                "en": "I'm sorry, I couldn't retrieve the information you requested. There may be an issue with the knowledge base.",
                "sv": "Jag är ledsen, jag kunde inte hämta den information du begärde. Det kan finnas ett problem med kunskapsbasen."
            },
        }
        self._loaded_sources.append("built-in")

    def load_translations_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load translations from a JSON file.

        Args:
            file_path: Path to the JSON translation file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Translation file not found: {path}")
                return False

            with open(path, 'r', encoding='utf-8') as f:
                translations = json.load(f)

            # Merge translations
            for key, langs in translations.items():
                if key not in self._messages:
                    self._messages[key] = {}

                for lang, text in langs.items():
                    self._messages[key][lang] = text

            self._loaded_sources.append(str(path))
            logger.info(f"Loaded translations from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
            return False

    def load_translations_for_customer(self, customer_id: str) -> bool:
        """
        Load customer-specific translations.

        Args:
            customer_id: The customer identifier

        Returns:
            True if loaded successfully, False otherwise
        """
        customer_i18n_path = project_path("customers", customer_id, "i18n")
        if not customer_i18n_path.exists():
            logger.debug(f"No i18n directory for customer: {customer_id}")
            return False

        success = False
        for lang_file in customer_i18n_path.glob("*.json"):
            if self.load_translations_from_file(lang_file):
                success = True

        return success

    def set_language(self, language: str) -> bool:
        """
        Set the current language for messages.

        Args:
            language: Language code (e.g., "en", "sv")

        Returns:
            True if language was set, False if not supported
        """
        if language in self._supported_languages:
            self._current_language = language
            logger.info(f"Language set to: {language}")
            return True
        else:
            logger.warning(f"Language not supported: {language}")
            return False

    def get_message(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get a translated message by key.

        Args:
            key: The message identifier
            language: Optional language override (defaults to current language)
            **kwargs: Format variables to insert into the message

        Returns:
            The translated and formatted message
        """
        lang = language or self._current_language

        # First try the requested language
        if key in self._messages and lang in self._messages[key]:
            message = self._messages[key][lang]

        # Fall back to English
        elif key in self._messages and DEFAULT_LANGUAGE in self._messages[key]:
            message = self._messages[key][DEFAULT_LANGUAGE]
            logger.debug(f"Falling back to {DEFAULT_LANGUAGE} for message: {key}")

        # Key not found
        else:
            fallback_key = "error.message_not_found"
            if fallback_key in self._messages and lang in self._messages[fallback_key]:
                message = self._messages[fallback_key][lang].format(key=key)
            else:
                message = f"Message not found: {key}"
            logger.warning(message)

        try:
            return message.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format parameter in message {key}: {e}")
            return message
        except Exception as e:
            logger.error(f"Error formatting message {key}: {e}")
            return message

    @property
    def current_language(self) -> str:
        """Get the current language code."""
        return self._current_language

    @property
    def supported_languages(self) -> List[str]:
        """Get the list of supported languages."""
        return list(self._supported_languages)

    def add_message(self, key: str, translations: Dict[str, str]) -> None:
        """
        Add a new message to the translation dictionary.

        Args:
            key: The message identifier
            translations: Dict mapping language codes to message strings
        """
        if key not in self._messages:
            self._messages[key] = {}

        for lang, text in translations.items():
            self._messages[key][lang] = text

        logger.debug(f"Added message: {key}")

    def load_all_translations(self) -> None:
        """Load all available translations from the i18n directory."""
        i18n_dir = project_path("i18n")
        if not i18n_dir.exists():
            logger.info("Creating i18n directory")
            i18n_dir.mkdir(exist_ok=True)

        for lang_file in i18n_dir.glob("*.json"):
            self.load_translations_from_file(lang_file)

# Singleton instance
i18n_manager = I18nManager()

# Convenience function for getting translated messages
def get_message(key: str, **kwargs) -> str:
    """
    Get a translated message by key with variable substitution.

    Args:
        key: The message identifier
        **kwargs: Variables to format into the message

    Returns:
        The translated and formatted message
    """
    return i18n_manager.get_message(key, **kwargs)

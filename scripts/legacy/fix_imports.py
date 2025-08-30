#!/usr/bin/env python3
"""
Fix import references across the project after core module reorganization.

This script:
1. Finds files that import from old module locations
2. Updates those imports to reference the new module structure
3. Reports all changes made
"""
import argparse
import os
import re
from pathlib import Path

# Old module paths and their new locations
IMPORT_MAP = {
    r"from core\.paths import": "from core.config.paths import",
    r"from core\.settings import": "from core.config.settings import",
    r"from core\.defaults import": "from core.config.defaults import",
    r"from core\.customer_service import": "from core.services.customer_service import",
    r"from core\.component_registry import": "from core.utils.component_registry import",
    r"from core\.exceptions import": "from core.utils.exceptions import",
    r"from core\.messages import": "from core.utils.i18n import",
    r"from core\.conversation import": "from core.rag.conversation import",
    r"from core\.context_formatter import": "from core.rag.context_formatter import",
    r"from core\.prompt_manager import": "from core.rag.prompt_manager import",
    r"from core\.classify import": "from core.rag.classify import",
    r"import core\.paths": "import core.config.paths",
    r"import core\.settings": "import core.config.settings",
    r"import core\.defaults": "import core.config.defaults",
    r"import core\.customer_service": "import core.services.customer_service",
    r"import core\.component_registry": "import core.utils.component_registry",
    r"import core\.exceptions": "import core.utils.exceptions",
    r"import core\.messages": "import core.utils.i18n",
    # Variable names that need to be updated
    r"\bcustomer_manager\b": "customer_service"
}

# Files to exclude from processing
EXCLUDE_PATHS = {
    "core/__init__.py",  # We manually updated this file
    "__pycache__",
    ".venv",
    ".git",
    "ai_qna_assistant.egg-info",
}

def should_process(file_path):
    """Check if we should process this file path."""
    return (
        file_path.endswith((".py")) and
        not any(exclude in str(file_path) for exclude in EXCLUDE_PATHS)
    )

def update_imports_in_file(file_path, dry_run=False):
    """Update imports in a single file based on the import map."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        original_content = content
        changes_made = 0

        # Apply all replacements
        for old_pattern, new_import in IMPORT_MAP.items():
            # Count occurrences before replacement
            matches = len(re.findall(old_pattern, content))
            if matches > 0:
                content = re.sub(old_pattern, new_import, content)
                changes_made += matches

        # Only write back if changes were made and not in dry-run mode
        if changes_made > 0:
            print(f"{file_path}: {changes_made} import references updated")

            if not dry_run:
                with open(file_path, 'w') as file:
                    file.write(content)

            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory, dry_run=False):
    """Recursively process all Python files in the directory."""
    total_files = 0
    updated_files = 0

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(exclude in d for exclude in EXCLUDE_PATHS)]

        for file in files:
            file_path = os.path.join(root, file)

            if should_process(file_path):
                total_files += 1
                if update_imports_in_file(file_path, dry_run):
                    updated_files += 1

    return total_files, updated_files

def main():
    parser = argparse.ArgumentParser(description="Fix import references after core module reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Only print changes, don't modify files")
    parser.add_argument("--path", default=".", help="Root directory to process (default: current directory)")
    args = parser.parse_args()

    target_dir = args.path
    print(f"Processing Python files in {target_dir}" + (" (DRY RUN)" if args.dry_run else ""))

    total_files, updated_files = process_directory(target_dir, args.dry_run)

    print(f"\n✅ Processed {total_files} Python files")
    print(f"✅ Updated {updated_files} files with new import references")

    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Script to reorganize the configuration modules in the project.

This script reorganizes the configuration-related files and directories to create
a more logical structure:

1. Moves openai client code from core/config.py to core/llm/openai.py
2. Consolidates all configuration handling into a single core/config directory
3. Updates imports and references to maintain compatibility
"""

import os
import shutil
import re
from pathlib import Path

# Define project root path
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "core"

# Files to modify or move
CONFIG_PY = CORE_DIR / "config.py"
CONFIG_DIR = CORE_DIR / "config"
CONFIGURATION_DIR = CORE_DIR / "configuration"

def backup_file(file_path):
    """Create a backup of a file before modifying it."""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")

def safe_read_file(file_path):
    """Safely read a file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # If all encodings fail, use bytes and decode with errors='replace'
    print(f"Warning: Could not determine encoding for {file_path}, using binary mode with replace")
    return file_path.read_bytes().decode('utf-8', errors='replace')

def update_imports(file_path, old_import, new_import):
    """Update import statements in a file."""
    if not file_path.exists():
        print(f"Warning: File {file_path} not found, skipping import updates")
        return

    backup_file(file_path)
    try:
        content = safe_read_file(file_path)
        updated_content = content.replace(old_import, new_import)

        if content != updated_content:
            file_path.write_text(updated_content, encoding='utf-8')
            print(f"Updated imports in {file_path}")
    except Exception as e:
        print(f"Error updating imports in {file_path}: {e}")

def find_files_with_import(directory, import_statement):
    """Find all Python files containing a specific import statement."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = Path(root) / filename
                try:
                    content = safe_read_file(file_path)
                    if import_statement in content:
                        files.append(file_path)
                except Exception as e:
                    print(f"Warning: Could not check imports in {file_path}: {e}")
    return files

def move_configuration_service():
    """Move configuration service to the config directory."""
    # Check if configuration service exists
    service_path = CONFIGURATION_DIR / "service.py"
    config_init = CONFIG_DIR / "__init__.py"

    if not service_path.exists():
        print(f"Configuration service not found at {service_path}")
        return

    # Create backup
    backup_file(service_path)
    backup_file(config_init)

    # Read service content
    service_content = safe_read_file(service_path)

    # Update imports inside service.py
    service_content = service_content.replace(
        "from core.config.settings import settings",
        "from core.config.settings import settings"
    )

    # Write to new location
    new_service_path = CONFIG_DIR / "service.py"
    new_service_path.write_text(service_content, encoding='utf-8')

    # Add service to config/__init__.py exports
    init_content = safe_read_file(config_init)

    if "from core.config.service import" not in init_content:
        # Find the __all__ list
        all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', init_content, re.DOTALL)
        if all_match:
            # Extract current __all__ content
            all_content = all_match.group(1)
            # Add ConfigurationService to __all__
            if "'ConfigurationService'" not in all_content and '"ConfigurationService"' not in all_content:
                new_all = all_content.rstrip() + ",\n    'ConfigurationService'\n"
                init_content = init_content.replace(all_match.group(1), new_all)

            # Add import statement
            if "from core.config.service import ConfigurationService" not in init_content:
                import_section_end = init_content.find("__all__")
                init_content = (
                    init_content[:import_section_end] +
                    "from core.config.service import ConfigurationService\n\n" +
                    init_content[import_section_end:]
                )

            # Write updated init file
            config_init.write_text(init_content, encoding='utf-8')
            print(f"Updated {config_init} to include ConfigurationService")

    print(f"Moved configuration service to {new_service_path}")

def update_configuration_imports():
    """Update imports for configuration services across the project."""
    files_to_update = find_files_with_import(PROJECT_ROOT, "from core.config.service import")

    for file_path in files_to_update:
        update_imports(
            file_path,
            "from core.config.service import",
            "from core.config.service import"
        )

    # Also update any imports using core.settings
    settings_files = find_files_with_import(PROJECT_ROOT, "from core.config.settings import")
    for file_path in settings_files:
        update_imports(
            file_path,
            "from core.config.settings import",
            "from core.config.settings import"
        )

def fix_openai_client_imports():
    """Fix the imports after moving get_client from config.py to openai.py"""
    files_to_update = find_files_with_import(PROJECT_ROOT, "from core.llm import get_llm_client")

    for file_path in files_to_update:
        update_imports(
            file_path,
            "from core.llm import get_llm_client",
            "from core.llm import get_llm_client"
        )
        # Also update any function calls
        update_imports(
            file_path,
            "client = get_llm_client()",
            "client = get_llm_client()"
        )

def update_test_excel_end_to_end():
    """Specifically fix the test_excel_end_to_end.py script that has import issues."""
    file_path = PROJECT_ROOT / "scripts" / "test_excel_end_to_end.py"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return

    backup_file(file_path)
    try:
        content = safe_read_file(file_path)

        # Fix the broken import line - look for the pattern more carefully
        content = re.sub(
            r'from core\.config\.py import\s+#.*',
            'from core.llm import get_llm_client  # Fixed import',
            content
        )

        # Also fix any client = get_llm_client() calls
        content = re.sub(
            r'client\s*=\s*get_client\(\)',
            'client = get_llm_client()',
            content
        )

        file_path.write_text(content, encoding='utf-8')
        print(f"Fixed imports in {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def main():
    """Main function to execute the reorganization."""
    print("Starting configuration reorganization...")

    # 1. Create backups of key files
    backup_file(CONFIG_PY)

    # 2. Move configuration service to config directory
    move_configuration_service()

    # 3. Update imports across the project
    update_configuration_imports()

    # 4. Fix OpenAI client imports
    fix_openai_client_imports()

    # 5. Fix the test_excel_end_to_end.py script
    update_test_excel_end_to_end()

    # 6. Check if we can remove the configuration directory (if empty)
    if CONFIGURATION_DIR.exists():
        items = list(CONFIGURATION_DIR.iterdir())
        if len(items) <= 1 and any(item.name == "__init__.py" for item in items):
            init_path = CONFIGURATION_DIR / "__init__.py"
            if init_path.exists():
                backup_file(init_path)

            print(f"Configuration directory is empty except for __init__.py, you can safely delete it:")
            print(f"rm -rf {CONFIGURATION_DIR}")
        else:
            print(f"Note: Configuration directory still contains files. Review and migrate them manually.")

    print("\nReorganization complete!")
    print("\nNext steps:")
    print("1. Test the application to ensure everything works properly")
    print("2. Remove the backup files (.bak) after confirming functionality")
    print("3. Update any documentation to reflect the new structure")

if __name__ == "__main__":
    main()

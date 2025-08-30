#!/usr/bin/env python3
"""
Generate project documentation and summary reports.

This script analyzes Python modules in the project and generates:
1. Markdown reference docs in docs/api/
2. Architecture overview files in docs/architecture/
3. Usage example stubs in docs/examples/
4. Guide stubs in docs/guides/
5. A component registry summary at docs/component_registry.json
"""
import os
import sys
import inspect
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Project root
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Docs subfolders
DOCS_DIR = PROJ_ROOT / "docs"
API_DIR = DOCS_DIR / "api"
ARCH_DIR = DOCS_DIR / "architecture"
EXAMPLES_DIR = DOCS_DIR / "examples"
GUIDES_DIR = DOCS_DIR / "guides"
REGISTRY_FILE = DOCS_DIR / "component_registry.json"

for d in (API_DIR, ARCH_DIR, EXAMPLES_DIR, GUIDES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def extract_module_docs(module_path: str) -> Dict[str, Any]:
    """
    Extract docstrings and public API structure from a Python module.

    Returns a dict with module doc, classes, functions.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        return {"module": module_path, "error": str(e)}

    data: Dict[str, Any] = {
        "name": module_path,
        "doc": inspect.getdoc(module) or "",
        "file": inspect.getfile(module),
        "classes": {},
        "functions": {}
    }

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_path or name.startswith("_"):
            continue
        cls_doc = inspect.getdoc(obj) or ""
        methods = {}
        for mname, mobj in inspect.getmembers(obj, inspect.isfunction):
            if mobj.__module__ != module_path or mname.startswith("_"):
                continue
            methods[mname] = {
                "signature": str(inspect.signature(mobj)),
                "doc": inspect.getdoc(mobj) or ""
            }
        data["classes"][name] = {"doc": cls_doc, "methods": methods}

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module_path or name.startswith("_"):
            continue
        data["functions"][name] = {
            "signature": str(inspect.signature(obj)),
            "doc": inspect.getdoc(obj) or ""
        }

    return data


def write_markdown(module_docs: Dict[str, Any], out_dir: Path) -> Path:
    """
    Write API reference markdown for given module.
    """
    name = module_docs["name"].replace('.', '_')
    out_file = out_dir / f"{name}.md"
    with out_file.open('w', encoding='utf-8') as f:
        f.write(f"# `{module_docs['name']}` API Reference\n\n")
        if module_docs.get("doc"):
            f.write(module_docs["doc"] + "\n\n")
        if module_docs.get("classes"):
            f.write("## Classes\n\n")
            for cls, info in module_docs["classes"].items():
                f.write(f"### `{cls}`\n{info['doc']}\n\n")
                if info.get("methods"):
                    f.write("#### Methods\n\n")
                    for m, minfo in info["methods"].items():
                        f.write(f"- `{m}{minfo['signature']}`: {minfo['doc']}\n")
                    f.write("\n")
        if module_docs.get("functions"):
            f.write("## Functions\n\n")
            for fn, finfo in module_docs["functions"].items():
                f.write(f"- `{fn}{finfo['signature']}`: {finfo['doc']}\n")
    return out_file


def write_architecture(module_docs: Dict[str, Any], out_dir: Path) -> Path:
    """
    Generate an architecture overview stub for this module.
    """
    name = module_docs["name"].replace('.', '_')
    out_file = out_dir / f"{name}.md"
    with out_file.open('w') as f:
        f.write(f"# Architecture overview: `{module_docs['name']}`\n\n")
        f.write("Describe the role of this module in the overall architecture, its main components and interactions.\n")
    return out_file


def write_example_stub(module_docs: Dict[str, Any], out_dir: Path) -> Path:
    """
    Generate a usage example stub for this module.
    """
    name = module_docs["name"].replace('.', '_')
    out_file = out_dir / f"{name}_example.md"
    with out_file.open('w') as f:
        f.write(f"# Example usage: `{module_docs['name']}`\n\n")
        f.write("```python\n# TODO: Add import and usage example for this module\n```\n")
    return out_file


def write_guide_stub(module_docs: Dict[str, Any], out_dir: Path) -> Path:
    """
    Create a guide stub if module introduces a new pattern or feature.
    """
    name = module_docs["name"].replace('.', '_')
    out_file = out_dir / f"{name}_guide.md"
    with out_file.open('w') as f:
        f.write(f"# Guide: `{module_docs['name']}`\n\n")
        f.write("Provide a step-by-step guide or tutorial for using this module.\n")
    return out_file


def discover_modules(root: Path, prefix: str) -> List[str]:
    """
    Recursively discover importable modules under a directory.
    """
    modules = []
    for item in root.rglob('*.py'):
        rel = item.relative_to(PROJ_ROOT)
        parts = rel.with_suffix('').parts
        if parts[-1].startswith('_'):
            continue
        if 'tests' in parts or 'docs' in parts or 'scripts' in parts:
            continue
        modules.append('.'.join(parts))
    return sorted(set(modules))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modules', nargs='+', help='Specific modules to document')
    args = parser.parse_args()

    if args.modules:
        mods = args.modules
    else:
        mods = discover_modules(PROJ_ROOT / 'core', 'core') + \
               discover_modules(PROJ_ROOT / 'pipeline', 'pipeline') + \
               discover_modules(PROJ_ROOT / 'cli', 'cli')

    registry = {}
    for mod in mods:
        docs = extract_module_docs(mod)
        registry[mod] = docs
        try:
            write_markdown(docs, API_DIR)
            write_architecture(docs, ARCH_DIR)
            write_example_stub(docs, EXAMPLES_DIR)
            write_guide_stub(docs, GUIDES_DIR)
        except Exception as e:
            print(f"Failed writing docs for {mod}: {e}")

    # Dump registry
    with REGISTRY_FILE.open('w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)

    print("Docs generation complete.")


if __name__ == '__main__':
    main()

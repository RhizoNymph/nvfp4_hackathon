#!/usr/bin/env python3
"""
Fetch CUTLASS docs from crawl API and organize into a directory tree.
"""

import json
import os
import re
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path

# CRAWL_ID can be set via environment variable or defaults to the last known ID
CRAWL_ID = os.environ.get("CUTLASS_CRAWL_ID", "019b39b3-cc0a-765a-88b2-2fd70d2ce7af")
BASE_URL = f"http://127.0.0.1:3002/v2/crawl/{CRAWL_ID}"
OUTPUT_DIR = Path("/home/nymph/Code/ai/nvfp4-hackathon/docs/cutlass")

# Strip this prefix from URLs to get relative paths
URL_PREFIX = "/cutlass/media/docs/pythonDSL/"


def fetch_all_pages():
    """Fetch all pages of crawl data."""
    all_data = []
    url = BASE_URL
    page = 1
    
    while url:
        print(f"Fetching page {page}...")
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            all_data.extend(data["data"])
            print(f"  Got {len(data['data'])} items (total: {len(all_data)})")
        
        # Get next page URL
        url = data.get("next")
        page += 1
    
    return all_data


def url_to_filepath(url: str) -> Path:
    """Convert a URL to a local file path."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Strip the prefix
    if path.startswith(URL_PREFIX):
        path = path[len(URL_PREFIX):]
    
    # Handle root URL
    if not path or path == "/":
        path = "index.md"
    else:
        # Convert .html to .md
        if path.endswith(".html"):
            path = path[:-5] + ".md"
        elif not path.endswith(".md"):
            # If no extension, add index.md
            path = path.rstrip("/") + "/index.md" if path else "index.md"
    
    # Clean up path
    path = path.lstrip("/")
    
    return OUTPUT_DIR / path


def sanitize_markdown(content: str) -> str:
    """Clean up markdown content."""
    if not content:
        return ""
    
    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    return content.strip()


def save_document(item: dict) -> bool:
    """Save a single document to the appropriate file."""
    metadata = item.get("metadata", {})
    markdown = item.get("markdown", "")
    
    url = metadata.get("url", "")
    title = metadata.get("title", "")
    
    if not url:
        return False
    
    filepath = url_to_filepath(url)
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata header
    header = f"# {title}\n\n"
    header += f"> Source: {url}\n\n"
    if metadata.get("description"):
        header += f"> {metadata['description']}\n\n"
    header += "---\n\n"
    
    content = header + sanitize_markdown(markdown)
    
    # Write file
    filepath.write_text(content, encoding="utf-8")
    return True


def create_index(docs_info: list):
    """Create a main index file listing all documents."""
    # Group by directory
    by_dir = {}
    for info in docs_info:
        path = info["path"]
        dir_path = path.parent
        if dir_path not in by_dir:
            by_dir[dir_path] = []
        by_dir[dir_path].append(info)
    
    # Create main index
    index_content = "# CUTLASS Python DSL Documentation\n\n"
    index_content += "This directory contains the CUTLASS Python DSL documentation.\n\n"
    index_content += "## Contents\n\n"
    
    for dir_path in sorted(by_dir.keys()):
        rel_path = dir_path.relative_to(OUTPUT_DIR) if dir_path != OUTPUT_DIR else Path(".")
        if str(rel_path) == ".":
            continue
        index_content += f"- [{rel_path}]({rel_path}/)\n"
    
    index_path = OUTPUT_DIR / "_index.md"
    index_path.write_text(index_content, encoding="utf-8")
    print(f"\nCreated index at {index_path}")


def main():
    print(f"Fetching CUTLASS docs from crawl {CRAWL_ID}...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fetch all data
    all_data = fetch_all_pages()
    print(f"\nTotal documents fetched: {len(all_data)}")
    
    # Save each document
    docs_info = []
    saved = 0
    errors = 0
    
    for item in all_data:
        try:
            if save_document(item):
                filepath = url_to_filepath(item.get("metadata", {}).get("url", ""))
                docs_info.append({
                    "path": filepath,
                    "title": item.get("metadata", {}).get("title", ""),
                    "url": item.get("metadata", {}).get("url", "")
                })
                saved += 1
        except Exception as e:
            print(f"Error saving document: {e}")
            errors += 1
    
    print(f"\nSaved {saved} documents, {errors} errors")
    
    # Create index
    create_index(docs_info)
    
    # Show directory structure
    print("\nDirectory structure:")
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir():
            file_count = len(list(subdir.glob("*.md")))
            print(f"  {subdir.name}/ ({file_count} files)")
        else:
            print(f"  {subdir.name}")


if __name__ == "__main__":
    main()


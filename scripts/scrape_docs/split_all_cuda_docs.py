#!/usr/bin/env python3
"""
Split all CUDA documentation files into sections.
Processes all index.md files in docs/cuda subdirectories.
"""

import subprocess
import sys
from pathlib import Path


def find_index_files(base_dir):
    """Find all index.md files in subdirectories that haven't been split yet."""
    base_path = Path(base_dir)
    index_files = []
    
    for index_file in base_path.rglob("index.md"):
        # Skip PDF directory files (they're just metadata)
        if "pdf" in str(index_file.parts):
            continue
        
        # Skip root index.md
        if index_file.parent == base_path:
            continue
        
        # Skip if it's already been processed (check if section files exist)
        try:
            index_dir = index_file.parent
            # Check if there are already numbered section files (e.g., 1-*.md or 1.md)
            existing_sections = list(index_dir.glob("[0-9]-*.md")) + list(index_dir.glob("[0-9].md"))
            if existing_sections:
                continue  # Skip silently if already split
            
            # Also check if index has section links (already processed)
            content = index_file.read_text(encoding='utf-8')
            if "## Contents" in content and "This document is split into sections" in content:
                continue  # Skip silently if already processed
            
            # Include all index.md files - let split_sections.py decide if they can be split
            index_files.append(index_file)
        except Exception as e:
            print(f"Warning: Error checking {index_file}: {e}")
            continue
    
    return index_files


def split_file(index_file, script_path):
    """Run split_sections.py on a single index.md file."""
    index_dir = index_file.parent
    print(f"\n{'='*70}")
    print(f"Processing: {index_file}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(index_file), str(index_dir)],
            capture_output=True,
            text=True,
            check=False  # Don't fail on non-zero exit
        )
        output = result.stdout.strip()
        if output:
            print(output)
        
        # "No sections found" is not an error - just means file doesn't have sections to split
        if "No sections found" in output:
            return True  # Successfully processed (just no sections to split)
        
        if result.returncode != 0:
            print(f"Error processing {index_file}:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
        
        return True
    except Exception as e:
        print(f"Unexpected error processing {index_file}: {e}")
        return False


def main():
    base_dir = Path(__file__).parent.parent / "docs" / "cuda"
    script_path = Path(__file__).parent / "split_sections.py"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        sys.exit(1)
    
    print(f"Finding index.md files in {base_dir}...")
    index_files = find_index_files(base_dir)
    
    if not index_files:
        print("No index.md files found that need splitting.")
        return
    
    print(f"\nFound {len(index_files)} file(s) to process:")
    for f in index_files:
        print(f"  - {f}")
    
    print("\nStarting processing...")
    success_count = 0
    fail_count = 0
    
    for index_file in index_files:
        if split_file(index_file, script_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


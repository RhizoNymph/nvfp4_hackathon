#!/usr/bin/env python3
"""
Split all CUTLASS documentation files into sections.
Processes all .md files in docs/cutlass subdirectories.
"""

import subprocess
import sys
from pathlib import Path


def find_markdown_files(base_dir):
    """Find all markdown files in subdirectories that haven't been split yet."""
    base_path = Path(base_dir)
    md_files = []
    
    for md_file in base_path.rglob("*.md"):
        # Skip index files
        if md_file.name == "index.md" or md_file.name == "_index.md":
            continue
        
        # Skip files that are already in section subdirectories (they're already split sections)
        # Check if parent's parent has a section subdirectory with this file
        md_dir = md_file.parent
        parent_dir = md_dir.parent if md_dir != base_path else None
        
        # If this file is in a subdirectory that looks like a section directory, skip it
        # Section directories are typically named after the original file
        if parent_dir and parent_dir != base_path:
            # Check if parent has a section subdirectory
            potential_section_dir = parent_dir / md_dir.name
            if potential_section_dir.exists() and md_dir == potential_section_dir:
                # This file is already in a section directory, skip
                continue
        
        # Skip if this file's directory has a section subdirectory (already split)
        md_dir = md_file.parent
        section_subdir = md_dir / md_file.stem
        if section_subdir.exists() and section_subdir.is_dir():
            # Check if it has section files
            section_files = list(section_subdir.glob("*.md"))
            if section_files:
                continue  # Already split
        
        # Skip if there's an index.md in a section subdirectory
        index_in_section = section_subdir / "index.md"
        if index_in_section.exists():
            content = index_in_section.read_text(encoding='utf-8')
            if "## Contents" in content and "This document is split into sections" in content:
                continue  # Already processed
        
        # Only process files in the base directory or direct subdirectories (not section subdirs)
        # Files should be at: docs/cutlass/*.md or docs/cutlass/*/*.md (not docs/cutlass/*/sections/*.md)
        rel_path = md_file.relative_to(base_path)
        path_parts = rel_path.parts
        
        # Skip if file is more than 2 levels deep (already in a section directory)
        # Format: base_dir/file.md (1 level) or base_dir/subdir/file.md (2 levels) is OK
        # But base_dir/subdir/section/file.md (3 levels) should be skipped
        if len(path_parts) > 2:
            continue
        
        # Also skip if the file is in a directory that looks like a section directory
        # Section directories typically contain an index.md with "This document is split"
        md_dir = md_file.parent
        if md_dir != base_path:
            # Check if this directory has an index.md indicating it's a section directory
            index_file = md_dir / "index.md"
            if index_file.exists():
                try:
                    content = index_file.read_text(encoding='utf-8')
                    if "This document is split into sections" in content:
                        continue  # This is a section directory, skip files in it
                except:
                    pass
        
        # Include the file
        md_files.append(md_file)
    
    return md_files


def split_file(md_file, script_path):
    """Run split_cutlass_sections.py on a single markdown file."""
    md_dir = md_file.parent
    # Create a subdirectory for sections
    sections_dir = md_dir / md_file.stem
    
    print(f"\n{'='*70}")
    print(f"Processing: {md_file}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(md_file), str(sections_dir)],
            capture_output=True,
            text=True,
            check=False
        )
        output = result.stdout.strip()
        if output:
            print(output)
        
        # "No sections found" is not an error - just means file doesn't have sections to split
        if "No sections found" in output:
            return True  # Successfully processed (just no sections to split)
        
        if result.returncode != 0:
            print(f"Error processing {md_file}:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
        
        return True
    except Exception as e:
        print(f"Unexpected error processing {md_file}: {e}")
        return False


def main():
    base_dir = Path(__file__).parent.parent / "docs" / "cutlass"
    script_path = Path(__file__).parent / "split_cutlass_sections.py"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        sys.exit(1)
    
    print(f"Finding markdown files in {base_dir}...")
    md_files = find_markdown_files(base_dir)
    
    if not md_files:
        print("No markdown files found that need splitting.")
        return
    
    print(f"\nFound {len(md_files)} file(s) to process:")
    for f in md_files:
        print(f"  - {f}")
    
    print("\nStarting processing...")
    success_count = 0
    fail_count = 0
    
    for md_file in md_files:
        if split_file(md_file, script_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


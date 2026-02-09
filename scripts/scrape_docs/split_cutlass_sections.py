#!/usr/bin/env python3
"""
Split CUTLASS markdown documentation files by section headers.
Sections are identified by markdown headers (Title[#](url)) followed by equals signs.
"""

import re
import sys
from pathlib import Path


def find_section_boundaries(content_lines):
    """Find line numbers where top-level sections start."""
    boundaries = []
    
    # Pattern to match markdown section headers: "Title[#](url)" followed by equals signs
    # Format: "Overview[#](url)" or "Why CUTLASS DSLs?[#](url)"
    # Followed by blank/anchor lines, then equals signs
    
    for i in range(len(content_lines) - 4):
        line_raw = content_lines[i]
        line = line_raw.strip()
        
        # Check if line matches section header pattern: text followed by [#](url)
        # Pattern: any text, then [#](url)
        section_header_pattern = re.compile(r'^(.+?)\[#\]\([^)]+\)')
        match = section_header_pattern.match(line)
        
        if not match:
            continue
        
        # Extract title
        title = match.group(1).strip()
        
        # Skip if it's just navigation or metadata (common patterns)
        if any(skip in title.lower() for skip in ['skip to', 'back to', 'light dark', 'nvidia cutlass']):
            continue
        
        # Check lines ahead for equals signs (up to 4 lines ahead)
        found_equals = False
        for offset in range(1, 5):
            if i + offset >= len(content_lines):
                break
            check_line_raw = content_lines[i + offset]
            check_line = check_line_raw.strip()
            
            # Skip blank lines, single backslash lines, and anchor lines
            if not check_line or check_line == '\\' or (check_line.startswith('[') and ']' in check_line and not check_line.startswith('[#')):
                continue
            
            # Remove trailing backslash if present
            if check_line.endswith('\\'):
                check_line = check_line[:-1].strip()
            
            # Check if this line is all equals signs
            if check_line and all(c == '=' for c in check_line):
                found_equals = True
                break
        
        if found_equals:
            boundaries.append((i, title))
    
    return boundaries


def extract_section_title(line):
    """Extract the title from a section header line."""
    # Line format: "Title[#](url)"
    title_match = re.match(r'^(.+?)\[#\]', line.strip())
    if title_match:
        return title_match.group(1).strip()
    return None


def sanitize_filename(title):
    """Convert a title to a filename-safe string."""
    if not title:
        return ""
    # Convert to lowercase, replace spaces and special chars with hyphens
    filename = re.sub(r'[^\w\s-]', '', title.lower())
    filename = re.sub(r'[-\s]+', '-', filename)
    filename = filename.strip('-')
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def split_file(input_file, output_dir):
    """Split a markdown file into separate files by top-level sections."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the file
    content = input_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Find section boundaries
    boundaries = find_section_boundaries(lines)
    
    if not boundaries:
        print(f"No sections found in {input_file}")
        return
    
    # Only split if there are 2+ sections (single section files don't need splitting)
    if len(boundaries) < 2:
        print(f"Only {len(boundaries)} section found in {input_file}, skipping split")
        return
    
    print(f"Found {len(boundaries)} sections in {input_file}")
    
    # Extract header (everything before first section)
    header_end = boundaries[0][0] if boundaries else len(lines)
    header_lines = lines[:header_end]
    
    # Process each section
    section_files = []
    for i, (start_line, title) in enumerate(boundaries):
        # Determine end line (start of next section or end of file)
        end_line = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        
        # Extract section content
        section_lines = lines[start_line:end_line]
        
        # Create filename with title
        title_slug = sanitize_filename(title)
        if title_slug:
            filename = f"{title_slug}.md"
        else:
            filename = f"section-{i+1}.md"
        
        # Combine header and section content
        full_content = '\n'.join(header_lines + section_lines)
        
        # Write to file
        output_file = output_path / filename
        output_file.write_text(full_content, encoding='utf-8')
        
        section_files.append((title, output_file))
        print(f"  Created {output_file.name}: {title}")
    
    # Create/update index.md
    create_index(output_path, section_files, input_path.stem, header_lines)
    
    return section_files


def create_index(output_dir, section_files, doc_name, header_lines):
    """Create or update index.md with links to all sections."""
    index_path = output_dir / "index.md"
    
    # Build index content from header
    index_content = '\n'.join(header_lines)
    if header_lines:
        index_content += '\n\n'
    
    index_content += "## Contents\n\n"
    index_content += "This document is split into sections for easier navigation:\n\n"
    
    for title, file_path in section_files:
        index_content += f"- [{title}]({file_path.name})\n"
    
    index_path.write_text(index_content, encoding='utf-8')
    print(f"\nCreated index at {index_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: split_cutlass_sections.py <input_file> [output_dir]")
        print("  If output_dir is not specified, files are created in the same directory as input_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(input_file).parent)
    
    split_file(input_file, output_dir)


if __name__ == "__main__":
    main()


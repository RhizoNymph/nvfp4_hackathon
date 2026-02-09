#!/usr/bin/env python3
"""
Split markdown documentation files by top-level section numbers.
Each top-level section (e.g., "4.") and all its subsections go into a separate file.
"""

import re
import sys
from pathlib import Path


def find_section_boundaries(content_lines):
    """Find line numbers where top-level sections start."""
    boundaries = []
    
    top_level_pattern = re.compile('^(\\d+)\\\\')
    for i in range(len(content_lines) - 4):
        line_raw = content_lines[i]
        line = line_raw.strip()
        
        # Check if line ends with backslash (markdown continuation)
        # Check the raw line - last non-whitespace character should be backslash
        stripped_raw = line_raw.rstrip()
        line_ends_with_backslash = stripped_raw and stripped_raw[-1] == '\\'
        
        # For pattern matching, remove trailing backslash if present
        line_for_match = line
        if line_ends_with_backslash:
            # Remove the trailing backslash for matching
            line_for_match = stripped_raw[:-1].strip()
        elif line.endswith('\\'):
            line_for_match = line[:-1].strip()
        
        match = top_level_pattern.match(line_for_match)
        if not match:
            continue
        
        # Check lines ahead for equals signs (up to 4 lines ahead to handle various formats)
        found_equals = False
        for offset in range(1, 5):
            if i + offset >= len(content_lines):
                break
            check_line_raw = content_lines[i + offset]
            check_line = check_line_raw.strip()
            # Skip blank lines, single backslash lines, and anchor lines (starting with [)
            if not check_line or check_line == '\\' or (check_line.startswith('[') and ']' in check_line):
                continue
            # Remove trailing backslash if present (markdown continuation)
            if check_line.endswith('\\'):
                check_line = check_line[:-1].strip()
            # Check if this line is all equals signs (after removing trailing backslash)
            if check_line and all(c == '=' for c in check_line):
                found_equals = True
                break
        
        if found_equals:
            section_num = int(match.group(1))
            boundaries.append((i, section_num))
    
    return boundaries


def extract_section_title(line):
    """Extract the title from a section header line."""
    # Line format: "4\. Application Profiling[anchor]" or "1\. [Introduction](url)"
    # First try to extract from markdown link format: [Title](url)
    link_match = re.search(r'\[(.+?)\]\(', line)
    if link_match:
        return link_match.group(1).strip()
    
    # Fallback: Match: number, backslash-dot, space, title, then [ or end
    title_match = re.match(r'^\d+\\\.\s*(.+?)(?:\[|$)', line.strip())
    if title_match:
        return title_match.group(1).strip()
    # Fallback: try without backslash
    title_match = re.match(r'^\d+\.\s*(.+?)(?:\[|$)', line.strip())
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
    return filename


def split_file(input_file, output_dir):
    """Split a markdown file into separate files by top-level sections."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the file - BACKUP original content first
    original_content = input_path.read_text(encoding='utf-8')
    lines = original_content.split('\n')
    
    # Find section boundaries
    boundaries = find_section_boundaries(lines)
    
    if not boundaries:
        print(f"No sections found in {input_file}")
        return
    
    print(f"Found {len(boundaries)} sections in {input_file}")
    
    # Extract header (everything before first section)
    header_end = boundaries[0][0] if boundaries else len(lines)
    header_lines = lines[:header_end]
    
    # Process each section
    section_files = []
    for i, (start_line, section_num) in enumerate(boundaries):
        # Determine end line (start of next section or end of file)
        end_line = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        
        # Extract section content
        section_lines = lines[start_line:end_line]
        
        # Get section title
        title = extract_section_title(section_lines[0]) if section_lines else f"Section {section_num}"
        
        # Create filename with title
        title_slug = sanitize_filename(title)
        if title_slug:
            filename = f"{section_num}-{title_slug}.md"
        else:
            filename = f"{section_num}.md"
        
        # Combine header and section content
        full_content = '\n'.join(header_lines + section_lines)
        
        # Write to file
        output_file = output_path / filename
        output_file.write_text(full_content, encoding='utf-8')
        
        section_files.append((section_num, title, output_file))
        print(f"  Created {output_file.name}: {title}")
    
    # Create/update index.md (but preserve original content by restoring it after)
    create_index(output_path, section_files, input_path.stem, header_lines)
    
    # Restore original content to index.md (the create_index overwrote it)
    # Actually, we want the index to have the table of contents, so don't restore
    
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
    
    for section_num, title, file_path in section_files:
        index_content += f"- [{section_num}. {title}]({file_path.name})\n"
    
    index_path.write_text(index_content, encoding='utf-8')
    print(f"\nCreated index at {index_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: split_sections_v2.py <input_file> [output_dir]")
        print("  If output_dir is not specified, files are created in the same directory as input_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(input_file).parent)
    
    split_file(input_file, output_dir)


if __name__ == "__main__":
    main()


import sys
import os
import re

def filter_ncu(input_file, search_term):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Prepare output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_filtered{ext}"

    # Pattern to identify the start of ANY kernel block
    # It looks for the characteristic (G,G,G)x(B,B,B) pattern
    kernel_start_re = re.compile(r".* \(\d+, \d+, \d+\)x\(\d+, \d+, \d+\), Context")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return

    output_lines = []
    # Always keep the very first line (the file-wide header)
    output_lines.append(lines[0])

    current_block = []
    keep_current_block = False

    # Start from index 1 to skip the global header
    for line in lines[1:]:
        # Check if this line is the start of a new kernel
        if kernel_start_re.match(line):
            # If the PREVIOUS block was a match, commit it to our output
            if keep_current_block:
                output_lines.extend(current_block)
            
            # Reset for the new block
            current_block = [line]
            # Check if this new kernel header contains our search term
            keep_current_block = search_term in line
        else:
            # If we aren't at a new kernel start, just collect the line
            current_block.append(line)

    # Don't forget to check the very last block processed
    if keep_current_block:
        output_lines.extend(current_block)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"Done! Found kernels matching '{search_term}'.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Use your CUTLASS term as default
    target = "cutlass3x"
    
    if len(sys.argv) < 2:
        print("Usage: python filter_ncu.py <file.txt> [search_term]")
    else:
        file_path = sys.argv[1]
        if len(sys.argv) > 2:
            target = sys.argv[2]
        
        filter_ncu(file_path, target)
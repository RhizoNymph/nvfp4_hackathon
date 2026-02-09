import sys
import os
import re
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

def parse_kernel_chunk(args):
    file_path, start_byte, end_byte = args
    
    # Patterns
    section_re = re.compile(r"^\s+Section: (.+)")
    metric_re = re.compile(r"^\s+(.*?)\s{2,}(.*?)\s{2,}([\d,\.]+)")
    speedup_re = re.compile(r"Estimated Speedup:\s+([\d\.]+)%")
    prefix_re = re.compile(r"^\s*(OPT|INF|ERR)\s+(.*)")
    continuation_re = re.compile(r"^\s{5,}(.*)")

    metrics = {}
    speedups = {}
    explanations = [] # List of (section, text)
    
    current_section = "Header"
    in_explanation = False
    current_exp_buffer = []

    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start_byte)
        chunk = f.read(end_byte - start_byte)
        
    for line in chunk.splitlines():
        if not line.strip(): 
            in_explanation = False
            continue
            
        sec_match = section_re.match(line)
        if sec_match:
            current_section = sec_match.group(1).strip()
            in_explanation = False
            continue

        m_match = metric_re.match(line)
        if m_match and "Metric Name" not in line:
            name, unit, val_str = m_match.groups()
            try:
                val = float(val_str.replace(',', ''))
                if current_section not in metrics: metrics[current_section] = {}
                metrics[current_section][name.strip()] = {'unit': unit.strip(), 'value': val}
            except ValueError: pass
            in_explanation = False
            continue

        sp_match = speedup_re.search(line)
        if sp_match:
            if current_section not in speedups: speedups[current_section] = []
            speedups[current_section].append(float(sp_match.group(1)))

        pref_match = prefix_re.match(line)
        if pref_match:
            in_explanation = True
            content = pref_match.group(2).strip()
            if "Estimated Speedup" not in content:
                current_exp_buffer = [content]
            else:
                current_exp_buffer = []
            continue

        if in_explanation:
            cont_match = continuation_re.match(line)
            if cont_match:
                content = cont_match.group(1).strip()
                if content: current_exp_buffer.append(content)
            else:
                if current_exp_buffer:
                    full_text = " ".join(current_exp_buffer)
                    explanations.append((current_section, full_text))
                in_explanation = False
                current_exp_buffer = []

    return {'metrics': metrics, 'speedups': speedups, 'explanations': explanations}

def smart_aggregate_explanations(raw_exps):
    """
    raw_exps: list of (section, text)
    Returns: dict of {section: set(summarized_texts)}
    """
    # { (section, template): [ [val1, val2...], [val1, val2...] ] }
    groups = defaultdict(list)
    
    # Regex to find numbers/percentages: 12.3, 1,234, 95%
    num_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?%?)'

    for sec, text in raw_exps:
        # Find all numbers
        vals = re.findall(num_pattern, text)
        # Create a template by replacing numbers with {}
        template = re.sub(num_pattern, '{}', text)
        groups[(sec, template)].append(vals)

    final_results = defaultdict(set)
    for (sec, template), val_lists in groups.items():
        if not val_lists: continue
        
        # We need to average each "slot" across all instances
        num_slots = len(val_lists[0])
        avg_vals = []
        
        for i in range(num_slots):
            raw_vals = []
            is_pct = False
            for v_list in val_lists:
                if i < len(v_list):
                    clean_v = v_list[i].replace(',', '').replace('%', '')
                    try:
                        raw_vals.append(float(clean_v))
                        if '%' in v_list[i]: is_pct = True
                    except: pass
            
            if raw_vals:
                avg = sum(raw_vals) / len(raw_vals)
                formatted = f"{avg:.2f}%" if is_pct else f"{avg:,.2f}"
                avg_vals.append(formatted)
            else:
                avg_vals.append("???")

        # Fill the template back in
        try:
            summary = template.format(*avg_vals)
            final_results[sec].add(summary)
        except IndexError:
            final_results[sec].add(template) # Fallback

    return final_results

def main(input_file):
    if not os.path.exists(input_file): return

    print("Indexing kernel locations...")
    kernel_start_re = re.compile(r".* \(\d+, \d+, \d+\)x\(\d+, \d+, \d+\), Context")
    offsets = []
    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line: break
            if kernel_start_re.match(line): offsets.append(pos)
    offsets.append(os.path.getsize(input_file))
    tasks = [(input_file, offsets[i], offsets[i+1]) for i in range(len(offsets)-1)]
    
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(parse_kernel_chunk, tasks), total=len(tasks), desc="Processing"):
            results.append(res)

    # Standard Aggregation
    final_metrics = defaultdict(lambda: defaultdict(lambda: {'unit': '', 'values': []}))
    final_speedups = defaultdict(list)
    all_raw_explanations = []

    for res in results:
        for sec, metrics in res['metrics'].items():
            for m_name, m_info in metrics.items():
                final_metrics[sec][m_name]['unit'] = m_info['unit']
                final_metrics[sec][m_name]['values'].append(m_info['value'])
        for sec, sps in res['speedups'].items():
            final_speedups[sec].extend(sps)
        all_raw_explanations.extend(res['explanations'])

    # Smart Aggregation for Explanations
    final_explanations = smart_aggregate_explanations(all_raw_explanations)

    output_file = input_file.replace(".txt", "_stats.txt")
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("NCU AGGREGATED PERFORMANCE SUMMARY (Averaged over all kernels)\n")
        out.write("="*80 + "\n\n")

        # Top Bottlenecks
        if final_speedups:
            summary = sorted([(s, sum(v)/len(v)) for s, v in final_speedups.items()], key=lambda x: x[1], reverse=True)
            out.write("RANKED BOTTLENECKS (Highest Potential Speedup First):\n")
            out.write(f"{'Section Name':<50} | {'Avg. Speedup %':<15}\n" + "-"*70 + "\n")
            for sec, val in summary:
                out.write(f"{sec:<50} | {val:>13.2f}%\n")
            out.write("\n" + "="*80 + "\n\n")

        for sec_name, metrics in sorted(final_metrics.items()):
            if sec_name == "Header": continue
            out.write(f"### Section: {sec_name}\n")
            out.write(f"{'-'*45} ----------- ------------\n")
            for m_name, data in sorted(metrics.items()):
                if not data['values']: continue
                avg = sum(data['values']) / len(data['values'])
                out.write(f"{m_name:<45} {data['unit']:<11} {avg:<12.2f}\n")
            
            if sec_name in final_speedups:
                avg_sp = sum(final_speedups[sec_name]) / len(final_speedups[sec_name])
                out.write(f"\n>> GLOBAL AVERAGE ESTIMATED SPEEDUP: {avg_sp:.2f}%\n")
            
            if sec_name in final_explanations:
                out.write("\nAdvice & Analysis (Averaged Trends):\n")
                for exp in sorted(final_explanations[sec_name]):
                    out.write(f" - {exp}\n\n")
            out.write("="*80 + "\n\n")

    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1: main(sys.argv[1])
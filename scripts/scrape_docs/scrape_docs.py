#!/usr/bin/env python3
"""
Automated script to scrape CUDA and/or CUTLASS documentation from Firecrawl API.
Handles starting crawls, polling for completion, and organizing results.
"""

import argparse
import json
import time
import requests
import sys
import subprocess
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:3002"
CRAWL_V2_ENDPOINT = f"{BASE_URL}/v2/crawl"
CRAWL_V1_ENDPOINT = f"{BASE_URL}/v1/crawl"


def start_cuda_crawl():
    """Start CUDA documentation crawl."""
    payload = {
        "url": "https://docs.nvidia.com/cuda/archive/13.0.0/",
        "scrapeOptions": {
            "formats": ["markdown"]
        }
    }
    
    print("Starting CUDA documentation crawl...")
    response = requests.post(CRAWL_V2_ENDPOINT, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    data = response.json()
    
    if not data.get("success"):
        raise Exception(f"Failed to start crawl: {data}")
    
    crawl_id = data.get("id")
    print(f"CUDA crawl started: {crawl_id}")
    return crawl_id


def start_cutlass_crawl():
    """Start CUTLASS documentation crawl."""
    payload = {
        "url": "https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html",
        "scrapeOptions": {
            "formats": ["markdown"],
            "waitFor": 2000
        },
        "crawlEntireDomain": True,
        "includePaths": [
            ".*pythonDSL/.*"
        ]
    }
    
    print("Starting CUTLASS documentation crawl...")
    # Note: CUTLASS uses v1 endpoint based on user's example
    response = requests.post(f"{BASE_URL}/v1/crawl", json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    data = response.json()
    
    if not data.get("success"):
        raise Exception(f"Failed to start crawl: {data}")
    
    crawl_id = data.get("id")
    print(f"CUTLASS crawl started: {crawl_id}")
    return crawl_id


def check_status(crawl_id, use_v1=False):
    """Check crawl status. Returns (status, data)."""
    # Status is always checked via v1 endpoint
    endpoint = f"{CRAWL_V1_ENDPOINT}/{crawl_id}"
    
    try:
        response = requests.get(endpoint, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        # v1 endpoint returns status directly
        status = data.get("status", "unknown")
        return status, data
    except Exception as e:
        print(f"Error checking status: {e}")
        return "error", None


def wait_for_completion(crawl_id, use_v1=False, poll_interval=5, max_wait=3600):
    """Wait for crawl to complete, polling periodically."""
    print(f"Waiting for crawl {crawl_id} to complete...")
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f"Crawl {crawl_id} did not complete within {max_wait} seconds")
        
        status, data = check_status(crawl_id, use_v1=use_v1)
        
        if status == "completed":
            print(f"Crawl completed in {elapsed:.1f} seconds")
            return data
        elif status == "scraping" or status == "pending":
            print(f"  Status: {status} (elapsed: {elapsed:.1f}s)", end='\r')
            time.sleep(poll_interval)
        elif status == "error" or status == "failed":
            raise Exception(f"Crawl {crawl_id} failed with status: {status}")
        else:
            print(f"  Unknown status: {status}, waiting...")
            time.sleep(poll_interval)


def fetch_results(crawl_id, use_v1=False):
    """Fetch all results from completed crawl."""
    # Results are fetched via v2 endpoint
    endpoint = f"{CRAWL_V2_ENDPOINT}/{crawl_id}"
    
    print(f"Fetching results from {endpoint}...")
    all_data = []
    url = endpoint
    page = 1
    
    while url:
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            all_data.extend(data["data"])
            print(f"  Fetched page {page}: {len(data['data'])} items (total: {len(all_data)})")
        
        # Get next page URL
        url = data.get("next")
        if url:
            page += 1
    
    print(f"Total items fetched: {len(all_data)}")
    return all_data


def organize_cuda_docs(crawl_id):
    """Organize CUDA docs using the existing script."""
    print("\nOrganizing CUDA documentation...")
    script_path = Path(__file__).parent / "organize_cuda_docs.py"
    
    import subprocess
    import os
    env = os.environ.copy()
    env["CUDA_CRAWL_ID"] = crawl_id
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error organizing CUDA docs:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def organize_cutlass_docs(crawl_id):
    """Organize CUTLASS docs using the existing script."""
    print("\nOrganizing CUTLASS documentation...")
    script_path = Path(__file__).parent / "organize_cutlass_docs.py"
    
    import subprocess
    import os
    env = os.environ.copy()
    env["CUTLASS_CRAWL_ID"] = crawl_id
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error organizing CUTLASS docs:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(description="Scrape and organize CUDA and/or CUTLASS documentation")
    parser.add_argument("--cuda", action="store_true", help="Scrape CUDA documentation")
    parser.add_argument("--cutlass", action="store_true", help="Scrape CUTLASS documentation")
    parser.add_argument("--cuda-id", type=str, help="Use existing CUDA crawl ID (skip starting new crawl)")
    parser.add_argument("--cutlass-id", type=str, help="Use existing CUTLASS crawl ID (skip starting new crawl)")
    parser.add_argument("--skip-wait", action="store_true", help="Skip waiting for completion (assume already completed)")
    parser.add_argument("--skip-organize", action="store_true", help="Skip organizing docs (just fetch crawl IDs)")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    if not args.cuda and not args.cutlass:
        parser.print_help()
        sys.exit(1)
    
    cuda_crawl_id = None
    cutlass_crawl_id = None
    
    # Start crawls
    if args.cuda:
        if args.cuda_id:
            cuda_crawl_id = args.cuda_id
            print(f"Using existing CUDA crawl ID: {cuda_crawl_id}")
        else:
            cuda_crawl_id = start_cuda_crawl()
    
    if args.cutlass:
        if args.cutlass_id:
            cutlass_crawl_id = args.cutlass_id
            print(f"Using existing CUTLASS crawl ID: {cutlass_crawl_id}")
        else:
            cutlass_crawl_id = start_cutlass_crawl()
    
    # Wait for completion
    if not args.skip_wait:
        if cuda_crawl_id:
            wait_for_completion(cuda_crawl_id, use_v1=False, poll_interval=args.poll_interval)
        if cutlass_crawl_id:
            # CUTLASS uses v1 for status checking
            wait_for_completion(cutlass_crawl_id, use_v1=True, poll_interval=args.poll_interval)
    
    # Organize docs
    if not args.skip_organize:
        if cuda_crawl_id:
            if organize_cuda_docs(cuda_crawl_id):
                print("\nSplitting CUDA documentation into sections...")
                split_script = Path(__file__).parent / "split_all_cuda_docs.py"
                subprocess.run([sys.executable, str(split_script)], check=False)
        
        if cutlass_crawl_id:
            if organize_cutlass_docs(cutlass_crawl_id):
                print("\nSplitting CUTLASS documentation into sections...")
                split_script = Path(__file__).parent / "split_all_cutlass_docs.py"
                subprocess.run([sys.executable, str(split_script)], check=False)
    
    # Print summary
    print("\n" + "="*70)
    print("Crawl Summary:")
    if cuda_crawl_id:
        print(f"  CUDA crawl ID: {cuda_crawl_id}")
        print(f"  Status URL: {CRAWL_V1_ENDPOINT}/{cuda_crawl_id}")
        print(f"  Results URL: {CRAWL_V2_ENDPOINT}/{cuda_crawl_id}")
    if cutlass_crawl_id:
        print(f"  CUTLASS crawl ID: {cutlass_crawl_id}")
        print(f"  Status URL: {CRAWL_V1_ENDPOINT}/{cutlass_crawl_id}")
        print(f"  Results URL: {CRAWL_V2_ENDPOINT}/{cutlass_crawl_id}")
    print("="*70)


if __name__ == "__main__":
    import os
    main()


# Documentation Scraping Guide

This guide explains how to scrape and organize NVIDIA CUDA and CUTLASS documentation using the Firecrawl API.

## Overview

The scraping process involves:
1. **Starting a crawl** - Submit a crawl request to Firecrawl API
2. **Polling for completion** - Wait for the crawl to finish (status changes from "scraping" to "completed")
3. **Fetching results** - Download all scraped content when ready
4. **Organizing docs** - Save files into a directory tree structure
5. **Splitting sections** - Split large documents into smaller, browsable sections

## Firecrawl API Endpoints

- **Start crawl**: `POST http://127.0.0.1:3002/v2/crawl`
- **Check status**: `GET http://127.0.0.1:3002/v1/crawl/{id}`
- **Get results**: `GET http://127.0.0.1:3002/v2/crawl/{id}`

## CUDA Documentation

### Manual Process

1. **Start the crawl**:
```bash
curl -X POST http://127.0.0.1:3002/v2/crawl \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://docs.nvidia.com/cuda/archive/13.0.0/",
    "scrapeOptions": {
      "formats": ["markdown"]
    }
  }'
```

2. **Response** (save the `id`):
```json
{
  "success": true,
  "id": "019b2af4-4d39-75ab-9b62-41c351b7a4d2",
  "url": "http://127.0.0.1:3002/v1/crawl/019b2af4-4d39-75ab-9b62-41c351b7a4d2"
}
```

3. **Check status** (poll until `status` is `"completed"`):
```bash
curl http://127.0.0.1:3002/v1/crawl/019b2af4-4d39-75ab-9b62-41c351b7a4d2
```

4. **Fetch and organize**:
```bash
python3 scripts/organize_cuda_docs.py
python3 scripts/split_all_cuda_docs.py
```

### Automated Process

Use the `scrape_docs.py` script:
```bash
python3 scripts/scrape_docs.py --cuda
```

This will:
1. Start the crawl
2. Poll for completion
3. Organize the docs
4. Split sections automatically

## CUTLASS Documentation

### Manual Process

1. **Start the crawl**:
```bash
curl -X POST http://127.0.0.1:3002/v1/crawl \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html",
    "scrapeOptions": {
      "formats": ["markdown"],
      "waitFor": 2000
    },
    "crawlEntireDomain": true,
    "includePaths": [
      ".*pythonDSL/.*"
    ]
  }'
```

2. **Response** (save the `id`):
```json
{
  "success": true,
  "id": "019b39b3-cc0a-765a-88b2-2fd70d2ce7af",
  "url": "http://127.0.0.1:3002/v1/crawl/019b39b3-cc0a-765a-88b2-2fd70d2ce7af"
}
```

3. **Check status** and wait for completion

4. **Fetch and organize**:
```bash
python3 scripts/organize_cutlass_docs.py
python3 scripts/split_all_cutlass_docs.py
```

### Automated Process

Use the `scrape_docs.py` script:
```bash
python3 scripts/scrape_docs.py --cutlass
```

This will:
1. Start the crawl
2. Poll for completion
3. Organize the docs
4. Split sections automatically

## Automated Script Usage

The `scrape_docs.py` script automates the entire process:

### Basic Usage

```bash
# Scrape CUDA docs
python3 scripts/scrape_docs.py --cuda

# Scrape CUTLASS docs
python3 scripts/scrape_docs.py --cutlass

# Scrape both
python3 scripts/scrape_docs.py --cuda --cutlass
```

### Advanced Options

```bash
# Use existing crawl IDs (skip starting new crawls)
python3 scripts/scrape_docs.py --cuda-id 019b2af4-4d39-75ab-9b62-41c351b7a4d2

# Skip waiting (assume crawls already completed)
python3 scripts/scrape_docs.py --cuda-id <id> --skip-wait

# Just start crawls, don't organize
python3 scripts/scrape_docs.py --cuda --skip-organize

# Custom polling interval
python3 scripts/scrape_docs.py --cuda --poll-interval 10
```

## Script Order

### For CUDA Docs:
1. `scrape_docs.py --cuda` (or manual crawl + organize_cuda_docs.py)
2. `split_all_cuda_docs.py` (splits large files into sections)

### For CUTLASS Docs:
1. `scrape_docs.py --cutlass` (or manual crawl + organize_cutlass_docs.py)
2. `split_all_cutlass_docs.py` (splits files with multiple sections)

### For Both:
```bash
# Automated (recommended)
python3 scripts/scrape_docs.py --cuda --cutlass

# Manual (if you prefer step-by-step)
python3 scripts/scrape_docs.py --cuda --cutlass --skip-organize
python3 scripts/organize_cuda_docs.py
python3 scripts/organize_cutlass_docs.py
python3 scripts/split_all_cuda_docs.py
python3 scripts/split_all_cutlass_docs.py
```

## Output Structure

- **CUDA docs**: `docs/cuda/` - Organized by topic, split by numbered sections
- **CUTLASS docs**: `docs/cutlass/` - Organized by topic, split by markdown headers

## Notes

- CUDA sections use numbered format: `1\. Overview`, `2\. Preface`, etc.
- CUTLASS sections use markdown headers: `Overview[#](url)`, `Why CUTLASS DSLs?[#](url)`, etc.
- The splitting scripts automatically skip files with only 1 section
- Large documents are split into separate files for easier browsing by agents


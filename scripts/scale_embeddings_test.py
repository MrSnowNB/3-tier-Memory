#!/usr/bin/env python3
"""
Scaling Test Wrapper for Ollama Embeddings

Executes the stress test across different concurrency levels
to find the maximum stable embedding throughput up to 144 workers.

Usage:
    python scripts/scale_embeddings_test.py --max-concurrency 144 --duration 30

This script:
- Runs baseline test (2 workers, 30s)
- Scales through [4,8,16,32,64,128] workers (30s each)
- Attempts 144 workers if prior levels succeed
- Aggregates results into logs/embedding_scale_test.csv
"""

import subprocess
import time
import csv
import logging
from pathlib import Path
import argparse
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_stress_test(concurrency: int, duration: int = 30, batch_size: int = 8) -> dict:
    """Run stress test for given concurrency level."""
    logger.info(f"🏃 Running stress test: concurrency={concurrency}, duration={duration}s")

    cmd = [
        "python", "scripts/stress_embeddings.py",
        "--concurrency", str(concurrency),
        "--duration", str(duration),
        "--batch-size", str(batch_size),
        "--report-interval", str(duration),  # Match duration for single summary
        "--model", "nomic-embed-text"  # Will be auto-detected to use available models
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)

        logger.info(f"Test completed with exit code: {result.returncode}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")

        # Try to parse the JSON summary if it exists
        summary_file = Path("docs/phase2_stress_summary.json")
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            return summary
        else:
            logger.error("No summary file found")
            return {"result": "FAILED", "error": "No summary file"}

    except subprocess.TimeoutExpired:
        logger.error(f"Test timed out for concurrency {concurrency}")
        return {"result": "TIMEOUT", "concurrency": concurrency}
    except Exception as e:
        logger.error(f"Test failed for concurrency {concurrency}: {e}")
        return {"result": "ERROR", "concurrency": concurrency, "error": str(e)}

def aggregate_results(results: list, output_csv: str):
    """Aggregate all results into a CSV file."""
    logger.info(f"📊 Aggregating results to {output_csv}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'concurrency', 'result', 'duration_actual', 'total_requests',
            'overall_error_rate', 'p95_baseline', 'p95_final',
            'throughput_req_per_sec', 'throughput_embed_per_sec',
            'memory_trend_mb_per_minute', 'models_used'
        ])

        for result in results:
            if 'overall_metrics' in result:
                row = [
                    result.get('config', {}).get('concurrency', 'unknown'),
                    result.get('result', 'UNKNOWN'),
                    result.get('duration', {}).get('actual_seconds', 0),
                    result.get('overall_metrics', {}).get('total_requests', 0),
                    result.get('overall_metrics', {}).get('overall_error_rate', 0),
                    result.get('overall_metrics', {}).get('p95_baseline', 0),
                    result.get('overall_metrics', {}).get('p95_final', 0),
                    result.get('overall_metrics', {}).get('throughput_req_per_sec', 0),
                    result.get('overall_metrics', {}).get('throughput_embed_per_sec', 0),
                    result.get('overall_metrics', {}).get('memory_trend_mb_per_minute', 0),
                    ','.join(result.get('models', ['unknown']))
                ]
            else:
                # Fallback for failed/timed out tests
                row = [
                    result.get('concurrency', 'unknown'),
                    result.get('result', 'UNKNOWN'),
                    0, 0, 0, 0, 0, 0, 0, 0, 'unknown'
                ]

            writer.writerow(row)

def main():
    """Main execution for scaling test."""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama Embeddings Scaling Test")
    parser.add_argument("--max-concurrency", type=int, default=144, help="Maximum concurrency to test")
    parser.add_argument("--duration", type=int, default=30, help="Duration per test level in seconds")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for requests")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline test")

    args = parser.parse_args()

    logger.info("🚀 STARTING OLLAMA EMBEDDINGS SCALING TEST")
    logger.info(f"Max concurrency: {args.max_concurrency}")
    logger.info(f"Duration per level: {args.duration}s")
    logger.info(f"Batch size: {args.batch_size}")

    # Test levels: baseline (2) + scaling levels
    if args.baseline_only:
        concurrency_levels = [2]
    else:
        # Start with 2, then 4,8,16,32,... up to max
        concurrency_levels = [2]
        level = 4
        while level <= args.max_concurrency:
            concurrency_levels.append(level)
            level *= 2

        # Add the final level if it's not a power of 2
        if args.max_concurrency not in concurrency_levels:
            concurrency_levels.append(args.max_concurrency)

    logger.info(f"Testing concurrency levels: {concurrency_levels}")

    results = []
    success_levels = []

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    for concurrency in concurrency_levels:
        result = run_stress_test(concurrency, args.duration, args.batch_size)
        results.append(result)

        if result.get('result') == 'PASSED':
            success_levels.append(concurrency)
            logger.info(f"✅ Concurrency {concurrency}: PASSED")
        else:
            logger.warning(f"❌ Concurrency {concurrency}: {result.get('result', 'FAILED')}")

        # Small delay between tests
        time.sleep(5)

    # Aggregate results
    output_csv = "logs/embedding_scale_test.csv"
    aggregate_results(results, output_csv)

    # Summary
    max_stable_concurrency = max(success_levels) if success_levels else 0
    logger.info("🎯 SCALING TEST COMPLETED")
    logger.info(f"📊 Results saved to {output_csv}")
    logger.info(f"🎖️  Maximum stable concurrency: {max_stable_concurrency}")
    logger.info(f"📈 Successful levels: {success_levels}")

    # Create documentation
    doc_content = f"""# Ollama Embeddings Scaling Test Results

## Summary
- **Maximum Stable Concurrency**: {max_stable_concurrency} workers
- **Test Duration per Level**: {args.duration}s
- **Batch Size**: {args.batch_size}
- **Successful Levels**: {', '.join(map(str, success_levels))}

## Detailed Results
See `logs/embedding_scale_test.csv` for complete metrics.

## Recommendations
The recommended operating range is up to {max_stable_concurrency} concurrent embedding workers
for stable performance without degradation or runner errors.

## Timestamp
{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
"""

    with open("docs/phase2_embedding_scale_results.md", 'w') as f:
        f.write(doc_content)

    logger.info("📝 Documentation saved to docs/phase2_embedding_scale_results.md")

    return 0 if max_stable_concurrency >= 144 else 1

if __name__ == "__main__":
    exit(main())

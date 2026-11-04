"""Low-cost API load testing.

Default is GET /health only. This never calls Gemini and avoids paid traffic.
Example: python load_test.py --url http://localhost:8000 --requests 150 --concurrency 10
"""
import argparse
import concurrent.futures
import json
import statistics
import time

import httpx


def one_request(url: str, timeout: float) -> dict:
    start = time.perf_counter()
    try:
        response = httpx.get(url, timeout=timeout)
        return {"status": response.status_code, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
    except httpx.RequestError:
        return {"status": "network_error", "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


def run(url: str, requests: int, concurrency: int, timeout: float) -> dict:
    if requests < 1 or concurrency < 1 or requests > 10000 or concurrency > 100:
        raise ValueError("requests must be 1..10000 and concurrency 1..100")
    target = url.rstrip("/") + "/health"
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: one_request(target, timeout), range(requests)))
    seconds = max(time.perf_counter() - started, 1e-6)
    latencies = sorted(item["latency_ms"] for item in results)
    successes = sum(item["status"] == 200 for item in results)
    def percentile(p: float) -> float:
        return latencies[min(len(latencies) - 1, max(0, int((len(latencies) - 1) * p)))]
    return {
        "endpoint": "/health", "requests": requests, "concurrency": concurrency,
        "successes": successes, "failed": requests - successes,
        "success_rate": round(successes / requests, 4),
        "rps": round(requests / seconds, 2),
        "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99), "average_ms": round(statistics.mean(latencies), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load-test health endpoint only (no Gemini API calls).")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.requests, args.concurrency, args.timeout), indent=2))

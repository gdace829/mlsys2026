"""
FlashInfer-Bench Local Benchmark Runner (Debug Version).
Enhanced with error tracking to capture COMPILE_ERROR details.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    # 调试阶段：减少迭代次数，确保快速反馈
    if config is None:
        config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)

    trace_set_path = get_trace_set_path()
    print(f"[*] Loading dataset from: {trace_set_path}")
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    print(f"[*] Found {len(workloads)} workloads for {solution.definition}")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    try:
        benchmark = Benchmark(bench_trace_set, config)
        # 核心：run_all 可能会抛出底层编译异常
        result_trace_set = benchmark.run_all(dump_traces=True)
    except Exception as e:
        print("\n" + "!"*60)
        print("CRITICAL ERROR CAPTURED DURING BENCHMARK RUN:")
        traceback.print_exc()
        print("!"*60 + "\n")
        return {}

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            
            # 如果是 COMPILE_ERROR，尝试打印 trace 里的错误日志
            if entry["status"] == "COMPILE_ERROR":
                print(f"[!] Workload {trace.workload.uuid} failed with COMPILE_ERROR")
            
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    if not results:
        return

    for def_name, traces in results.items():
        print(f"\nResults for {def_name}:")
        success_count = 0
        for workload_uuid, result in traces.items():
            status = result.get("status")
            if status == "SUCCESS":
                success_count += 1
            
            latency = result.get("latency_ms", 0)
            print(f"  - {workload_uuid[:8]}: {status} | Latency: {latency:.4f} ms")
        
        print(f"\nSummary: {success_count}/{len(traces)} workloads passed.")


def main():
    """Pack solution and run benchmark."""
    try:
        print("--- Phase 1: Packing Solution ---")
        solution_path = pack_solution()

        print("\n--- Phase 2: Loading Solution ---")
        solution = Solution.model_validate_json(solution_path.read_text())
        print(f"OK: {solution.name} targetting {solution.definition}")

        print("\n--- Phase 3: Running Benchmark ---")
        results = run_benchmark(solution)

        if not results:
            print("\n[FAILED] No results returned. Check the error log above.")
            return

        print_results(results)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
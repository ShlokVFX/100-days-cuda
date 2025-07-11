import dataclasses
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any
from collections import OrderedDict
from statistics import geometric_mean

import torch.cuda

from utils import set_seed
try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from submission import custom_kernel
from reference import check_implementation, generate_input

WARMUP_RUNS = 10
TIMED_RUNS = 100


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, 'w')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)
    
    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def copy_kv_cache(module, kv_cache_shape):
    """
    Creates a copy of the KVCache module manually.
    """
    copied_module = type(module)(kv_cache_shape)
    
    # Copy parameters
    params = OrderedDict()
    for name, param in module.named_parameters():
        params[name] = param.clone().requires_grad_(param.requires_grad).cuda()
        
    # Copy buffers
    buffers = OrderedDict()
    for name, buff in module.named_buffers():
        print(f"Buff name: {name}, shape: {buff.shape}")
        buffers[name] = buff.clone().cuda()
    
    # Assign params and buffers to copied module
    copied_module.load_state_dict(params, strict=False)
    copied_module.load_state_dict(buffers, strict=False)
    copied_module.seq_len = module.seq_len
    
    return copied_module.cuda()


def get_test_cases(file_name: str) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    return tests


def warm_up(test: TestCase):
    config, data, kv_cache = generate_input(**test.args)
    config_copy = copy_config_weights(config)
    start = time.perf_counter()
    while time.perf_counter() - start < 0.2:
        custom_kernel((config_copy, data, kv_cache))
        torch.cuda.synchronize()


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg)**2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


def copy_config_weights(config):
    """
    Creates a copy of the Config object with cloned weight tensors.
    """
    return dataclasses.replace(
        config,
        Q_proj_down_weight=config.Q_proj_down_weight.clone().cuda(),
        Q_proj_up_weight=config.Q_proj_up_weight.clone().cuda(),
        KV_proj_down_weight=config.KV_proj_down_weight.clone().cuda(),
        KV_proj_up_weight=config.KV_proj_up_weight.clone().cuda()
    )


def run_testing(logger: PopcornOutput, tests: list[TestCase]):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)

        config, data, kv_cache = generate_input(**test.args)
        kv_cache_copy = copy_kv_cache(kv_cache, config.kv_cache_shape)

        torch.cuda.synchronize()
        submission_output = custom_kernel((config, data, kv_cache))
        torch.cuda.synchronize()
        error = check_implementation((config, data, kv_cache_copy), submission_output)
        if error:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", error)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    durations = []
    # generate input data once
    config, data, kv_cache = generate_input(**test.args)
    # first, one obligatory correctness check; also triggers triton compile for the given shape
    kv_cache_copy = copy_kv_cache(kv_cache, config.kv_cache_shape)
    config_copy = copy_config_weights(config)
    with torch.no_grad():
        output = custom_kernel((config, data, kv_cache))
        error = check_implementation((config_copy, data, kv_cache_copy), output)
    if error:
        return error

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    with torch.no_grad():
        for i in range(max_repeats):
            if recheck:
                config, data, kv_cache = generate_input(**test.args)
                kv_cache_copy = copy_kv_cache(kv_cache, config.kv_cache_shape)
                config_copy = copy_config_weights(config)
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = custom_kernel((config, data, kv_cache))
            torch.cuda.synchronize()
            end = time.perf_counter_ns()

            if recheck:
                error = check_implementation((config_copy, data, kv_cache_copy), output)
                if error:
                    return error

            del output
            durations.append(end-start)

            if i > 1:
                stats = calculate_stats(durations)
                if stats.err / stats.mean < 0.01 or stats.mean *  stats.runs > max_time_ns:
                    break

    return calculate_stats(durations)


def run_benchmarking(logger: PopcornOutput, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    warm_up(tests[0])
    passed = True
    logger.log("benchmark-count", len(tests))
    scores = []
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = benchmark(test, False, 100, 10e9)
        if isinstance(result, Stats):
            if result.mean / 1_000_000_000 > 1:
                factor = 1_000_000_000
                suffix = "s"
            elif result.mean / 1_000_000 > 1:
                factor = 1_000_000
                suffix = "ms"
            elif result.mean / 1_000 > 1:
                factor = 1_000
                suffix = "µs"
            else:
                factor = 1
                suffix = "µs"
            logger.print(f"⏱ {result.mean / factor:.2f} ± {result.std / factor:.3f} {suffix}")
            logger.print(
                f"⚡ {result.best / factor:.2f} {suffix} 🐌 {result.worst / factor:.2f} {suffix}"
            )
            scores.append(result.mean)

            # for field in dataclasses.fields(Stats):
            #     logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)

    if passed:
        logger.log("check", "pass")
        if result.mean / 1_000_000_000 > 1:
            factor = 1_000_000_000
            suffix = "s"
        elif result.mean / 1_000_000 > 1:
            factor = 1_000_000
            suffix = "ms"
        elif result.mean / 1_000 > 1:
            factor = 1_000
            suffix = "µs"
        else:
            factor = 1
            suffix = "µs"
        logger.print(f"Your score: {geometric_mean(scores) / factor:.2f} {suffix} 🥳")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    tests = get_test_cases(sys.argv[2])

    with PopcornOutput(int(fd)) as logger:
        seed = os.getenv("POPCORN_SEED")
        seed = int(seed) if seed else 42
        set_seed(seed)

        if mode == "test":
            return run_testing(logger, tests)

        if mode == "benchmark":
            return run_benchmarking(logger, tests)
        
        if mode == "leaderboard":
            warm_up(tests[0])
            result = benchmark(tests[-1], True, 100, 30e9)
            passed = True
            scores = []
            if isinstance(result, Stats):
                if result.mean / 1_000_000_000 > 1:
                    factor = 1_000_000_000
                    suffix = "s"
                elif result.mean / 1_000_000 > 1:
                    factor = 1_000_000
                    suffix = "ms"
                elif result.mean / 1_000 > 1:
                    factor = 1_000
                    suffix = "µs"
                else:
                    factor = 1
                    suffix = "µs"
                logger.print(f"⏱ {result.mean / factor:.2f} ± {result.std / factor:.3f} {suffix}")
                logger.print(
                    f"⚡ {result.best / factor:.2f} {suffix} 🐌 {result.worst / factor:.2f} {suffix}"
                )
                scores.append(result.mean)
            else:
                passed = False
                logger.log(f"status", "fail")
                logger.log(f"error", str(result)) #TODO: Make sure result implements __str__?
        
            logger.log("check", "pass" if passed else "fail")
            if passed:
                if result.mean / 1_000_000_000 > 1:
                    factor = 1_000_000_000
                    suffix = "s"
                elif result.mean / 1_000_000 > 1:
                    factor = 1_000_000
                    suffix = "ms"
                elif result.mean / 1_000 > 1:
                    factor = 1_000
                    suffix = "µs"
                else:
                    factor = 1
                    suffix = "µs"
                logger.print(f"Your score: {geometric_mean(scores) / factor:.2f} {suffix} 🥳")
        else:
            # TODO: Implement script and profile mode
            return 2


if __name__ == "__main__":
    sys.exit(main())
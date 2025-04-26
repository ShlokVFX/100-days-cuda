import re
from statistics import geometric_mean

benchmark_text = """
benchmark-count: 18
benchmark.0.spec: k: 7168; m: 1024; n: 1536; seed: 8135
⏱ 373.82 ± 2.867 µs
⚡ 372.11 µs 🐌 394.21 µs
benchmark.1.spec: k: 1536; m: 1024; n: 3072; seed: 6251
⏱ 88.50 ± 1.787 µs
⚡ 86.64 µs 🐌 101.48 µs
benchmark.2.spec: k: 7168; m: 1024; n: 576; seed: 12346
benchmark.2.status: fail
benchmark.2.error: Number of mismatched elements: 588850\nERROR at (0, 0): -588.0 -170.0\nERROR at (0, 1): -460.0 -14.1875\nERROR at (0, 2): 130.0 45.0\nERROR at (0, 3): 176.0 50.0\nERROR at (0, 4): -332.0 -102.5\n... and 588845 more mismatched elements.
benchmark.3.spec: k: 256; m: 1024; n: 7168; seed: 5364
⏱ 49.73 ± 1.415 µs
⚡ 48.38 µs 🐌 58.73 µs
benchmark.4.spec: k: 2048; m: 1024; n: 7168; seed: 6132
⏱ 236.56 ± 3.325 µs
⚡ 233.77 µs 🐌 265.14 µs
benchmark.5.spec: k: 7168; m: 1024; n: 4608; seed: 7531
⏱ 420.58 ± 7.260 µs
⚡ 406.86 µs 🐌 474.76 µs
benchmark.6.spec: k: 2304; m: 1024; n: 7168; seed: 12345
⏱ 262.37 ± 3.637 µs
⚡ 259.12 µs 🐌 295.23 µs
benchmark.7.spec: k: 7168; m: 1024; n: 512; seed: 6563
benchmark.7.status: fail
benchmark.7.error: Number of mismatched elements: 523436\nERROR at (0, 0): 294.0 -3.78125\nERROR at (0, 1): 374.0 77.0\nERROR at (0, 2): -237.0 -5.875\nERROR at (0, 3): -828.0 -172.0\nERROR at (0, 4): 186.0 106.5\n... and 523431 more mismatched elements.
benchmark.8.spec: k: 512; m: 1024; n: 4096; seed: 17512
⏱ 45.09 ± 1.407 µs
⚡ 43.10 µs 🐌 53.53 µs
benchmark.9.spec: k: 7168; m: 6144; n: 1536; seed: 6543
⏱ 839.45 ± 16.626 µs
⚡ 817.23 µs 🐌 966.35 µs
benchmark.10.spec: k: 1536; m: 6144; n: 3072; seed: 234
⏱ 371.84 ± 4.672 µs
⚡ 365.49 µs 🐌 387.36 µs
benchmark.11.spec: k: 7168; m: 6144; n: 576; seed: 9863
benchmark.11.status: fail
benchmark.11.error: Number of mismatched elements: 3532979\nERROR at (0, 0): -165.0 -61.25\nERROR at (0, 1): 418.0 115.5\nERROR at (0, 2): 256.0 54.75\nERROR at (0, 3): -504.0 -94.5\nERROR at (0, 4): 172.0 -2.875\n... and 3532974 more mismatched elements.
benchmark.12.spec: k: 256; m: 6144; n: 7168; seed: 764243
⏱ 170.20 ± 3.067 µs
⚡ 166.38 µs 🐌 189.56 µs
benchmark.13.spec: k: 2048; m: 6144; n: 7168; seed: 76547
⏱ 1.10 ± 0.019 ms
⚡ 1.07 ms 🐌 1.24 ms
benchmark.14.spec: k: 7168; m: 6144; n: 4608; seed: 65436
⏱ 2.49 ± 0.021 ms
⚡ 2.46 ms 🐌 2.57 ms
benchmark.15.spec: k: 2304; m: 6144; n: 7168; seed: 452345
⏱ 1.23 ± 0.019 ms
⚡ 1.19 ms 🐌 1.33 ms
benchmark.16.spec: k: 7168; m: 6144; n: 512; seed: 12341
benchmark.16.status: fail
benchmark.16.error: Number of mismatched elements: 3140482\nERROR at (0, 0): -73.0 2.859375\nERROR at (0, 1): 221.0 62.25\nERROR at (0, 2): -148.0 -56.0\nERROR at (0, 3): -404.0 -64.5\nERROR at (0, 4): 219.0 72.5\n... and 3140477 more mismatched elements.
benchmark.17.spec: k: 512; m: 6144; n: 4096; seed: 45245
⏱ 193.67 ± 4.720 µs
⚡ 187.95 µs 🐌 214.98 µs
check: fail
"""

# Extract all mean times with units (µs or ms)
# This regex captures the value and the unit separately
mean_times_with_units = re.findall(r'⏱\s*([\d.]+)\s*±.*?(µs|ms)', benchmark_text)

# Convert all times to microseconds based on their unit
times_in_microseconds = []
for value, unit in mean_times_with_units:
    time = float(value)
    if unit == "ms":
        time *= 1000  # convert ms to µs
    times_in_microseconds.append(time)

# Calculate geometric mean
geo_mean = geometric_mean(times_in_microseconds)

# Output
print("Collected mean times (µs):", times_in_microseconds)
print("Geometric mean (µs):", geo_mean)

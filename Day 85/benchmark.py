import re
from statistics import geometric_mean

benchmark_text = """
benchmark.0.spec: k: 7168; m: 1024; n: 1536; seed: 8135
⏱ 349.02 ± 4.372 µs
⚡ 346.44 µs 🐌 391.30 µs
benchmark.1.spec: k: 1536; m: 1024; n: 3072; seed: 6251
⏱ 147.07 ± 2.758 µs
⚡ 145.08 µs 🐌 172.37 µs
benchmark.2.spec: k: 7168; m: 1024; n: 576; seed: 12346
⏱ 351.54 ± 1.922 µs
⚡ 349.01 µs 🐌 358.57 µs
benchmark.3.spec: k: 256; m: 1024; n: 7168; seed: 5364
⏱ 58.33 ± 1.291 µs
⚡ 57.04 µs 🐌 67.66 µs
benchmark.4.spec: k: 2048; m: 1024; n: 7168; seed: 6132
⏱ 300.75 ± 5.922 µs
⚡ 296.18 µs 🐌 352.60 µs
benchmark.5.spec: k: 7168; m: 1024; n: 4608; seed: 7531
⏱ 699.50 ± 12.812 µs
⚡ 694.45 µs 🐌 822.45 µs
benchmark.6.spec: k: 2304; m: 1024; n: 7168; seed: 12345
⏱ 334.25 ± 6.173 µs
⚡ 330.70 µs 🐌 391.43 µs
benchmark.7.spec: k: 7168; m: 1024; n: 512; seed: 6563
⏱ 345.17 ± 1.642 µs
⚡ 343.42 µs 🐌 351.02 µs
benchmark.8.spec: k: 512; m: 1024; n: 4096; seed: 17512
⏱ 65.33 ± 1.381 µs
⚡ 64.03 µs 🐌 75.29 µs
benchmark.9.spec: k: 7168; m: 6144; n: 1536; seed: 6543
⏱ 1.37 ± 0.032 ms
⚡ 1.36 ms 🐌 1.68 ms
benchmark.10.spec: k: 1536; m: 6144; n: 3072; seed: 234
⏱ 615.67 ± 7.004 µs
⚡ 611.66 µs 🐌 678.30 µs
benchmark.11.spec: k: 7168; m: 6144; n: 576; seed: 9863
⏱ 690.71 ± 15.851 µs
⚡ 686.32 µs 🐌 845.98 µs
benchmark.12.spec: k: 256; m: 6144; n: 7168; seed: 764243
⏱ 268.79 ± 3.861 µs
⚡ 265.59 µs 🐌 297.94 µs
benchmark.13.spec: k: 2048; m: 6144; n: 7168; seed: 76547
⏱ 1.83 ± 0.025 ms
⚡ 1.81 ms 🐌 2.06 ms
benchmark.14.spec: k: 7168; m: 6144; n: 4608; seed: 65436
⏱ 4.21 ± 0.027 ms
⚡ 4.18 ms 🐌 4.36 ms
benchmark.15.spec: k: 2304; m: 6144; n: 7168; seed: 452345
⏱ 2.06 ± 0.028 ms
⚡ 2.04 ms 🐌 2.32 ms
benchmark.16.spec: k: 7168; m: 6144; n: 512; seed: 12341
⏱ 681.19 ± 11.662 µs
⚡ 677.45 µs 🐌 795.41 µs
benchmark.17.spec: k: 512; m: 6144; n: 4096; seed: 45245
⏱ 292.22 ± 3.780 µs
⚡ 289.43 µs 🐌 323.37 µs
check: pass
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

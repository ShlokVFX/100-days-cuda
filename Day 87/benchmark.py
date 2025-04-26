import re
from statistics import geometric_mean

benchmark_text = """
benchmark.0.spec: k: 7168; m: 1024; n: 1536; seed: 8135
⏱ 336.39 ± 10.02 µs
⚡ 320.407 µs 🐌 358.292 µs
benchmark.1.spec: k: 1536; m: 1024; n: 3072; seed: 6251
⏱ 79.18 ± 1.15 µs
⚡ 78.4180 µs 🐌 82.5870 µs
benchmark.2.spec: k: 7168; m: 1024; n: 576; seed: 12346
⏱ 328.02 ± 0.55 µs
⚡ 326.902 µs 🐌 328.866 µs
benchmark.3.spec: k: 256; m: 1024; n: 7168; seed: 5364
⏱ 30.96 ± 0.08 µs
⚡ 30.8300 µs 🐌 31.0710 µs
benchmark.4.spec: k: 2048; m: 1024; n: 7168; seed: 6132
⏱ 201.76 ± 0.97 µs
⚡ 200.696 µs 🐌 204.143 µs
benchmark.5.spec: k: 7168; m: 1024; n: 4608; seed: 7531
⏱ 354.99 ± 12.62 µs
⚡ 343.138 µs 🐌 387.077 µs
benchmark.6.spec: k: 2304; m: 1024; n: 7168; seed: 12345
⏱ 226.92 ± 2.23 µs
⚡ 223.828 µs 🐌 230.002 µs
benchmark.7.spec: k: 7168; m: 1024; n: 512; seed: 6563
⏱ 296.94 ± 1.25 µs
⚡ 294.990 µs 🐌 298.878 µs
benchmark.8.spec: k: 512; m: 1024; n: 4096; seed: 17512
⏱ 28.23 ± 0.13 µs
⚡ 27.9430 µs 🐌 28.4240 µs
benchmark.9.spec: k: 7168; m: 6144; n: 1536; seed: 6543
⏱ 688.80 ± 24.69 µs
⚡ 631.071 µs 🐌 728.172 µs
benchmark.10.spec: k: 1536; m: 6144; n: 3072; seed: 234
⏱ 310.10 ± 11.16 µs
⚡ 296.351 µs 🐌 325.980 µs
benchmark.11.spec: k: 7168; m: 6144; n: 576; seed: 9863
⏱ 332.37 ± 16.34 µs
⚡ 315.115 µs 🐌 369.156 µs
benchmark.12.spec: k: 256; m: 6144; n: 7168; seed: 764243
⏱ 126.20 ± 0.48 µs
⚡ 125.404 µs 🐌 126.847 µs
benchmark.13.spec: k: 2048; m: 6144; n: 7168; seed: 76547
⏱ 900.84 ± 32.96 µs
⚡ 856.504 µs 🐌 942.659 µs
benchmark.14.spec: k: 7168; m: 6144; n: 4608; seed: 65436
⏱ 2.00 ± 0.05 ms
⚡ 1.918 ms 🐌 2.079 ms
benchmark.15.spec: k: 2304; m: 6144; n: 7168; seed: 452345
⏱ 1.02 ± 0.04 ms
⚡ 978.461 µs 🐌 1.083 ms
benchmark.16.spec: k: 7168; m: 6144; n: 512; seed: 12341
⏱ 329.65 ± 18.10 µs
⚡ 309.662 µs 🐌 366.070 µs
benchmark.17.spec: k: 512; m: 6144; n: 4096; seed: 45245
⏱ 144.37 ± 1.75 µs
⚡ 142.443 µs 🐌 149.177 µs
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

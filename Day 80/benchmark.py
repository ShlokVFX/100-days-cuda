import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 803 ± 0.8 µs
 ⚡ 798 µs 🐌 813 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 229 ± 0.2 µs
 ⚡ 228 µs 🐌 238 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 690 ± 1.1 µs
 ⚡ 673 µs 🐌 728 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 83.0 ± 0.13 µs
 ⚡ 81.6 µs 🐌 92.1 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 520 ± 0.6 µs
 ⚡ 516 µs 🐌 567 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 1253 ± 1.3 µs
 ⚡ 1248 µs 🐌 1376 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 585 ± 1.0 µs
 ⚡ 580 µs 🐌 653 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 676 ± 0.7 µs
 ⚡ 661 µs 🐌 704 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 107 ± 0.1 µs
 ⚡ 106 µs 🐌 115 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 2.34 ± 0.003 ms
 ⚡ 2.31 ms 🐌 2.63 ms

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 966 ± 2.2 µs
 ⚡ 957 µs 🐌 1152 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 1034 ± 1.0 µs
 ⚡ 1028 µs 🐌 1114 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 380 ± 0.9 µs
 ⚡ 365 µs 🐌 418 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 2.87 ± 0.003 ms
 ⚡ 2.84 ms 🐌 3.06 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 6.49 ± 0.006 ms
 ⚡ 6.44 ms 🐌 6.74 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 3.23 ± 0.003 ms
 ⚡ 3.19 ms 🐌 3.31 ms

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 1030 ± 1.0 µs
 ⚡ 1026 µs 🐌 1098 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 435 ± 1.2 µs
 ⚡ 426 µs 🐌 482 µs
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

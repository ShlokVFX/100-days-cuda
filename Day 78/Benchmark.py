import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 3.42 ± 0.003 ms
 ⚡ 3.35 ms 🐌 3.61 ms

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 1375 ± 2.4 µs
 ⚡ 1330 µs 🐌 1469 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 1337 ± 4.3 µs
 ⚡ 1267 µs 🐌 1480 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 540 ± 1.8 µs
 ⚡ 503 µs 🐌 600 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 4.24 ± 0.004 ms
 ⚡ 4.20 ms 🐌 4.33 ms

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 9.91 ± 0.010 ms
 ⚡ 9.84 ms 🐌 10.0 ms

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 4.78 ± 0.005 ms
 ⚡ 4.73 ms 🐌 4.89 ms

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 1178 ± 3.8 µs
 ⚡ 1116 µs 🐌 1297 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 613 ± 1.5 µs
 ⚡ 581 µs 🐌 665 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 24.7 ± 0.02 ms
 ⚡ 24.6 ms 🐌 25.2 ms

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 9.12 ± 0.009 ms
 ⚡ 9.07 ms 🐌 9.40 ms

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 8.09 ± 0.008 ms
 ⚡ 8.02 ms 🐌 8.20 ms

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 3.19 ± 0.003 ms
 ⚡ 3.14 ms 🐌 3.39 ms

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 28.7 ± 0.03 ms
 ⚡ 28.6 ms 🐌 28.8 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 75.0 ± 0.06 ms
 ⚡ 74.9 ms 🐌 75.1 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 32.4 ± 0.03 ms
 ⚡ 32.3 ms 🐌 32.8 ms

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 7.99 ± 0.008 ms
 ⚡ 7.95 ms 🐌 8.03 ms

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 3.64 ± 0.004 ms
 ⚡ 3.59 ms 🐌 3.88 ms
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

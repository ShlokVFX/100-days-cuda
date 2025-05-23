import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 467 ± 0.5 µs
 ⚡ 460 µs 🐌 497 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 108 ± 0.2 µs
 ⚡ 106 µs 🐌 126 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 488 ± 0.8 µs
 ⚡ 463 µs 🐌 510 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 45.7 ± 0.10 µs
 ⚡ 44.8 µs 🐌 53.9 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 220 ± 0.7 µs
 ⚡ 210 µs 🐌 253 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 546 ± 3.1 µs
 ⚡ 506 µs 🐌 786 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 246 ± 0.9 µs
 ⚡ 234 µs 🐌 285 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 462 ± 1.2 µs
 ⚡ 448 µs 🐌 490 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 51.6 ± 0.12 µs
 ⚡ 50.1 µs 🐌 59.7 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 938 ± 1.7 µs
 ⚡ 907 µs 🐌 1005 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 391 ± 1.1 µs
 ⚡ 374 µs 🐌 437 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 523 ± 1.0 µs
 ⚡ 505 µs 🐌 605 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 177 ± 0.7 µs
 ⚡ 155 µs 🐌 194 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1222 ± 68.9 µs
 ⚡ 1093 µs 🐌 8.01 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.61 ± 0.004 ms
 ⚡ 2.53 ms 🐌 2.72 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1288 ± 3.0 µs
 ⚡ 1231 µs 🐌 1389 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 593 ± 93.4 µs
 ⚡ 489 µs 🐌 9.84 ms

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 179 ± 0.6 µs
 ⚡ 172 µs 🐌 215 µs
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

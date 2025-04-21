import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 467 ± 0.6 µs
 ⚡ 462 µs 🐌 512 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 108 ± 0.2 µs
 ⚡ 107 µs 🐌 129 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 487 ± 1.0 µs
 ⚡ 465 µs 🐌 509 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 47.4 ± 0.18 µs
 ⚡ 46.7 µs 🐌 64.7 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 220 ± 0.8 µs
 ⚡ 209 µs 🐌 259 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 547 ± 1.3 µs
 ⚡ 525 µs 🐌 651 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 245 ± 0.7 µs
 ⚡ 234 µs 🐌 284 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 463 ± 1.2 µs
 ⚡ 450 µs 🐌 493 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 52.4 ± 0.13 µs
 ⚡ 51.1 µs 🐌 63.7 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 955 ± 2.0 µs
 ⚡ 925 µs 🐌 1082 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 397 ± 1.2 µs
 ⚡ 379 µs 🐌 436 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 529 ± 1.1 µs
 ⚡ 513 µs 🐌 618 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 187 ± 0.7 µs
 ⚡ 165 µs 🐌 218 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1162 ± 2.5 µs
 ⚡ 1116 µs 🐌 1254 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.63 ± 0.003 ms
 ⚡ 2.56 ms 🐌 2.75 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1315 ± 5.2 µs
 ⚡ 1236 µs 🐌 1755 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 504 ± 1.2 µs
 ⚡ 493 µs 🐌 608 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 183 ± 0.8 µs
 ⚡ 175 µs 🐌 213 µs
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

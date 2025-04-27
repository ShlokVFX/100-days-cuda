import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 318 ± 0.3 µs
 ⚡ 316 µs 🐌 343 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 70.7 ± 0.22 µs
 ⚡ 69.2 µs 🐌 91.9 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 318 ± 0.3 µs
 ⚡ 316 µs 🐌 328 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 41.4 ± 0.13 µs
 ⚡ 40.4 µs 🐌 51.0 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 187 ± 0.4 µs
 ⚡ 185 µs 🐌 230 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 342 ± 0.9 µs
 ⚡ 336 µs 🐌 422 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 207 ± 0.5 µs
 ⚡ 204 µs 🐌 253 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 284 ± 0.4 µs
 ⚡ 282 µs 🐌 325 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 38.1 ± 0.14 µs
 ⚡ 36.9 µs 🐌 48.8 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 681 ± 2.5 µs
 ⚡ 659 µs 🐌 904 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 308 ± 0.5 µs
 ⚡ 302 µs 🐌 338 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 322 ± 0.8 µs
 ⚡ 317 µs 🐌 397 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 138 ± 0.4 µs
 ⚡ 133 µs 🐌 162 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 872 ± 1.2 µs
 ⚡ 854 µs 🐌 950 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1973 ± 2.9 µs
 ⚡ 1930 µs 🐌 2.22 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 982 ± 1.4 µs
 ⚡ 957 µs 🐌 1084 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 321 ± 0.8 µs
 ⚡ 318 µs 🐌 392 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 152 ± 0.4 µs
 ⚡ 145 µs 🐌 176 µs
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

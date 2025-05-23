import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 319 ± 0.4 µs
 ⚡ 317 µs 🐌 352 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 71.7 ± 0.22 µs
 ⚡ 70.5 µs 🐌 92.4 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 317 ± 0.3 µs
 ⚡ 316 µs 🐌 329 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 40.7 ± 0.15 µs
 ⚡ 39.5 µs 🐌 51.8 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 184 ± 0.5 µs
 ⚡ 181 µs 🐌 228 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 343 ± 1.1 µs
 ⚡ 331 µs 🐌 428 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 205 ± 0.6 µs
 ⚡ 202 µs 🐌 260 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 284 ± 0.4 µs
 ⚡ 281 µs 🐌 326 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 37.7 ± 0.15 µs
 ⚡ 36.2 µs 🐌 50.3 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 668 ± 2.3 µs
 ⚡ 655 µs 🐌 876 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 299 ± 0.5 µs
 ⚡ 294 µs 🐌 331 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 321 ± 0.9 µs
 ⚡ 317 µs 🐌 404 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 133 ± 0.4 µs
 ⚡ 129 µs 🐌 163 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 850 ± 1.6 µs
 ⚡ 832 µs 🐌 984 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1927 ± 1.9 µs
 ⚡ 1900 µs 🐌 2.04 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 953 ± 1.7 µs
 ⚡ 936 µs 🐌 1091 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 322 ± 0.7 µs
 ⚡ 319 µs 🐌 394 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 146 ± 0.4 µs
 ⚡ 141 µs 🐌 175 µs
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

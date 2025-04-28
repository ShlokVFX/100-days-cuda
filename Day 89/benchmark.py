import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 311 ± 0.4 µs
 ⚡ 310 µs 🐌 348 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 69.6 ± 0.24 µs
 ⚡ 68.6 µs 🐌 92.0 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 303 ± 0.3 µs
 ⚡ 301 µs 🐌 317 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 39.3 ± 0.15 µs
 ⚡ 38.1 µs 🐌 49.3 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 183 ± 0.4 µs
 ⚡ 179 µs 🐌 226 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 338 ± 1.3 µs
 ⚡ 329 µs 🐌 440 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 205 ± 1.0 µs
 ⚡ 202 µs 🐌 296 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 274 ± 0.5 µs
 ⚡ 273 µs 🐌 319 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 37.8 ± 0.16 µs
 ⚡ 36.1 µs 🐌 48.8 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 656 ± 2.1 µs
 ⚡ 640 µs 🐌 840 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 294 ± 0.4 µs
 ⚡ 289 µs 🐌 326 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 310 ± 0.8 µs
 ⚡ 306 µs 🐌 388 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 130 ± 0.3 µs
 ⚡ 126 µs 🐌 156 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 854 ± 1.4 µs
 ⚡ 836 µs 🐌 954 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1965 ± 2.0 µs
 ⚡ 1938 µs 🐌 2.05 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 961 ± 1.4 µs
 ⚡ 936 µs 🐌 1048 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 307 ± 0.8 µs
 ⚡ 305 µs 🐌 382 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 146 ± 0.3 µs
 ⚡ 141 µs 🐌 161 µs
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

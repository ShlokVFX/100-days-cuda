import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 409 ± 0.4 µs
 ⚡ 407 µs 🐌 448 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 109 ± 0.1 µs
 ⚡ 107 µs 🐌 117 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 410 ± 0.4 µs
 ⚡ 408 µs 🐌 421 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 54.0 ± 1.06 µs
 ⚡ 51.6 µs 🐌 158 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 263 ± 0.4 µs
 ⚡ 258 µs 🐌 296 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 496 ± 1.6 µs
 ⚡ 478 µs 🐌 624 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 292 ± 0.5 µs
 ⚡ 287 µs 🐌 330 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 403 ± 0.4 µs
 ⚡ 401 µs 🐌 414 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 52.9 ± 0.12 µs
 ⚡ 51.6 µs 🐌 60.7 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 989 ± 1.9 µs
 ⚡ 964 µs 🐌 1138 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 436 ± 0.9 µs
 ⚡ 407 µs 🐌 475 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 510 ± 1.1 µs
 ⚡ 491 µs 🐌 578 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 197 ± 0.5 µs
 ⚡ 192 µs 🐌 213 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1296 ± 1.7 µs
 ⚡ 1223 µs 🐌 1375 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.95 ± 0.003 ms
 ⚡ 2.91 ms 🐌 3.03 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1457 ± 1.6 µs
 ⚡ 1375 µs 🐌 1502 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 467 ± 0.8 µs
 ⚡ 450 µs 🐌 504 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 219 ± 0.4 µs
 ⚡ 213 µs 🐌 233 µs
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

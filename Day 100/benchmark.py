import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 158 ± 0.2 µs
 ⚡ 157 µs 🐌 168 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 50.2 ± 0.12 µs
 ⚡ 49.4 µs 🐌 56.7 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 153 ± 0.2 µs
 ⚡ 152 µs 🐌 162 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 34.2 ± 0.10 µs
 ⚡ 33.2 µs 🐌 41.2 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 121 ± 0.1 µs
 ⚡ 120 µs 🐌 132 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 202 ± 0.7 µs
 ⚡ 191 µs 🐌 235 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 134 ± 0.1 µs
 ⚡ 131 µs 🐌 142 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 153 ± 0.2 µs
 ⚡ 152 µs 🐌 160 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 32.4 ± 0.12 µs
 ⚡ 31.5 µs 🐌 39.5 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 392 ± 1.6 µs
 ⚡ 368 µs 🐌 447 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 188 ± 0.9 µs
 ⚡ 178 µs 🐌 233 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 186 ± 0.6 µs
 ⚡ 174 µs 🐌 209 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 107 ± 0.3 µs
 ⚡ 103 µs 🐌 119 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 539 ± 1.8 µs
 ⚡ 509 µs 🐌 595 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1137 ± 2.2 µs
 ⚡ 1091 µs 🐌 1211 µs

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 590 ± 1.8 µs
 ⚡ 546 µs 🐌 645 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 171 ± 0.5 µs
 ⚡ 161 µs 🐌 181 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 117 ± 0.2 µs
 ⚡ 112 µs 🐌 124 µs
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

import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 234 ± 0.2 µs
 ⚡ 233 µs 🐌 243 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 68.0 ± 0.13 µs
 ⚡ 67.1 µs 🐌 77.2 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 232 ± 0.2 µs
 ⚡ 230 µs 🐌 243 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 41.3 ± 0.11 µs
 ⚡ 40.2 µs 🐌 49.7 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 163 ± 0.2 µs
 ⚡ 159 µs 🐌 179 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 284 ± 0.6 µs
 ⚡ 270 µs 🐌 298 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 181 ± 0.2 µs
 ⚡ 177 µs 🐌 195 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 231 ± 0.2 µs
 ⚡ 230 µs 🐌 245 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 37.6 ± 0.10 µs
 ⚡ 36.5 µs 🐌 46.0 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 551 ± 1.4 µs
 ⚡ 534 µs 🐌 627 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 255 ± 0.5 µs
 ⚡ 243 µs 🐌 272 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 260 ± 0.5 µs
 ⚡ 249 µs 🐌 274 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 132 ± 0.3 µs
 ⚡ 129 µs 🐌 145 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 741 ± 1.6 µs
 ⚡ 687 µs 🐌 800 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1642 ± 2.4 µs
 ⚡ 1583 µs 🐌 1703 µs

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 829 ± 1.6 µs
 ⚡ 787 µs 🐌 887 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 243 ± 0.4 µs
 ⚡ 239 µs 🐌 257 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 147 ± 0.4 µs
 ⚡ 141 µs 🐌 159 µs
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

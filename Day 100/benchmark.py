import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 150 ± 0.2 µs
 ⚡ 149 µs 🐌 168 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 49.1 ± 0.12 µs
 ⚡ 48.0 µs 🐌 57.9 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 146 ± 0.2 µs
 ⚡ 145 µs 🐌 160 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 32.0 ± 0.16 µs
 ⚡ 31.2 µs 🐌 42.4 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 116 ± 0.2 µs
 ⚡ 113 µs 🐌 135 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 186 ± 0.7 µs
 ⚡ 177 µs 🐌 228 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 128 ± 0.5 µs
 ⚡ 125 µs 🐌 175 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 144 ± 0.2 µs
 ⚡ 143 µs 🐌 160 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 30.2 ± 0.17 µs
 ⚡ 29.1 µs 🐌 40.3 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 352 ± 0.9 µs
 ⚡ 336 µs 🐌 405 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 168 ± 0.7 µs
 ⚡ 162 µs 🐌 207 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 165 ± 0.4 µs
 ⚡ 159 µs 🐌 184 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 96.7 ± 0.41 µs
 ⚡ 91.5 µs 🐌 117 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 477 ± 1.1 µs
 ⚡ 454 µs 🐌 527 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1074 ± 1.5 µs
 ⚡ 1049 µs 🐌 1164 µs

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 529 ± 0.9 µs
 ⚡ 497 µs 🐌 562 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 157 ± 0.5 µs
 ⚡ 151 µs 🐌 185 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 105 ± 0.2 µs
 ⚡ 102 µs 🐌 116 µs
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

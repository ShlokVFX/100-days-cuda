import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 146 ± 0.2 µs
 ⚡ 144 µs 🐌 161 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 50.4 ± 0.14 µs
 ⚡ 48.3 µs 🐌 61.2 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 145 ± 0.2 µs
 ⚡ 144 µs 🐌 158 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 34.5 ± 0.13 µs
 ⚡ 33.5 µs 🐌 43.0 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 118 ± 0.2 µs
 ⚡ 116 µs 🐌 131 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 193 ± 0.9 µs
 ⚡ 182 µs 🐌 232 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 130 ± 0.3 µs
 ⚡ 127 µs 🐌 159 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 146 ± 0.1 µs
 ⚡ 145 µs 🐌 159 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 33.3 ± 0.09 µs
 ⚡ 32.4 µs 🐌 40.1 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 374 ± 1.2 µs
 ⚡ 351 µs 🐌 417 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 181 ± 1.0 µs
 ⚡ 172 µs 🐌 231 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 178 ± 0.6 µs
 ⚡ 170 µs 🐌 205 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 108 ± 0.2 µs
 ⚡ 104 µs 🐌 122 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 503 ± 1.5 µs
 ⚡ 481 µs 🐌 554 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 1106 ± 2.4 µs
 ⚡ 1051 µs 🐌 1167 µs

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 558 ± 1.5 µs
 ⚡ 530 µs 🐌 610 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 166 ± 0.5 µs
 ⚡ 161 µs 🐌 189 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 118 ± 0.1 µs
 ⚡ 116 µs 🐌 125 µs
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

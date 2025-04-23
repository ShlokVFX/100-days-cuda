import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 493 ± 0.6 µs
 ⚡ 490 µs 🐌 540 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 116 ± 0.2 µs
 ⚡ 114 µs 🐌 137 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 530 ± 1.8 µs
 ⚡ 484 µs 🐌 550 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 48.6 ± 0.14 µs
 ⚡ 47.3 µs 🐌 57.5 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 231 ± 0.9 µs
 ⚡ 217 µs 🐌 274 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 583 ± 1.4 µs
 ⚡ 551 µs 🐌 691 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 253 ± 0.8 µs
 ⚡ 243 µs 🐌 289 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 498 ± 2.3 µs
 ⚡ 478 µs 🐌 546 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 55.9 ± 0.15 µs
 ⚡ 54.3 µs 🐌 65.2 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 964 ± 1.8 µs
 ⚡ 938 µs 🐌 1081 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 409 ± 1.1 µs
 ⚡ 393 µs 🐌 458 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 560 ± 1.9 µs
 ⚡ 548 µs 🐌 738 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 196 ± 0.7 µs
 ⚡ 176 µs 🐌 221 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1211 ± 2.8 µs
 ⚡ 1164 µs 🐌 1289 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.83 ± 0.069 ms
 ⚡ 2.69 ms 🐌 9.66 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1366 ± 2.2 µs
 ⚡ 1316 µs 🐌 1429 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 533 ± 1.3 µs
 ⚡ 525 µs 🐌 656 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 191 ± 0.6 µs
 ⚡ 184 µs 🐌 225 µs
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

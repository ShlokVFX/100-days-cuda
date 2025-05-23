import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 421 ± 0.4 µs
 ⚡ 417 µs 🐌 458 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 102 ± 0.1 µs
 ⚡ 101 µs 🐌 115 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 423 ± 0.4 µs
 ⚡ 421 µs 🐌 433 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 53.8 ± 0.14 µs
 ⚡ 52.7 µs 🐌 64.4 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 256 ± 0.5 µs
 ⚡ 250 µs 🐌 300 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 467 ± 1.3 µs
 ⚡ 451 µs 🐌 568 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 288 ± 0.5 µs
 ⚡ 279 µs 🐌 325 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 397 ± 0.4 µs
 ⚡ 395 µs 🐌 422 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 48.5 ± 0.13 µs
 ⚡ 47.6 µs 🐌 59.2 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 928 ± 2.3 µs
 ⚡ 902 µs 🐌 1135 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 408 ± 0.6 µs
 ⚡ 398 µs 🐌 437 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 443 ± 0.8 µs
 ⚡ 428 µs 🐌 504 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 190 ± 0.7 µs
 ⚡ 182 µs 🐌 229 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1213 ± 1.9 µs
 ⚡ 1151 µs 🐌 1301 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.78 ± 0.003 ms
 ⚡ 2.73 ms 🐌 2.96 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1363 ± 1.7 µs
 ⚡ 1286 µs 🐌 1450 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 421 ± 0.9 µs
 ⚡ 418 µs 🐌 507 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 214 ± 0.4 µs
 ⚡ 209 µs 🐌 225 µs
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

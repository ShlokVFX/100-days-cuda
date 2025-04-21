import re
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 418 ± 0.4 µs
 ⚡ 415 µs 🐌 443 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 99.1 ± 0.12 µs
 ⚡ 98.5 µs 🐌 110 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 420 ± 0.4 µs
 ⚡ 419 µs 🐌 428 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 51.0 ± 0.11 µs
 ⚡ 50.0 µs 🐌 57.3 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 255 ± 0.4 µs
 ⚡ 249 µs 🐌 282 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 468 ± 0.8 µs
 ⚡ 442 µs 🐌 509 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 286 ± 0.7 µs
 ⚡ 278 µs 🐌 330 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 394 ± 0.4 µs
 ⚡ 392 µs 🐌 417 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 45.9 ± 0.11 µs
 ⚡ 45.1 µs 🐌 54.4 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 946 ± 2.8 µs
 ⚡ 912 µs 🐌 1169 µs

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 413 ± 0.6 µs
 ⚡ 396 µs 🐌 428 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 451 ± 0.6 µs
 ⚡ 436 µs 🐌 479 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 183 ± 0.3 µs
 ⚡ 179 µs 🐌 199 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 1236 ± 6.2 µs
 ⚡ 1169 µs 🐌 1802 µs

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 2.80 ± 0.003 ms
 ⚡ 2.77 ms 🐌 2.84 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 1386 ± 3.0 µs
 ⚡ 1338 µs 🐌 1619 µs

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 420 ± 0.6 µs
 ⚡ 415 µs 🐌 466 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 212 ± 0.5 µs
 ⚡ 206 µs 🐌 229 µs
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

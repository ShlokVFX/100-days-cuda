import re
import math
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 706 ± 2.3 µs
 ⚡ 702 µs 🐌 709 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 265 ± 2.3 µs
 ⚡ 261 µs 🐌 274 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 365 ± 1.6 µs
 ⚡ 362 µs 🐌 367 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 187 ± 1.8 µs
 ⚡ 182 µs 🐌 213 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 595 ± 5.6 µs
 ⚡ 583 µs 🐌 631 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 1460 ± 13.2 µs
 ⚡ 1429 µs 🐌 1531 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 664 ± 6.5 µs
 ⚡ 640 µs 🐌 687 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 339 ± 3.3 µs
 ⚡ 331 µs 🐌 394 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 179 ± 1.7 µs
 ⚡ 175 µs 🐌 192 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 2.37 ± 0.014 ms
 ⚡ 2.34 ms 🐌 2.39 ms

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 955 ± 5.7 µs
 ⚡ 944 µs 🐌 963 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 1452 ± 13.8 µs
 ⚡ 1434 µs 🐌 1479 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 490 ± 4.6 µs
 ⚡ 480 µs 🐌 511 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 2.45 ± 0.023 ms
 ⚡ 2.34 ms 🐌 2.59 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 5.89 ± 0.057 ms
 ⚡ 5.62 ms 🐌 6.37 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 2.87 ± 0.026 ms
 ⚡ 2.84 ms 🐌 2.94 ms

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 1259 ± 10.6 µs
 ⚡ 1246 µs 🐌 1280 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 492 ± 4.9 µs
 ⚡ 481 µs 🐌 509 µs
"""

# Extract all mean times after ⏱ using regex
mean_times_microseconds = re.findall(r'⏱\s*([\d.]+)\s*±', benchmark_text)

# Convert all times to float (and ms to µs where needed)
times_in_microseconds = []
for time in mean_times_microseconds:
    value = float(time)
    if value < 10:  # assume it's in milliseconds if it's very small
        value *= 1000  # convert ms to µs
    times_in_microseconds.append(value)

# Calculate geometric mean
geo_mean = geometric_mean(times_in_microseconds)

# Output
print("Collected mean times (µs):", times_in_microseconds)
print("Geometric mean (µs):", geo_mean)

import re
import math
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 2.79 ± 0.003 ms
 ⚡ 2.78 ms 🐌 2.85 ms

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 588 ± 0.6 µs
 ⚡ 585 µs 🐌 629 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 2.78 ± 0.002 ms
 ⚡ 2.77 ms 🐌 2.79 ms

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 203 ± 0.3 µs
 ⚡ 198 µs 🐌 214 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 1467 ± 2.2 µs
 ⚡ 1431 µs 🐌 1524 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 3.14 ± 0.003 ms
 ⚡ 3.09 ms 🐌 3.37 ms

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 1648 ± 2.0 µs
 ⚡ 1618 µs 🐌 1701 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 2.62 ± 0.003 ms
 ⚡ 2.61 ms 🐌 2.76 ms

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 223 ± 0.6 µs
 ⚡ 213 µs 🐌 239 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 5.81 ± 0.006 ms
 ⚡ 5.66 ms 🐌 5.90 ms

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 2.38 ± 0.003 ms
 ⚡ 2.32 ms 🐌 2.46 ms

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 2.95 ± 0.003 ms
 ⚡ 2.90 ms 🐌 3.03 ms

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 907 ± 1.7 µs
 ⚡ 858 µs 🐌 974 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 7.11 ± 0.007 ms
 ⚡ 7.04 ms 🐌 7.25 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 16.9 ± 0.02 ms
 ⚡ 16.7 ms 🐌 17.1 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 7.98 ± 0.008 ms
 ⚡ 7.92 ms 🐌 8.19 ms

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 2.79 ± 0.003 ms
 ⚡ 2.78 ms 🐌 2.99 ms

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 1128 ± 2.8 µs
 ⚡ 1083 µs 🐌 1191 µs
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

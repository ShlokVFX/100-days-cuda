import re
import math
from statistics import geometric_mean

benchmark_text = """
k: 7168; m: 1024; n: 1536; seed: 8135
 ⏱ 705 ± 0.7 µs
 ⚡ 700 µs 🐌 741 µs

k: 1536; m: 1024; n: 3072; seed: 6251
 ⏱ 277 ± 0.5 µs
 ⚡ 269 µs 🐌 317 µs

k: 7168; m: 1024; n: 576; seed: 12346
 ⏱ 369 ± 0.6 µs
 ⚡ 365 µs 🐌 401 µs

k: 256; m: 1024; n: 7168; seed: 5364
 ⏱ 207 ± 0.6 µs
 ⚡ 202 µs 🐌 243 µs

k: 2048; m: 1024; n: 7168; seed: 6132
 ⏱ 620 ± 0.7 µs
 ⚡ 601 µs 🐌 660 µs

k: 7168; m: 1024; n: 4608; seed: 7531
 ⏱ 1472 ± 4.6 µs
 ⚡ 1424 µs 🐌 1892 µs

k: 2304; m: 1024; n: 7168; seed: 12345
 ⏱ 666 ± 1.0 µs
 ⚡ 649 µs 🐌 705 µs

k: 7168; m: 1024; n: 512; seed: 6563
 ⏱ 341 ± 0.8 µs
 ⚡ 334 µs 🐌 371 µs

k: 512; m: 1024; n: 4096; seed: 17512
 ⏱ 194 ± 0.4 µs
 ⚡ 191 µs 🐌 231 µs

k: 7168; m: 6144; n: 1536; seed: 6543
 ⏱ 2.44 ± 0.070 ms
 ⚡ 2.30 ms 🐌 9.39 ms

k: 1536; m: 6144; n: 3072; seed: 234
 ⏱ 983 ± 1.5 µs
 ⚡ 943 µs 🐌 1025 µs

k: 7168; m: 6144; n: 576; seed: 9863
 ⏱ 1436 ± 2.0 µs
 ⚡ 1398 µs 🐌 1486 µs

k: 256; m: 6144; n: 7168; seed: 764243
 ⏱ 497 ± 1.4 µs
 ⚡ 468 µs 🐌 556 µs

k: 2048; m: 6144; n: 7168; seed: 76547
 ⏱ 2.64 ± 0.099 ms
 ⚡ 2.41 ms 🐌 12.4 ms

k: 7168; m: 6144; n: 4608; seed: 65436
 ⏱ 6.02 ± 0.097 ms
 ⚡ 5.63 ms 🐌 15.3 ms

k: 2304; m: 6144; n: 7168; seed: 452345
 ⏱ 2.89 ± 0.004 ms
 ⚡ 2.80 ms 🐌 2.98 ms

k: 7168; m: 6144; n: 512; seed: 12341
 ⏱ 1283 ± 3.1 µs
 ⚡ 1225 µs 🐌 1356 µs

k: 512; m: 6144; n: 4096; seed: 45245
 ⏱ 503 ± 0.9 µs
 ⚡ 489 µs 🐌 546 µs
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

import re
from statistics import geometric_mean

benchmark_text = """
benchmark.0.spec: k: 7168; m: 1024; n: 1536; seed: 8135
⏱ 317.88 ± 2.695 µs
⚡ 314.74 µs 🐌 333.22 µs
benchmark.1.spec: k: 1536; m: 1024; n: 3072; seed: 6251
⏱ 71.11 ± 2.071 µs
⚡ 69.50 µs 🐌 89.23 µs
benchmark.2.spec: k: 7168; m: 1024; n: 576; seed: 12346
⏱ 321.32 ± 1.552 µs
⚡ 319.09 µs 🐌 327.43 µs
benchmark.3.spec: k: 256; m: 1024; n: 7168; seed: 5364
⏱ 40.98 ± 1.098 µs
⚡ 39.93 µs 🐌 48.60 µs
benchmark.4.spec: k: 2048; m: 1024; n: 7168; seed: 6132
⏱ 184.78 ± 3.952 µs
⚡ 181.55 µs 🐌 220.23 µs
benchmark.5.spec: k: 7168; m: 1024; n: 4608; seed: 7531
⏱ 339.92 ± 6.171 µs
⚡ 333.10 µs 🐌 389.15 µs
benchmark.6.spec: k: 2304; m: 1024; n: 7168; seed: 12345
⏱ 205.09 ± 4.319 µs
⚡ 200.54 µs 🐌 243.54 µs
benchmark.7.spec: k: 7168; m: 1024; n: 512; seed: 6563
⏱ 287.36 ± 4.182 µs
⚡ 283.97 µs 🐌 326.79 µs
benchmark.8.spec: k: 512; m: 1024; n: 4096; seed: 17512
⏱ 37.34 ± 1.103 µs
⚡ 36.16 µs 🐌 45.66 µs
benchmark.9.spec: k: 7168; m: 6144; n: 1536; seed: 6543
⏱ 676.41 ± 19.578 µs
⚡ 656.71 µs 🐌 837.81 µs
benchmark.10.spec: k: 1536; m: 6144; n: 3072; seed: 234
⏱ 297.14 ± 3.022 µs
⚡ 292.56 µs 🐌 315.50 µs
benchmark.11.spec: k: 7168; m: 6144; n: 576; seed: 9863
⏱ 322.64 ± 6.415 µs
⚡ 316.19 µs 🐌 377.88 µs
benchmark.12.spec: k: 256; m: 6144; n: 7168; seed: 764243
⏱ 137.75 ± 2.818 µs
⚡ 134.02 µs 🐌 157.63 µs
benchmark.13.spec: k: 2048; m: 6144; n: 7168; seed: 76547
⏱ 877.97 ± 17.166 µs
⚡ 851.80 µs 🐌 1013.80 µs
benchmark.14.spec: k: 7168; m: 6144; n: 4608; seed: 65436
⏱ 1.99 ± 0.017 ms
⚡ 1.95 ms 🐌 2.03 ms
benchmark.15.spec: k: 2304; m: 6144; n: 7168; seed: 452345
⏱ 979.14 ± 17.032 µs
⚡ 943.63 µs 🐌 1118.95 µs
benchmark.16.spec: k: 7168; m: 6144; n: 512; seed: 12341
⏱ 319.07 ± 4.999 µs
⚡ 316.69 µs 🐌 366.43 µs
benchmark.17.spec: k: 512; m: 6144; n: 4096; seed: 45245
⏱ 148.21 ± 2.697 µs
⚡ 144.82 µs 🐌 170.51 µs
check: pass
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

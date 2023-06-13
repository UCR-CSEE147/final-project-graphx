import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: python input.py <num_points> <min_val> <max_val>")
    sys.exit(1)

try:
    num_points = int(sys.argv[1])
    min_val = int(sys.argv[2])
    max_val = int(sys.argv[3])
except (ValueError, IndexError):
    print("Invalid arguments")
    sys.exit(1)

points = np.random.randint(min_val, max_val + 1, (num_points, 2))

with open("input.txt", "w") as file:
    file.write(f"{num_points}\n")
    for point in points:
        file.write(f"{point[0]}\t{point[1]}\n")

print(f"Generated {num_points} unique points and wrote to input.txt")
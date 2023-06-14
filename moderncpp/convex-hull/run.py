import sys
import subprocess

if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python3 run.py <num_points> <min_val> <max_val> [plot]")
    sys.exit(1)

try:
    num_points = int(sys.argv[1])
    min_val = int(sys.argv[2])
    max_val = int(sys.argv[3])
except (ValueError, IndexError):
    print("Invalid arguments")
    sys.exit(1)

subprocess.run(["python3", "input.py", str(num_points), str(min_val), str(max_val)])

subprocess.run(["./moderncpp", "input.txt"])

# specify "plot" as the 5th argument to plot the points and convex hull else leave it blank for no plot
if len(sys.argv) == 5 and sys.argv[4].lower() == "plot":
    subprocess.run(["python3", "plot.py"])
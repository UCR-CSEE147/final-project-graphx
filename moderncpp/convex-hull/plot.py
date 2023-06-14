import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

with open("input.txt", "r") as file:
    lines = file.readlines()

num_points = int(lines[0])
coordinates = [list(map(int, line.split())) for line in lines[1:]]
x = [point[0] for point in coordinates]
y = [point[1] for point in coordinates]

with open('output.txt', 'r') as file:
    hull_lines = file.readlines()

hull_coordinates = [list(map(int, line.split())) for line in hull_lines]

# Calculate the angle of each point with respect to the first point
reference_point = min(hull_coordinates, key=lambda p:(p[1], p[0]))
hull_coordinates = sorted(hull_coordinates, key=lambda p: math.atan2(p[1] - reference_point[1], p[0] - reference_point[0]))

fig, ax = plt.subplots()
ax.scatter(x, y, color='black', label='Points')

# Create a polygon from the hull coordinates
hull_polygon = Polygon(hull_coordinates, closed=True, edgecolor='grey', facecolor='none')
ax.add_patch(hull_polygon)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Convex Hull of {num_points} Points')
plt.savefig('plot.png')
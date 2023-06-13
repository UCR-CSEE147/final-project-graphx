#include <stdio.h>
#include <stdlib.h>

#include "support.h"

#define BLOCK_SIZE 256

typedef struct {
    int x;
    int y;
} Point;

__device__ int side(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

__global__ void convexHullKernel(Point* points, Point* result, int n) {
    if (n < 3) return;

    int l = 0;
    for (int i = 1; i < n; i++)
        if (points[i].x < points[l].x)
            l = i;

    int p = l, q;
    int resultSize = 0;
    do {
        result[resultSize++] = points[p];
        q = (p + 1) % n;
        for (int i = 0; i < n; i++) {
            if (side(points[p], points[i], points[q]) == 2)
               q = i;
        }
        p = q;
    } while (p != l);
}

void convexHull(Point* points, Point* result, int n)
{
    dim3 dimGrid((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    convexHullKernel<<<dimGrid, dimBlock>>>(points, result, n);
}
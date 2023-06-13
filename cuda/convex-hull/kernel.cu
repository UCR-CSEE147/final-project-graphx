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

__global__ void convexHullKernel(Point* points, Point* result, int n) 
{
    
}

void convexHull(Point* points, Point* result, int n)
{
    dim3 dimGrid((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    convexHullKernel<<<dimGrid, dimBlock>>>(points, result, n);
}
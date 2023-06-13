#include <stdio.h>
#include <stdlib.h>

#include "support.h"

#define BLOCK_SIZE 256

struct Point {
    int x, y;
};

__global__ void findLeftMost(Point* points, int size, int* leftMostIndex)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = threadIdx.x;
    if (index >= size) return;

    // Shared memory for each block
    __shared__ int localIndex[BLOCK_SIZE];
    __shared__ float localX[BLOCK_SIZE];
    __shared__ float localY[BLOCK_SIZE];

    // Load points into shared memory
    localIndex[t] = index;
    localX[t] = points[index].x;
    localY[t] = points[index].y; // tie breaker
    __syncthreads();

    // Perform reduction within each block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (t < stride && index + stride < size)
        {
            if (localX[t] > localX[t + stride] || 
                (localX[t] == localX[t + stride] && localY[t] > localY[t + stride]))
            {
                localX[t] = localX[t + stride];
                localY[t] = localY[t + stride];
                localIndex[t] = localIndex[t + stride];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (t == 0)
    {
        int oldIndex, newIndex;
        do {
            oldIndex = atomicAdd(leftMostIndex, 0);
            if (localX[0] < points[oldIndex].x || 
                (localX[0] == points[oldIndex].x && localY[0] < points[oldIndex].y))
                newIndex = localIndex[0];
            else
                newIndex = oldIndex;
        } while (atomicCAS(leftMostIndex, oldIndex, newIndex) != oldIndex); // compare and swap
    }
}


int side(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;
    return (val > 0) ? 1 : -1;
}

void giftWrapping(Point* points, int size, Point* hull, int* hullSize)
{
    if (size < 3)
    {
        hull = points;
        *hullSize = size;
        return;
    }

    /*
    int leftMost = 0;
    for (int i = 1; i < size; i++)
        if (points[i].x < points[leftMost].x)
            leftMost = i;
    */
    
    Point* d_points;
    cudaMalloc((void**) &d_points, size * sizeof(Point));
    cudaMemcpy(d_points, points, size * sizeof(Point), cudaMemcpyHostToDevice);

    int* d_leftMostIndex;
    cudaMalloc((void**) &d_leftMostIndex, sizeof(int));

    dim3 dim_block(BLOCK_SIZE);
    dim3 dim_grid((size - 1) / BLOCK_SIZE + 1);
    Timer timer;
    startTime(&timer);
    findLeftMost<<<dim_grid, dim_block>>>(d_points, size, d_leftMostIndex);
    stopTime(&timer);
    printf("Elapsed Time: %f s\n", elapsedTime(timer));
    cudaDeviceSynchronize();

    int leftMost;
    cudaMemcpy(&leftMost, d_leftMostIndex, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_leftMostIndex);
    
    printf("Leftmost: %d\n", leftMost);

    startTime(&timer);
    int p = leftMost, q;
    do
    {
        hull[*hullSize] = points[p];
        q = (p + 1) % size;
        for (int i = 0; i < size; i++)
        {
            if (side(points[p], points[i], points[q]) == -1)
                q = i;
        }
        p = q;
        (*hullSize)++;
    } while (p != leftMost);
    stopTime(&timer);
    printf("Elapsed Time: %f s\n", elapsedTime(timer));
}

int main(int argc, char** argv)
{
    // Timer timer;

    if (argc != 2)
    {
        printf("Usage: ./cuda <input file>\n");
        exit(1);
    }

    FILE* fin = fopen(argv[1], "r");

    int size;
    fscanf(fin, "%d", &size);

    Point* points = (Point*)malloc(size * sizeof(Point));
    for (int i = 0; i < size; i++)
        fscanf(fin, "%d %d", &points[i].x, &points[i].y);

    fclose(fin);

    Point* hull = (Point*)malloc(size * sizeof(Point));
    int hullSize = 0;

    //startTime(&timer);
    giftWrapping(points, size, hull, &hullSize);
    //stopTime(&timer);

    FILE* fout = fopen("output.txt", "w");
    //fprintf(fout, "%d\n", hullSize);
    for (int i = 0; i < hullSize; i++)
        fprintf(fout, "%d %d\n", hull[i].x, hull[i].y);
    
    fclose(fout);

    //printf("Elapsed Time: %f s\n", elapsedTime(timer));

    free(points);
    free(hull);

    return 0;
}
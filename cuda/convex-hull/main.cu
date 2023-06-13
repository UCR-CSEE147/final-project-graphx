#include <stdio.h>
#include <stdlib.h>

#include "kernel.cu"
#include "support.h"

int main(int argc, char** argv)
{
    Timer timer;
    cudaError_t cuda_ret;

    // Check if argument is given
    if (argc != 2)
    {
        printf("Usage: %s <input file>\n", argv[0]);
        exit(0);
    }

    // Open input file from argument
    FILE* fin = fopen(argv[1], "r");
    if (fin == NULL)
    {
        printf("Error opening file %s\n", argv[1]);
        exit(1);
    }

    int n = 0;
    // Read number of points in file
    fscanf(fin, "%d", &n);

    // Declare host points array and read in points from input file
    //Point* h_points = (Point*) malloc( sizeof(Point) * n );

    Point *points, *result;
    cudaMallocManaged(&points, sizeof(Point) * n);
    cudaMallocManaged(&result, sizeof(Point) * n);

    for (unsigned int i = 0; i < n; i++) { fscanf(fin, "%d\t%d", &points[i].x, &points[i].y); }

    // Close input file
    fclose(fin);

    // Start timer
    startTime(&timer);
    convexHull(points, result, n);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(cuda_ret));
    stopTime(&timer);

    // Print time elapsed and write points to output file
    printf("Time elapsed: %f s\n", elapsedTime(timer));
    FILE* fout = fopen("hull.txt", "w");
    fprintf(fout, "%d\n", n);
    for (unsigned int i = 0; i < n; i++) { fprintf(fout, "%d\t%d\n", result[i].x, result[i].y); }
    fclose(fout);

    // Free memory
    //free(h_points);
    cudaFree(points);
    cudaFree(result);

    return 0;
}
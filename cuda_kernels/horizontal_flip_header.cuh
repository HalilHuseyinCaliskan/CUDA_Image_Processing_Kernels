#ifndef HORIZONTAL_FLIP
#define HORIZONTAL_FLIP

#include <cuda_runtime.h>

__global__ void horizontal_flip(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int idx = y_index * width + x_index;
        int idx_flip = y_index * width + (width - x_index -1);

        output_matrix[idx] = input_matrix[idx_flip];

    }

}

#endif

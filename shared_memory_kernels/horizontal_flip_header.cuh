#ifndef HORIZONTAL_FLIP
#define HORIZONTAL_FLIP

#include <cuda_runtime.h>
#define THREAD_SIZE 16

__global__ void horizontal_flip(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    __shared__ unsigned char shared_matrix[THREAD_SIZE * THREAD_SIZE];

    int shared_index_x = threadIdx.x;
    int shared_index_y = threadIdx.y;

    int x_index = threadIdx.x + THREAD_SIZE * blockIdx.x;
    int y_index = threadIdx.y + THREAD_SIZE * blockIdx.y;

    if (x_index < width && y_index < height){

        shared_matrix[shared_index_y * THREAD_SIZE + shared_index_x] = input_matrix[y_index * width + x_index];

    }

    __syncthreads();

    if (x_index < width && y_index < height){

        int idx = y_index * width + (width -x_index -1);
        int idx_flip = shared_index_y * THREAD_SIZE + shared_index_x;

        output_matrix[idx] = shared_matrix[idx_flip];

    }

}

#endif

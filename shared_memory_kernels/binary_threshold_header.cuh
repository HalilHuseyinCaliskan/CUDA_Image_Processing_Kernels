#ifndef BINARY_THRESHOLD
#define BINARY_THRESHOLD
#define THREAD_SIZE 16
#include <cuda_runtime.h>

__global__ void binary_threshold(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height, unsigned char threshold_value, unsigned char target_value){

    __shared__ unsigned char shared_matrix[THREAD_SIZE * THREAD_SIZE];

    int shared_id_x = threadIdx.x;
    int shared_id_y = threadIdx.y;

    int x_index = threadIdx.x + THREAD_SIZE * blockIdx.x;
    int y_index = threadIdx.y + THREAD_SIZE * blockIdx.y;

    int total_index = y_index * width + x_index;

    if (x_index < width && y_index < height){

        shared_matrix[shared_id_y * THREAD_SIZE + shared_id_x] = input_matrix[total_index];

    }

    __syncthreads();

    if (x_index < width && y_index < height){
        
        output_matrix[total_index] = shared_matrix[shared_id_y * THREAD_SIZE + shared_id_x] > threshold_value ? target_value : 0 ;

    }

}

#endif

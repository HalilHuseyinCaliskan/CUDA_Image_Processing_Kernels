#ifndef BINARY_THRESHOLD
#define BINARY_THRESHOLD

#include <cuda_runtime.h>

__global__ void binary_threshold(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height, unsigned char threshold_value, unsigned char target_value){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = y_index * width + x_index;

    if (x_index < width && y_index < height){
        
        output_matrix[total_index] = input_matrix[total_index] > threshold_value ? target_value : 0 ;

    }

}

#endif

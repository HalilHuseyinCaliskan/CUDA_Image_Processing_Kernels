#ifndef BGR_TO_GRAY
#define BGR_TO_GRAY

#include <cuda_runtime.h>
#define THREAD_SIZE 16

__global__ void bgr_to_gray(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    __shared__ unsigned char shared_matrix[THREAD_SIZE * THREAD_SIZE * 3];

    int shared_id_x = threadIdx.x;
    int shared_id_y = threadIdx.y;

    int x_index = threadIdx.x + THREAD_SIZE * blockIdx.x;
    int y_index = threadIdx.y + THREAD_SIZE * blockIdx.y;

    int total_index = y_index * width + x_index;

    if (x_index < width && y_index < height){

        int bgr_index = 3 * total_index;
        int shared_id = (shared_id_y * THREAD_SIZE + shared_id_x) * 3;

        shared_matrix[shared_id] = input_matrix[bgr_index]; 
        shared_matrix[shared_id + 1] = input_matrix[bgr_index + 1]; 
        shared_matrix[shared_id + 2] = input_matrix[bgr_index + 2]; 

    }

    __syncthreads();

    if (x_index < width && y_index < height){

        int shared_id = (shared_id_y * THREAD_SIZE + shared_id_x) * 3;

        unsigned char blue = shared_matrix[shared_id]; 
        unsigned char green = shared_matrix[shared_id + 1]; 
        unsigned char red = shared_matrix[shared_id + 2]; 
        
        unsigned char gray_value = (unsigned char)(0.114f * blue + 0.587f * green + 0.299f * red);

        output_matrix[total_index] = gray_value;

        

    }

}

#endif

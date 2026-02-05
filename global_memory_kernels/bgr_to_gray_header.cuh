#ifndef BGR_TO_GRAY
#define BGR_TO_GRAY

#include <cuda_runtime.h>

__global__ void bgr_to_gray(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = y_index * width + x_index;

    if (x_index < width && y_index < height){

        int bgr_index = 3 * total_index;
        unsigned char blue = input_matrix[bgr_index];
        unsigned char green = input_matrix[bgr_index + 1];
        unsigned char red = input_matrix[bgr_index + 2];

        unsigned char gray_value = (unsigned char)(0.114f * blue + 0.587f * green + 0.299f * red);

        output_matrix[total_index] = gray_value;

    }

}

#endif

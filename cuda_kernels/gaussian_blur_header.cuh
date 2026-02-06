#ifndef GAUSSIAN_BLUR
#define GAUSSIAN_BLUR

#include <cuda_runtime.h>

__global__ void gaussian_blur(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = y_index * width + x_index;

    unsigned char kernel[3][3] = {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };
    
    int w = 0;
    int weightde_sum = 0;
    int sum = 0;

    if (x_index < width && y_index < height){

        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <= 1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_X = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                weightde_sum += input_matrix[control_y * width + control_X] * kernel[i + 1][j + 1];
                sum += kernel[i + 1][j + 1];
            }
        }

        output_matrix[total_index] = (unsigned char)(weightde_sum/sum);

    }

}

#endif

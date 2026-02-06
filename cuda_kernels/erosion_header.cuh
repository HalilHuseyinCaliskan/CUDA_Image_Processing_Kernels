#ifndef EROSION
#define EROSION

#include <cuda_runtime.h>

__global__ void erosion(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = y_index * width + x_index;

    int control = 255;

    if (x_index < width && y_index < height){

        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <=1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_x = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                int current_value = input_matrix[control_y * width + control_x];

                if (current_value < control){

                    control = current_value;
                    break;

                }

            }
        }

        output_matrix[total_index] = control;

    }

}


#endif

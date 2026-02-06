#ifndef MEDIAN_BLUR
#define MEDIAN_BLUR

#include <cuda_runtime.h>

__device__ void bubble_sort(unsigned char dizi[], int length){

    for(int i = 0; i < length; i++){
        for(int j = i + 1; j < length; j++){
            unsigned char temp = dizi[i];
            unsigned char value = dizi[j];
            if (value < temp){
                dizi[j] = temp;
                dizi[i] = value;
            }
        }
    }
}

__global__ void median_blur(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = y_index * width + x_index;

    unsigned char dizi[9];
    int control = 0;
    int length = 9;

    if (x_index < width && y_index < height){

        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <= 1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_X = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                dizi[control++] = input_matrix[control_y * width + control_X];
            }
        }
        bubble_sort(dizi,length);

        output_matrix[total_index] = dizi[4];

    }

}

#endif

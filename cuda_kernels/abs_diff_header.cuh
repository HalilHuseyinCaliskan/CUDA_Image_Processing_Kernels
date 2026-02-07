#ifndef ABS_DIFF
#define ABS_DIFF

__global__ void abs_diff(unsigned char *input_matrix1, unsigned char *input_matrix2, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        output_matrix[total_index] = abs(input_matrix1[total_index] - input_matrix2[total_index]);

    }

}

#endif

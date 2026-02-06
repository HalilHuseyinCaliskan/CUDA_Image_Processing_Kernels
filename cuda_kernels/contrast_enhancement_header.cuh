#ifndef CONTRSAT_ENHANCEMENT
#define CONTRSAT_ENHANCEMENT

__global__ void contrast_enhancement(unsigned char *input_matrix, int width, int height, float alpha, int beta){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;
        float pixel_value = input_matrix[total_index] * alpha + (float)beta;
        float value = fminf(fmaxf(pixel_value,0.0f),255.0f);
        input_matrix[total_index] = (unsigned char)value;

    }

}

#endif

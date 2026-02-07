contour_detection_header.cuh#ifndef CONTOUR_DETECTION
#define CONTOUR_DETECTION

__global__ void contour_detection(unsigned char *input_matrix, unsigned char *binary_matrix,int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        int value = binary_matrix[total_index];

        if(value > 0){
            int bgr_index = total_index * 3;
            input_matrix[bgr_index] = 0;
            input_matrix[bgr_index + 1] = 0;
            input_matrix[bgr_index + 2] = 255;
        }

    }

} 

#endif

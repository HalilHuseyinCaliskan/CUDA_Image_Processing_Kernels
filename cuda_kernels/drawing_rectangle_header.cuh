#ifndef DRAWING_RECTANGLE
#define DRAWING_RECTANGLE

__global__ void drawing_rectangle(unsigned char *input_matrix, int width, int height, int left, int right, int top, int bottom, unsigned char value){

    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int index_y = threadIdx.y + blockDim.y * blockIdx.y;

    int total_index = index_y * width + index_x;

    if (index_x < width && index_y < height){

        bool left_control = (index_x == left) && ((index_y <= bottom) && (index_y >= top));
        bool right_control = (index_x == right) && ((index_y <= bottom) && (index_y >= top));
        bool top_control = (index_y == top) && ((index_x <= right) && (index_x >= left));
        bool bottom_control = (index_y == bottom) && ((index_x <= right) && (index_x >= left));

        if (left_control || right_control || top_control || bottom_control){

            input_matrix[total_index] = value;

        }

    }

}

#endif

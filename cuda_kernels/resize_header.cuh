#ifndef RESIZE
#define RESIZE

__global__ void resize(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height, int new_width, int new_height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < new_width && y_index < new_height){

        int total_index = y_index * new_width + x_index;

        int x = roundf(x_index * (float)(width/new_width));
        int y = roundf(y_index * (float)(height/new_height));

        //int control_x = min(max(x,0),width -1);
        //int control_y = min(max(y,0),height -1);

        int old_index = y * width + x;

        output_matrix[total_index] = input_matrix[old_index];

    }

}

#endif

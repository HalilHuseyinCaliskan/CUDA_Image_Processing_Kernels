#ifndef SOBEL_FILTER
#define SOBEL_FILTER

__global__ void sobel_filter(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if(x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        int Gx[3][3] = {
            {-1,0,1},
            {-2,0,2},
            {-1,0,1}
        };

        int Gy[3][3] = {
            {-1,-2,-1},
            {0,0,0},
            {1,2,1}
        };

        int sum_x = 0;
        int sum_y = 0;

        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <= 1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_x = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                int pixel = input_matrix[control_y * width + control_x];

                sum_x += Gx[i + 1][j + 1] * pixel;
                sum_y += Gy[i + 1][j + 1] * pixel;

            }
        }

        int value = min(abs(sum_x) + abs(sum_y),255);

        output_matrix[total_index] = value;

    }

}

#endif

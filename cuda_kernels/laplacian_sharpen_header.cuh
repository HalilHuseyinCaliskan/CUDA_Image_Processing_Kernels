#ifndef LAPLACIAN_SHARPEN
#define LAPLACIAN_SHARPEN

__global__ void laplacian_sharpen(unsigned char *input_matrix, unsigned char *output_matrix,int width,int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if(x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        int kernel[3][3] = {
            {0,-1,0},
            {-1,4,-1},
            {0,-1,0}
        }; 

        int weighted_sum = 0;

        for(int i = -1; i <=1; i++ ){
            for(int j = -1; j <=1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_x = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                int w = kernel[i+1][j+1];
                weighted_sum += input_matrix[control_y * width + control_x] * w;

            }
        }

        int value = input_matrix[total_index] + weighted_sum;
        value = min(max(value,0),255);

        output_matrix[total_index] = (unsigned char)(value);

    }
    
}

#endif

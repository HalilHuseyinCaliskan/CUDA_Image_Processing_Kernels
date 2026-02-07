#ifndef MOTION_DETECTION
#define MOTION_DETECTION

__device__ void bgr_to_gray(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;
    

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        int bgr_index = 3 * total_index;
        unsigned char blue = input_matrix[bgr_index];
        unsigned char green = input_matrix[bgr_index + 1];
        unsigned char red = input_matrix[bgr_index + 2];

        unsigned char gray_value = (unsigned char)(0.114f * blue + 0.587f * green + 0.299f * red);

        output_matrix[total_index] = gray_value;

    }

}

__device__ void gaussian_blur(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;


    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        unsigned char kernel[3][3] = {
            {1,2,1},
            {2,4,2},
            {1,2,1}
        };
        
        int w = 0;
        int weightde_sum = 0;
        int sum = 0;

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

__device__ void abs_diff(unsigned char *input_matrix1, unsigned char *input_matrix2, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        output_matrix[total_index] = abs(input_matrix1[total_index] - input_matrix2[total_index]);

    }

}

__device__ void binary_threshold(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height, unsigned char threshold_value, unsigned char target_value){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;
        
        output_matrix[total_index] = input_matrix[total_index] > threshold_value ? target_value : 0 ;

    }

}

__device__ void erosion(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;

        int control = 255;

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

__device__ void image_transfer(unsigned char *prev_image, unsigned char *blured_image, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < height){

        int total_index = y_index * width + x_index;
        prev_image[total_index] = blured_image[total_index];

    }

}

__global__ void motion_detection(unsigned char *bgr_image, unsigned char *gray_image, unsigned char *blured_image, unsigned char *diffed_image,
 unsigned char *threshold_image, unsigned char *eroded_image, unsigned char *prev_image, int width, int height, unsigned char threshold_value, unsigned char target_value){

    bgr_to_gray(bgr_image,gray_image,width,height);
    gaussian_blur(gray_image,blured_image,width,height);
    abs_diff(blured_image,prev_image,diffed_image,width,height);
    binary_threshold(diffed_image,threshold_image,width,height,threshold_value,target_value);
    erosion(threshold_image,eroded_image,width,height);
    image_transfer(prev_image,blured_image,width,height);
    
}

#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

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

__global__ void blur(unsigned char *input_matrix, unsigned char *output_matrix, int width, int height){

    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = y_index * width + x_index;

    float sum = 0.0f;

    if (x_index < width && y_index < height){

        for(int i = -1; i <= 1; i++){

            for(int j = -1; j <= 1; j++){

                int x = x_index + j;
                int y = y_index + i;

                int control_x = min(max(x,0),width -1);
                int control_y = min(max(y,0),height -1);

                sum += (float)input_matrix[control_y * width + control_x];

            }
        }

        output_matrix[idx] = (unsigned char)(sum/9.0f);

    }

}


int main(){

    int width = 1920;
    int height = 1080;

    int size = width * height * sizeof(unsigned char);

    unsigned char *gray, *gray2, *filtered, *blured; 

    cudaMalloc((void**)&gray,size);
    cudaMalloc((void**)&gray2,size);
    cudaMalloc((void**)&filtered,size);
    cudaMalloc((void**)&blured,size);

    dim3 threadsPeerBlock(16,16);
    dim3 numBlocks((width + threadsPeerBlock.x -1)/threadsPeerBlock.x,(height + threadsPeerBlock.y -1)/threadsPeerBlock.y);

    cudaStream_t s1, s2;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cv::VideoCapture kamera_v0(0);
    cv::VideoCapture kamera_v1(1);

    if (!kamera_v0.isOpened() || !kamera_v1.isOpened()){
        std::cout<<"Kameralar açılamıyor";
        return -1;
    }

    cv::Mat frame_v0, frame_v1;
    cv::Mat result_v0(height,width,CV_8UC1);
    cv::Mat result_v1(height,width,CV_8UC1);


    while(true){

        kamera_v0 >> frame_v0;
        kamera_v1 >> frame_v1;

        if (frame_v0.empty() || frame_v1.empty()){
            std::cout<<"Goruntu yok";
            return -1;
        }
        cv::cvtColor(frame_v0,frame_v0,cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame_v1,frame_v1,cv::COLOR_BGR2GRAY);
        cv::resize(frame_v0,frame_v0,cv::Size(1920,1080));
        cv::resize(frame_v1,frame_v1,cv::Size(1920,1080));

        cudaMemcpyAsync(gray,frame_v0.data,size,cudaMemcpyHostToDevice,s1);
        cudaMemcpyAsync(gray2,frame_v1.data,size,cudaMemcpyHostToDevice,s2);

        sobel_filter<<<numBlocks,threadsPeerBlock,0,s1>>>(gray,filtered,width,height);
        blur<<<numBlocks,threadsPeerBlock,0,s2>>>(gray2,blured,width,height);

        cudaMemcpyAsync(result_v0.data,filtered,size,cudaMemcpyDeviceToHost,s1);
        cudaMemcpyAsync(result_v1.data,blured,size,cudaMemcpyDeviceToHost,s2);

        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);

        cv::imshow("frame_v0",result_v0);
        cv::imshow("frame_v1",result_v1);
        cv::waitKey(1);

    }

    cudaFree(gray);
    cudaFree(gray2);
    cudaFree(filtered);
    cudaFree(blured);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    return 0;

}

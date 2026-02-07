#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "resize_header.cuh"

int main(){

    cv::Mat resim = cv::imread("kedi.png",cv::IMREAD_COLOR);

    if (resim.empty()){
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(resim,gray,cv::COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;
    int size = width * height * sizeof(unsigned char);
    int new_width = 640;
    int new_height = 640;
    int size2 = new_width * new_height * sizeof(unsigned char);
    unsigned char *input_matrix,*output_matrix;


    cudaMallocManaged((void**)&input_matrix,size);
    cudaMallocManaged((void**)&output_matrix,size2);

    memcpy(input_matrix,gray.data,size);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((new_width + threadsPerBlock.x -1)/threadsPerBlock.x,(new_height + threadsPerBlock.y -1)/threadsPerBlock.y);

    resize<<<numBlocks,threadsPerBlock>>>(input_matrix,output_matrix,width,height,new_width,new_height);

    cudaDeviceSynchronize();

    cv::Mat resized(new_height,new_width,CV_8UC1);
    memcpy(resized.data,output_matrix,size2);

    cv::imshow("frame",resized);
    cv::waitKey(0);

    cudaFree(input_matrix);
    cudaFree(output_matrix);

    return 0;
}

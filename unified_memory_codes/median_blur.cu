#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "median_blur_header.cuh"

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
    unsigned char *input_matrix, *output_matrix;

    cudaMalloc((void**)&input_matrix,size);
    cudaMalloc((void**)&output_matrix,size);

    cudaMemcpy(input_matrix,gray.data,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    median_blur<<<numBlocks,threadsPerBlock>>>(input_matrix,output_matrix,width,height);

    cudaDeviceSynchronize();

    cv::Mat blured(height,width,CV_8UC1);
    cudaMemcpy(blured.data,output_matrix,size,cudaMemcpyDeviceToHost);

    cv::imshow("frame",blured);
    cv::waitKey(0);

    cudaFree(input_matrix);
    cudaFree(output_matrix);

    return 0;
}

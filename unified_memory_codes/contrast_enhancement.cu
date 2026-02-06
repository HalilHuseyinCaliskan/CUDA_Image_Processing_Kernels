#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "contrast_enhancement_header.cuh"

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
    unsigned char *input_matrix;
    float alpha = 1.2f;
    int beta = 4;


    cudaMallocManaged((void**)&input_matrix,size);

    memcpy(input_matrix,gray.data,size);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    contrast_enhancement<<<numBlocks,threadsPerBlock>>>(input_matrix,width,height,alpha,beta);

    cudaDeviceSynchronize();

    cv::Mat enhanced(height,width,CV_8UC1);
    memcpy(enhanced.data,input_matrix,size);

    cv::imshow("frame",enhanced);
    cv::waitKey(0);

    cudaFree(input_matrix);

    return 0;
}

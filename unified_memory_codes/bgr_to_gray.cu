#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "bgr_to_gray_header.cuh"

int main(){

    cv::Mat resim = cv::imread("kedi.png",cv::IMREAD_COLOR);

    if (resim.empty()){
        return -1;
    }

    int width = resim.cols;
    int height = resim.rows;
    int size = width * height * 3 * sizeof(unsigned char);
    int size2 = width * height * sizeof(unsigned char);
    unsigned char *input_matrix, *output_matrix;

    cudaMalloc((void**)&input_matrix,size);
    cudaMalloc((void**)&output_matrix,size2);

    memcpy(input_matrix,resim.data,size);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    bgr_to_gray<<<numBlocks,threadsPerBlock>>>(input_matrix,output_matrix,width,height);

    cudaDeviceSynchronize();

    cv::Mat gray(height,width,CV_8UC1);
    memcpy(gray.data,output_matrix,size2);

    cv::imshow("frame",gray);
    cv::waitKey(0);

    cudaFree(input_matrix);
    cudaFree(output_matrix);

    return 0;
}

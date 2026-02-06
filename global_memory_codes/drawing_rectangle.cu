#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "drawing_rectangle_header.cuh"

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
    int left = 200;
    int right = 600;
    int top = 300;
    int bottom = 400;
    unsigned char value = 0;

    cudaMalloc((void**)&input_matrix,size);

    cudaMemcpy(input_matrix,gray.data,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    drawing_rectangle<<<numBlocks,threadsPerBlock>>>(input_matrix,width,height, left, right, top, bottom, value);

    cudaDeviceSynchronize();

    cv::Mat draw(height,width,CV_8UC1);
    cudaMemcpy(draw.data,input_matrix,size,cudaMemcpyDeviceToHost);

    cv::imshow("frame",draw);
    cv::waitKey(0);

    cudaFree(input_matrix);

    return 0;
}

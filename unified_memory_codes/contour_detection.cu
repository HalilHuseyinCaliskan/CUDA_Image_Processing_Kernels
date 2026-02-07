#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "contour_detection_header.cuh"

int main(){

    cv::Mat resim = cv::imread("kedi.png",cv::IMREAD_COLOR);

    if (resim.empty()){
        return -1;
    }

    cv::Mat gray, binary;
    cv::cvtColor(resim,gray,cv::COLOR_BGR2GRAY);
    cv::threshold(gray,binary,50,255,cv::THRESH_BINARY);

    int width = gray.cols;
    int height = gray.rows;
    int size_bgr = width * height * 3 * sizeof(unsigned char);
    int size = width * height * sizeof(unsigned char);
    unsigned char *input_matrix, *output_matrix;

    cudaMallocManaged((void**)&input_matrix,size_bgr);
    cudaMallocManaged((void**)&output_matrix,size);

    memcpy(input_matrix,resim.data,size_bgr);
    cudaMemcpy(output_matrix,binary.data,size);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    contour_detection<<<numBlocks,threadsPerBlock>>>(input_matrix,output_matrix,width,height);

    cudaDeviceSynchronize();

    cv::Mat contour_detected(height,width,CV_8UC3);
    memcpy(contour_detected.data,input_matrix,size_bgr);

    cv::imshow("frame",contour_detected);
    cv::imshow("frame2",binary);
    cv::waitKey(0);

    cudaFree(input_matrix);
    cudaFree(output_matrix);

    return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include "motion_detection_header.cuh"

int main(){

    cv::VideoCapture cap(0);

    if (!cap.isOpened()){
        std::cout<<"Kamera açılmadı";
        return -1;
    }

    int width = 1920;
    int height = 1080;

    unsigned char *bgr_image, *gray_image, *blured_image, *diffed_image, *threshold_image, *dilated_image, *prev_image;

    int bgr_image_size = width * height * 3 * sizeof(unsigned char);
    int size = width * height * sizeof(unsigned char);

    unsigned char threshold_value = 45;
    unsigned char target_value = 255;

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x,(height + threadsPerBlock.y -1)/threadsPerBlock.y);

    cudaMalloc((void**)&bgr_image,bgr_image_size);
    cudaMalloc((void**)&gray_image,size);
    cudaMalloc((void**)&blured_image,size);
    cudaMalloc((void**)&diffed_image,size);
    cudaMalloc((void**)&threshold_image,size);
    cudaMalloc((void**)&dilated_image,size);
    cudaMalloc((void**)&prev_image,size);

    cv::Mat frame, prev_;
    cv::Mat result(height,width,CV_8UC1);

    cap >> prev_;

    cv::resize(prev_,prev_,cv::Size(1920,1080));
    cv::cvtColor(prev_,prev_,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_, prev_, cv::Size(5,5), 1.5);

    cudaMemcpy(prev_image,prev_.data,size,cudaMemcpyHostToDevice);

    while(true){

        cap >> frame;
        
        if (frame.empty()){
            std::cout<<"Görüntü boş";
            return -1;
        }

        cv::resize(frame,frame,cv::Size(1920,1080));

        cudaEventRecord(start);

        cudaMemcpy(bgr_image,frame.data,bgr_image_size,cudaMemcpyHostToDevice);

        motion_detection<<<numBlocks,threadsPerBlock>>>(bgr_image,gray_image,blured_image,diffed_image,threshold_image,dilated_image,prev_image,width,height,threshold_value,target_value);

        cudaMemcpy(result.data,eroded_image,size,cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);

        std::cout<<"Toplam sure: "<<time<<std::endl;

        cv::imshow("frame",result);
        cv::waitKey(1);

    }

    cudaFree(bgr_image);
    cudaFree(gray_image);
    cudaFree(blured_image);
    cudaFree(diffed_image);
    cudaFree(threshold_image);
    cudaFree(dilated_image);
    cudaFree(prev_image);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024
Purpose: Headerfile, Contains all function definitions/signatures

*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

int customGreyscale(const cv::Mat& src, cv::Mat& dst); //Q4
void sepiaFilter(const cv::Mat& src, cv::Mat& dst, bool applySepia);//Q5
int blur5x5_1(cv::Mat& src, cv::Mat& dst);//Q6A
int blur5x5_2(cv::Mat& src, cv::Mat& dst);//Q6B
int sobelX3x3(cv::Mat& src, cv::Mat& dst);//Q7
int sobelY3x3(cv::Mat& src, cv::Mat& dst);//Q7
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);//Q8
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);//Q9
void retainStrongColor(cv::Mat &src, cv::Mat &dst, double threshold);//Q11
int selectiveBlurInsideFaces(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);//Extension face privacy
int selectiveBlurOutsideFaces(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);//Q11
int thermalPalette(cv::Mat& src, cv::Mat& dst); //Extension
int mirrorVertically(cv::Mat &src, cv::Mat &dst);//Extension
void negativeImage(cv::Mat &src, cv::Mat &dst);//Q11
void embossingEffect(cv::Mat &src, cv::Mat &dst);//Extension
void addSparkles(cv::Mat &src, cv::Mat &dst);//Extension
int blackOutOutsideFaces(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);//extension
#endif // FILTER_H
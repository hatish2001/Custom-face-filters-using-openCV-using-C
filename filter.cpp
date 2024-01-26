/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024
Purpose: Contains all the filter functions and their implementations which is called by vidDisplay.cpp

*/

#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"
using namespace cv;

/*
Summary: Applies a custom greyscale transformation to an input image.

This function takes an input image in BGR format, and for each pixel,
it subtracts the red channel value from 255 and assigns the resulting
value to all three color channels, creating a custom greyscale effect.
The output image is stored in the provided destination matrix. 

 Input: 
 @param src Input image in BGR format.
 @param dst Output matrix where the custom greyscale version of the input image is stored. 

 @return Returns 0 on success, or -1 if the input image is empty.
*/

int customGreyscale(const Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Create an output matrix with the same size as the input
    dst = Mat(src.size(), CV_8UC3);

    // Iterate through each pixel and apply custom greyscale transformation
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Get the RGB values of the current pixel
            Vec3b color = src.at<Vec3b>(i, j);

            // Calculate the greyscale value using a custom transformation
            uchar greyValue = 255 - color[2];  // Subtract red channel from 255

            // Set the greyscale value for all three channels
            dst.at<Vec3b>(i, j) = Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0; // Return success
}

/*
 summary: Applies a Sepia tone filter to an input image if specified.

  This function takes an input image in BGR format and, if the 'applySepia'
  parameter is true, it iterates through each pixel, calculates the Sepia
  tone values, and assigns the new values to each pixel. If 'applySepia' is
  false, the function simply clones the input image.
  The output image is stored in the provided destination matrix.
  
  @param src Input image in BGR format.
  @param dst Output matrix where the result is stored.
  @param applySepia Flag indicating whether to apply the Sepia filter (true) or not (false).

  @return Returns 0 on success, or -1 if the input image is empty.
 */
void sepiaFilter(const Mat& src, Mat& dst, bool applySepia) {
    if (applySepia) {
        // Iterate through each pixel and apply Sepia tone transformation
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                // Get the RGB values of the current pixel
                Vec3b color = src.at<Vec3b>(i, j);

                // Calculate the Sepia tone values
                uchar newRed = saturate_cast<uchar>(0.393 * color[2] + 0.769 * color[1] + 0.189 * color[0]);
                uchar newGreen = saturate_cast<uchar>(0.349 * color[2] + 0.686 * color[1] + 0.168 * color[0]);
                uchar newBlue = saturate_cast<uchar>(0.272 * color[2] + 0.534 * color[1] + 0.131 * color[0]);

                // Set the new values for all three channels
                dst.at<Vec3b>(i, j) = Vec3b(newBlue, newGreen, newRed);
            }
        }
        //applyVignette(dst);
    } else {
        // If Sepia filter is not applied, simply copy the input image
        dst = src.clone();
    }
}

/*
    Apply a 5x5 blur filter to the input image.

    This function takes an input image in BGR format and applies a 5x5 blur filter to each pixel.
    The filter kernel is defined with weights that decrease with distance from the center.
    The blurred result is stored in the provided destination matrix.

    @param src Input image in BGR format.
    @param dst Output matrix where the blurred version of the input image is stored.

    @return Returns 0 on success, or -1 if the input image is empty.
*/

int blur5x5_1(Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Ensure the output matrix has the same size as the input
    dst = src.clone();

    int kernel[5][5] = {{1, 2, 4, 2, 1},
                        {2, 4, 8, 4, 2},
                        {4, 8, 16, 8, 4},
                        {2, 4, 8, 4, 2},
                        {1, 2, 4, 2, 1}};

    int kernelSum = 100; // Sum of all kernel values (for normalization)

    // Iterate through each pixel and apply the blur filter
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the filter for each color channel
            for (int x = -2; x <= 2; ++x) {
                for (int y = -2; y <= 2; ++y) {
                    Vec3b color = src.at<Vec3b>(i + x, j + y);

                    sumB += color[0] * kernel[x + 2][y + 2];
                    sumG += color[1] * kernel[x + 2][y + 2];
                    sumR += color[2] * kernel[x + 2][y + 2];
                }
            }

            // Normalize and assign the blurred values to the destination image
            dst.at<Vec3b>(i, j) = Vec3b(sumB / kernelSum, sumG / kernelSum, sumR / kernelSum);
        }
    }

    
    return 0; // Return success
}


/*
Apply a separable 5x5 blur filter to the input image using two 1D filters.

This function takes an input image in BGR format and applies a separable 5x5 blur filter to it. 
The blur is achieved by convolving the image with two 1D filters: one vertically and one horizontally.
The kernel used for both filters is [1, 2, 4, 2, 1], and the sum of the kernel values is 10 for normalization.

@param src Input image in BGR format.
@param dst Output matrix where the blurred version of the input image is stored.

@return Returns 0 on success or -1 if the input image is empty.
*/

int blur5x5_2(Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Ensure the output matrix has the same size as the input
    dst = src.clone();

    // Apply separable 1x5 filters vertically and horizontally
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; // Sum of all kernel values (for normalization)

    // Apply vertical filter
    for (int j = 2; j < src.cols - 2; ++j) {
        for (int i = 2; i < src.rows - 2; ++i) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the filter for each color channel
            for (int k = -2; k <= 2; ++k) {
                Vec3b color = src.at<Vec3b>(i + k, j);
                sumB += color[0] * kernel[k + 2];
                sumG += color[1] * kernel[k + 2];
                sumR += color[2] * kernel[k + 2];
            }

            // Normalize and assign the blurred values to the destination image
            dst.at<Vec3b>(i, j) = Vec3b(sumB / kernelSum, sumG / kernelSum, sumR / kernelSum);
        }
    }

    // Apply horizontal filter
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the filter for each color channel
            for (int k = -2; k <= 2; ++k) {
                Vec3b color = dst.at<Vec3b>(i, j + k);
                sumB += color[0] * kernel[k + 2];
                sumG += color[1] * kernel[k + 2];
                sumR += color[2] * kernel[k + 2];
            }

            // Normalize and assign the blurred values to the destination image
            dst.at<Vec3b>(i, j) = Vec3b(sumB / kernelSum, sumG / kernelSum, sumR / kernelSum);
        }
    }

    return 0; // Return success
}

/*
Applies a 3x3 Sobel X filter to an input image in BGR format.

This function calculates the Sobel gradient in the X direction for each pixel in the input image.
The output is stored in the provided destination matrix. The function uses a separable 1x3 Sobel
X filter with kernel values {-1, 0, 1}. The resulting gradients are stored as short integers in a 
3-channel matrix, where each channel represents the gradient in the Blue, Green, and Red channels.

@param src Input image in BGR format.
@param dst Output matrix where the Sobel X gradients are stored.

@return Returns 0 on success, or -1 if the input image is empty.
*/
int sobelX3x3(Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Ensure the output matrix has the same size as the input
    dst = Mat(src.size(), CV_16SC3);

    // Apply separable 1x3 Sobel X filter
    int kernel[3] = {-1, 0, 1};

    // Apply Sobel X filter
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            Vec3b colorLeft = src.at<Vec3b>(i, j - 1);
            Vec3b colorRight = src.at<Vec3b>(i, j + 1);

            short sumB = colorRight[0] - colorLeft[0];
            short sumG = colorRight[1] - colorLeft[1];
            short sumR = colorRight[2] - colorLeft[2];

            // Assign the Sobel X value to the destination image
            dst.at<Vec3s>(i, j) = Vec3s(sumB, sumG, sumR);
        }
    }

    return 0; // Return success
}

/*
  Applies a 3x3 Sobel Y filter to an input image.
  
  This function takes an input image in BGR format and computes the Sobel Y gradient
  using a separable 1x3 filter. The output is a matrix where each pixel contains
  the Sobel Y gradient for each color channel.
  
  @param src Input image in BGR format.
  @param dst Output matrix where the Sobel Y gradient is stored for each color channel.
  
  @return Returns 0 on success, or -1 if the input image is empty.
*/
int sobelY3x3(Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Ensure the output matrix has the same size as the input
    dst = Mat(src.size(), CV_16SC3);

    // Apply separable 1x3 Sobel Y filter
    int kernel[3] = {-1, 0, 1};

    // Apply Sobel Y filter
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            Vec3b colorUp = src.at<Vec3b>(i - 1, j);
            Vec3b colorDown = src.at<Vec3b>(i + 1, j);

            short sumB = colorDown[0] - colorUp[0];
            short sumG = colorDown[1] - colorUp[1];
            short sumR = colorDown[2] - colorUp[2];

            // Assign the Sobel Y value to the destination image
            dst.at<Vec3s>(i, j) = Vec3s(sumB, sumG, sumR);
        }
    }

    return 0; // Return success
}

/*
  Calculates the magnitude of gradient vectors from two input matrices representing gradient components.
  The input matrices 'sx' and 'sy' are assumed to contain X and Y components of the gradients, respectively.
  The resulting gradient magnitude is mapped to color based on the angle of the gradient vector in the HSV color space.
  The color-mapped output is stored in the provided destination matrix 'dst'.

  @param sx Input matrix representing the X component of the gradient.
  @param sy Input matrix representing the Y component of the gradient.
  @param dst Output matrix where the color-mapped gradient magnitude is stored.

  @return Returns 0 on success, or -1 if either of the input images 'sx' or 'sy' is empty.
*/

int magnitude(Mat& sx, Mat& sy, Mat& dst) {
    // Check if the input images are valid
    if (sx.empty() || sy.empty()) {
        return -1; // Return an error code
    }

    // Ensure the output matrix has the same size as the input
    dst = Mat(sx.size(), CV_8UC3);

    // Calculate gradient magnitude using Euclidean distance and map to color
    for (int i = 0; i < sx.rows; ++i) {
        for (int j = 0; j < sx.cols; ++j) {
            short gradX = sx.at<Vec3s>(i, j)[0];
            short gradY = sy.at<Vec3s>(i, j)[0];

            // Calculate magnitude using Euclidean distance
            float magnitude = sqrt(static_cast<float>(gradX * gradX + gradY * gradY));

            // Map the angle of the gradient to a color in the HSV space
            float angle = atan2(static_cast<float>(gradY), static_cast<float>(gradX));
            angle = (angle < 0) ? (angle + CV_PI) : angle;
            angle *= 180.0 / CV_PI;

            // Set HSV values
            uchar hue = static_cast<uchar>(angle / 2);
            uchar saturation = 255;
            uchar value = static_cast<uchar>(magnitude);

            // Convert HSV to BGR
            Mat hsvPixel(1, 1, CV_8UC3, Scalar(hue, saturation, value));
            Mat bgrPixel;
            cvtColor(hsvPixel, bgrPixel, COLOR_HSV2BGR);

            // Assign the color to the destination image
            dst.at<Vec3b>(i, j) = bgrPixel.at<Vec3b>(0, 0);
        }
    }

    return 0; // Return success
}

/*
  Applies a blur followed by quantization to reduce color levels in an input image.

  This function takes an input image and applies a 5x5 separable blur filter.
  It then quantizes each color channel into the specified number of levels.
  The quantized image is stored in the provided destination matrix.

  @param src Input image in BGR format.
  @param dst Output matrix where the blurred and quantized version of the input image is stored.
  @param levels The number of quantization levels for each color channel.

  @return Returns 0 on success, or -1 if the input image is empty.
*/
int blurQuantize(Mat& src, Mat& dst, int levels) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Blur the image using the 5x5 separable blur filter
    blur5x5_2(src, dst);

    // Quantize each color channel into the specified number of levels
    int bucketSize = 255 / levels;

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                // Quantize each color channel
                int quantizedValue = (dst.at<Vec3b>(i, j)[c] / bucketSize) * bucketSize;

                // Update the pixel value with the quantized value
                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(quantizedValue);
            }
        }
    }

    return 0; // Return success
}

/*
  Applies a color retention filter to an input image.

  This function retains the strong color in the image while converting
  the less dominant colors to greyscale. The strength of the retained color
  is determined by a threshold, and the mean color of the image is used
  to identify the dominant color channel. The output image is a modified
  version of the input, with strong color retained and other areas
  converted to greyscale.

  @param src Input image in BGR format.
  @param dst Output matrix where the color-retained version of the input image is stored.
  @param threshold Threshold for determining strong color retention (default is 0.9).
*/

void retainStrongColor(cv::Mat &src, cv::Mat &dst, double threshold = 0.9) {
    // Check if the input image is valid
    if (src.empty()) {
        return;  // Return without processing if the image is empty
    }

    // Calculate the mean color of the image
    cv::Scalar meanColor = cv::mean(src);

    // Determine the dominant color channel
    int dominantChannel = 0;
    for (int i = 1; i < 3; ++i) {
        if (meanColor[i] > meanColor[dominantChannel]) {
            dominantChannel = i;
        }
    }

    // Retain the strong color and convert everything else to greyscale
    dst = src.clone();  // Create a copy of the input image

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            if (src.at<cv::Vec3b>(i, j)[dominantChannel] > threshold * meanColor[dominantChannel]) {
                // Retain the strong color in the output image
                dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j);
            } else {
                // Convert everything else to greyscale
                uchar grayValue = static_cast<uchar>(0.3 * src.at<cv::Vec3b>(i, j)[0] +
                                                    0.59 * src.at<cv::Vec3b>(i, j)[1] +
                                                    0.11 * src.at<cv::Vec3b>(i, j)[2]);
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(grayValue, grayValue, grayValue);
            }
        }
    }
}

/*
  Applies selective blur to an input image based on specified face regions.

  This function takes an input image, a vector of face rectangles, and
  selectively applies a Gaussian blur to the regions inside the detected faces.
  
  @param src Input image.
  @param dst Output matrix where the selectively blurred image is stored.
  @param faces Vector of rectangles representing the detected face regions.

  @return Returns 0 on success, or -1 if the input image is empty.
*/

int selectiveBlurInsideFaces(Mat &src, Mat &dst, std::vector<Rect> &faces) {
    // Ensure the input image is not empty
    if (src.empty()) {
        return -1;
    }

    // Create a mask to identify regions outside faces
    Mat mask(src.size(), CV_8U, Scalar(255));

    // Preserve the original unblurred face regions in the mask
    for (const Rect &face : faces) {
        rectangle(mask, face, Scalar(0), FILLED);
    }

    // Blur the regions outside faces
    Mat blurredImage;
    GaussianBlur(src, blurredImage, Size(21, 21), 0, 0);

    // Copy the blurred regions outside faces and the original face regions back to the output
    src.copyTo(dst, mask);
    blurredImage.copyTo(dst, ~mask);

    return 0;
}

/*
   Apply selective blur outside detected faces.

   This function takes an input image (`src`) and a vector of rectangles
   representing face regions (`faces`). It applies Gaussian blur to the areas
   outside the detected faces while keeping the original unblurred face regions.
   The result is stored in the output matrix (`dst`).

   @param src Input image in BGR format.
   @param dst Output matrix where the selectively blurred image is stored.
   @param faces Vector of rectangles representing detected face regions.

   @return Returns 0 on success, or -1 if the input image is empty.
*/
int selectiveBlurOutsideFaces(Mat &src, Mat &dst, std::vector<Rect> &faces) {
    // Ensure the input image is not empty
    if (src.empty()) {
        return -1;
    }

    // Create a mask to identify face regions
    Mat mask(src.size(), CV_8U, Scalar(0));

    // Preserve the original unblurred face regions in the mask
    for (const Rect &face : faces) {
        rectangle(mask, face, Scalar(255), FILLED);
    }

    // Blur the regions outside faces
    Mat blurredImage;
    GaussianBlur(src, blurredImage, Size(21, 21), 0, 0);

    // Copy the original face regions and the blurred regions outside faces back to the output
    src.copyTo(dst, mask);
    blurredImage.copyTo(dst, ~mask);

    return 0;
}

/*
  Applies a thermal palette transformation to the input image.

  This function takes an input color image (BGR format) and converts it
  to a grayscale image. Then, it applies a thermal colormap to visualize
  temperature variations, ranging from cool to warm. The result is stored
  in the provided destination matrix in RGB format.

  @param src Input color image in BGR format.
  @param dst Output matrix where the thermal palette version of the input image is stored.

  @return Returns 0 on success, or -1 if the input image is empty.
**/

int thermalPalette(Mat& src, Mat& dst) {
    // Check if the input image is valid
    if (src.empty()) {
        return -1; // Return an error code
    }

    // Convert the source image to a grayscale image
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Create a colormap for thermal visualization
    cv::Mat colormap(grey.size(), CV_8UC3);

    // Define the color mapping from cool to warm
    cv::applyColorMap(grey, colormap, cv::COLORMAP_JET);

    // Convert the colormap back to BGR
    cv::cvtColor(colormap, dst, cv::COLOR_BGR2RGB);

    return 0; // Return success
}

/*
  Applies vertical mirroring to an input image.

  This function takes an input image and produces a vertically mirrored version
  of it by flipping along the vertical axis. The resulting image is stored in
  the provided destination matrix.

  @param src Input image.
  @param dst Output matrix where the vertically mirrored image is stored.

  @return Returns 0 on success, or -1 if the input image is empty.
*/
int mirrorVertically(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;  // Return an error code
    }

    cv::flip(src, dst, 0);  // Flip along the vertical axis

    return 0;  // Return success
}

/*
   Apply a negative transformation to the input image.

   This function takes an input image in BGR format and creates a negative version
   by inverting each color channel. The resulting negative image is stored in the
   provided destination matrix.

   Parameters:
       - src: Input image in BGR format.
       - dst: Output matrix where the negative version of the input image is stored.
**/
void negativeImage(Mat &src, Mat &dst) {
    // Create a copy of the source image
    dst = src.clone();

    // Invert each color channel
    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            for (int c = 0; c < dst.channels(); ++c) {
                dst.at<cv::Vec3b>(i, j)[c] = 255 - dst.at<cv::Vec3b>(i, j)[c];
            }
        }
    }
}

/*
  Applies an embossing effect to the input image using the Sobel filter.

  This function takes a color image in BGR format as input and applies an embossing
  effect by convolving the image with Sobel filters in the X and Y directions.
  The resulting embossing effect is stored in the output matrix.

  @param src Input color image in BGR format.
  @param dst Output matrix where the embossed version of the input image is stored.
*/

void embossingEffect(Mat &src, Mat &dst) {
    // Convert the source image to greyscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Create Sobel filters for X and Y directions
    cv::Mat sobelX, sobelY;
    cv::Sobel(grey, sobelX, CV_16S, 1, 0);
    cv::Sobel(grey, sobelY, CV_16S, 0, 1);

    // Normalize the Sobel outputs to the range [0, 255]
    cv::normalize(sobelX, sobelX, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(sobelY, sobelY, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Direction vector for the embossing effect
    cv::Vec2f direction(0.7071, 0.7071);

    // Calculate the dot product with the selected direction
    cv::Mat embossingX = sobelX * direction[0];
    cv::Mat embossingY = sobelY * direction[1];

    // Combine the embossing effects for X and Y directions
    dst = embossingX + embossingY;

    // Normalize the result to the range [0, 255]
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
}

/*
  Apply sparkle effect to an input image.

  This function takes an input image in BGR format, converts it to greyscale,
  performs Canny edge detection, and adds sparkle effects to the image where
  strong edges are detected. The output image is stored in the provided 
  destination matrix.

  @param src Input image in BGR format.
  @param dst Output matrix where the image with added sparkles is stored.
*/

void addSparkles(cv::Mat &src, cv::Mat &dst) {
    // Convert the source image to greyscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(grey, edges, 50, 150);

    // Convert the edges to 3 channels
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);

    // Add sparkles to the image where there are strong edges
    cv::addWeighted(src, 1.0, edges, 0.5, 0, dst);
}


/*
  Applies a black-outside-faces effect to an input image.

  This function takes an input image (`src`) and a vector of rectangles representing
  detected faces (`faces`). It creates a binary mask where the face regions are set
  to white (255) and the rest is black (0). The original face regions are copied to
  the output image (`dst`), while the regions outside the faces are set to black.

  @param src Input image in BGR format.
  @param dst Output image where the original face regions are preserved, and the
             regions outside the faces are set to black.
  @param faces Vector of rectangles representing detected faces.
  
  @return Returns 0 on success, or -1 if the input image is empty.
*/

int blackOutOutsideFaces(Mat &src, Mat &dst, std::vector<Rect> &faces) {
    // Ensure the input image is not empty
    if (src.empty()) {
        return -1;
    }

    // Create a mask to identify face regions
    Mat mask(src.size(), CV_8U, Scalar(0));

    // Preserve the original face regions in the mask
    for (const Rect &face : faces) {
        rectangle(mask, face, Scalar(255), FILLED);
    }

    // Copy the original face regions to the output
    src.copyTo(dst, mask);

    // Set the intensity of pixels outside faces to zero (black)
    dst.setTo(Scalar(0), ~mask);

    return 0;
}
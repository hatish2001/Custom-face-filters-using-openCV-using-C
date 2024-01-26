/*

Authored by: Aadhi Aadhavan Balasubramanian, Harishraj Udaya Bhaskar
Date: 01/26/2024
Purpose: Main function, Driver code, Applied filter based on Keystroke

*/

/*
Main Function toggles filters on or off based on the following key strokes.
Keystrokes: 
'q': Quits the program. -- Task 2
's': Saves the current frame as an image with a timestamp. -- Task 2
'g': Toggles between color and grayscale modes. -- Task 3
'h': Toggles an alternative grayscale mode. -- Task 4
'a': Toggles sepia mode. -- Task 5
'b': Toggles blur (5x5) mode. -- Task 6A, 6B, implemeted for video using separable filters
'x': Toggles Sobel X mode. -- Task 7
'y': Toggles Sobel Y mode. -- Task 7
'm': Toggles magnitude mode. -- Task 8
'l': Toggles blur quantize mode. -- Task 9
'f': Toggles face detection mode. -- Task 10
'c': Toggles retain strong color mode. -- Task 11 -1
'p': Toggles blur inside faces mode. --Extension
'z': Toggles blur outside faces mode. -- Task 11 - 2
't': Toggles thermal palette mode. --Extension
'v': Toggles vertically mirror mode. --Extension
'n': Toggles negative image mode. --Task 11-3
'e': Toggles embossing effect mode. --Extension
'5': Toggles sparkles mode.--Extension
'6': Toggles black out outside faces mode. --Extension
*/


#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open a video capture object (0 for default camera)
    VideoCapture cap(0);

    // Check if the video capture is successful
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video capture device." << endl;
        return -1;
    }

    // Create a window for display
    namedWindow("Video Display", WINDOW_NORMAL);

    // Flag to indicate whether to display in grayscale
    bool grayscaleMode = false;
    bool customGreyscaleMode = false;
    bool sepiaMode = false;
    double vignetteStrength = 0.3;
    bool blur5x5Mode = false;
    bool sobelXMode = false;
    bool sobelYMode = false;
    bool magnitudeMode = false;
    bool blurQuantizeMode = false;
    bool faceDetectMode = false;
    bool retainStrongColorMode = false;
    bool blurInsideFacesMode = false;
    bool blurOutsideFacesMode = false;
    bool thermalPaletteMode = false;
    bool mirrorVerticallyMode = false;
    bool negativeImageMode = false;
    bool embossingEffectMode = false;
    bool sparklesMode = false;
    bool blackOutOutsideFacesMode = false;
   

    while (true) {
        Mat frame;
        Mat grey;
        Mat mirroredFrame;

        // Capture a new frame
        cap >> frame;

        // Convert to grayscale if in grayscale mode
        if (grayscaleMode) {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
        }
        if (customGreyscaleMode) {
            customGreyscale(frame, frame);
        } 
        if (sepiaMode) {
            sepiaFilter(frame, frame, true);
        }
        if (blur5x5Mode) {
            blur5x5_2(frame, frame);
        }
        if (sobelXMode) {
            Mat sobelXOutput;
            sobelX3x3(frame, sobelXOutput);
            // Convert to 8-bit for visualization
            convertScaleAbs(sobelXOutput, frame);
        }

        if (sobelYMode) {
            Mat sobelYOutput;
            sobelY3x3(frame, sobelYOutput);
            // Convert to 8-bit for visualization
            convertScaleAbs(sobelYOutput, frame);
        }
        else if (magnitudeMode) {
            Mat sobelXOutput, sobelYOutput;
            sobelX3x3(frame, sobelXOutput);
            sobelY3x3(frame, sobelYOutput);

            // Calculate gradient magnitude
            Mat gradientMagnitude;
            magnitude(sobelXOutput, sobelYOutput, gradientMagnitude);

            frame = gradientMagnitude; // Use the gradient magnitude as the processed frame
        }
        if (blurQuantizeMode) {
            blurQuantize(frame, frame, 10); // Use 10 levels as an example
        }
        if (faceDetectMode) {
            vector<Rect> faces;
            cvtColor(frame, grey, COLOR_BGR2GRAY);
            detectFaces(grey, faces);
            drawBoxes(frame, faces);
        }
        if (retainStrongColorMode) {
            double threshold = 0.9;
            retainStrongColor(frame, frame, threshold);
        }
        if (blurInsideFacesMode) {
            cvtColor(frame, grey, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(grey, faces);
            drawBoxes(frame, faces);
            selectiveBlurInsideFaces(frame, frame, faces);
            }
        if (blurOutsideFacesMode) {
            cvtColor(frame, grey, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(grey, faces);
            drawBoxes(frame, faces);
            selectiveBlurOutsideFaces(frame, frame, faces);
            }
        if (thermalPaletteMode) {
            thermalPalette(frame, frame);
        }
        if (mirrorVerticallyMode) {
            // Mirror the frame vertically
            mirrorVertically(frame, mirroredFrame);

            // Display both original and vertically mirrored images side by side
            hconcat(frame, mirroredFrame, frame);
        }
        if (negativeImageMode) {
            negativeImage(frame, frame);
        }
        if (embossingEffectMode) {
            // Apply the embossing effect filter
            embossingEffect(frame, frame);
        }
        if (sparklesMode) {
            // Apply the sparkles filter
            addSparkles(frame, frame);
        }
        if (blackOutOutsideFacesMode) {
            cvtColor(frame, grey, COLOR_BGR2GRAY);
            vector<Rect> faces;
            detectFaces(grey, faces);
            drawBoxes(frame, faces);
            // Black out outside faces if in blackOutOutsideFacesMode
            blackOutOutsideFaces(frame, frame, faces);
            
        }



        // Display the captured frame
        imshow("Video Display", frame);

        // Check for user input
        char key = waitKey(1);

        // Quit if the user types 'q'
        if (key == 'q') {
            break;
        }

        // Save image with timestamp if the user types 's'
        if (key == 's') {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            // Create a timestamp string
            char timestamp[20];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", ltm);
            string filePath = "/Users/aadhi/Desktop/CV Class Codes/Project 1/Results/captured_image.jpg" + string(timestamp) + ".jpg";
            imwrite(filePath,  frame);
            cout << "Image saved" << endl;
        }

        // Toggle between color and grayscale modes
        if (key == 'g') {
            grayscaleMode = !grayscaleMode;
            sepiaMode = false;
            customGreyscaleMode = false;
            blur5x5Mode = false;
        }

        if (key == 'h') {
            customGreyscaleMode = !customGreyscaleMode;
            sepiaMode = false;
            grayscaleMode = false;
            blur5x5Mode = false;

        }

        if (key == 'a') {
            sepiaMode = !sepiaMode;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to sepia
            grayscaleMode = false;
            blur5x5Mode = false;
        }

        if (key == 'b') {
            blur5x5Mode = !blur5x5Mode;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }

        if (key == 'x') {
            sobelXMode = !sobelXMode;
            sobelYMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }

        if (key == 'y') {
            sobelYMode = !sobelYMode;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'm') {
            magnitudeMode = !magnitudeMode;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'l') {
            blurQuantizeMode =!blurQuantizeMode;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'f') {
            faceDetectMode = !faceDetectMode;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'c') {
            retainStrongColorMode = !retainStrongColorMode;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'p') {
            blurInsideFacesMode = !blurInsideFacesMode;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'z') {
            blurOutsideFacesMode = !blurOutsideFacesMode;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 't') {
            thermalPaletteMode =!thermalPaletteMode;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'v') {
            mirrorVerticallyMode = !mirrorVerticallyMode;
            thermalPaletteMode =false;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'n') {
            negativeImageMode = !negativeImageMode;
            mirrorVerticallyMode = false;
            thermalPaletteMode =false;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == 'e') {
            embossingEffectMode = !embossingEffectMode;
            negativeImageMode = false;
            mirrorVerticallyMode = false;
            thermalPaletteMode =false;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == '5') {
            sparklesMode = !sparklesMode;
            embossingEffectMode = false;
            negativeImageMode = false;
            mirrorVerticallyMode = false;
            thermalPaletteMode =false;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
        if (key == '6') {
            blackOutOutsideFacesMode = !blackOutOutsideFacesMode;
            sparklesMode = false;
            embossingEffectMode = false;
            negativeImageMode = false;
            mirrorVerticallyMode = false;
            thermalPaletteMode =false;
            blurOutsideFacesMode = false;
            blurInsideFacesMode = false;
            retainStrongColorMode = false;
            faceDetectMode = false;
            blurQuantizeMode =false;
            magnitudeMode = false;
            sobelYMode = false;
            sobelXMode = false;
            blur5x5Mode = false;
            grayscaleMode = false;
            customGreyscaleMode = false; // Turn off custom greyscale mode when switching to blur
            sepiaMode = false; // Turn off sepia mode when switching to blur
        }
    }

    // Release the video capture object and close the window
    cap.release();
    destroyAllWindows();

    return 0;
}

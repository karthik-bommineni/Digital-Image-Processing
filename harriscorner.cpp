#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void harrisCornerDetector() {

    Mat image, gray;
    Mat output, output_norm, output_norm_scaled;

    // Loading the actual image
    //image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/squares.png", IMREAD_COLOR);
    image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/house.jpg", IMREAD_COLOR);

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }
    cv::imshow("Original image", image);

    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detecting corners using the cornerHarris built in function
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output,
        3,              // Neighborhood size
        3,              // Aperture parameter for the Sobel operator
        0.04);          // Harris detector free parameter

    // Normalizing - Convert corner values to integer so they can be drawn
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(output_norm, output_norm_scaled);

    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 100) {
                circle(image, Point(i, j), 4, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }

    // Displaying the result
    cv::resize(image, image, cv::Size(), 1.5, 1.5);
    cv::imshow("Output Harris", image);
    cv::waitKey();   
}

int main()
{

    harrisCornerDetector();

    return 0;
}

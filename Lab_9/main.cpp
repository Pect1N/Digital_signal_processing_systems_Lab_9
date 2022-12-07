#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double** mattodouble(Mat inputImage)
{
    double** ptr = new double* [inputImage.rows];
    for (int i = 0; i < inputImage.rows; ++i)
    {
        ptr[i] = new double[inputImage.cols];
        for (int j = 0; j < inputImage.cols; ++j)
            ptr[i][j] = inputImage.at<uchar>(i, j);
    }
    return ptr;
}

Mat doubletomat(double** inputImage)
{
    Mat result = imread("lenna.png", IMREAD_GRAYSCALE);
    for (int i = 0; i < 512; ++i)
        for (int j = 0; j < 512; ++j)
            result.at<uchar>(i, j) = inputImage[i][j];
    return result;
}

int main(void)
{
	Mat inputImage = imread("lenna.png", IMREAD_GRAYSCALE);
    Mat result;

    const int length = 512;
    const int sigma = 1;
    const int size = 7;
    const float min = 0.1;
    const float max = 0.2;

    double** image_double = mattodouble(inputImage);
    double** gauss_image = mattodouble(inputImage);
    double** GradientModule = mattodouble(inputImage);
    double** GradientVector = mattodouble(inputImage);
    double** zeroing_image = mattodouble(inputImage);
    double** filtered_image = mattodouble(inputImage);

    double GaussMask[size][size] = {};
    double FilterResultX = 0, FilterResultY = 0;

    int shift = 0;
    int SobelFilterX[3][3] = { { -1, 0, 1}, { -2, 0, 2}, { -1, 0, 1} };
    int SobelFilterY[3][3] = { { -1, -2, -1}, { 0, 0, 0}, { 1, 2, 1} };

	imshow("Input image", inputImage);

    // создание фильтра
    shift = size / 2;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            GaussMask[x][y] = (1 / (2 * 3, 14 * pow(sigma, 2))) * exp(-(pow((x - shift), 2) + pow((y - shift), 2)) / (2 * pow(sigma, 2)));
        }
    }

    // применение фильтра
    for (int y = shift; y < length - shift; ++y)
    {
        for (int x = shift; x < length - shift; ++x)
        {
            gauss_image[y][x] = 0;
            for (int i = -3; i < shift + 1; ++i)
                for (int j = -3; j < shift + 1; ++j)
                    gauss_image[y][x] += image_double[y + i][x + j] * GaussMask[shift + i][shift + j];
        }
    }
    result = doubletomat(gauss_image);
    imshow("Gauss image", result);

    // фильтр Собеля
    for (int y = 1; y < length - 1; ++y)
    {
        int counter = 0;
        for (int x = 1; x < length - 1; ++x)
        {
            FilterResultX = 0;
            FilterResultY = 0;
            for (int i = -1; i < 2; ++i)
            {
                for (int j = -1; j < 2; ++j)
                {
                    FilterResultX += gauss_image[y + i][x + j] * SobelFilterX[1 + i][1 + j];
                    FilterResultY += gauss_image[y + i][x + j] * SobelFilterY[1 + i][1 + j];
                }
            }
            // модуль градиента
            GradientModule[y][x] = sqrt(pow(FilterResultX, 2) + pow(FilterResultY, 2));
            // ориентация вектора градиента
            GradientVector[y][x] = atan2(FilterResultY, FilterResultX);
        }
    }
    result = doubletomat(GradientModule);
    imshow("Gradient image", result);

    // обнуление пикселей
    for (int y = shift; y < length - shift; ++y)
    {
        for (int x = shift; x < length - shift; ++x)
        {
            double dx = cos(GradientVector[y][x] * 180 / M_PI);
            double dy = sin(GradientVector[y][x] * 180 / M_PI);
            dx = (dx > 0) ? 1 : ((dx < 0) ? -1 : 0);
            dy = (dy > 0) ? -1 : ((dy < 0) ? 1 : 0);
            if (GradientModule[y + int(dy)][x + int(dx)] <= GradientModule[y][x])
                zeroing_image[y + int(dy)][x + int(dx)] = 0;
            if (GradientModule[y - int(dy)][x - int(dx)] <= GradientModule[y][x])
                zeroing_image[y - int(dy)][x - int(dx)] = 0;
            zeroing_image[y][x] = GradientModule[y][x];
        }
    }
    result = doubletomat(zeroing_image);
    imshow("Zeroing image", result);

    // применение порога
    for (int y = shift; y < length - shift; ++y)
    {
        for (int x = shift; x < length - shift; ++x)
        {
            if (zeroing_image[y][x] >= 255 * max)
                filtered_image[y][x] = 255;
            else if (zeroing_image[y][x] < 255 * min)
                filtered_image[y][x] = 0;
            else
                filtered_image[y][x] = 127;
        }
    }
    result = doubletomat(filtered_image);
    imshow("Filtered image", result);

    // восстановление
    for (int y = shift; y < length - shift; ++y)
    {
        for (int x = shift; x < length - shift; ++x)
            if (filtered_image[y][x] == 127)
            {
                for (int i = -1; i < 2; ++i)
                    for (int j = -1; j < 2; ++j)
                        if (filtered_image[y + i][x + j] == 255)
                            filtered_image[y][x] = 255;
                if (filtered_image[y][x] == 127)
                    filtered_image[y][x] = 0;
            }
    }
    result = doubletomat(filtered_image);
    imshow("Result image", result);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
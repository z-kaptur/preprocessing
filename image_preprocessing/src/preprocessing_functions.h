/**
* @file preprocessing_functions.h
* @brief  Preprocessing namespace with functions and types for image preprocessing.
*/

#ifndef PREPROCESSING_FUNCTIONS_H
#define PREPROCESSING_FUNCTIONS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

/**
 *  Enum for available types of filters.
 *  Default char values for easier commandline arguments parsing.
 */
enum FilterType {
	gaussian = 'g', 
	sobel = 's',
	median = 'm'
};

/**
*  Struct containing both chrominances.
*/
struct Chrominances {
	cv::Mat u;
	cv::Mat v;
};

/**
*  Struct containing luminance and chrominances.
*/
struct YuvImage {
	cv::Mat luminance;
	Chrominances chrominances;
};

namespace preprocessing
{
	/**
	 * @brief Filtering grayscale image.
	 * @param grayscale_img input image (cv::Mat), modified in the function 
	 * @param type filter type (gaussian, median or sobel)
	 */
	void Filter(cv::Mat& grayscale_img, FilterType type);

	/**
	* @brief Subtracting mean value from all the points in the input image.
	* @param grayscale_img input image (cv::Mat), modified in the function
	*/
	void SubtractMean(cv::Mat& grayscale_img);

	/**
	* @brief Total inversion of pixel values in a grayscale image (negative image)
	* @param grayscale_img input image (cv::Mat), modified in the function
	*/
	void ConvertToNegative(cv::Mat& grayscale_img);

	/**
	 * @brief Converting a BGR image to YUV (chrominances decimated 2x2)
	 * @param input_img BGR (3-channel) image (cv::Mat)
	 * @returns YUV image
	 */
	YuvImage ConvertToYuv(const cv::Mat& input_img);

	/**
	 * @brief Calculating pca parameters based on maximum number of components
	 * @param data cv::Mat with input images as rows (flattened to 1 dimension)
	 * @param max_components maximum number of components to be retained
	 * @returns  cv::PCA structure with computed mean, eigenvector and eigenvalues
	 */
	cv::PCA PcaBase(cv::Mat& data, int max_components);

	/**
	* @brief Calculating pca parameters based on percentage of variance to be retained
	* @param data cv::Mat with input images as rows (flattened to 1 dimension)
	* @param retained_variance percentage of variance to be retained
	* @returns  cv::PCA structure with computed mean, eigenvector and eigenvalues
	*/
	cv::PCA PcaBase(cv::Mat& data, double retained_variance);
	
	/**
	 * @brief Calculating psnr between two images
	 * @param img1 first image (cv::Mat)
	 * @param img2 second image (cv::Mat)
	 */
	double CompareImages(cv::Mat img1, cv::Mat img2);

	/**
	 * @brief Normalization of image to 8 bit representation (pixel values 0-255)
	 * @param input input image (modified in the function)
	 */
	void Normalize8Bit(cv::Mat& input);

}

#endif // !PREPROCESSING_FUNCTIONS_H
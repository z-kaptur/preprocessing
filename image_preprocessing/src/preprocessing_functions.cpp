/**
 * @file preprocessing_functions.cpp
 * @brief  Implementation of functions for image preprocessing.
 *
 */

#include "preprocessing_functions.h"
using namespace cv;
using namespace std;

namespace preprocessing
{
	/**
	 * Filter processes 1-channel Mat. 
	 * When a 3-channel Mat is passed, it is converted to grayscale and then processed.
	 * Incorrect filter type should raise invalid argument error.
	 */
	void Filter(Mat& grayscale_img, const FilterType type) 
	{
		if (type == sobel) {
			//Applying horizontal and vertical sobel filters and calculating the average.
			Mat sobel_x, sobel_y;
			Mat sobel_x_8u, sobel_y_8u;
			Sobel(grayscale_img, sobel_x, CV_32F, 1, 0, 3);
			Sobel(grayscale_img, sobel_y, CV_32F, 0, 1, 3);
			convertScaleAbs(sobel_x, sobel_x_8u);
			convertScaleAbs(sobel_y, sobel_y_8u);
			addWeighted(sobel_x_8u, 0.5, sobel_y_8u, 0.5, 0, grayscale_img);
		}
		if (type == median) {
			medianBlur(grayscale_img, grayscale_img, 5);
		}
		if (type == gaussian) {
			GaussianBlur(grayscale_img, grayscale_img, Size(5, 5), 0, 0);
		}
	}

	/**
	 * Subtracting mean works for Mat with any number of channels and data type (CV_8U/ CV_32F)
	 */
	void SubtractMean(Mat& grayscale_img) 
	{
		grayscale_img = grayscale_img - mean(grayscale_img);
	}

	/**
	* ConvertToNegative processes 1-channel Mat.
	* When a 3-channel Mat is passed, it is converted to grayscale and then processed.
	* Works for CV_8U (pixel values 0-255) and CV_32F (pixel values 0-1), otherwise throws an invalid argument error.
	*/
	void ConvertToNegative(Mat& grayscale_img)
	{
		if(grayscale_img.channels()==3) cvtColor(grayscale_img, grayscale_img, CV_BGR2GRAY);

		if(grayscale_img.type()==CV_8U) grayscale_img = 255 - grayscale_img;
		else if (grayscale_img.type() == CV_32F) grayscale_img = 1 - grayscale_img;
		else throw invalid_argument("Cannot convert to negative: uncorrect Mat type (only CV_8U and CV_32F accepted)");
	}

	/**
	 * ConvertToYuv processes 3-channel mat (BGR).
	 * When a grayscale Mat is passed, it should throw an invalid argument error.
	 * Returns YuvImage object: full-size luminance and 2 chrominances decimated 2x2
	 */
	YuvImage ConvertToYuv(const Mat& input_img) 
	{
		Mat temp_yuv_image;
		cvtColor(input_img, temp_yuv_image, CV_BGR2YUV);

		Mat yuv_mat[3];
		yuv_mat[0] = Mat::zeros(temp_yuv_image.cols, temp_yuv_image.rows, CV_16S);
		yuv_mat[1] = Mat::zeros(temp_yuv_image.cols, temp_yuv_image.rows, CV_16S);
		yuv_mat[2] = Mat::zeros(temp_yuv_image.cols, temp_yuv_image.rows, CV_16S);
		split(temp_yuv_image, yuv_mat);

		// decimating chrominances
		Chrominances chrominances;
		resize(yuv_mat[1], chrominances.u, { 0, 0 }, 0.5, 0.5);
		resize(yuv_mat[2], chrominances.v, { 0, 0 }, 0.5, 0.5);

		return{ yuv_mat[0], chrominances };
	}

	PCA PcaBase(Mat& data, int max_components) {

		PCA pca;
		pca(data, Mat(), PCA::DATA_AS_ROW,	max_components);
		return pca;
	}

	PCA PcaBase(Mat& data, double retained_variance) {

		PCA pca;
		pca(data, Mat(), PCA::DATA_AS_ROW, retained_variance);
		return pca;
	}


	/**
	 * Normalization works for all Mat types (apart from CV_8S) 
	 */
	void Normalize8Bit (Mat& input)
	{
		double min, max, delta = 0;
		minMaxLoc(input, &min, &max);
		delta = max - min;
		input -= min;
		input = input.mul(255 / delta);
		if(input.type()!=CV_8U) input.convertTo(input, CV_8U);
		
	}

	/** 
	 * Calculates psnr between the two grayscale pictures 
	 * based on mean squared error between corresponding pixels values.
	 * When 3-channel Mat is passed it is converted to grayscale.
	 */
	double CompareImages(Mat img1, Mat img2)
	{
		if(img1.channels()!=img2.channels())
		{
			if(img1.channels()==3) cvtColor(img1, img1, CV_BGR2GRAY);
			else cvtColor(img2, img2, CV_BGR2GRAY);
		}

		Normalize8Bit(img1);
		Normalize8Bit(img2);

		Mat diff;
		absdiff(img1, img2, diff);      
		
		diff.convertTo(diff, CV_32F); 
		diff = diff.mul(diff);          
	
		waitKey(0);
		Scalar sum_diff = sum(diff);        

		double total_sq_diff = 0;

		for (auto channel_diff : sum_diff.val) total_sq_diff += channel_diff;
	
		if (total_sq_diff <= 1e-10) return numeric_limits<double>::infinity();
		
		double  mean_squared_error = total_sq_diff / static_cast<double>(img1.channels() * img2.total());
		double psnr = 10.0*log10((255 * 255) / mean_squared_error);
		return psnr;
		
	}
}
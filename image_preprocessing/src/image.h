/**
* @file image.h
* @brief Image class.
*/
#ifndef IMAGE_H
#define IMAGE_H

#include<memory>
#include "preprocessing_functions.h"
#include<string>
#include <opencv2/core/core.hpp>
#include <tuple>
#include <iostream>

/**
 * @brief Struct representing processing configuration - operations to be performed during image processing.
 */
struct ProcessingConfiguration {

	/**
	 * @brief Default ProcessingConfiguration constructor.
	 */
	ProcessingConfiguration();

	/**
	 * @brief ProcessingConfiguration constructor.
	 * @param format image format - CV_LOAD_IMAGE_GRAYSCALE/CV_LOAD_IMAGE_COLOR
	 * @param filter filter flag - enables use of filters
	 * @param filter_types vector of filter types to be performed (in order)
	 * @param mean subtracting mean flag
	 * @param negative converting to negative flag
	 * @param pca pca flag
	 */
	ProcessingConfiguration(int format, bool filter, std::vector<FilterType> filter_types, bool mean, bool negative, bool pca);

	int format;
	bool filter;
	std::vector<FilterType> filter_types;
	bool mean;
	bool negative;
	bool pca;
};

/**
 * @brief Class representing a single image
 */
class Image {

public:

	/**
	 * @brief Image constructor.
	 * @param path path to the image file
	 * @param label image label (category)
	 */
	Image(const std::string path, int label);

	/**
	* @brief Image constructor.
	* @param img image matrix (cv::Mat)
	* @param label image label (category)
	*/
	Image(cv::Mat img, int label);

	/**
	 * @brief Setting configuration for all the image objects.
	 */
	static void SetCfg(ProcessingConfiguration& new_cfg);

	/**
	 * @brief Preparing data for primal components analysis.
	 */
	std::shared_ptr<cv::Mat> PcaPrepare();

	/**
	 * @brief Processing and formatting original image data according to processing configuration.
	 * @returns processed data in the form of vector of floats
	 */
	std::shared_ptr<std::vector<float>> ProcesssAndFormatData();

	/**
	* @brief Processing and formatting original image data according to processing configuration (with pca analysis).
	* @returns processed data in the form of vector of floats
	*/
	std::shared_ptr<std::vector<float>> ProcesssAndFormatData(cv::PCA& pca_vector);
	
	auto GetOriginal() const { return *original; }
	int GetSize() const { return static_cast<int>(original->total()); }
	int GetLabel() const { return label; }

private:
	static ProcessingConfiguration cfg; 
	std::unique_ptr<cv::Mat> original; /**< Original image */
	std::shared_ptr<cv::Mat> processed; /**< Processed image ready for pca: 1-dimension Mat (memory allocation only if pca is chosen in cfg) */
	std::shared_ptr<std::vector<float>> formatted; /**< Processed and formatted image data */
	int label; /**< Image label/category */

	/**
	* @brief Formatting image matrix (cv::Mat) to vector of floats and saving it in member variable (formatted). 
	*/
	void FormatMatForNn(cv::Mat&) const;

	/**
	* @brief Formatting grayscale image matrix and chrominances to vector of floats and saving it in member variable (formatted).
	* @param grayscale pointer to matrix (cv::Mat) with grayscale image
	* @param color pointer to chrominances structure
	*/
	void FormatDataForNn(std::unique_ptr<cv::Mat> grayscale, std::unique_ptr<Chrominances> color);

	/**
	* @brief Performing image processing according to configuration
	* @returns tuple with pointers to grayscale Mar and pointer to Chrominances structure (nullptr if color option is not chosen)
	*/
	std::tuple<std::unique_ptr<cv::Mat>,std::unique_ptr<Chrominances>> Process() const;


};


#endif // !IMAGE_H


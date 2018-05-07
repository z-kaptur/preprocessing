/**
* @file data_loader.h
* @brief Loading and managing images (DataLoader class).
*/

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "image.h"
#include<string>
#include <Windows.h>
#include <random>
#include<fstream>

/**
 * @brief Reading images, storing and managing image data.
 * The class is used for searching for image files in given paths, storing images in a vector, 
 * loading data prepared for neural network input and saving processed data to file.
 */
class DataLoader {

public:
	/**
	 * @brief DataLoader constructor.
	 * @param path path to the folder containing subfolders with categorized images
	 * @param num_categories number of categories
	 * @param cfg ProcessingConfiguration struct containing processing options
	 * @param extensions vector of strings corresponding to allowed file extensions. Default: .JPEG, .jpg, .jpeg, .png, .bmp
	 */
	DataLoader(std::string path, 
		const int num_categories, 
		ProcessingConfiguration cfg,
		std::vector<std::string>  extensions = {".JPEG",".jpg",".jpeg",".png",".bmp"});

	/**
	 * @brief Searching and reading image data from path.
	 * @param random_shuffle flag for shuffling the data in random order (after reading)
	 * @returns number of images successfully read
	 */
	int ReadData(bool random_shuffle = false);

	/**
	 * @brief Loading next image processed and formatted for neural network input.
	 */
	std::shared_ptr<std::vector<float>> LoadNextImage();

	/**
	 * @brief Saving all the processed and formatted images to a file.
	 * @param path path where the file should be saved
	 */
	void SaveFormattedData(std::string path);

	int GetNumImages() const { return num_images; }
	
	/**
	 * @brief Reading previosly saved processed data from a file.
	 * @param path path to the file
	 * @param data_vector vector of vector of floats to be filled with image data
	 * @param labels vector of ints to be filled with corresponding labels
	 */
	static void ReadVector(std::string path, std::vector<std::vector<float> >& data_vector, std::vector<int>& labels);

private:
	std::vector<std::unique_ptr<Image>> images; /**< Vector of pointers to loaded images */
	std::string path; /**< Path to the folder containing image data */
	int num_categories; /**< Number of image categories */
	ProcessingConfiguration cfg; /**< Image processing options */
	std::vector<std::string> allowed_extentions; /**< Allowed file extensions when searching for image files */
	int num_images; /**< Number of images read */
	int current_index; /**< Current index of image - for loading images one by one */
	cv::PCA pca_vector; /**< Pca parameters for whole vector of images */

	/**
	 * @brief Searching for files with allowed extensions in a folder.
	 * @param path path to the folder
	 * @param filenames string vector to be filled with filenames of image files in a folder
	 */
	void ReadFilenames(const std::string& path, std::vector<std::string>& filenames);

	/**
	 * @brief Reading all image files in a folder and adding them to images vector.
	 * @param filenames string vector of all filenames that should be read
	 * @param label label of images within the folder
	 */
	int ReadAllFromDirectory(std::vector<std::string>& filenames, int label);
	
	/**
	 * @brief Random shuffle of images in the vector.
	 */
	void ShuffleImages();

	/**
	* @brief Calculate PCA parameters for set of images in the member vector images
	*/
	cv::PCA PcaCalculate();


};


#endif // !DATA_LOADER_H

/**
* @file data_loader.cpp
* @brief Loading and managing images (DataLoader class) - implementation.
*/
#include "data_loader.h"
 

using namespace std;
using namespace cv;

DataLoader::DataLoader(string path, const int num_categories, ProcessingConfiguration cfg,vector<string>  extensions): 
path(move(path)), num_categories(num_categories), cfg(move(cfg)), allowed_extentions(std::move(extensions)), num_images(0), current_index(0)
{
	if (num_categories < 2) throw invalid_argument("DataLoader constructor: There must be at least 2 categories for classification");
	Image::SetCfg(cfg);
}

/**
 * Reading data from folder in path member variable. 
 * Folder should contain subfolders corresponding to particular image categories. 
 * The categories have to be mapped to integers (0,num_categories-1)
 * Folder structure should be as follows:
 *	path
 *		0
 *			image1.jpg
 *			...
 *			imagex.jpg
 *		1
 *			image1.jpg
 *			...
 *			imagex.jpg
 *		...
 *		num_images-1
 *			image1.jpg
 *			...
 *			imagex.jpg
 *	
 *	Filenames are irrevelant, any files with allowed extensions will be read. All images must have the same size. 
 */
int DataLoader::ReadData(bool random_shuffle) {
	vector<vector<string>> filenames;
	vector<string> filenames_in_dir;
	int num_files = 0;

	for (int i = 0; i < num_categories; i++)
	{
		cerr << path + to_string(i) + "/";
		ReadFilenames(path + to_string(i) + "/", filenames_in_dir);
		num_files += ReadAllFromDirectory(filenames_in_dir, i);
	}

	if (cfg.pca) pca_vector = PcaCalculate();
	if (random_shuffle) ShuffleImages();
	num_images = num_files;
	return num_files;
}


std::shared_ptr<std::vector<float>> DataLoader::LoadNextImage()
{
	auto current_image = (cfg.pca) ? images[current_index]->ProcesssAndFormatData() : images[current_index]->ProcesssAndFormatData(pca_vector);
	current_index++;
	if (current_index == num_images)
	{
		current_index = 0;
		ShuffleImages();
	}
	return current_image;
}


/**
 * File format:
 * |number of images (int)| number of images x ||number of image points (int)| number of image points x |image point value (float)|| number of images x |image label (int)|
 *
 */
void DataLoader::SaveFormattedData(std::string path)
{
	if (cfg.pca) pca_vector = PcaCalculate();

	ofstream file(path, std::ios::out | std::ofstream::binary | ios::trunc);

	file.write(reinterpret_cast<const char *>(&num_images), sizeof(num_images));

	shared_ptr<vector<float>> formatted_vector;

	for (auto& img : images) {
		if (cfg.pca)  formatted_vector = img->ProcesssAndFormatData(pca_vector);
		else formatted_vector = img->ProcesssAndFormatData();

		int size = static_cast<int>(formatted_vector->size());
		file.write(reinterpret_cast<const char *>(&size), sizeof(size));

		for (float item : *formatted_vector) {
			file.write(reinterpret_cast<const char *>(&item), sizeof(float));
		}
	}
	for (auto& img : images) {
		int label = img->GetLabel();
		file.write(reinterpret_cast<const char*>(&label), sizeof(label));
	}

	file.close();
}

/**
* File format:
* |number of images (int)| number of images x ||number of image points (int)| number of image points x |image point value (float)|| number of images x |image label (int)|
*
*/
void DataLoader::ReadVector(std::string path, std::vector<std::vector<float>>& data_vector, vector<int>& labels)
{
	ifstream file(path, std::ios::in | std::ifstream::binary);

	int size = 0;
	file.read(reinterpret_cast<char *>(&size), sizeof(size));

	data_vector.resize(size);
	labels.resize(size);
	for (int i = 0; i < size; i++) {
		int size2 = 0;
		file.read(reinterpret_cast<char *>(&size2), sizeof(size2));
		float f;
		for (int k = 0; k < size2; k++) {
			file.read(reinterpret_cast<char *>(&f), sizeof(float));
			data_vector[i].push_back(f);
		}
	}
	for (int i = 0; i < size; i++) {
		int label = 0;
		file.read(reinterpret_cast<char *>(&label), sizeof(label));
		labels[i] = label;
	}
	file.close();
}

/**
 * Works on WindowsOS
 * Adds files with correct extensions to filenames vector.
 * Empty or non-existent folder causes invalid argument exception.
 */
void DataLoader::ReadFilenames(const string& path, vector<string>& filenames) {
	HANDLE dir;
	WIN32_FIND_DATA file_data;
	filenames.clear();

	// Project -> Properties -> General -> Character Set -> Use Multi-Byte Character Set

	if ((dir = FindFirstFile((path + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
	{
		throw invalid_argument("There is no folder with a given path (" + path + ") or it is empty");
		return; /* No files found */
	}
	do {
		const string file_name = file_data.cFileName;
		bool correct_type = false;
		for (auto& extension : allowed_extentions)
		{
			if (file_name.find(extension) != string::npos) correct_type = true;
		}
		const string full_file_name = path + file_name;

		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		if (!correct_type) continue;
		
		filenames.push_back(full_file_name);

	} while (FindNextFile(dir, &file_data));

	if (filenames.empty()) throw invalid_argument("There is no folder with a given path (" + path + ") or it is empty");

	FindClose(dir);
}


/**
 * All the images have to have the same size.
 */
int DataLoader::ReadAllFromDirectory(vector<string>& filenames, int label) {
	int data_dimension = -1;
	for (auto file : filenames)
	{
		try {
			images.emplace_back(make_unique<Image>(file, label));
		}
		catch(const invalid_argument& e)
		{
			cerr << e.what() << endl;
			throw;
		}
		catch(const exception& e)
		{
			cerr << "Error while adding an image to a list" << endl;
			cerr << "Error message: " << e.what() << endl;
			throw;
		}

		auto current_dim = images.back()->GetSize();

		if ((current_dim != data_dimension) && data_dimension != -1) throw invalid_argument("Inconsistent data size!");

		data_dimension = current_dim;

	}
	return static_cast<int>(filenames.size()); //number of images read
}


void DataLoader::ShuffleImages()
{
	shuffle(images.begin(), images.end(), default_random_engine());
}

PCA DataLoader::PcaCalculate()
{
	Mat m;
	for (auto& img : images) m.push_back(*img->PcaPrepare());

	return preprocessing::PcaBase(m, 100);
}




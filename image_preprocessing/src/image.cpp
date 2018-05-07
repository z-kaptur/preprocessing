/**
* @file image.cpp
* @brief Implementation of Image class.
*/

#include "image.h"

using namespace std;
using namespace cv;


/**
 * format - CV_LOAD_IMAGE_GRAYSCALE
 * filter - false
 * mean - false
 * negative - false
 * pca - false
 */
ProcessingConfiguration::ProcessingConfiguration() :
	format(CV_LOAD_IMAGE_GRAYSCALE), filter(false), filter_types({}), mean(false), negative(false), pca(false)
{
}

ProcessingConfiguration::ProcessingConfiguration(int format, bool filter, vector<FilterType> filter_types, bool mean, bool negative, bool pca) :
	format(format), filter(filter), filter_types(filter_types), mean(mean), negative(negative), pca(pca)
{
}

/**
 * Initial configuration values
 */
ProcessingConfiguration Image::cfg (CV_LOAD_IMAGE_GRAYSCALE, false, {}, false, false, false);

void Image::SetCfg(ProcessingConfiguration& new_cfg)
{
	Image::cfg = new_cfg;
	if (new_cfg.format != CV_LOAD_IMAGE_COLOR && new_cfg.format != CV_LOAD_IMAGE_GRAYSCALE)
	{
		throw invalid_argument("Invalid format, only CV_LOAD_IMAGE_COLOR (" + to_string(CV_LOAD_IMAGE_COLOR) +
			") or CV_LOAD_IMAGE_GRAYSCALE (" + to_string(CV_LOAD_IMAGE_GRAYSCALE) + ") available");
	}
	
};



Image::Image(const string path, int label): original(make_unique<Mat>(imread(path, cfg.format))), formatted(nullptr), label(label)
{
	if (!original || (original->cols == 0 && original->rows == 0)) throw invalid_argument("Image constructor: Invalid path: "+path+" , image could not be read");
}

Image::Image(Mat img, int label): original(make_unique<Mat>(img)), formatted(nullptr), label(label)
{
	if (!original || (original->cols == 0 && original->rows == 0)) throw invalid_argument("Image constructor: Empty Mat");
}

/**
 * Accepts Mats with format CV_8U (pixel values 0-255) or CV_32F (pixel values 0-1)
 */
void Image::FormatMatForNn(Mat& img) const
{
	Mat out_mat;
	if (img.type() == CV_8U) {
		img.convertTo(out_mat, CV_32F);
		out_mat /= 256;
	}
	else out_mat = img;
	for (int i = 0; i < out_mat.rows; ++i) {
		formatted->insert(formatted->end(), out_mat.ptr<float>(i), out_mat.ptr<float>(i) + out_mat.cols);
	}
}


void Image::FormatDataForNn(unique_ptr<Mat> grayscale, unique_ptr<Chrominances> color)
{
	if (formatted) formatted->clear();
	FormatMatForNn(*grayscale);

	if (color) {
		FormatMatForNn(color->u);
		FormatMatForNn(color->v);
	}
}

std::tuple<std::unique_ptr<cv::Mat>, std::unique_ptr<Chrominances>> Image::Process() const
{
	unique_ptr<Mat> grayscale = nullptr;
	unique_ptr<Chrominances> color = nullptr;

	if (cfg.format == CV_LOAD_IMAGE_COLOR) {
		YuvImage yuvImg = preprocessing::ConvertToYuv(*original);
		grayscale = make_unique<Mat>(yuvImg.luminance);
		color = make_unique<Chrominances>(yuvImg.chrominances);
	}
	else {
		grayscale = make_unique<Mat>(*original);
	}

	if (cfg.mean) preprocessing::SubtractMean(*grayscale);

	if (cfg.filter) {
		for (auto f : cfg.filter_types) {
			preprocessing::Filter(*grayscale, f);
		}
	}

	if (cfg.negative) preprocessing::ConvertToNegative(*grayscale);
	
	return make_tuple(move(grayscale), move(color));
}

/**
 * Processes image and transforms 2-D Mat into 1-D Mat.
 */
shared_ptr<Mat> Image::PcaPrepare()
{
	auto processed_img_with_color = Process();
	const auto processed_img = move(get<0>(processed_img_with_color));

	Mat dst(1, processed_img->rows*processed_img->cols, CV_32F);

	dst = processed_img->reshape(1, 1);

	processed = make_shared<Mat>(dst);
	
	return processed;
}


shared_ptr<vector<float>> Image::ProcesssAndFormatData()
{
	if (formatted == nullptr) {
		formatted = make_shared<vector<float>>();
		unique_ptr<Mat> grayscale;
		unique_ptr<Chrominances> color;
		tie(grayscale, color) = Process();
		
		FormatDataForNn(move(grayscale), move(color));
	}

	return formatted;

}

/**
 * Overloaded function used when pca is included.
 */
shared_ptr<vector<float>> Image::ProcesssAndFormatData(PCA& pca_vector)
{
	if (formatted == nullptr) {
		formatted = make_shared<vector<float>>();
		Mat point = pca_vector.project((*processed).reshape(1,1));
		FormatDataForNn(make_unique<Mat>(point), nullptr);
	}
	return formatted;
}




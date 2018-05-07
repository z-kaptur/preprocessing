/**
* @file preprocessing_functions_tests.cpp
* @brief Unit tests for preprocessing functions.
*/

#include "Catch.h"
#include "../../image_preprocessing/src/preprocessing_functions.h"
using namespace std;
using namespace cv;

TEST_CASE("Filter() should not change the input Mat size and shape") {
	Mat test = imread("../../image_preprocessing/tests/samples/1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	auto cols_before = test.cols;
	auto rows_before = test.rows;
	auto size_before = test.total();
	preprocessing::Filter(test, sobel);
	auto cols_after = test.cols;
	auto rows_after = test.rows;
	auto size_after = test.total();
	
	auto is_same = (cols_before == cols_after) && (rows_before == rows_after) && (size_before == size_after);
	REQUIRE(is_same);
}

TEST_CASE("When 3-channel Mat passed to Filter() then it returns grayscale Mat") {
	Mat test = imread("../../image_preprocessing/tests/samples/1.jpg",CV_LOAD_IMAGE_COLOR);
	preprocessing::Filter(test, sobel);

	REQUIRE(test.channels()==1);
}


TEST_CASE("When the same grayscale images passed to CompareImages() then it returns +inf psnr value") {
	Mat test = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	auto psnr = preprocessing::CompareImages(test, test);

	REQUIRE(psnr==numeric_limits<double>::infinity());

}

TEST_CASE("When the same color images passed to CompareImages() then it returns +inf psnr value") {
	Mat test = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_COLOR);
	auto psnr = preprocessing::CompareImages(test, test);

	REQUIRE(psnr == numeric_limits<double>::infinity());
}

TEST_CASE("When the same images (one read as 3-channel, second as 1-channel image) passed to CompareImgaes() then it returns psnr greater than 50dB") {
	Mat test_color = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat test_grayscale = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	auto psnr = preprocessing::CompareImages(test_color, test_grayscale);

	REQUIRE(psnr > 50);
}


TEST_CASE("When pca image is reconstructed, then its psnr is greater than 30") {
	vector<Mat> images;
	string path = "../../image_preprocessing/tests/samples/pca/";
	vector<string> files = { "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg","9.jpg","10.jpg" };

	for (auto const& file : files) images.push_back(imread(path + file, CV_LOAD_IMAGE_GRAYSCALE));

	Mat images_for_pca(static_cast<int>(images.size()), images[0].rows*images[0].cols, CV_32F);

	//cerr << "rows" << images_for_pca.rows << " cols" << images_for_pca.cols << endl;
	for (auto i = 0; i < images.size(); i++)
	{
		Mat image_row = images[i].clone().reshape(1, 1);
		Mat row_i = images_for_pca.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	Mat test_img = imread("../../image_preprocessing/tests/samples/pca/2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_img_row = test_img.clone().reshape(1, 1);

	SECTION("pca based on variance") {
		PCA pca = preprocessing::PcaBase(images_for_pca, 1.0);
		
		auto projection = pca.project(test_img_row);
		auto reconstruction = pca.backProject(projection);
		auto reconstruction_2d = reconstruction.clone().reshape(1, 64);
		auto psnr = preprocessing::CompareImages(test_img, reconstruction_2d);
		REQUIRE(psnr > 30);
	}

	SECTION("pca based on max components") {
		PCA pca = preprocessing::PcaBase(images_for_pca, 0);

		auto projection = pca.project(test_img_row);
		auto reconstruction = pca.backProject(projection);
		auto reconstruction_2d = reconstruction.clone().reshape(1, 64);
		auto psnr = preprocessing::CompareImages(test_img, reconstruction_2d);
		REQUIRE(psnr > 30);
	}
}

TEST_CASE("When 1-channel Mat passed to ConvertToYuv then throw invalid argument error") {
	Mat test_grayscale = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	REQUIRE_THROWS_AS(preprocessing::ConvertToYuv(test_grayscale),invalid_argument);
}

TEST_CASE("ConvertToYuv() should return structure with full-size luminance and 4x decimated chrominances") {
	Mat test_color = imread("../../image_preprocessing/tests/samples/1.jpg", CV_LOAD_IMAGE_COLOR);
	auto yuv_image = preprocessing::ConvertToYuv(test_color);

	auto y_total = yuv_image.luminance.total();
	auto u_total = yuv_image.chrominances.u.total();
	auto v_total = yuv_image.chrominances.v.total();

	REQUIRE(test_color.total() == 3 * y_total);
	REQUIRE(u_total == v_total);
	REQUIRE(y_total == 4 * v_total);
}
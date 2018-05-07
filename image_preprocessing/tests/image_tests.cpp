/**
* @file data_loader_tests.cpp
* @brief Unit tests for Image class.
*/

#define CATCH_CONFIG_MAIN

#include "Catch.h"
#include "../../image_preprocessing/src/image.h"
#include <chrono>

using namespace std;
using namespace cv;

TEST_CASE("When invalid path passed to Image constructor then throw an invalid_argument exception") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);
	
	REQUIRE_THROWS_AS(make_unique<Image>("../../image_preprocessing/tests/samples/doesnotexist.jpg",1),invalid_argument);
}

TEST_CASE("When 64x64 grayscale image read then ProcesssAndFormatData() returns 4096-element vector ") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	cfg.filter = true;
	cfg.filter_types = { median,gaussian,sobel };
	cfg.mean = true;
	cfg.negative = true;
	Image::SetCfg(cfg);

	Image testImg("../../image_preprocessing/tests/samples/1.jpg",1);
	
	auto formatted = *testImg.ProcesssAndFormatData();
	REQUIRE(formatted.size() == 4096);
}

TEST_CASE("When 64x64 color image read then ProcesssAndFormatData() returns 6144-element vector ") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_COLOR;
	cfg.filter = true;
	cfg.filter_types = { median,gaussian,sobel };
	cfg.mean = true;
	cfg.negative = true;
	Image::SetCfg(cfg);

	Image test_img("../../image_preprocessing/tests/samples/1.jpg",1);

	auto formatted = *test_img.ProcesssAndFormatData();
	REQUIRE(formatted.size() == 6144);
}

TEST_CASE("When image processing requested second time then processing time should be shorter") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_COLOR;
	cfg.filter = true;
	cfg.filter_types = { median,gaussian,sobel };
	cfg.mean = true;
	cfg.negative = true;
	Image::SetCfg(cfg);

	Image test_img("../../image_preprocessing/tests/samples/1.jpg", 1);

	auto start = chrono::high_resolution_clock::now();
	auto formatted = test_img.ProcesssAndFormatData();
	auto finish = chrono::high_resolution_clock::now();
	auto first_time = finish - start;

	start = chrono::high_resolution_clock::now();
	formatted = test_img.ProcesssAndFormatData();
	finish = chrono::high_resolution_clock::now();
	auto second_time = finish - start;

	REQUIRE(second_time.count()<first_time.count());
}

TEST_CASE("When trying to set uncorrect configuration then invalid_argument exception ") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_UNCHANGED;
	cfg.filter = true;
	cfg.filter_types = { median,gaussian,sobel };
	cfg.mean = true;
	cfg.negative = true;
	
	REQUIRE_THROWS_AS(Image::SetCfg(cfg), invalid_argument);
}



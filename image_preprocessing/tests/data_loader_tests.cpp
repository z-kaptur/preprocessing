/**
* @file data_loader_tests.cpp
* @brief Unit tests for DataLoader class.
*/

#include "Catch.h"
#include "../../image_preprocessing/src/data_loader.h"
using namespace std;
using namespace cv;

TEST_CASE("When one folder is missing then ReadData() throws an exception") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);

	DataLoader data_loader("../../image_preprocessing/tests/samples/folder_missing/", 5, cfg);

	REQUIRE_THROWS_AS(data_loader.ReadData(), invalid_argument);
}

TEST_CASE("When any folder is empty then ReadData() throws an exception") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);

	DataLoader data_loader("../../image_preprocessing/tests/samples/folder_empty/", 5, cfg);

	REQUIRE_THROWS_AS(data_loader.ReadData(), invalid_argument);
}


TEST_CASE("When not every image has the same size then ReadData() throws an exception") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);

	DataLoader data_loader("../../image_preprocessing/tests/samples/inconsistent_size/", 5, cfg);

	REQUIRE_THROWS_AS(data_loader.ReadData(), invalid_argument);
}


TEST_CASE("When ReadData() comes across a file of wrong type then it ignores it") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);

	DataLoader data_loader("../../image_preprocessing/tests/samples/different_types/", 5, cfg);

	REQUIRE(data_loader.ReadData()==15);
	REQUIRE(data_loader.GetNumImages() == 15);
}


TEST_CASE("When there are 50 files in the folders then ReadData() returns 50") {
	ProcessingConfiguration cfg;
	cfg.format = CV_LOAD_IMAGE_GRAYSCALE;
	Image::SetCfg(cfg);

	DataLoader data_loader("../../image_preprocessing/tests/samples/50/", 5, cfg);

	REQUIRE(data_loader.ReadData() == 50);
	REQUIRE(data_loader.GetNumImages() == 50);
}

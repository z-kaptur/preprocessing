#include "image.h"
#include "preprocessing_functions.h"
#include "data_loader.h"
#include <tclap/CmdLine.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <map>
using namespace std;


int main(int argc, char** argv)
{
	try {

		TCLAP::CmdLine cmd("", ' ', "1");


		TCLAP::ValueArg<std::string> i_path("i", "input", "Path to the folder with data", true, "", "string");
		TCLAP::ValueArg<std::string> num_labels("l", "labels", "Number of type of labels (categories)", true, "", "int");
		
		TCLAP::ValueArg<std::string> filters("f", "filter", "Name of filters to be applied (sobel(s)/gaussian(g)/median(m))", false, "", "string");

		TCLAP::ValueArg<std::string> pca_type("p", "pca", "Type of pca analysis", false, "", "string");
		TCLAP::ValueArg<std::string> pca_components("e", "components", "Max number of pca components", false, "", "int");
		TCLAP::ValueArg<std::string> pca_variance("v", "variance", "Pca reatained variance", false, "", "double");
		TCLAP::ValueArg<std::string> s_path("s", "save", "Save path", false, "", "string");

		cmd.add(i_path);
		cmd.add(num_labels);
		cmd.add(filters);
		cmd.add(pca_type);
		cmd.add(pca_components);
		cmd.add(pca_variance);
		cmd.add(s_path);

		TCLAP::SwitchArg negative_switch("n", "negative", "Change the image to negative", cmd, false);
		TCLAP::SwitchArg mean_switch("m", "mean", "Subtract mean", cmd, false);
		TCLAP::SwitchArg color_switch("c", "color", "Read data as color images", cmd, false);

		cmd.parse(argc, argv);


		string input_path = i_path.getValue();
		int num_categories = stoi(num_labels.getValue());
		bool filter = !filters.getValue().empty();
		string filter_t = filters.getValue();
		bool mean = mean_switch.getValue();
		bool negative = negative_switch.getValue();
		bool pca = !pca_type.getValue().empty();
		bool save = !s_path.getValue().empty();
		string save_path = s_path.getValue();
		bool color = color_switch.getValue();

		int type = (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
		
		filter_t.erase(std::remove(filter_t.begin(), filter_t.end(), ' '), filter_t.end());
		vector<char> filter_vector(filter_t.begin(), filter_t.end());
		vector<FilterType> filter_type;

		for (auto f : filter_vector) filter_type.push_back(static_cast<FilterType>(f));

		ProcessingConfiguration cfg(type, filter, filter_type, mean, negative, pca);
		DataLoader data_loader(input_path, num_categories, cfg);
		data_loader.ReadData();
		cerr << "Read succesfully" << endl;
		if (save) data_loader.SaveFormattedData(save_path);
		vector<vector<float>> test;
		vector<int> labels;
		cerr << test.size() << ":" << data_loader.GetNumImages() << endl;
		DataLoader::ReadVector(save_path, test, labels);
		cerr << test.size() << ":" << data_loader.GetNumImages() << endl;
		cerr << labels.size() << ":" << data_loader.GetNumImages() << endl;
		if (test.size() != data_loader.GetNumImages()) throw exception("Saving went wrong");
		if (labels.size() != data_loader.GetNumImages()) throw exception("Saving went wrong");
	}
	catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}
	catch (exception& e)
	{
		cerr << "error: " << e.what() << endl;
	}
	return 0;
}
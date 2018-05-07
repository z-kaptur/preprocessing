# Image preprocessing for neural networks
Console application - image data preprocessing (filtering, PCA, changing color models etc.) and formatting for the use in a neural network. 
Unit testing using Catch framework.

## Usage:
```
   image_preprocessing.exe  [-c] [-m] [-n] -s <string> [-v <double>] [-e
                            <int>] [-p <string>] [-f <string>] -l <int> -i
                            <string> [--] [--version] [-h]
Where:

   -c,  --color
     Read data as color images

   -m,  --mean
     Subtract mean

   -n,  --negative
     Change the image to negative

   -s <string>,  --save <string>
     (required)  Save path

   -v <double>,  --variance <double>
     Pca reatained variance

   -e <int>,  --components <int>
     Max number of pca components

   -p <string>,  --pca <string>
     Type of pca analysis

   -f <string>,  --filter <string>
     Name of filters to be applied (sobel(s)/gaussian(g)/median(m))

   -l <int>,  --labels <int>
     (required)  Number of type of labels (categories)

   -i <string>,  --input <string>
     (required)  Path to the folder with data

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.
```

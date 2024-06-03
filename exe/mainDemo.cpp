/*
* Copyright (c) 2016, Gabor Adam Fodor <fogggab@yahoo.com>
* All rights reserved.
*
* License:
*
* This program is provided for scientific and educational purposed only.
* Feel free to use and/or modify it for such purposes, but you are kindly
* asked not to redistribute this or derivative works in source or executable
* form. A license must be obtained from the author of the code for any other use.
*
*/
#include <platypus/CradleFunctions.h>
#include <platypus/TextureRemoval.h>

/**
* This piece of demo code demonstrates the combined cradle removal and wood-grain separation 
* for an image specified by the user.
* If the image specified does not exist, the code results in an error.
*
* Parameters:
*   arg[1]				path+filename of input image to be processed
*
* Output:
*  "solution.png"				x-ray with cradle and wood-grain removed, saved as grayscale png
*  "difference.png"				cradle component, before wood-grain removal
*  "difference_textrem.png"		cradle component, after wood-grain removal
**/

int main(int argc, char** argv)
{
	//Read in image file as grayscale
	cv::Mat img_orig = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	cv::Mat img(img_orig.rows, img_orig.cols, CV_32F);
	cv::Mat out;

	//Conversion to float
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			img.at<float>(i, j) = (float)img_orig.at<uchar>(i, j);
		}
	}
	
	cv::Mat nointensity, mask, cradle;
	CradleFunctions::MarkedSegments ms;
	
	//Initialize mask file
	mask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
	
	//Step 1: Run cradle removal algorithm
	CradleFunctions::removeCradle(img, nointensity, cradle, mask, ms);
	
	//Step 2: Separate wood grain
	TextureRemoval::textureRemove(nointensity, mask, out, ms);
	
	cv::imwrite("solution.png", out);
	cv::imwrite("difference.png", img - out);
	cv::imwrite("difference_textrem.png", (nointensity - out + 50));
}
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

/**
* This piece of demo code demonstrates cradle removal for an image specified by the user.
* If the image specified does not exist, the code results in an error.
*
* Parameters:
*   arg[1]				path+filename of input image to be processed
*   arg[2]				path+filename of output images (optional; defaults to "out" if not specified)
*   arg[3] & arg[4]		number of horizontal and vertical pieces in the cradled image (optional)
*
* Output:
*  "<outfilename>_original.png"		original, unprocessed image file saved as grayscale png
*  "<outfilename>_nointensity.png"	x-ray with cradle removed, saved as grayscale png
*  "<outfilename>_cradle.png"		cradle component of the separation, saved as grayscale png
*  "<outfilename>_mask.png"			cradle mask file, marking location of cradle pieces
*  "<outfilename>.msf"				an auxiliary file containing processing information, usable for texture removal
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
	std::vector<int> v, h;

	//Initialize mask file
	mask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
	CradleFunctions::MarkedSegments ms;

	//Check if number of cradle pieces specified
	if (argc >= 5){
		int nrh = std::atoi(argv[3]);
		int nrv = std::atoi(argv[4]);
		if (nrh < 0 || nrv <0 || (nrh == 0 && nrv == 0)){
			//Run cradle detection, making an automated estimation of the number of horizontal/vertical cradle pieces
			CradleFunctions::cradledetect(img, mask, v, h);
		}
		else{
			//Run cradle detection, looking for 'nrh' horizontal and 'nrv' vertical cradle pieces
			CradleFunctions::cradledetect(img, mask, nrv, nrh, v, h);
		}
	}
	else{
		//Run cradle detection, making an automated estimation of the number of horizontal/vertical cradle pieces
		CradleFunctions::cradledetect(img, mask, v, h);
	}

	//Run cradle removal algorithm, using approximate locations of cradle pieces stored in 'v' and 'h'
	CradleFunctions::removeCradle(img, nointensity, cradle, mask, v, h, ms);

	std::string filename;
	if (argc < 3)
		filename = "out"; //Use a default name for the output files
	else
		filename = argv[2];


	//Write out image files
	cv::imwrite(filename + "_original.png", img);
	cv::imwrite(filename + "_mask.png", mask);
	cv::imwrite(filename + "_nointensity.png", nointensity);
	cv::imwrite(filename + "_cradle.png", cradle);

	//Write out marked segment file (contains processing information from cradle removal stage, used for the texture removal)
	CradleFunctions::writeMarkedSegmentsFile(filename + ".msf", ms);
}
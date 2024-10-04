#pragma once
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

#ifndef CRADLEFUNCTIONS_H
#define CRADLEFUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>


/**
* Collection of all functions that make cradle removal possible.
* The main functions are cradledetect(), that attempts to find the approximate position
* of horizontal/vertical cradle pieces and removeCradle(), that adjusts the intensity of
* the detected cradle pieces, separating the input image into a cradle and a x-ray part.
**/

namespace CradleFunctions{

	//Masking parameters used during cradle/texture removal
	constexpr unsigned char V_MASK = 1;		//Marks that pixel is part of a vertical cradle piece
	constexpr unsigned char H_MASK = 2;		//Marks that pixel is part of a horizontal cradle piece
	constexpr unsigned char DEFECT = 4;		//Marks that pixel should be ignored during processing due to material defect

	//Four possible categorizations of a pixel (part of horizontal or vertical cradle, both, or none)
	const int NONCRADLE = 0;
	const int HORIZONTAL_DIR = 1;
	const int VERTICAL_DIR = 2;
	const int CROSS_DIR = 3;

	//Structure used to store cradled/non-cradled pixel pairs
	struct cradle_sample_pairs{
		int start, end;					//Start/end indices of samples
		std::vector<float> ncu;			//Non-cradled upper part
		std::vector<float> cu;			//Cradled upper part
		std::vector<float> ncl;			//Non-cradled lower part
		std::vector<float> cl;			//Cradled lower part
	};

	//Marks individual cradle segments in a mask & their central location.
	//Used to provide extra info to the Platypus interface and to aid the
	//texture removal functions.
	struct MarkedSegments{
		int pieces;									//Total number of cradle segments in the image
		cv::Mat piece_mask;							//A mask pairing each pixel with a cradle segment identifier
		std::vector<int> piece_type;				//Type of the cradle segment (nocradle, horizontal, vertical, cross-section)
		std::vector<std::vector<int>> pieceIDh;		//Array of cradle segment identifiers corresponding to each horizontal cradle piece
		std::vector<std::vector<int>> pieceIDv;		//Array of cradle segment identifiers corresponding to each vertical cradle piece
		std::vector<cv::Point2i> piece_middle;		//Middle coordinate of each cradle segment (needed by the Platypus interface)
	};

	//Callback functions for the interface
	struct Callbacks
	{
		virtual bool progress(int value, int total) const = 0;
	};

	void removeCradle(
		const cv::Mat &in,			//Input grayscale float X-ray image
		cv::Mat &out,				//Cradle removed X-ray is saved out here
		cv::Mat &cradle,			//Cradle component after separation saved out here
		cv::Mat &mask,				//Mask containing marked vertical/horizontal cradle positions
		MarkedSegments &ms			//MarkedSegment structure will contain processing information
	);

	void removeCradle(
		const cv::Mat &in,			//Input grayscale float X-ray image
		cv::Mat &out,				//Cradle removed X-ray is saved out here
		cv::Mat &cradle,			//Cradle component after separation saved out here
		cv::Mat &mask,				//Mask containing marked vertical/horizontal cradle positions
		std::vector<int> &vrange,	//Approximate position of vertical cradle pieces, in pairs of (X_start1, X_end1,..,X_startN, X_endN)
		std::vector<int> &hrange,	//Approximate position of horizontal cradle pieces, in pairs of (Y_start1, Y_end1,..,Y_startM, Y_endM)
		MarkedSegments &ms			//MarkedSegment structure will contain processing information
	);

	void removeVertical(
		const cv::Mat &img,									//Input grayscale float X-ray image
		cv::Mat &mask,										//Mask containing marked vertical and/or horizontal cradle positions
		cv::Mat &cradle,									//Cradle component after separation saved out here
		std::vector<std::vector<int>> &midpos_points,		//Center of vertical cradle pieces
		std::vector<int> s,									//Width of vertical cradle pieces
		std::vector<std::vector<std::vector<float>>> &vm,	//Saves out parameters of the fitted multiplicative model, used for processing cross-sections
		MarkedSegments &ms									//MarkedSegment structure will contain processing information
	);
	void removeHorizontal(
		const cv::Mat &img,									//Input grayscale float X-ray image
		cv::Mat &mask,										//Mask containing marked horizontal and/or vertical cradle positions
		cv::Mat &cradle,									//Cradle component after separation saved out here
		std::vector<std::vector<int>> &midpos_points,		//Center of horizontal cradle pieces
		std::vector<int> s,									//Width of horizontal cradle pieces
		std::vector<std::vector<std::vector<float>>> &hm,	//Saves out parameters of the fitted multiplicative model, used for processing cross-sections
		MarkedSegments &ms									//MarkedSegment structure will contain processing information
	);
	void removeCrossSection(
		const cv::Mat &img,									//Input grayscale float X-ray image
		cv::Mat &mask,										//Mask containing marked horizontal and/or vertical cradle positions
		cv::Mat &cradle,									//Cradle component after separation saved out here
		std::vector<int> &hrange,							//Width of horizontal cradle pieces
		std::vector<int> &vrange,							//Width of vertical cradle pieces
		std::vector<std::vector<int>> &midposh_points,		//Center of horizontal cradle pieces
		std::vector<std::vector<int>> &midposv_points,		//Center of vertical cradle pieces
		std::vector<std::vector<std::vector<float>>> &hm,	//Parameters of the fitted multiplicative model for horizontal cradle pieces
		std::vector<std::vector<std::vector<float>>> &vm,	//Parameters of the fitted multiplicative model for vertical cradle pieces
		MarkedSegments &ms									//MarkedSegment structure will contain processing information
	);

	//Guided cradle detection
	void cradledetect(
		const cv::Mat &in,				//Input grayscale float X-ray image
		const cv::Mat &mask,			//On return, mask will contain marked horizontal and/or vertical cradle positions
		int vn, int hn,					//Number of vertical (vn) and horizontal (hn) cradle pieces the algorithm has to find
		std::vector<int> &vrange,		//Position of vertical cradle pieces, in pairs of 2: (X_start1, X_end1,..,X_startN, X_endN)
		std::vector<int> &hrange		//Position of vertical cradle pieces, in pairs of 2: (Y_start1, Y_end1,..,Y_startM, Y_endM)
	);

	//Blind cradle detection
	void cradledetect(
		const cv::Mat &in,				//Input grayscale float X-ray image
		const cv::Mat &mask,			//On return, mask will contain marked horizontal and/or vertical cradle positions
		std::vector<int> &vrange,		//Position of vertical cradle pieces, in pairs of 2: (X_start1, X_end1,..,X_startN, X_endN)
		std::vector<int> &hrange		//Position of vertical cradle pieces, in pairs of 2: (Y_start1, Y_end1,..,Y_startM, Y_endM)
	);

	//Functions used for estimating rotation angle of horizontal/vertical cradle pieces
	std::vector<double> findRadonTransformAngle(const cv::Mat &img, cv::Mat &mask, std::vector<double> &thetav, int sx, int ex, int sy, int ey, int noflag);
	cv::Mat getRadonforAngle(cv::Mat &d, cv::Mat &mask, double theta, int sx, int ex, int sy, int ey, int noflag);
	std::vector<float> getEdges(cv::Mat &R, double angle, int type, int s);
	cv::Mat get_edgeshape(const cv::Mat &r, int side, double h, double l, int usecdf);
	void backProjection(cv::Mat &m, cv::Mat &cradle, cv::Mat &mask, double angle, int s0, int s1, int shift, int flag, int noflag);
	void backProjectionMask(std::vector<float> &m, cv::Mat &mask, double angle, int s0, int s1, int shift, int flag, int noflag);
	std::vector<int> pointBackProjection(int b, double angle, int s0, int s1, cv::Mat &upline, cv::Mat &downline);

	//Auxiliary functions
	std::vector<std::vector<int>> markVerticalCradle(const cv::Mat &img, cv::Mat &mask, std::vector<int> &hrange, int s);
	std::vector<std::vector<int>> markHorizontalCradle(const cv::Mat &img, cv::Mat &mask, std::vector<int> &vrange, int s);
	void createMaskVertical(cv::Mat &mask, std::vector<int> &vrange, int s);
	void removeMaskVertical(cv::Mat &mask, std::vector<int> &vrange, int s);
	void removeEdgeArtifact(const cv::Mat &img, cv::Mat &cradle, int dir, int stx, int enx, int sty, int eny);
	cv::Mat flipVertical(cv::Mat &in);
	float max(cv::Mat &m);
	float min(cv::Mat &m);
	int maxi(cv::Mat &m);
	int mini(cv::Mat &m);
	double cotan(double i);
	std::vector<float> linearFitting(std::vector<float> &x, std::vector<float> &y);
	float getMean(std::vector<float> &x);
	float getMean(cv::Mat &im);
	float getMedian(cv::Mat &im);
	float getMedian(std::vector<float> &v);
	float getVariance(std::vector<float> &v);
	float getVariance(cv::Mat v);
	void writeMarkedSegmentsFile(std::string name, MarkedSegments ms);
	MarkedSegments readMarkedSegmentsFile(std::string name);

	//Interface function
	void setCallbacks(const Callbacks *callbacks);
	const Callbacks *callbacks();
	bool progress(int value, int total);
}
#endif

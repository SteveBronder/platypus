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
#include <fstream>

/**
* Collection of all functions that make cradle removal possible.
* The main functions are cradledetect(), that attempts to find the approximate position
* of horizontal/vertical cradle pieces and removeCradle(), that adjusts the intensity of
* the detected cradle pieces, separating the input image into a cradle and a x-ray part.
**/
#ifndef M_PI
#define M_PI 3.1415927
#endif
#ifndef RAD
#define RAD(A)  (M_PI*((double)(A))/180.0)
#endif
namespace CradleFunctions{
	static const int MAXIMA = 0;
	static const int MINIMA = 1;
	static const Callbacks *s_callbacks;

	//Remove cradle intensity from X-ray
	void removeCradle(
		const cv::Mat &in,			//Input grayscale float X-ray image
		cv::Mat &out,				//Cradle removed X-ray is saved out here
		cv::Mat &cradle,			//Cradle component after separation saved out here
		cv::Mat &mask,				//Mask containing marked vertical/horizontal cradle positions
		MarkedSegments &ms			//MarkedSegment structure will contain processing information
		){

		//Estimate position of vertical/horizontal cradle piece position
		std::vector<int> vrange, hrange;

		cradledetect(in, mask, vrange, hrange);				//Find number of cradle pieces blindly

		//Call removal function
		removeCradle(in, out, cradle, mask, vrange, hrange, ms);
	}

	//Remove cradle intensity from X-ray
	void removeCradle(
		const cv::Mat &in,			//Input grayscale float X-ray image
		cv::Mat &out,				//Cradle removed X-ray is saved out here
		cv::Mat &cradle,			//Cradle component after separation saved out here
		cv::Mat &mask,				//Mask containing marked vertical/horizontal cradle positions
		std::vector<int> &vrange,	//Approximate position of vertical cradle pieces, in pairs of (X_start1, X_end1,..,X_startN, X_endN) 
		std::vector<int> &hrange,	//Approximate position of horizontal cradle pieces, in pairs of (Y_start1, Y_end1,..,Y_startM, Y_endM) 
		MarkedSegments &ms			//MarkedSegment structure will contain processing information
		){
		//Initialize cradle part
		cradle = cv::Mat(in.rows, in.cols, CV_32F, cv::Scalar(0));

		//Mark cradle piece mask
		createMaskVertical(mask, vrange, 0);
		std::vector<std::vector<int>> hmidpos = markHorizontalCradle(in, mask, hrange, -1);
		removeMaskVertical(mask, vrange, 0);
		std::vector<std::vector<int>> vmidpos = markVerticalCradle(in, mask, vrange, -1);

		//Fitted model parameters
		std::vector<std::vector<std::vector<float>>> vm, hm;
		std::vector<int> widthh, widthv;

		//Fill up width h
		widthh = std::vector<int>(hrange.size() / 2);
		for (int i = 0; i < hrange.size() / 2; i++){
			widthh[i] = hrange[i * 2 + 1] - hrange[i * 2];
		}

		//Fill up width v
		widthv = std::vector<int>(vrange.size() / 2);
		for (int i = 0; i < vrange.size() / 2; i++){
			widthv[i] = vrange[i * 2 + 1] - vrange[i * 2];
		}

		//Initialize structure (used by the interface)
		ms.pieces = 0;
		ms.pieceIDh = std::vector<std::vector<int>>(widthh.size());
		ms.pieceIDv = std::vector<std::vector<int>>(widthv.size());
		ms.piece_type = std::vector<int>();
		ms.piece_mask = cv::Mat(in.rows, in.cols, CV_16U, cv::Scalar(0));
		ms.piece_middle = std::vector<cv::Point2i>();

		//Remove horizontal
		removeHorizontal(in, mask, cradle, hmidpos, widthh, hm, ms);

		//Remove vertical
		removeVertical(in, mask, cradle, vmidpos, widthv, vm, ms);

		//Remove cross sections
		removeCrossSection(in, mask, cradle, widthh, widthv, hmidpos, vmidpos, hm, vm, ms);

		out = in - cradle;
	}

	//Mark horizontal cradle pieces in the mask image
	std::vector<std::vector<int>> markHorizontalCradle(
		const cv::Mat &img,			// Input image
		cv::Mat &mask,				// Mask image
		std::vector<int> &vrange,	// Position of horizontal cradle pieces
		int s						// Parameter used for smoothing filters; corresponds to about 20% of cradle piece width
		// If set to -1, this value is determined on the fly by the code
		){
		//Grad filtering
		cv::Mat grad;
		int L = 20;

		cv::Mat hkern(2 * L, 1, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}'
		for (int i = L; i < 2 * L; i++){
			hkern.at<float>(i, 0) = -1;
		}
		cv::filter2D(img, grad, CV_32F, hkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

		//Initialize angles
		std::vector<double> theta;
		double a = 80.0;
		while (a <= 100.0){
			theta.push_back(a);
			a += 0.10;
		}

		std::vector<std::vector<int>> midposition(vrange.size() / 2);

		//Only return start/end points of middle line
		std::vector<std::vector<int>> res(vrange.size() / 2);

		//Check if we need to adapt s
		if (s == -1){
			//Set s as a function of the average cradle-piece thickness
			float avg = 0;
			for (int i = 0; i < vrange.size() / 2; i++){
				avg += (vrange[i * 2 + 1] - vrange[i * 2]);
			}
			avg /= vrange.size() / 2;
			s = avg * 0.2;
		}

		//Cover all horizontal cradles
		for (int i = 0; i < vrange.size() / 2; i++){

			midposition[i] = std::vector<int>(img.cols);

			//Set adaptively value of s
			if (vrange[i * 2] == 0 || vrange[i * 2 + 1] == img.rows - 1){
				s = (vrange[i * 2 + 1] - vrange[i * 2]) * 0.2;
			}
			else{
				s = (vrange[i * 2 + 1] - vrange[i * 2]) * 0.1;
			}
			int step = std::max(1, s / 40);
			std::vector<double> radon;
			float angle1, angle2;
			float bestc;
			int stk, enk;

			if (vrange[i * 2] != 0){
				radon = findRadonTransformAngle(grad, mask, theta, vrange[i * 2] - s, vrange[i * 2] + s, 0, (img).cols, (V_MASK | DEFECT));
				angle1 = radon[0] * M_PI / 180;

				//Find best position for cradle edge
				stk = -s;
				float bestc = 0;
				for (int k = -s; k <= s; k += step){
					float cost = 0;
					for (int j = 0; j < img.cols; j++){
						int py = vrange[2 * i] + j * std::cos(angle1) + k;
						if (py >= 0 && py < mask.rows - 1){
							if ((mask.at<char>(py, j) & (V_MASK | DEFECT)) == 0){
								cost += grad.at<float>(py, j)*grad.at<float>(py, j);
							}
							if ((mask.at<char>(py + 1, j) & (V_MASK | DEFECT)) == 0){
								cost += grad.at<float>(py + 1, j)*grad.at<float>(py + 1, j);
							}
						}
					}
					if (cost > bestc){
						bestc = cost;
						stk = k;
					}
				}
			}
			else{
				//Marked segment is right along the edge - take 90 degree angle and fix position
				angle1 = M_PI / 2;
				stk = 0;
			}

			//Mark position of cradle edge - upper
			for (int j = 0; j < img.cols; j++){
				int py = vrange[2 * i] + j * std::cos(angle1) + stk;
				if (py < 0){
					py = 0;
				}
				if (py > mask.rows - 1){
					py = mask.rows - 1;
				}
				midposition[i][j] = py;
			}

			if (vrange[i * 2 + 1] != img.rows - 1){
				radon = findRadonTransformAngle(grad, mask, theta, vrange[i * 2 + 1] - s, vrange[i * 2 + 1] + s, 0, img.cols, (V_MASK | DEFECT));
				angle2 = radon[0] * M_PI / 180;

				//Find best position for cradle edge
				enk = -s;
				bestc = 0;
				for (int k = -s; k <= s; k += step){
					float cost = 0;
					for (int j = 0; j < img.cols; j++){
						int py = vrange[2 * i + 1] + j * std::cos(angle2) + k;
						if (py >= 0 && py < img.rows - 1){
							if ((mask.at<char>(py, j) & (V_MASK | DEFECT)) == 0){
								cost += grad.at<float>(py, j)*grad.at<float>(py, j);
							}
							if ((mask.at<char>(py + 1, j) & (V_MASK | DEFECT)) == 0){
								cost += grad.at<float>(py + 1, j)*grad.at<float>(py + 1, j);
							}
						}
					}
					if (cost > bestc){
						bestc = cost;
						enk = k;
					}
				}
			}
			else{
				//Marked segment is right along the edge - take 90 degree angle and fix position
				angle2 = M_PI / 2;
				enk = 0;
			}

			//Mark position of cradle edge - lower
			for (int j = 0; j < img.cols; j++){
				int py = vrange[2 * i + 1] + j * std::cos(angle2) + enk;
				if (py < 0){
					py = 0;
				}
				if (py > mask.rows - 1){
					py = mask.rows - 1;
				}
				//Mark segment mask
				for (int k = midposition[i][j]; k <= py; k++){
					mask.at<char>(k, j) |= H_MASK;
				}
				midposition[i][j] += py;
			}

			//Mark middle of cradle
			for (int j = 0; j < img.cols; j++){
				midposition[i][j] /= 2;
			}

			//Only return start/end points of middle line
			res[i] = std::vector<int>(4);
			res[i][0] = 0;
			res[i][1] = midposition[i][0];
			res[i][2] = img.cols - 1;
			res[i][3] = midposition[i][img.cols - 1];
		}

		return res;
	}

	//Mark vertical cradle pieces in the mask image
	std::vector<std::vector<int>> markVerticalCradle(
		const cv::Mat &img,			// Input image
		cv::Mat &mask,				// Mask image
		std::vector<int> &vrange,	// Position of horizontal cradle pieces
		int s						// Parameter used for smoothing filters; corresponds to about 20% of cradle piece width
		// If set to -1, this value is determined on the fly by the code
		){
		//Filter image horizontal/vertical
		cv::Mat grad;
		int L = 20;

		cv::Mat vkern(1, 2 * L, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}
		for (int i = L; i < 2 * L; i++){
			vkern.at<float>(0, i) = -1;
		}
		cv::filter2D(img, grad, CV_32F, vkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

		//Initialize angles
		std::vector<double> theta;
		double a = -20.0;
		while (a <= 20.0){
			theta.push_back(a);
			a += 0.10;
		}

		std::vector<std::vector<int>> midposition(vrange.size() / 2);

		//Only return start/end points of middle line
		std::vector<std::vector<int>> res(vrange.size() / 2);

		//Check if we need to adapt s
		if (s == -1){
			//Set s as a function of the average cradle-piece thickness
			float avg = 0;
			for (int i = 0; i < vrange.size() / 2; i++){
				avg += (vrange[i * 2 + 1] - vrange[i * 2]);
			}
			avg /= vrange.size() / 2;
			s = avg * 0.2;
		}

		//Cover all vertical cradles
		for (int i = 0; i < vrange.size() / 2; i++){

			midposition[i] = std::vector<int>(img.rows);

			//Set adaptively value of s
			if (vrange[i * 2] == 0 || vrange[i * 2 + 1] == img.cols - 1){
				s = (vrange[i * 2 + 1] - vrange[i * 2]) * 0.2;
			}
			else{
				s = (vrange[i * 2 + 1] - vrange[i * 2]) * 0.1;
			}
			int step = std::max(1, s / 40);

			std::vector<double> radon;
			double angle1, angle2, bestc;
			int stk, enk;

			if (vrange[i * 2] != 0){
				radon = findRadonTransformAngle(grad, mask, theta, 0, img.rows, vrange[i * 2] - s, vrange[i * 2] + s, H_MASK | DEFECT);
				angle1 = radon[0] * M_PI / 180;

				//Find best position for cradle edge
				stk = -s;
				bestc = 0;
				for (int k = -s; k <= s; k += step){
					float cost = 0;
					for (int j = 0; j < img.rows; j++){
						int py = vrange[2 * i] + j * std::sin(angle1) + k;
						if (py >= 0 && py < mask.cols - 1){
							if ((mask.at<char>(j, py) & (H_MASK | DEFECT)) == 0){
								cost += grad.at<float>(j, grad.cols - py)*grad.at<float>(j, py);
							}
							if ((mask.at<char>(j, py + 1) & (H_MASK | DEFECT)) == 0){
								cost += grad.at<float>(j, py + 1)*grad.at<float>(j, py + 1);
							}
						}
					}
					if (cost > bestc){
						bestc = cost;
						stk = k;
					}
				}
			}
			else{
				//Marked segment is right along the edge - take 90 degree angle and fix position
				angle1 = 0;
				stk = 0;
			}

			//Mark position of cradle edge - upper
			for (int j = 0; j < img.rows; j++){
				int py = vrange[2 * i] + j * std::sin(angle1) + stk;
				if (py < 0)
					py = 0;
				if (py > mask.cols - 1)
					py = mask.cols - 1;
				midposition[i][j] = py;
			}

			if (vrange[i * 2 + 1] != img.cols - 1){
				radon = findRadonTransformAngle(grad, mask, theta, 0, img.rows, vrange[i * 2 + 1] - s, vrange[i * 2 + 1] + s, (H_MASK | DEFECT));
				angle2 = radon[0] * M_PI / 180;

				//Find best position for cradle edge
				enk = -s;
				bestc = 0;
				for (int k = -s; k <= s; k += step){
					float cost = 0;
					for (int j = 0; j < img.rows; j++){
						int py = vrange[2 * i + 1] + j * std::sin(angle2) + k;
						if (py >= 0 && py < mask.cols - 1){
							if ((mask.at<char>(j, py) & (H_MASK | DEFECT)) == 0){
								cost += grad.at<float>(j, grad.cols - py)*grad.at<float>(j, py);
							}
							if ((mask.at<char>(j, py + 1) & (H_MASK | DEFECT)) == 0){
								cost += grad.at<float>(j, py + 1)*grad.at<float>(j, py + 1);
							}
						}
					}
					if (cost > bestc){
						bestc = cost;
						enk = k;
					}
				}
			}
			else{
				//Marked segment is right along the edge - take 90 degree angle and fix position
				angle2 = 0;
				enk = 0;
			}

			//Mark position of cradle edge - lower
			for (int j = 0; j < img.rows; j++){
				int py = std::round(vrange[2 * i + 1] + j * std::sin(angle2) + enk);
				if (py < 0)
					py = 0;
				if (py > mask.cols - 1)
					py = mask.cols - 1;

				//Mark segment mask
				for (int k = midposition[i][j]; k <= py; k++){
					mask.at<char>(j, k) |= V_MASK;
				}
				midposition[i][j] += py;
			}

			//Mark middle of cradle
			for (int j = 0; j < img.rows; j++){
				midposition[i][j] /= 2;
			}
			//Only return start/end points of middle line
			res[i] = std::vector<int>(4);
			res[i][0] = 0;
			res[i][1] = midposition[i][0];
			res[i][2] = img.rows - 1;
			res[i][3] = midposition[i][img.rows - 1];
		}

		return res;
	}

	//Remove cradle from cross-sections
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
		){

		cv::Mat smooth, filtered;
		int fs = 5;
		smooth = cv::Mat(fs, fs, CV_32F, 1.0 / fs / fs); //5x5 uniform blur filter

		//Store filtered image
		cv::filter2D(img, filtered, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Do a checkup to make sure there is no invalid cradle pixel i the image
		for (int i = 0; i < cradle.rows; i++){
			for (int j = 0; j < cradle.cols; j++){
				if (cradle.at<float>(i, j) != cradle.at<float>(i, j)){
					cradle.at<float>(i, j) = 0;
				}
			}
		}

		//Create middle line masks for all horizontal/vertical pieces
		int htot = hrange.size();	//Total number of horizontal pieces
		int vtot = vrange.size();	//Total number of horizontal pieces
		std::vector<std::vector<int>> midposh(htot);
		std::vector<std::vector<int>> midposv(vtot);

		//Cover all horizontal cradles
		for (int i = 0; i < htot; i++){

			//Create midpos vector (interpolate two points for all columns)
			midposh[i] = std::vector<int>(img.cols);
			int x1 = midposh_points[i][0];
			int y1 = midposh_points[i][1];
			int x2 = midposh_points[i][2];
			int y2 = midposh_points[i][3];

			if (x2 == x1){
				//This is a vertical line -> invalid for a horizontal cradle piece
				return; //Stuff went wrong
			}
			else{
				float m = (y2 - y1) * 1.0 / (x2 - x1);
				//Fill up midpoints
				for (int j = 0; j < img.cols; j++){
					midposh[i][j] = m * (j - x1) + y1;
				}
			}
		}

		//Cover all vertical cradles
		for (int i = 0; i < vtot; i++){

			//Create midpos vector (interpolate two points for all columns)
			midposv[i] = std::vector<int>(img.rows);
			int x1 = midposv_points[i][0];
			int y1 = midposv_points[i][1];
			int x2 = midposv_points[i][2];
			int y2 = midposv_points[i][3];

			if (x2 == x1){
				//This is a vertical line -> invalid for a horizontal cradle piece
				return; //Stuff went wrong
			}
			else{
				float m = (y2 - y1) * 1.0 / (x2 - x1);
				//Fill up midpoints
				for (int j = 0; j < img.rows; j++){
					midposv[i][j] = m * (j - x1) + y1;
				}
			}
		}

		//Cover all cross section cradles
		for (int j = 0; j < vrange.size(); j++){
			for (int i = 0; i < hrange.size(); i++){

				// progress/abort
				if (!progress(j * vrange.size() + i, vrange.size() * hrange.size()))
					return;

				//Find pixels considered to be part of cross section
				int sx, sy, msx, msy;
				for (int k = 0; k < midposv[j].size(); k++){
					if (midposh[i][midposv[j][k]] == k){
						msy = midposv[j][k];
						msx = k;
					}
				}

				//Increase number of pieces
				ms.pieces++;
				ms.piece_type.push_back(CROSS_DIR);
				ms.pieceIDh[i].push_back(ms.pieces);
				ms.pieceIDv[j].push_back(ms.pieces);

				//Mark middle
				ms.piece_middle.push_back(cv::Point2i(msx, msy));

				sx = msx;
				sy = msy;

				int widthv = vrange[j];
				int widthh = hrange[i];
				int sv = std::max((int)(widthv * 0.03), 2);
				int sh = std::max((int)(widthh * 0.03), 2);

				int minh = std::max(sx - widthh / 2 - sh, 0);
				int maxh = std::min(sx + widthh / 2 + sh, img.rows);
				int minv = std::max(sy - widthv / 2 - sv, 0);
				int maxv = std::min(sy + widthv / 2 + sv, img.cols);

				//Middle of previously identified cradle intersections is marked by (H_MASK | V_MASK)
				if ((mask.at<char>(sx, sy) & (H_MASK | V_MASK)) == (H_MASK | V_MASK)){
					int stx, enx, sty, eny, ok;

					//Search upwards
					stx = sx - 1;
					ok = 0;
					while (ok == 0){
						ok = 1;
						if (stx != -1){
							for (int k = minv; k < maxv; k++){
								if ((mask.at<char>(stx, k) & (H_MASK | V_MASK)) == (H_MASK | V_MASK)){
									ok = 0;
									stx--;
									break;
								}
							}
						}
					}

					//Search downwards
					enx = sx + 1;
					ok = 0;
					while (ok == 0){
						ok = 1;
						if (enx != img.rows){
							for (int k = minv; k < maxv; k++){
								if ((mask.at<char>(enx, k)  & (H_MASK | V_MASK)) == (H_MASK | V_MASK)){
									ok = 0;
									enx++;
									break;
								}
							}
						}
					}

					//Search leftwards
					sty = sy - 1;
					ok = 0;
					while (ok == 0){
						ok = 1;
						if (sty != -1){
							for (int k = minh; k < maxh; k++){
								if ((mask.at<char>(k, sty) & (H_MASK | V_MASK)) == (H_MASK | V_MASK)){
									ok = 0;
									sty--;
									break;
								}
							}
						}
					}

					//Search rightwards
					eny = sy + 1;
					ok = 0;
					while (ok == 0){
						ok = 1;
						if (eny != img.cols){
							for (int k = minh; k < maxh; k++){
								if ((mask.at<char>(k, eny) & (H_MASK | V_MASK)) == (H_MASK | V_MASK)){
									ok = 0;
									eny++;
									break;
								}
							}
						}
					}

					stx = std::max(0, stx);
					enx = std::min(img.rows - 1, enx);
					sty = std::max(0, sty);
					eny = std::min(img.cols - 1, eny);

					int prev = -1;
					for (int k = 0; k < vm[j].size(); k++){
						if (vm[j][k][4] < msx)
							prev = k;
					}
					int postv = -1;
					for (int k = vm[j].size() - 1; k >= 0; k--){
						if (vm[j][k][4] > msx)
							postv = k;
					}
					int preh = -1;
					for (int k = 0; k < hm[i].size(); k++){
						if (hm[i][k][4] < msy)
							preh = k;
					}
					int posth = -1;
					for (int k = hm[i].size() - 1; k >= 0; k--){
						if (hm[i][k][4] > msy)
							posth = k;
					}

					//Remove cradle part
					for (int k = stx; k <= enx; k++){
						for (int l = sty; l <= eny; l++){
							if (cradle.at<float>(k, l) == 0 && ((mask.at<char>(k, l) & DEFECT) == 0)){

								float val = filtered.at<float>(k, l);

								float c1h, c1v, c2h, c2v, c3h, c3v, c4h, c4v;
								float chpre, chpost, cvpre, cvpost;
								float whpre, whpost, wvpre, wvpost;

								if (preh != -1){
									c1h = hm[i][preh][1] * val + hm[i][preh][0];
									c4h = hm[i][preh][3] * val + hm[i][preh][2];
									whpre = 1.0 / (1 + 1.0*(l - sty));

									//Take weighted average of approximations
									chpre = (k - stx) * 1.0 / (enx - stx)*(c4h - c1h) + c1h;

									if (chpre != chpre){
										chpre = 0;
										whpre = 0;
									}
								}
								else{
									chpre = 0;
									whpre = 0;
								}

								if (posth != -1){
									c2h = hm[i][posth][1] * val + hm[i][posth][0];
									c3h = hm[i][posth][3] * val + hm[i][posth][2];
									whpost = 1.0 / (1 + 1.0*(eny - l));

									//Take weighted average of approximations
									chpost = (k - stx) * 1.0 / (enx - stx)*(c3h - c2h) + c2h;

									if (chpost != chpost){
										chpost = 0;
										whpost = 0;
									}
								}
								else{
									chpost = 0;
									whpost = 0;
								}

								if (prev != -1){
									c1v = vm[j][prev][1] * val + vm[j][prev][0];
									c2v = vm[j][prev][3] * val + vm[j][prev][2];
									wvpre = 1.0 / (1 + 1.0*(k - stx));

									//Take weighted average of approximations
									cvpre = (l - sty) * 1.0 / (eny - sty)*(c2v - c1v) + c1v;

									if (cvpre != cvpre){
										cvpre = 0;
										wvpre = 0;
									}

								}
								else{
									cvpre = 0;
									wvpre = 0;
								}
								if (postv != -1){
									c3v = vm[j][postv][1] * val + vm[j][postv][0];
									c4v = vm[j][postv][3] * val + vm[j][postv][2];
									wvpost = 1.0 / (1 + 1.0*(enx - k));

									//Take weighted average of approximations
									cvpost = (l - sty) * 1.0 / (eny - sty)*(c4v - c3v) + c3v;

									if (cvpost != cvpost){
										cvpost = 0;
										wvpost = 0;
									}
								}
								else{
									cvpost = 0;
									wvpost = 0;
								}

								ms.piece_mask.at<ushort>(k, l) = ms.pieces;
								cradle.at<float>(k, l) = filtered.at<float>(k, l) - (whpre*chpre + whpost*chpost + wvpre*cvpre + wvpost*cvpost) *1.0 / (whpre + whpost + wvpre + wvpost);
							}
						}
					}


					//Clean up black lines
					int hwidth = (enx - stx) * 0.1;
					int vwidth = (eny - sty) * 0.1;

					removeEdgeArtifact(img, cradle, TextureRemoval::HORIZONTAL, stx - hwidth, stx + hwidth, sty, eny);
					removeEdgeArtifact(img, cradle, TextureRemoval::HORIZONTAL, enx - hwidth, enx + hwidth, sty, eny);
					removeEdgeArtifact(img, cradle, TextureRemoval::VERTICAL, stx, enx, sty - vwidth, sty + vwidth);
					removeEdgeArtifact(img, cradle, TextureRemoval::VERTICAL, stx, enx, eny - vwidth, eny + vwidth);
				}

			}
		}
	}

	//Remove vertical cradle pieces and save out correction model used for later usage
	void removeVertical(
		const cv::Mat &img,									//Input grayscale float X-ray image
		cv::Mat &mask,										//Mask containing marked vertical and/or horizontal cradle positions
		cv::Mat &cradle,									//Cradle component after separation saved out here
		std::vector<std::vector<int>> &midpos_points,		//Center of vertical cradle pieces
		std::vector<int> s,									//Width of vertical cradle pieces
		std::vector<std::vector<std::vector<float>>> &vm,	//Saves out parameters of the fitted multiplicative model, used for processing cross-sections
		MarkedSegments &ms									//MarkedSegment structure will contain processing information
	){
		//Set avg_s as a function of the average cradle-piece thickness
		float avg = 0;
		int avg_s;
		for (int i = 0; i < s.size(); i++){
			avg += s[i];
		}
		avg /= s.size();
		avg_s = std::max(3, (int)(avg * 0.2));

		//Directional smoothing of image
		cv::Mat smooth, filtered;
		smooth = cv::Mat(avg_s, 1, CV_32F, cv::Scalar(1.0 / avg_s));
		cv::filter2D(img, filtered, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		int vtot = midpos_points.size();	//Total number of vertical pieces
		std::vector<std::vector<int>> midpos(vtot);
		vm = std::vector<std::vector<std::vector<float>>>(vtot);

		//Cover all vertical cradles
		for (int i = 0; i < vtot; i++){
			// progress/abort
			if (!progress(i, vtot))
				return;

			//Set adaptively value of s
			int sfm = s[i] * 0.1;

			//Create midpos vector (interpolate two points for all columns)
			midpos[i] = std::vector<int>(img.rows);
			int x1 = midpos_points[i][0];
			int y1 = midpos_points[i][1];
			int x2 = midpos_points[i][2];
			int y2 = midpos_points[i][3];

			if (x2 == x1){
				//This is a vertical line -> invalid for a horizontal cradle piece
				return; //Stuff went wrong
			}
			else{
				float m = (y2 - y1) * 1.0 / (x2 - x1);
				//Fill up midpoints
				for (int j = 0; j < img.rows; j++){
					midpos[i][j] = m * (j - x1) + y1;
				}
			}

			int step = std::min(3, std::max(sfm / 5, 1));

			//Pairwise samples for fitting (upper and lower edges)
			std::vector<cradle_sample_pairs> segment_samples(100);
			cradle_sample_pairs sample = segment_samples[0];
			int segment_cnt = 0;
			int segment_seek = 1;

			//Sample cradle/noncradle pairs
			for (int j = 0; j < img.rows; j++){

				//Find start/end of cradle part
				int p1, p2;
				p1 = p2 = midpos[i][j];

				while (p1 > 0 && (mask.at<char>(j, p1) & V_MASK) == V_MASK)
					p1--;
				while (p2 < mask.cols - 1 && (mask.at<char>(j, p2) & V_MASK) == V_MASK)
					p2++;

				int start = std::max(0, p1 - sfm);
				int end = std::min(img.cols - 1, p2 + sfm);

				//Check if contains horizontal mask
				int maskfound = 0;
				for (int k = start; k <= end; k++){
					if ((mask.at<char>(j, k) & H_MASK) != 0){
						maskfound++;
					}
				}

				if (maskfound > 0){
					//Mark as vertical cradle (for cross section later on)
					for (int k = p1; k <= p2; k++){
						mask.at<char>(j, k) |= V_MASK;
					}

					if (segment_seek == 0){
						//Vertical mask part reached
						sample.end = j;
						segment_samples[segment_cnt] = sample;
						segment_cnt++;
						segment_seek = 1;
					}
				}
				else{
					//The current column contains no vertical cradle part
					//Initializ new segment
					if (segment_seek == 1){
						segment_seek = 0;

						sample = segment_samples[segment_cnt];
						sample.start = j;
						sample.end = -1;

						//Initialize structure arrays
						sample.ncu = std::vector<float>(img.rows);
						std::fill(sample.ncu.begin(), sample.ncu.end(), -1);
						sample.ncl = std::vector<float>(img.rows);
						std::fill(sample.ncl.begin(), sample.ncl.end(), -1);
						sample.cu = std::vector<float>(img.rows);
						std::fill(sample.cu.begin(), sample.cu.end(), -1);
						sample.cl = std::vector<float>(img.rows);
						std::fill(sample.cl.begin(), sample.cl.end(), -1);
					}

					if (p1 - 2 * sfm >= 0){
						//Sample above cradle
						std::vector<float> medi(sfm + 1);
						std::fill(medi.begin(), medi.end(), -1);
						sample.ncu[j] = -1;
						for (int z = std::max(0, p1 - 2 * sfm); z <= p1 - sfm; z++){
							if ((mask.at<char>(j, z) & (H_MASK | DEFECT)) == 0){
								medi[z - (p1 - 2 * sfm)] = filtered.at<float>(j, z);
							}
						}
						sample.ncu[j] = getMedian(medi);

						if ((mask.at<char>(j, p1 + sfm) & (H_MASK | DEFECT)) == 0){
							sample.cu[j] = filtered.at<float>(j, p1 + sfm);
						}
						else{
							sample.cu[j] = -1;
						}
					}

					if (p2 + 2 * sfm < mask.cols){
						//Sample below cradle
						std::vector<float> medi(sfm + 1);
						std::fill(medi.begin(), medi.end(), -1);
						sample.ncl[j] = -1;
						for (int z = p2 + sfm; z < std::min(p2 + 2 * sfm, filtered.cols); z++){
							if ((mask.at<char>(j, z) & (H_MASK | DEFECT)) == 0){
								medi[z - p2 - sfm] = filtered.at<float>(j, z);
							}
						}
						sample.ncl[j] = getMedian(medi);

						if ((mask.at<char>(j, p2 - sfm) & (H_MASK | DEFECT)) == 0){
							sample.cl[j] = filtered.at<float>(j, p2 - sfm);
						}
						else{
							sample.cl[j] = -1;
						}
					}
				}
			}

			//Add end to the last segment part
			if (sample.end == -1){
				//Vertical mask part reached
				sample.end = img.rows - 1;
				segment_samples[segment_cnt] = sample;
				segment_cnt++;
			}

			vm[i] = std::vector<std::vector<float>>(segment_cnt);

			//Fit model on each segment
			for (int s = 0; s < segment_cnt; s++){

				sample = segment_samples[s];

				//Increment counter for total number of pieces
				ms.pieces++;
				ms.piece_type.push_back(VERTICAL_DIR);
				ms.pieceIDv[i].push_back(ms.pieces);

				//Mark middle
				ms.piece_middle.push_back(cv::Point2i((sample.end + sample.start) / 2, midpos[i][(sample.end + sample.start) / 2]));

				std::vector<float> lin_model_midu(2), lin_model_midl(2);

				lin_model_midu = linearFitting(sample.cu, sample.ncu);
				if (getMedian(sample.ncl) != -1){
					lin_model_midl = linearFitting(sample.cl, sample.ncl);
				}
				else{
					lin_model_midl[0] = lin_model_midu[0];
					lin_model_midl[1] = lin_model_midu[1];
				}
				if (getMedian(sample.ncu) == -1){
					lin_model_midu[0] = lin_model_midl[0];
					lin_model_midu[1] = lin_model_midl[1];
				}

				std::vector<float> v1, v2;
				for (int zz = 0; zz < sample.cl.size(); zz++){
					if (sample.cl[zz] != -1 && sample.ncl[zz] != -1){
						v1.push_back(sample.cl[zz]);
						v2.push_back(sample.ncl[zz]);
					}
				}
				for (int zz = 0; zz < sample.cu.size(); zz++){
					if (sample.cu[zz] != -1 && sample.ncu[zz] != -1){
						v1.push_back(sample.cu[zz]);
						v2.push_back(sample.ncu[zz]);
					}
				}

				//If fitting on both upper and lower parts is bad - the constant factor is negative
				if (lin_model_midu[0] > 0 && lin_model_midl[0] > 0){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[0] > 0){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[0] > 0){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}

				if (lin_model_midu[1] > 1.2 && lin_model_midl[1] > 1.2){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[1] > 1.2){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[1] > 1.2){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}

				//If fitting on both upper and lower parts is bad - multiplicative factor is too small
				if (lin_model_midu[1] < 0.9 && lin_model_midl[1] < 0.9){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[1] < 0.9){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[1] < 0.9){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}

				//Save out fitted model for later usage
				vm[i][s] = std::vector<float>(5);
				vm[i][s][0] = lin_model_midu[0];
				vm[i][s][1] = lin_model_midu[1];
				vm[i][s][2] = lin_model_midl[0];
				vm[i][s][3] = lin_model_midl[1];
				vm[i][s][4] = (sample.end + sample.start) / 2;

				//Remove cradle
				std::vector<int> p1v(sample.end - sample.start + 1), p2v(sample.end - sample.start + 1);
				for (int j = sample.start; j <= sample.end; j++){

					//Find start/end of cradle part
					int p1, p2;
					p1 = p2 = midpos[i][j];

					while (p1 > 0 && (mask.at<char>(j, p1) & V_MASK) == V_MASK)
						p1--;
					while (p2 < mask.cols - 1 && (mask.at<char>(j, p2) & V_MASK) == V_MASK)
						p2++;

					p1v[j - sample.start] = p1;
					p2v[j - sample.start] = p2;

					//Mark as vertical cradle
					for (int k = p1; k <= p2; k++){
						mask.at<char>(j, k) |= V_MASK;
					}

					//Remove intensity from middle of cradle based on interpolation of the two edge profiles
					for (int k = p1 + sfm / 2; k < p2 - sfm / 2; k++){
						if (((mask.at<char>(j, k)) & (H_MASK | DEFECT)) == 0){
							float pv = img.at<float>(j, k);

							//Get the two estimations based on the edge profiles
							float epv1 = lin_model_midu[1] * pv + lin_model_midu[0];
							float epv2 = lin_model_midl[1] * pv + lin_model_midl[0];

							//Take weighted average of approximations
							float iv = (k - p1 - sfm) * 1.0 / (p2 - p1 - 2 * sfm)*(epv2 - epv1) + epv1;

							ms.piece_mask.at<ushort>(j, k) = ms.pieces;
							cradle.at<float>(j, k) = pv - iv;
						}
					}
				}

				std::vector<float> edgemap;
				std::vector<int> cnt;
				float minv, maxv;
				int first, last, separation;

				//Remove edge - upper
				edgemap = std::vector<float>(2 * sfm + 1);
				cnt = std::vector<int>(2 * sfm + 1);

				//Model cradle edge
				for (int j = sample.start; j <= sample.end; j++){
					int mid = p1v[j - sample.start];
					int lpmin = std::max(p1v[j - sample.start] - sfm, 0);
					int lpmax = std::min(p1v[j - sample.start] + sfm, filtered.cols - 1);
					for (int l = lpmin; l <= lpmax; l++){
						if ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0){
							edgemap[mid - l + sfm] += filtered.at<float>(j, l);
							cnt[mid - l + sfm]++;
						}
					}
				}
				for (int j = 0; j < edgemap.size(); j++){
					if (cnt[j] != 0){
						edgemap[j] /= cnt[j];
					}
				}

				first = 0;
				while (first < edgemap.size() && cnt[first] == 0) first++;
				last = edgemap.size() - 1;
				while (last >= 0 && cnt[last] == 0) last--;

				//Find proper edge of the cradle
				separation = first;
				minv = edgemap[first];
				maxv = edgemap[first];
				for (int j = first; j < last; j++){
					if (edgemap[j] < minv)
						minv = edgemap[j];
					if (edgemap[j] > maxv)
						maxv = edgemap[j];
				}
				while (edgemap[separation] >(minv + maxv) / 2){
					separation++;
				}

				if (first < last){

					//Remove edge
					for (int j = sample.start; j <= sample.end; j++){
						int bk = 0;
						float mcost = 1e20;

						//Find position that best fits the edge
						for (int k = -step; k <= step; k++){
							int lpmin = std::max(p1v[j - sample.start] - sfm, 0);
							int lpmax = std::min(p1v[j - sample.start] + sfm, filtered.cols - 1);
							int mid = p1v[j - sample.start];
							int c = 0;
							float cost = 0;


							float edgemean = 0;
							float samplemean = 0;
							//Get means of the two edge profiles
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
									samplemean += filtered.at<float>(j, l);
									edgemean += edgemap[pos];
									c++;
								}
							}

							samplemean /= c;
							edgemean /= c;

							//Check how well it fits
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size()){
									if (edgemap[pos] != 0 && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
										float tmp = (edgemap[pos] - edgemean) - (filtered.at<float>(j, l) - samplemean);
										cost += tmp*tmp;
									}
								}
							}
							cost /= c;

							if (cost < mcost){
								mcost = cost;
								bk = k;
							}
						}

						//Remove intensity
						int lpmin = std::max(p1v[j - sample.start] - sfm, 0);
						int lpmax = std::min(p1v[j - sample.start] + sfm, filtered.cols - 1);
						int mid = p1v[j - sample.start];

						float ref = lin_model_midu[1] * filtered.at<float>(j, lpmax) + lin_model_midu[0];
						float dif = filtered.at<float>(j, lpmax) - ref;

						float a, b;
						//Check if the edge is dropping or is just flat cradle
						if (std::abs(edgemap[last] - edgemap[first]) < std::abs(dif) *0.5){
							//Flat cradle, no edge
							a = 0;
							b = dif;
						}
						else{
							//Regular decaying edge
							a = dif / (edgemap[first] - edgemap[last]);
							b = dif - a * edgemap[first];
						}

						for (int l = lpmin; l < lpmax; l++){
							int pos = mid - l + sfm + bk;
							if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
								cradle.at<float>(j, l) = a*edgemap[pos] + b;
								if (pos <= separation){
									ms.piece_mask.at<ushort>(j, l) = ms.pieces;
								}
							}
						}
					}
				}

				//Remove edge - lower
				edgemap = std::vector<float>(2 * sfm + 1);
				cnt = std::vector<int>(2 * sfm + 1);

				//Model cradle edge
				for (int j = sample.start; j <= sample.end; j++){
					int mid = p2v[j - sample.start];
					int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
					int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.cols - 1);
					for (int l = lpmin; l <= lpmax; l++){
						if ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0){
							edgemap[mid - l + sfm] += filtered.at<float>(j, l);
							cnt[mid - l + sfm]++;
						}
					}
				}
				for (int j = 0; j < edgemap.size(); j++){
					if (cnt[j] != 0){
						edgemap[j] /= cnt[j];
					}
				}

				first = 0;
				while (first < edgemap.size() && cnt[first] == 0) first++;
				last = edgemap.size() - 1;
				while (last >= 0 && cnt[last] == 0) last--;
				
				//Find proper edge of the cradle
				separation = last - 1;
				minv = edgemap[separation];
				maxv = edgemap[separation];
				for (int j = first; j < last; j++){
					if (edgemap[j] < minv)
						minv = edgemap[j];
					if (edgemap[j] > maxv)
						maxv = edgemap[j];
				}
				while (edgemap[separation] >(minv + maxv) / 2){
					separation--;
				}

				if (first < last){

					//Remove edge
					for (int j = sample.start; j <= sample.end; j++){
						int bk = 0;
						float mcost = 1e20;

						//Find position that best fits the edge
						for (int k = -step; k <= step; k++){
							int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
							int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.cols - 1);
							int mid = p2v[j - sample.start];
							int c = 0;
							float cost = 0;

							float edgemean = 0;
							float samplemean = 0;
							//Get means of the two edge profiles
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
									samplemean += filtered.at<float>(j, l);
									edgemean += edgemap[pos];
									c++;
								}
							}

							samplemean /= c;
							edgemean /= c;

							//Check how well it fits
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
									if (edgemap[pos] != 0){
										float tmp = (edgemap[pos] - edgemean) - (filtered.at<float>(j, l) - samplemean);
										cost += tmp*tmp;
									}
								}
							}
							cost /= c;

							if (cost < mcost){
								mcost = cost;
								bk = k;
							}
						}

						//Remove intensity
						int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
						int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.cols - 1);
						int mid = p2v[j - sample.start];

						float ref = lin_model_midl[1] * filtered.at<float>(j, lpmin) + lin_model_midl[0];// filtered.at<float>(lpmax, j);
						float dif = filtered.at<float>(j, lpmin) - ref;
						float a, b;

						//Check if the edge is dropping or is just flat cradle
						if (std::abs(edgemap[last] - edgemap[first]) < std::abs(dif) *0.5){
							//Flat cradle, no edge
							a = 0;
							b = dif;
						}
						else{
							//Regular decaying edge
							a = dif / (edgemap[last] - edgemap[first]);
							b = dif - a * edgemap[last];
						}

						for (int l = lpmin; l < lpmax; l++){
							int pos = mid - l + sfm + bk;
							if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(j, l) & (H_MASK | DEFECT)) == 0)){
								cradle.at<float>(j, l) = a*edgemap[pos] + b;
								if (pos >= separation){
									ms.piece_mask.at<ushort>(j, l) = ms.pieces;
								}
							}
						}
					}
				}
			}
		}
	}
	
	//Remove horizontal cradle pieces and save out correction model used for later usage
	void removeHorizontal(
		const cv::Mat &img,									//Input grayscale float X-ray image
		cv::Mat &mask,										//Mask containing marked horizontal and/or vertical cradle positions
		cv::Mat &cradle,									//Cradle component after separation saved out here
		std::vector<std::vector<int>> &midpos_points,		//Center of horizontal cradle pieces
		std::vector<int> s,									//Width of horizontal cradle pieces
		std::vector<std::vector<std::vector<float>>> &hm,	//Saves out parameters of the fitted multiplicative model, used for processing cross-sections
		MarkedSegments &ms									//MarkedSegment structure will contain processing information
	){

		//Set avg_s as a function of the average cradle-piece thickness
		float avg = 0;
		int avg_s;
		for (int i = 0; i < s.size(); i++){
			avg += s[i];
		}
		avg /= s.size();
		avg_s = std::max(3, (int)(avg * 0.2));

		//Directional smoothing of image
		cv::Mat smooth, filtered;
		smooth = cv::Mat(1, avg_s, CV_32F, cv::Scalar(1.0 / avg_s));
		cv::filter2D(img, filtered, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		int vtot = midpos_points.size();	//Total number of horizontal pieces
		std::vector<std::vector<int>> midpos(vtot);
		hm = std::vector<std::vector<std::vector<float>>>(vtot);

		//Cover all horizontal cradles
		for (int i = 0; i < vtot; i++){
			// progress/abort
			if (!progress(i, vtot))
				return;

			//Set adaptively value of s
			int sfm = s[i] * 0.1;

			//Create midpos vector (interpolate two points for all columns)
			midpos[i] = std::vector<int>(img.cols);
			int x1 = midpos_points[i][0];
			int y1 = midpos_points[i][1];
			int x2 = midpos_points[i][2];
			int y2 = midpos_points[i][3];

			if (x2 == x1){
				//This is a vertical line -> invalid for a horizontal cradle piece
				return; //Stuff went wrong
			}
			else{
				float m = (y2 - y1) * 1.0 / (x2 - x1);
				//Fill up midpoints
				for (int j = 0; j < img.cols; j++){
					midpos[i][j] = m * (j - x1) + y1;
				}
			}

			//Pairwise samples for fitting (upper and lower edges)
			std::vector<cradle_sample_pairs> segment_samples(100);
			cradle_sample_pairs sample = segment_samples[0];
			int segment_cnt = 0;
			int segment_seek = 1;

			//Sample cradle/noncradle pairs
			for (int j = 0; j < img.cols; j++){

				//Find start/end of cradle part
				int p1, p2;
				p1 = p2 = midpos[i][j];

				while (p1 > 0 && (mask.at<char>(p1, j) & H_MASK) == H_MASK)
					p1--;
				while (p2 < mask.rows - 1 && (mask.at<char>(p2, j) & H_MASK) == H_MASK)
					p2++;

				int start = std::max(0, p1 - sfm);
				int end = std::min(img.rows - 1, p2 + sfm);

				//Check if contains vertical mask
				int maskfound = 0;
				for (int k = start; k <= end; k++){
					if ((mask.at<char>(k, j) & V_MASK) != 0){
						maskfound++;
					}
				}

				if (maskfound > 0){
					if (segment_seek == 0){
						//Vertical mask part reached
						sample.end = j;
						segment_samples[segment_cnt] = sample;
						segment_cnt++;
						segment_seek = 1;
					}
				}
				else{
					//The current column contains no vertical cradle part
					//Initializ new segment
					if (segment_seek == 1){
						segment_seek = 0;

						sample = segment_samples[segment_cnt];
						sample.start = j;
						sample.end = -1;

						//Initialize structure arrays
						sample.ncu = std::vector<float>(img.cols);
						std::fill(sample.ncu.begin(), sample.ncu.end(), -1);
						sample.cu = std::vector<float>(img.cols);
						std::fill(sample.cu.begin(), sample.cu.end(), -1);
						sample.ncl = std::vector<float>(img.cols);
						std::fill(sample.ncl.begin(), sample.ncl.end(), -1);
						sample.cl = std::vector<float>(img.cols);
						std::fill(sample.cl.begin(), sample.cl.end(), -1);
					}

					//Sample above cradle
					if (p1 - 2 * sfm >= 0){
						std::vector<float> medi(sfm + 1);
						std::fill(medi.begin(), medi.end(), -1);
						sample.ncu[j] = -1;
						for (int z = std::max(0, p1 - 2 * sfm); z <= p1 - sfm; z++){
							if ((mask.at<char>(z, j) & (V_MASK | DEFECT)) == 0){
								medi[z - (p1 - 2 * sfm)] = filtered.at<float>(z, j);
							}
						}
						sample.ncu[j] = getMedian(medi);

						if ((mask.at<char>(p1 + sfm, j) & (V_MASK | DEFECT)) == 0){
							sample.cu[j] = filtered.at<float>(p1 + sfm, j);
						}
						else{
							sample.cu[j] = -1;
						}
					}

					//Sample below cradle
					if (p2 + 2 * sfm < filtered.rows){
						std::vector<float> medi(sfm + 1);
						std::fill(medi.begin(), medi.end(), -1);
						sample.ncl[j] = -1;
						for (int z = p2 + sfm; z < std::min(p2 + 2 * sfm, filtered.rows); z++){
							if ((mask.at<char>(z, j) & (V_MASK | DEFECT)) == 0){
								medi[z - p2 - sfm] = filtered.at<float>(z, j);
							}
						}
						sample.ncl[j] = getMedian(medi);

						if ((mask.at<char>(p2 - sfm, j) & (V_MASK | DEFECT)) == 0){
							sample.cl[j] = filtered.at<float>(p2 - sfm, j);
						}
						else{
							sample.cl[j] = -1;
						}
					}
				}
			}

			//Add end to the last segment part
			if (sample.end == -1){
				//Vertical mask part reached
				sample.end = img.cols - 1;
				segment_samples[segment_cnt] = sample;
				segment_cnt++;
			}

			//Initialize fitted model array
			hm[i] = std::vector<std::vector<float>>(segment_cnt);

			//Fit model on each segment
			for (int s = 0; s < segment_cnt; s++){

				sample = segment_samples[s];

				//Mark middle
				ms.piece_middle.push_back(cv::Point2i(midpos[i][(sample.end + sample.start) / 2], (sample.end + sample.start) / 2));

				//Increment counter for total number of pieces
				ms.pieces++;
				ms.piece_type.push_back(HORIZONTAL_DIR);
				ms.pieceIDh[i].push_back(ms.pieces);

				std::vector<float> lin_model_midu(2), lin_model_midl(2);

				lin_model_midu = linearFitting(sample.cu, sample.ncu);
				if (getMedian(sample.ncl) != -1){
					lin_model_midl = linearFitting(sample.cl, sample.ncl);
				}
				else{
					lin_model_midl[0] = lin_model_midu[0];
					lin_model_midl[1] = lin_model_midu[1];
				}
				if (getMedian(sample.ncu) == -1){
					lin_model_midu[0] = lin_model_midl[0];
					lin_model_midu[1] = lin_model_midl[1];
				}

				//If fitting on both upper and lower parts is bad - constant factor is positive
				if (lin_model_midu[0] > 0 && lin_model_midl[0] > 0){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[0] > 0){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[0] > 0){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}


				//If fitting on both upper and lower parts is bad - multiplicative factor is too big
				if (lin_model_midu[1] > 1.2 && lin_model_midl[1] > 1.2){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[1] > 1.2){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[1] > 1.2){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}

				//If fitting on both upper and lower parts is bad - multiplicative factor is too small
				if (lin_model_midu[1] < 0.9 && lin_model_midl[1] < 0.9){
					//Revert back to additive model
					if (getMedian(sample.ncu) != -1){
						lin_model_midu[0] = getMedian(sample.ncu) - getMedian(sample.cu);
						lin_model_midu[1] = 1.0;
					}
					else{
						lin_model_midu[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midu[1] = 1.0;
					}
					if (getMedian(sample.ncl) != -1){
						lin_model_midl[0] = getMedian(sample.ncl) - getMedian(sample.cl);
						lin_model_midl[1] = 1.0;
					}
					else{
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}
				else{
					if (lin_model_midu[1] < 0.9){
						lin_model_midu[0] = lin_model_midl[0];
						lin_model_midu[1] = lin_model_midl[1];
					}
					if (lin_model_midl[1] < 0.9){
						lin_model_midl[0] = lin_model_midu[0];
						lin_model_midl[1] = lin_model_midu[1];
					}
				}

				//Save out fitted model for later usage
				hm[i][s] = std::vector<float>(5);
				hm[i][s][0] = lin_model_midu[0];
				hm[i][s][1] = lin_model_midu[1];
				hm[i][s][2] = lin_model_midl[0];
				hm[i][s][3] = lin_model_midl[1];
				hm[i][s][4] = (sample.end + sample.start) / 2;

				//Remove intensity from middle of cradle based on interpolation of the two edge profiles
				std::vector<int> p1v(sample.end - sample.start + 1), p2v(sample.end - sample.start + 1);
				for (int j = sample.start; j <= sample.end; j++){

					//Find start/end of cradle part
					int p1, p2;
					p1 = p2 = midpos[i][j];

					while (p1 > 0 && (mask.at<char>(p1, j) & H_MASK) == H_MASK)
						p1--;
					while (p2 < mask.rows - 1 && (mask.at<char>(p2, j) & H_MASK) == H_MASK)
						p2++;

					p1v[j - sample.start] = p1;
					p2v[j - sample.start] = p2;

					int start = std::max(0, p1 - sfm);
					int end = std::min(img.rows - 1, p2 + sfm);

					//Remove intensity from middle of cradle based on interpolation of the two edge profiles
					for (int k = p1 + sfm / 2; k < p2 - sfm / 2; k++){
						if (((mask.at<char>(k, j)) & (V_MASK | DEFECT)) == 0){
							float pv = img.at<float>(k, j);

							//Get the two estimations based on the edge profiles
							float epv1 = lin_model_midu[1] * pv + lin_model_midu[0];
							float epv2 = lin_model_midl[1] * pv + lin_model_midl[0];

							//Take weighted average of approximations
							float iv = (k - p1 - sfm) * 1.0 / (p2 - p1 - 2 * sfm)*(epv2 - epv1) + epv1;

							ms.piece_mask.at<ushort>(k, j) = ms.pieces;
							cradle.at<float>(k, j) = pv - iv;
							mask.at<char>(k, j) |= H_MASK;
						}
					}
				}

				std::vector<float> edgemap;
				std::vector<std::vector<float>> edgesample;
				std::vector<int> cnt;
				float minv, maxv;
				int first, last, separation;
				int step = std::min(3, std::max(sfm / 5, 1));

				//Remove edge - upper
				edgesample = std::vector<std::vector<float>>(2 * sfm + 1);
				for (int j = 0; j < edgesample.size(); j++){
					edgesample[j] = std::vector<float>(sample.end - sample.start + 1);
					std::fill(edgesample[j].begin(), edgesample[j].end(), -1);
				}
				edgemap = std::vector<float>(2 * sfm + 1);
				cnt = std::vector<int>(2 * sfm + 1);

				//Remove cradle edge
				for (int j = sample.start; j <= sample.end; j++){
					int mid = p1v[j - sample.start];
					int lpmin = std::max(p1v[j - sample.start] - sfm, 0);
					int lpmax = std::min(p1v[j - sample.start] + sfm, filtered.rows - 1);
					for (int l = lpmin; l <= lpmax; l++){
						if ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0){
							edgesample[mid - l + sfm][j - sample.start] = filtered.at<float>(l, j);
							cnt[mid - l + sfm]++;
						}
					}
				}
				for (int j = 0; j < edgemap.size(); j++){
					if (cnt[j] != 0){
						edgemap[j] = getMedian(edgesample[j]);
					}
				}
				first = 0;
				while (first < edgemap.size() && cnt[first] == 0) first++;
				last = edgemap.size() - 1;
				while (last >= 0 && cnt[last] == 0) last--;

				//Find proper edge of the cradle
				separation = first;
				minv = edgemap[first];
				maxv = edgemap[first];
				for (int j = first; j < last; j++){
					if (edgemap[j] < minv)
						minv = edgemap[j];
					if (edgemap[j] > maxv)
						maxv = edgemap[j];
				}
				while (edgemap[separation] >(minv + maxv) / 2){
					separation++;
				}
				
				if (first < last){

					//Remove edge
					for (int j = sample.start; j <= sample.end; j++){
						//Adjust sampled edgemap
						int mid, lpmin, lpmax;

						int bk = 0;
						float mcost = 1e20;

						//Find position that best fits the edge
						for (int k = -step; k <= step; k++){
							int lpmin = std::max(p1v[j - sample.start] - sfm, 0);
							int lpmax = std::min(p1v[j - sample.start] + sfm, filtered.rows - 1);
							int mid = p1v[j - sample.start];
							int c = 0;
							float cost = 0;


							float edgemean = 0;
							float samplemean = 0;
							//Get means of the two edge profiles
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
									samplemean += filtered.at<float>(l, j);
									edgemean += edgemap[pos];
									c++;
								}
							}

							samplemean /= c;
							edgemean /= c;

							//Check how well it fits
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
									if (cnt[pos] != 0){
										float tmp = (edgemap[pos] - edgemean) - (filtered.at<float>(l, j) - samplemean);
										cost += tmp*tmp;
									}
								}
							}
							cost /= c;

							if (cost < mcost){
								mcost = cost;
								bk = k;
							}
						}

						//Remove intensity
						lpmin = std::max(p1v[j - sample.start] - sfm, 0);
						lpmax = std::min(p1v[j - sample.start] + sfm, filtered.rows - 1);
						mid = p1v[j - sample.start];

						float ref = lin_model_midu[1] * filtered.at<float>(lpmax, j) + lin_model_midu[0];
						float dif = filtered.at<float>(lpmax, j) - ref;

						float a, b;
						//Check if the edge is dropping or is just flat cradle
						if (std::abs(edgemap[last] - edgemap[first]) < std::abs(dif) *0.5){
							//Flat cradle, no edge
							a = 0;
							b = dif;
						}
						else{
							//Regular decaying edge
							a = dif / (edgemap[first] - edgemap[last]);
							b = dif - a * edgemap[first];
						}

						for (int l = lpmin; l < lpmax; l++){
							int pos = mid - l + sfm + bk;
							if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
								cradle.at<float>(l, j) = a*edgemap[pos] + b;
								if (pos <= separation){
									ms.piece_mask.at<ushort>(l, j) = ms.pieces;
								}
							}
						}
					}
				}

				//Remove edge - lower
				edgemap = std::vector<float>(2 * sfm + 1);
				cnt = std::vector<int>(2 * sfm + 1);

				//Model cradle edge
				for (int j = sample.start; j <= sample.end; j++){
					int mid = p2v[j - sample.start];
					int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
					int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.rows - 1);
					for (int l = lpmin; l <= lpmax; l++){
						if ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0){
							edgemap[mid - l + sfm] += filtered.at<float>(l, j);
							cnt[mid - l + sfm]++;
						}
					}
				}
				for (int j = 0; j < edgemap.size(); j++){
					if (cnt[j] != 0){
						edgemap[j] /= cnt[j];
					}
				}

				first = 0;
				while (first < edgemap.size() && cnt[first] == 0) first++;
				last = edgemap.size() - 1;
				while (last >= 0 && cnt[last] == 0) last--;
				
				//Find proper edge of the cradle
				separation = last - 1;
				minv = edgemap[separation];
				maxv = edgemap[separation];
				for (int j = first; j < last; j++){
					if (edgemap[j] < minv)
						minv = edgemap[j];
					if (edgemap[j] > maxv)
						maxv = edgemap[j];
				}
				while (edgemap[separation] >(minv + maxv) / 2){
					separation--;
				}

				if (first < last){

					//Remove edge
					for (int j = sample.start; j <= sample.end; j++){
						int bk = 0;
						float mcost = 1e20;

						//Find position that best fits the edge
						for (int k = -step; k <= step; k++){
							int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
							int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.rows - 1);
							int mid = p2v[j - sample.start];
							int c = 0;
							float cost = 0;

							float edgemean = 0;
							float samplemean = 0;
							//Get means of the two edge profiles
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
									samplemean += filtered.at<float>(l, j);
									edgemean += edgemap[pos];
									c++;
								}
							}

							samplemean /= c;
							edgemean /= c;

							//Check how well it fits
							for (int l = lpmin; l < lpmax; l++){
								int pos = mid - l + sfm + k;
								if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
									if (edgemap[pos] != 0){
										float tmp = (edgemap[pos] - edgemean) - (filtered.at<float>(l, j) - samplemean);
										cost += tmp*tmp;
									}
								}
							}
							cost /= c;

							if (cost < mcost){
								mcost = cost;
								bk = k;
							}
						}

						//Remove intensity
						int lpmin = std::max(p2v[j - sample.start] - sfm, 0);
						int lpmax = std::min(p2v[j - sample.start] + sfm, filtered.rows - 1);
						int mid = p2v[j - sample.start];

						float ref = lin_model_midl[1] * filtered.at<float>(lpmin, j) + lin_model_midl[0]; //filtered.at<float>(lpmax, j);
						float dif = filtered.at<float>(lpmin, j) - ref;
						float a, b;

						//Check if the edge is dropping or is just flat cradle
						if (std::abs(edgemap[last] - edgemap[first]) < std::abs(dif) *0.5){
							//Flat cradle, no edge
							a = 0;
							b = dif;
						}
						else{
							//Regular decaying edge
							a = dif / (edgemap[last] - edgemap[first]);
							b = dif - a * edgemap[last];
						}

						for (int l = lpmin; l < lpmax; l++){
							int pos = mid - l + sfm + bk;
							if (pos >= 0 && pos < edgemap.size() && ((mask.at<char>(l, j) & (V_MASK | DEFECT)) == 0)){
								cradle.at<float>(l, j) = a*edgemap[pos] + b;
								if (pos >= separation){
									ms.piece_mask.at<ushort>(l, j) = ms.pieces;
								}
							}
						}
					}
				}
			}
		}
	}

	//Recursive watershed algorithm that marks all pixels of image 'val' smaller then threshold th,
	//propagating from starting point (i,j), marked in binary image 'mark'
	void watershed(int i, int j, int th, cv::Mat &val, cv::Mat &mark){
		if (i > 0 && mark.at<float>(i - 1, j) == 0 && th > val.at<float>(i - 1, j)){
			mark.at<float>(i - 1, j) = 1;
			watershed(i - 1, j, th, val, mark);
		}
		if (i < val.rows - 1 && mark.at<float>(i + 1, j) == 0 && th > val.at<float>(i + 1, j)){
			mark.at<float>(i + 1, j) = 1;
			watershed(i + 1, j, th, val, mark);
		}
		if (j > 0 && mark.at<float>(i, j - 1) == 0 && th > val.at<float>(i, j - 1)){
			mark.at<float>(i, j - 1) = 1;
			watershed(i, j - 1, th, val, mark);
		}
		if (j < val.cols - 1 && mark.at<float>(i, j + 1) == 0 && th > val.at<float>(i, j + 1)){
			mark.at<float>(i, j + 1) = 1;
			watershed(i, j + 1, th, val, mark);
		}
	}

	//Function returns a value 'alpha' that approximately minimizes the optimization problem
	// alpha = arg min (Var[img - alpha * (cradle .* mark)])
	// where .* denotes a pixel-wise product of two images
	float findAlpha(cv::Mat &img, cv::Mat &cradle, cv::Mat &mark){
		float ba, minv, step;
		cv::Mat rec(cradle.rows, cradle.cols, CV_32F, cv::Scalar(0));

		for (int i = 0; i < cradle.rows; i++){
			for (int j = 0; j < cradle.cols; j++){
				if (mark.at<float>(i, j) == 0)
					rec.at<float>(i, j) = cradle.at<float>(i, j);
				else
					rec.at<float>(i, j) = 0;
			}
		}
		minv = getVariance(img - rec);
		ba = 0;
		step = 0.1;

		for (float alpha = step; alpha <= 1; alpha += step){
			//Fill up weighted
			for (int i = 0; i < cradle.rows; i++){
				for (int j = 0; j < cradle.cols; j++){
					if (mark.at<float>(i, j) == 0)
						rec.at<float>(i, j) = cradle.at<float>(i, j);
					else
						rec.at<float>(i, j) = alpha * cradle.at<float>(i, j);
				}
			}

			//Check if lower variance
			float cost = getVariance(img - rec);
			if (cost < minv){
				ba = alpha;
				minv = cost;
			}
		}
		return ba;
	}

	//Function responsable for detecting and correcting, when possible, for overcorrections in border areas
	//of cross-sections. The detection is based on trying to identify strong, black lines in these areas 
	//and if they are present, findin the right smoothing parameter that removes these artifacts.
	void removeEdgeArtifact(const cv::Mat &img, cv::Mat &cradle, int dir, int stx, int enx, int sty, int eny){
		cv::Mat select, dct, rec, lowp, input, lowpassori, orig;

		if (stx < 0)
			stx = 0;
		if (enx > img.rows - 1){
			enx = img.rows - 1;
		}
		if (sty < 0)
			sty = 0;
		if (eny > img.cols - 1){
			eny = img.cols - 1;
		}

		if ((enx - stx) % 2 == 1)
			enx--;
		if ((eny - sty) % 2 == 1)
			eny--;

		input = img(cv::Range(stx, enx), cv::Range(sty, eny)) - cradle(cv::Range(stx, enx), cv::Range(sty, eny));
		select = input;

		//Count how large the black line is
		std::vector<float> in_sum;
		if (dir == TextureRemoval::HORIZONTAL){
			in_sum = std::vector<float>(select.rows);
			for (int i = 0; i < select.rows; i++){
				for (int j = 0; j < select.cols; j++){
					in_sum[i] += select.at<float>(i, j);
				}
			}
			for (int i = 0; i < in_sum.size(); i++){
				in_sum[i] /= select.cols;
			}
		}
		else{
			in_sum = std::vector<float>(select.cols);
			for (int i = 0; i < select.rows; i++){
				for (int j = 0; j < select.cols; j++){
					in_sum[j] += select.at<float>(i, j);
				}
			}
			for (int i = 0; i < in_sum.size(); i++){
				in_sum[i] /= select.rows;
			}
		}
		float medv = getMedian(in_sum);

		//Check number of pixels below 0.7x the median
		int wb = 0;
		for (int i = 0; i < in_sum.size(); i++){
			if (in_sum[i] < medv * 0.7)
				wb++;
		}

		if (wb == 0)
			return;

		//Localize the black line position better
		cv::Mat kernel, dest;
		if (dir == TextureRemoval::HORIZONTAL){
			kernel = cv::getGaborKernel(cv::Size(2 * wb, select.cols * 0.9), 5, 0, 2 * wb, 0.01, 0);
			cv::transpose(kernel, kernel);
			cv::filter2D(select, dest, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		}
		else{
			kernel = cv::getGaborKernel(cv::Size(2 * wb, select.rows * 0.9), 5, 0, 2 * wb, 0.01, 0);
			cv::filter2D(select, dest, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		}

		//Low-pass filtering
		cv::blur(select, lowp, cv::Size(std::max(int(select.rows*0.05), 2), std::max(int(select.cols*0.05), 2)));
		select = select - lowp;

		//DCT filtering
		cv::dct(select, dct); //Forward dct transform
		if (dir == TextureRemoval::HORIZONTAL){
			//Horizontal frequency filtering
			for (int j = 0; j < std::max((int)(dct.cols * 0.1), 1); j++){
				for (int i = 1; i < dct.rows; i++){
					dct.at<float>(i, j) = 0;
				}
			}
		}
		else{
			//Vertical frequency filtering
			for (int j = 0; j < std::max((int)(dct.rows * 0.1), 1); j++){
				for (int i = 1; i < dct.cols; i++){
					dct.at<float>(j, i) = 0;
				}
			}
		}
		cv::dct(dct, rec, cv::DCT_INVERSE); //Inverse dct transform

		//Reconstruction difference map
		rec = select - rec;
		dest = dest - mean(dest);
		
		//Apply strong smooth filter on rec
		cv::blur(rec, rec, cv::Size(wb, wb));

		//Find min value position
		float v = dest.at<float>(0, 0);
		int px, py;
		px = py = 0;

		for (int i = stx; i < enx; i++){
			for (int j = sty; j < eny; j++){
				if (v > dest.at<float>(i - stx, j - sty)){
					v = dest.at<float>(i - stx, j - sty);
					px = i - stx;
					py = j - sty;
				}
			}
		}

		//Watershed for marking where we adopt the cradle mask
		cv::Mat mark(rec.rows, rec.cols, CV_32F, cv::Scalar(0));
		std::vector<int> vx(rec.rows*rec.cols), vy(rec.rows*rec.cols);
		int start = 0, end = 1;
		float th = 0;
		vx[0] = px;
		vy[0] = py;

		while (start < end){
			int i = vx[start];
			int j = vy[start];
			if (i > 0 && mark.at<float>(i - 1, j) == 0 && th > rec.at<float>(i - 1, j)){
				mark.at<float>(i - 1, j) = 1;
				vx[end] = i - 1;
				vy[end] = j;
				end++;
			}
			if (i < rec.rows - 1 && mark.at<float>(i + 1, j) == 0 && th > rec.at<float>(i + 1, j)){
				mark.at<float>(i + 1, j) = 1;
				vx[end] = i + 1;
				vy[end] = j;
				end++;
			}
			if (j > 0 && mark.at<float>(i, j - 1) == 0 && th > rec.at<float>(i, j - 1)){
				mark.at<float>(i, j - 1) = 1;
				vx[end] = i;
				vy[end] = j - 1;
				end++;
			}
			if (j < rec.cols - 1 && mark.at<float>(i, j + 1) == 0 && th > rec.at<float>(i, j + 1)){
				mark.at<float>(i, j + 1) = 1;
				vx[end] = i;
				vy[end] = j + 1;
				end++;
			}
			start++;
		}

		cv::Mat cradleseg = cradle(cv::Range(stx, enx), cv::Range(sty, eny));
		cv::Mat imgseg = img(cv::Range(stx, enx), cv::Range(sty, eny));
		float alpha = findAlpha(imgseg, cradleseg, mark);

		for (int i = stx; i < enx; i++){
			for (int j = sty; j < eny; j++){
				if (mark.at<float>(i - stx, j - sty) == 0){
					mark.at<float>(i - stx, j - sty) = cradle.at<float>(i, j);
				}
				else{
					mark.at<float>(i - stx, j - sty) = alpha * cradle.at<float>(i, j);
				}
			}
		}

		//Find ideal smoothing amount
		float minv = -1;
		int bestv = 3;
		for (int i = 3; i < 2 * wb; i += 2){
			cv::Mat bmark, check;
			cv::blur(mark, bmark, cv::Size(i, i));
			check = imgseg - bmark;

			float var = getVariance(check);
			if (minv == -1){
				minv = var;
			}
			if (var < minv){
				minv = var;
				bestv = i;
			}
		}

		//Blur it a bit to remove edge artifacts
		cv::blur(mark, mark, cv::Size(bestv, bestv));

		for (int i = stx; i < enx; i++){
			for (int j = sty; j < eny; j++){
				cradle.at<float>(i, j) = mark.at<float>(i - stx, j - sty);
			}
		}

		return;
	}

	//Randon transform used to find cradle tilting angle
	std::vector<double> findRadonTransformAngle(const cv::Mat &img, cv::Mat &mask, std::vector<double> &thetav, int sx, int ex, int sy, int ey, int noflag){
		int W = (ey - sy), H = (ex - sx);
		int maxv = (int)(std::sqrt(W*W + H*H)) + 1;
		double center_x = sy + W / 2;
		double center_y = sx + H / 2;
		cv::Mat acc(2 * maxv, thetav.size(), CV_32F, cv::Scalar(0));

		//Calculate Radon transform
		for (int y = sx; y < sx + H; y++){
			for (int x = sy; x < sy + W; x++){
				if (y >= 0 && y < mask.rows && x >= 0 && x < mask.cols && ((mask.at<char>(y, x) & noflag) == 0)){
					for (int i = 0; i < thetav.size(); i++){
						double theta = thetav[i];
						if (theta < 0)
							theta += 360;
						if (theta > 360)
							theta -= 360;
						int r = (x - center_x) *1.0 * cos(RAD(theta)) - (y - center_y) *1.0 * sin(RAD(theta));
						acc.at<float>((r + maxv), i) = acc.at<float>((r + maxv), i) + (img).at<float>(y, x);
					}
				}
			}
		}

		//Get squared values + sum up for each angle
		cv::Mat sum(1, thetav.size(), CV_32F, cv::Scalar(0));
		for (int i = 0; i < thetav.size(); i++){
			for (int j = 0; j < 2 * maxv; j++){
				acc.at<float>(j, i) = acc.at<float>(j, i)*acc.at<float>(j, i);
				sum.at<float>(0, i) += acc.at<float>(j, i);
			}
		}

		//Find best angle
		int ind = 0;
		for (int i = 1; i < thetav.size(); i++){
			if (sum.at<float>(0, ind) < sum.at<float>(0, i))
				ind = i;
		}
		std::vector<double> res;
		res.push_back(thetav[ind]);

		//Find maxima for given angle
		int ind2 = 0;
		for (int i = 1; i < 2 * maxv; i++){
			if (acc.at<float>(i, ind) > acc.at<float>(ind2, ind))
				ind2 = i;
		}

		res.push_back(ind2 - maxv);
		return res;
	}

	//Marks approximate vertical cradle position in the mask as vertical cradle so that it won't interfere when estimating horizontal cradle piece position
	void createMaskVertical(cv::Mat &mask, std::vector<int> &vrange, int s){
		for (int i = 0; i < vrange.size() / 2; i++){
			for (int j = 0; j < (mask).rows; j++){
				for (int k = std::max(0, vrange[i * 2] - s); k < std::min(mask.cols - 1, vrange[i * 2 + 1] + s); k++){
					mask.at<char>(j, k) |= V_MASK;
				}
			}
		}
	}

	//Undoes the effect of createMaskVertical()
	void removeMaskVertical(cv::Mat &mask, std::vector<int> &vrange, int s){
		for (int i = 0; i < vrange.size() / 2; i++){
			for (int j = 0; j < (mask).rows; j++){
				for (int k = std::max(0, vrange[i * 2] - s); k < std::min(mask.cols - 1, vrange[i * 2 + 1] + s); k++){
					mask.at<char>(j, k) ^= V_MASK;
				}
			}
		}
	}

	//Cradle detection method, returning approximate horizontal/vertical cradle positions in 'vrange' and 'hrange'
	void cradledetect(const cv::Mat &in, const cv::Mat &mask, std::vector<int> &vrange, std::vector<int> &hrange){

		//Filter gradients for horizontal/vertical
		int L = 20;
		//Vertical filtering
		cv::Mat vkern(1, 2 * L, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}
		for (int i = L; i < 2 * L; i++){
			vkern.at<float>(0, i) = -1;
		}
		cv::Mat hkern(2 * L, 1, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}'
		for (int i = L; i < 2 * L; i++){
			hkern.at<float>(i, 0) = -1;
		}

		cv::Mat ghimg, gvimg;
		cv::filter2D(in, gvimg, CV_32F, hkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
		cv::filter2D(in, ghimg, CV_32F, vkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

		std::vector<int> solv;
		std::vector<int> solh;
		cv::Mat dest;
		
		//Sum up vertical elements
		cv::Mat vsum(1, ghimg.cols, CV_32F, cv::Scalar(0));
		for (int j = 0; j < ghimg.cols; j++){
			for (int i = 0; i < ghimg.rows; i++){
				if ((mask.at<char>(i, j) & DEFECT) != DEFECT){
					vsum.at<float>(0, j) = vsum.at<float>(0, j) + ghimg.at<float>(i, j);
				}
			}
		}

		//Smooth filtering
		int s = std::max(3.0, std::min(10.0, std::max(gvimg.rows, gvimg.cols) / 230.0));	// 3 <= s <= 10
		cv::Mat smooth(1, s, CV_32F, 1.0 / s);
		cv::filter2D(vsum, dest, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Normalize - extract mean of the signal
		float mean = 0;
		for (int i = 0; i < (ghimg).cols; i++){
			mean += dest.at<float>(0, i);
		}
		mean /= (ghimg).cols;
		dest = dest - mean;

		std::vector<int> peaks, peak_type, allpeaksi;
		std::vector<double> peak_val, allpeaks;
		float maxpeak, minpeak;

		//Get maximas
		for (int i = 1; i < dest.cols - 1; i++){
			if (dest.at<float>(0, i - 1) < dest.at<float>(0, i) &&
				dest.at<float>(0, i + 1) < dest.at<float>(0, i)){
				allpeaksi.push_back(i);
				allpeaks.push_back(dest.at<float>(0, i));
			}
		}
		//Find max peak
		maxpeak = allpeaks[0];
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] > maxpeak){
				maxpeak = allpeaks[i];
			}
		}

		//Select peaks that are larger than half of the max peak
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] > maxpeak / 2){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MAXIMA);
			}
		}

		//Clear arrays
		allpeaks.clear();
		allpeaksi.clear();

		//Get minimas
		for (int i = 1; i < dest.cols - 1; i++){
			if (dest.at<float>(0, i - 1) > dest.at<float>(0, i) &&
				dest.at<float>(0, i + 1) > dest.at<float>(0, i)){
				allpeaksi.push_back(i);
				allpeaks.push_back(dest.at<float>(0, i));
			}
		}

		//Find min peak
		minpeak = allpeaks[0];
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] < minpeak){
				minpeak = allpeaks[i];
			}
		}

		//Select peaks that are smaller than half of the min peak
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] < minpeak / 2){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MINIMA);
			}
		}

		//Sort maximas based on position
		for (int i = 0; i < peaks.size(); i++){
			for (int j = i + 1; j < peaks.size(); j++){
				if (peaks[i] > peaks[j]){
					int tmp = peaks[i];
					peaks[i] = peaks[j];
					peaks[j] = tmp;
					tmp = peak_type[i];
					peak_type[i] = peak_type[j];
					peak_type[j] = tmp;
					float tmp2 = peak_val[i];
					peak_val[i] = peak_val[j];
					peak_val[j] = tmp2;
				}
			}
		}

		//Create array of start/end positions
		int state;
		if (peak_type[0] == MAXIMA){
			//Image starts with cradle part
			vrange.push_back(0);
			state = MAXIMA;
		}
		else{
			state = MINIMA;
		}
		for (int i = 0; i < peak_type.size(); i++){
			if (state == peak_type[i]){
				vrange.push_back(peaks[i]);
				if (state == MAXIMA)
					state = MINIMA;
				else if (state == MINIMA)
					state = MAXIMA;
			}
		}
		if (state == MAXIMA){
			//Image ends with a low point, mark last segment as cradle
			vrange.push_back(gvimg.cols);
		}

		//Sum up vertical elements
		cv::Mat hsum(1, (gvimg).rows, CV_32F, cv::Scalar(0));
		for (int j = 0; j < (gvimg).cols; j++){
			for (int i = 0; i < (gvimg).rows; i++){
				if ((mask.at<char>(i, j) & DEFECT) != DEFECT){
					hsum.at<float>(0, i) = hsum.at<float>(0, i) + (gvimg).at<float>(i, j);
				}
			}
		}

		//Smooth filtering
		cv::filter2D(hsum, dest, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		
		//Normalize - extract mean of the signal
		mean = 0;
		for (int i = 0; i < (gvimg).rows; i++){
			mean += dest.at<float>(0, i);
		}
		mean /= (gvimg).rows;
		//Extract mean
		dest = dest - mean;

		peaks.clear();
		peak_val.clear();
		peak_type.clear();
		allpeaksi.clear();
		allpeaks.clear();

		//Get maximas
		for (int i = 1; i < dest.cols - 1; i++){
			if (dest.at<float>(0, i - 1) < dest.at<float>(0, i) &&
				dest.at<float>(0, i + 1) < dest.at<float>(0, i)){
				allpeaksi.push_back(i);
				allpeaks.push_back(dest.at<float>(0, i));
			}
		}
		//Find max peak
		maxpeak = allpeaks[0];
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] > maxpeak){
				maxpeak = allpeaks[i];
			}
		}

		//Select peaks that are larger than half of the max peak
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] > maxpeak / 2){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MAXIMA);
			}
		}

		//Clear arrays
		allpeaks.clear();
		allpeaksi.clear();

		//Get minimas
		for (int i = 1; i < dest.cols - 1; i++){
			if (dest.at<float>(0, i - 1) > dest.at<float>(0, i) &&
				dest.at<float>(0, i + 1) > dest.at<float>(0, i)){
				allpeaksi.push_back(i);
				allpeaks.push_back(dest.at<float>(0, i));
			}
		}

		//Find min peak
		minpeak = allpeaks[0];
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] < minpeak){
				minpeak = allpeaks[i];
			}
		}

		//Select peaks that are smaller than half of the min peak
		for (int i = 0; i < allpeaks.size(); i++){
			if (allpeaks[i] < minpeak / 2){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MINIMA);
			}
		}

		//Sort maximas based on position
		for (int i = 0; i < peaks.size(); i++){
			for (int j = i + 1; j < peaks.size(); j++){
				if (peaks[i] > peaks[j]){
					int tmp = peaks[i];
					peaks[i] = peaks[j];
					peaks[j] = tmp;
					tmp = peak_type[i];
					peak_type[i] = peak_type[j];
					peak_type[j] = tmp;
					float tmp2 = peak_val[i];
					peak_val[i] = peak_val[j];
					peak_val[j] = tmp2;
				}
			}
		}

		//Create array of start/end positions
		if (peak_type[0] == MAXIMA){
			//Image starts with cradle part
			hrange.push_back(0);
			state = MAXIMA;
		}
		else{
			state = MINIMA;
		}
		for (int i = 0; i < peak_type.size(); i++){
			if (state == peak_type[i]){
				hrange.push_back(peaks[i]);
				if (state == MAXIMA)
					state = MINIMA;
				else if (state == MINIMA)
					state = MAXIMA;
			}
		}
		if (state == MAXIMA){
			//Image ends with a low point, mark last segment as cradle
			hrange.push_back(gvimg.rows);
		}
	}

	//Cradle detection method, returning approximate horizontal/vertical cradle positions in 'vrange' and 'hrange'
	//with number of vertical and horizontal pieces to be detected specified by 'vn' and 'hn' 
	void cradledetect(const cv::Mat &in, const cv::Mat &mask, int vn, int hn, std::vector<int> &vrange, std::vector<int> &hrange){

		//Filter gradients for horizontal/vertical
		int L = 20;
		//Vertical filtering
		cv::Mat vkern(1, 2 * L, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}
		for (int i = L; i < 2 * L; i++){
			vkern.at<float>(0, i) = -1;
		}
		cv::Mat hkern(2 * L, 1, CV_32F, 1);				//{-1, -1, -1, .. 1, 1, 1, 1}'
		for (int i = L; i < 2 * L; i++){
			hkern.at<float>(i, 0) = -1;
		}

		cv::Mat ghimg, gvimg;
		cv::filter2D(in, gvimg, CV_32F, hkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
		cv::filter2D(in, ghimg, CV_32F, vkern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

		std::vector<int> solv;
		std::vector<int> solh;
		cv::Mat dest;

		//Sum up vertical elements
		cv::Mat vsum(1, ghimg.cols, CV_32F, cv::Scalar(0));
		for (int j = 0; j < ghimg.cols; j++){
			for (int i = 0; i < ghimg.rows; i++){
				if ((mask.at<char>(i, j) & DEFECT) != DEFECT){
					vsum.at<float>(0, j) = vsum.at<float>(0, j) + ghimg.at<float>(i, j);
				}
			}
		}

		//Smooth filtering
		int s = std::max(3.0, std::min(10.0, std::max(gvimg.rows, gvimg.cols) / 230.0));	// 3 <= s <= 10
		cv::Mat smooth(1, s, CV_32F, 1.0 / s);
		cv::filter2D(vsum, dest, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Normalize - extract mean of the signal
		float mean = 0;
		for (int i = 0; i < (ghimg).cols; i++){
			mean += dest.at<float>(0, i);
		}
		mean /= (ghimg).cols;
		dest = dest - mean;

		std::vector<int> peaks, peak_type, allpeaksi;
		std::vector<double> peak_val, allpeaks, cost;
		int state, detected, selvn = vn;

		detected = 0;
		while (vn != detected){

			vrange.clear();

			//Get maximas
			for (int i = 1; i < dest.cols - 1; i++){
				if (dest.at<float>(0, i - 1) < dest.at<float>(0, i) &&
					dest.at<float>(0, i + 1) < dest.at<float>(0, i)){
					allpeaksi.push_back(i);
					allpeaks.push_back(dest.at<float>(0, i));
				}
			}

			//Sort by intensity
			for (int i = 0; i < allpeaks.size(); i++){
				for (int j = i + 1; j < allpeaks.size(); j++){
					if (allpeaks[i] < allpeaks[j]){
						int tmp = allpeaksi[i];
						allpeaksi[i] = allpeaksi[j];
						allpeaksi[j] = tmp;
						float tmp2 = allpeaks[i];
						allpeaks[i] = allpeaks[j];
						allpeaks[j] = tmp2;
					}
				}
			}

			//Take vn max peaks
			for (int i = 0; i < std::min(selvn, (int)allpeaks.size()); i++){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MAXIMA);
			}

			//Clear arrays
			allpeaks.clear();
			allpeaksi.clear();

			//Get minimas
			for (int i = 1; i < dest.cols - 1; i++){
				if (dest.at<float>(0, i - 1) > dest.at<float>(0, i) &&
					dest.at<float>(0, i + 1) > dest.at<float>(0, i)){
					allpeaksi.push_back(i);
					allpeaks.push_back(dest.at<float>(0, i));
				}
			}

			//Sort by intensity
			for (int i = 0; i < allpeaks.size(); i++){
				for (int j = i + 1; j < allpeaks.size(); j++){
					if (allpeaks[i] > allpeaks[j]){
						int tmp = allpeaksi[i];
						allpeaksi[i] = allpeaksi[j];
						allpeaksi[j] = tmp;
						float tmp2 = allpeaks[i];
						allpeaks[i] = allpeaks[j];
						allpeaks[j] = tmp2;
					}
				}
			}

			//Take vn max peaks
			for (int i = 0; i < std::min(selvn, (int)allpeaks.size()); i++){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MINIMA);
			}

			//Sort peaks based on position
			for (int i = 0; i < peaks.size(); i++){
				for (int j = i + 1; j < peaks.size(); j++){
					if (peaks[i] > peaks[j]){
						int tmp = peaks[i];
						peaks[i] = peaks[j];
						peaks[j] = tmp;
						tmp = peak_type[i];
						peak_type[i] = peak_type[j];
						peak_type[j] = tmp;
						float tmp2 = peak_val[i];
						peak_val[i] = peak_val[j];
						peak_val[j] = tmp2;
					}
				}
			}

			//Create array of start/end positions
			if (peak_type[0] == MAXIMA){
				//Image starts with cradle part
				vrange.push_back(0);
				cost.push_back(std::abs(peak_val[0]) * 2);
				state = MAXIMA;
			}
			else{
				state = MINIMA;
			}
			for (int i = 0; i < peak_type.size(); i++){
				if (state == peak_type[i]){
					vrange.push_back(peaks[i]);
					if (state == MINIMA){
						cost.push_back(std::abs(peak_val[i]));
					}
					else{
						cost[cost.size() - 1] += std::abs(peak_val[i]);
					}

					if (state == MAXIMA)
						state = MINIMA;
					else if (state == MINIMA)
						state = MAXIMA;
				}
			}
			if (state == MAXIMA){
				//Image ends with a low point, mark last segment as cradle
				vrange.push_back(gvimg.cols);
				cost[cost.size() - 1] *= 2;
			}

			//Reduce size to vn elements
			while (vrange.size() > 2 * vn){
				int mini = 0;
				float minc = cost[0];
				for (int i = 0; i < cost.size(); i++){
					if (cost[i] < minc){
						minc = cost[i];
						mini = i;
					}
				}
				//Shift over previous points
				for (int i = mini * 2 + 2; i < vrange.size(); i++){
					vrange[i - 2] = vrange[i];
				}
				//Resize
				vrange.resize(vrange.size() - 2);
			}
			detected = vrange.size() / 2;
			if (detected < vn){
				selvn++;
			}
		}


		//Sum up vertical elements
		cv::Mat hsum(1, (gvimg).rows, CV_32F, cv::Scalar(0));
		for (int j = 0; j < (gvimg).cols; j++){
			for (int i = 0; i < (gvimg).rows; i++){
				if ((mask.at<char>(i, j) & DEFECT) != DEFECT){
					hsum.at<float>(0, i) = hsum.at<float>(0, i) + (gvimg).at<float>(i, j);
				}
			}
		}


		//Smooth filtering
		cv::filter2D(hsum, dest, CV_32F, smooth, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Normalize - extract mean of the signal
		mean = 0;
		for (int i = 0; i < (gvimg).rows; i++){
			mean += dest.at<float>(0, i);
		}
		mean /= (gvimg).rows;
		//Extract mean
		dest = dest - mean;

		peaks.clear();
		peak_val.clear();
		peak_type.clear();
		allpeaksi.clear();
		allpeaks.clear();
		cost.clear();

		int selhn = hn;
		detected = 0;
		while (hn != detected){

			hrange.clear();

			//Get maximas
			for (int i = 1; i < dest.cols - 1; i++){
				if (dest.at<float>(0, i - 1) < dest.at<float>(0, i) &&
					dest.at<float>(0, i + 1) < dest.at<float>(0, i)){
					allpeaksi.push_back(i);
					allpeaks.push_back(dest.at<float>(0, i));
				}
			}

			//Sort by intensity
			for (int i = 0; i < allpeaks.size(); i++){
				for (int j = i + 1; j < allpeaks.size(); j++){
					if (allpeaks[i] < allpeaks[j]){
						int tmp = allpeaksi[i];
						allpeaksi[i] = allpeaksi[j];
						allpeaksi[j] = tmp;
						float tmp2 = allpeaks[i];
						allpeaks[i] = allpeaks[j];
						allpeaks[j] = tmp2;
					}
				}
			}

			//Take hn max peaks
			for (int i = 0; i < std::min(selhn, (int)allpeaks.size()); i++){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MAXIMA);
			}

			//Clear arrays
			allpeaks.clear();
			allpeaksi.clear();

			//Get minimas
			for (int i = 1; i < dest.cols - 1; i++){
				if (dest.at<float>(0, i - 1) > dest.at<float>(0, i) &&
					dest.at<float>(0, i + 1) > dest.at<float>(0, i)){
					allpeaksi.push_back(i);
					allpeaks.push_back(dest.at<float>(0, i));
				}
			}

			//Sort by intensity
			for (int i = 0; i < allpeaks.size(); i++){
				for (int j = i + 1; j < allpeaks.size(); j++){
					if (allpeaks[i] > allpeaks[j]){
						int tmp = allpeaksi[i];
						allpeaksi[i] = allpeaksi[j];
						allpeaksi[j] = tmp;
						float tmp2 = allpeaks[i];
						allpeaks[i] = allpeaks[j];
						allpeaks[j] = tmp2;
					}
				}
			}

			//Take hn max peaks
			for (int i = 0; i < std::min(selhn, (int)allpeaks.size()); i++){
				//Add to accepted list
				peaks.push_back(allpeaksi[i]);
				peak_val.push_back(allpeaks[i]);
				peak_type.push_back(MINIMA);
			}

			//Sort peaks based on position
			for (int i = 0; i < peaks.size(); i++){
				for (int j = i + 1; j < peaks.size(); j++){
					if (peaks[i] > peaks[j]){
						int tmp = peaks[i];
						peaks[i] = peaks[j];
						peaks[j] = tmp;
						tmp = peak_type[i];
						peak_type[i] = peak_type[j];
						peak_type[j] = tmp;
						float tmp2 = peak_val[i];
						peak_val[i] = peak_val[j];
						peak_val[j] = tmp2;
					}
				}
			}

			//Create array of start/end positions
			if (peak_type[0] == MAXIMA){
				//Image starts with cradle part
				hrange.push_back(0);
				cost.push_back(std::abs(peak_val[0]) * 2);
				state = MAXIMA;
			}
			else{
				state = MINIMA;
			}
			for (int i = 0; i < peak_type.size(); i++){
				if (state == peak_type[i]){
					hrange.push_back(peaks[i]);
					if (state == MINIMA){
						cost.push_back(std::abs(peak_val[i]));
					}
					else{
						cost[cost.size() - 1] += std::abs(peak_val[i]);
					}

					if (state == MAXIMA)
						state = MINIMA;
					else if (state == MINIMA)
						state = MAXIMA;
				}
			}
			if (state == MAXIMA){
				//Image ends with a low point, mark last segment as cradle
				hrange.push_back(gvimg.rows);
				cost[cost.size() - 1] *= 2;
			}

			//Reduce size to hn elements
			while (hrange.size() > 2 * hn){
				int mini = 0;
				float minc = cost[0];
				for (int i = 0; i < cost.size(); i++){
					if (cost[i] < minc){
						minc = cost[i];
						mini = i;
					}
				}
				//Shift over previous points
				for (int i = mini * 2 + 2; i < hrange.size(); i++){
					hrange[i - 2] = hrange[i];
				}
				//Resize
				hrange.resize(hrange.size() - 2);
			}


			detected = hrange.size() / 2;
			if (detected < hn){
				selhn++;
			}
		}

		return;
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	std::vector<float> getEdges(cv::Mat &R, double angle, int type, int s){
		//res[0] - dI
		//res[1-res.size()] : edgeshape

		//Local copy of r
		cv::Mat r = (R).clone();

		if (type == 2){
			r = flipVertical(r);
		}

		//Get histogram
		int channel = { 0 };
		int bins = s / 3;
		std::vector<int> cnt(bins + 2, 0);

		float minv = max(r), maxv = min(r);
		int shift = sin(angle / 180 * M_PI)*s;
		for (int i = (R).rows*1.0 / 2 - shift; i <= (R).rows*1.0 / 2 + shift; i++){
			if (r.at<float>(i, 0) > maxv)
				maxv = r.at<float>(i, 0);
			if (r.at<float>(i, 0) < minv)
				minv = r.at<float>(i, 0);
		}

		float range[2] = { minv, static_cast<float>(maxv*1.001) };
		float step = (range[1] - range[0]) / bins;

		//Calculate histogram
		for (int i = (R).rows*1.0 / 2 - shift; i <= (R).rows*1.0 / 2 + shift; i++){
			float val = r.at<float>(i, 0);
			cnt[(val - range[0]) / (range[1] - range[0])*bins + 1 + 0.5]++;
		}

		//Find min/max peak
		int imin, imax, vmin, vmax;
		imin = imax = -1;
		vmin = 0;
		vmax = 0;
		for (int i = 1; i <= bins; i++){
			//If peak..
			if (cnt[i] > cnt[i - 1] && cnt[i] > cnt[i + 1]){
				if (cnt[i] > vmin){
					if (cnt[i] > vmax){
						vmin = vmax;
						imin = imax;
						vmax = cnt[i];
						imax = i;
					}
					else{
						vmin = cnt[i];
						imin = i;
					}
				}
			}
		}
		double highintense = (imax - 1) * step + step / 2 + minv;
		double lowintense = (imin - 1) * step + step / 2 + minv;

		//Funky..
		if (highintense > lowintense){
			double tmp = highintense;
			highintense = lowintense;
			lowintense = tmp;
		}
		double dI = highintense - lowintense;

		int usecdf = 0;
		cv::Mat ledge, redge;
		ledge = get_edgeshape(r(cv::Range(r.rows / 2 - s, r.rows / 2), cv::Range(0, 1)), 1, highintense - step, lowintense + step, usecdf);		//left
		redge = get_edgeshape(r(cv::Range(r.rows / 2, r.rows / 2 + s), cv::Range(0, 1)), 2, highintense - step, lowintense + step, usecdf);		//right

		//Merge edgeshapes
		cv::Mat edgeshape(ledge.rows + redge.rows - 1, 1, CV_32F, cv::Scalar(0));
		for (int i = 0; i < ledge.rows; i++)
			edgeshape.at<float>(i, 0) = ledge.at<float>(i, 0);
		for (int i = 1; i < redge.rows; i++)
			edgeshape.at<float>(i + ledge.rows - 1, 0) = redge.at<float>(i, 0);
		edgeshape -= edgeshape.at<float>(edgeshape.rows - 1, 0);

		//Zero out until first negative elements
		int ind = -1;
		for (int i = 0; i < edgeshape.rows; i++){
			if (edgeshape.at<float>(i, 0) < 0){
				ind = i;
				break;
			}
		}
		if (ind != -1){
			for (int i = 0; i < ind; i++){
				edgeshape.at<float>(i, 0) = 0;
			}
		}

		// Zero out from first negative elements
		ind = -1;
		for (int i = 0; i < edgeshape.rows; i++){
			if (edgeshape.at<float>(i, 0) < 0){
				ind = i;
				break;
			}
		}
		if (ind != -1){
			for (int i = ind; i < edgeshape.rows; i++){
				edgeshape.at<float>(i, 0) = 0;
			}
		}

		dI = max(edgeshape) - min(edgeshape);
		if (type == 2){
			//Flip edgeshape
			edgeshape = flipVertical(edgeshape);
		}

		int L = (R).rows - edgeshape.rows;
		int Lf = std::floor(L*1.0 / 2);
		int Lc = std::ceil(L*1.0 / 2);

		//Extend edgeshape
		std::vector<float> res(1 + Lf + edgeshape.rows + Lc, 0);
		//res[0] = dI
		res[0] = dI;

		//Fill up values
		for (int i = 0; i < edgeshape.rows; i++){
			res[i + Lf + 1] = edgeshape.at<float>(i, 0);
		}

		return res;
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	void backProjection(cv::Mat &m, cv::Mat &cradle, cv::Mat &mask, double angle, int s0, int s1, int shift, int flag, int noflag){
		double theta = angle * M_PI / 180;
		int maxbin = std::floor((s0 - 1) / 2 / sin(theta));
		cv::Mat res(s0, s1, CV_32F, cv::Scalar(0));

		int j = (int)(-maxbin + (m).rows / 2);
		for (int i = -maxbin; i <= maxbin - 1; i++){
			if ((m).at<float>(j, 0) > 0){
				cv::Mat up, down;
				std::vector<int> v = pointBackProjection(i, angle, s0, s1, up, down);
				int l = v[0];
				int r = v[1];
				double delta = (m).at<float>(j, 0) / (r - l) / 2;
				for (int k = l; k <= r; k++){
					mask.at<char>(shift + up.at<int>(k - l, 0) - s0 / 2, k) |= flag;
					mask.at<char>(shift + down.at<int>(k - l, 0) - s0 / 2, k) |= flag;
					(cradle).at<float>(shift + up.at<int>(k - l, 0) - s0 / 2, k) += delta;
					(cradle).at<float>(shift + down.at<int>(k - l, 0) - s0 / 2, k) += delta;
				}
			}
			j++;
		}
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	void backProjectionMask(std::vector<float> &m, cv::Mat &mask, double angle, int s0, int s1, int shift, int flag, int noflag){
		double theta = angle * M_PI / 180;
		int maxbin = std::floor((s0 - 1) / 2 / sin(theta));
		cv::Mat res(s0, s1, CV_32F, cv::Scalar(0));

		int j = (int)(-maxbin + (m).size() / 2) + 1;
		for (int i = -maxbin; i <= maxbin - 1; i++){
			if (m[j] > 0){
				cv::Mat up, down;
				std::vector<int> v = pointBackProjection(i, angle, s0, s1, up, down);
				int l = v[0];
				int r = v[1];
				for (int k = l; k <= r; k++){
					mask.at<char>(shift + up.at<int>(k - l, 0) - s0 / 2, k) |= flag;
					mask.at<char>(shift + down.at<int>(k - l, 0) - s0 / 2, k) |= flag;
				}
			}
			j++;
		}
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	std::vector<int> pointBackProjection(int b, double angle, int s0, int s1, cv::Mat &upline, cv::Mat &downline){
		std::vector<int> range(2);
		range[0] = 0;
		range[1] = s1 - 1;
		int L = s0 + 1;
		int M = (s1 + 1) / 2;
		double theta = angle*M_PI / 180;
		std::vector<int> res(2);


		if (std::abs(angle - 90) < 0.1){
			upline = cv::Mat(s1, 1, CV_32S, cv::Scalar(1));
			downline = cv::Mat(s1, 1, CV_32S, cv::Scalar(1));
			upline *= std::ceil(L / 2 - b);
			downline *= std::floor(L / 2 - b);
			res[0] = range[0];
			res[1] = range[1];
		}
		else{
			if (angle < 90){
				res[0] = std::max((int)std::ceil(tan(theta)*(.5 + 1 / 2 / sin(theta) + b / sin(theta) - L / 2) + M), range[0]);
				res[1] = std::min((int)std::floor(tan(theta)*(L / 2 - 1 / 2 / sin(theta) + b / sin(theta)) + M), range[1]);
			}
			else{
				res[1] = std::min((int)std::floor(tan(theta)*(.5 + 1 / 2 / sin(theta) + b / sin(theta) - L / 2) + M), range[1]);
				res[0] = std::max((int)std::ceil(tan(theta)*(L / 2 - 1 / 2 / sin(theta) + b / sin(theta)) + M), range[0]);
			}
			if (res[0] > M || res[1] < M){
				res[0] = res[1] = 0;
				upline = cv::Mat(0, 0, CV_32S);
				downline = cv::Mat(0, 0, CV_32S);
			}
			else{
				std::vector<float> x(res[1] - res[0] + 1, 0);
				upline = cv::Mat(x.size(), 1, CV_32S, cv::Scalar(0));
				downline = cv::Mat(x.size(), 1, CV_32S, cv::Scalar(0));

				for (int i = 0; i < x.size(); i++){
					(upline).at<int>(i, 0) = std::ceil(L / 2 - b / std::sin(theta) + (res[0] - M + i)*cotan(theta) - 1 / 2 / std::sin(theta)) - 1;
					(downline).at<int>(i, 0) = std::floor(L / 2 - b / std::sin(theta) + (res[0] - M + i)*cotan(theta) + 1 / 2 / std::sin(theta));
				}
			}
		}
		return res;
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	cv::Mat get_edgeshape(const cv::Mat &r, int side, double h, double l, int usecdf){
		cv::Mat edgeshape(r.rows, 1, CV_32F, cv::Scalar(0));
		if (side == 1){
			//Left
			int ind1 = r.rows - 1;
			while (r.at<float>(ind1, 0) <= h) ind1--;
			int ind2 = r.rows - 1;
			while (r.at<float>(ind2 - 1, 0) - r.at<float>(ind2, 0) >= 0) ind2--;
			int ind = std::min(ind1, ind2);

			if (r.at<float>(ind, 0) < r.at<float>(r.rows - 1, 0)){
				ind = r.rows - 1;
				while (r.at<float>(ind, 0) <= r.at<float>(r.rows - 1, 0)) ind--;
			}
			if (ind == r.rows - 1){
				ind--;
			}

			//Fill edgeshape
			for (int i = ind; i < r.rows; i++)
				edgeshape.at<float>(i, 0) = r.at<float>(i, 0);
			return edgeshape;
		}
		else{

			//Right
			int ind1 = 0;
			while (r.at<float>(ind1, 0) >= l) ind1++;
			int ind2 = 1;
			while (r.at<float>(ind2 - 1, 0) - r.at<float>(ind2, 0) >= 0) ind2++;
			int ind = std::max(ind1, ind2) - 1;
			if (ind < 1)
				ind = 1;

			//Fill edgeshape
			for (int i = 0; i < ind; i++)
				edgeshape.at<float>(i, 0) = r.at<float>(i, 0);
			return edgeshape;
		}
	}

	//Functon used for profiling edge-shape of cradle pieces based on the Radon transform.
	//For more details and comments, look into the Matlab code, function getEdges()
	cv::Mat getRadonforAngle(cv::Mat &d, cv::Mat &mask, double theta, int sx, int ex, int sy, int ey, int noflag){
		int W = (ey - sy), H = (ex - sx);
		int maxv = (int)(std::sqrt(W*W + H*H)) + 1;
		double center_x = sy + W / 2;
		double center_y = sx + H / 2;
		cv::Mat acc(2 * maxv, 1, CV_32F, cv::Scalar(0));

		//Normalize
		if (theta < 0)
			theta += 360;
		if (theta > 360)
			theta -= 360;

		//Calculate Radon transform for single angle
		for (int y = sx; y < sx + H; y++){
			for (int x = sy; x < sy + W; x++){
				if ((mask.at<char>(y, x) & noflag) == 0){
					int r = (x - center_x) *1.0 * cos(RAD(theta)) - (y - center_y) *1.0 * sin(RAD(theta));
					acc.at<float>((r + maxv), 0) = acc.at<float>((r + maxv), 0) + (d).at<float>(y, x);
				}
			}
		}

		return acc;
	}

	//Flip vertical array
	cv::Mat flipVertical(cv::Mat &in){
		cv::Mat res(in.rows, 1, CV_32F);
		//Flip R
		for (int i = 0; i < (in.rows + 1) / 2; i++){
			res.at<float>(i, 0) = in.at<float>(in.rows - i - 1, 0);
			res.at<float>(in.rows - i - 1, 0) = in.at<float>(i, 0);
		}
		return res;
	}

	//Get max of a matrix
	float max(cv::Mat &m){
		float maxv = m.at<float>(0, 0);
		for (int i = 0; i < m.rows; i++){
			for (int j = 0; j < m.cols; j++){
				if (m.at<float>(i, j) > maxv)
					maxv = m.at<float>(i, j);
			}
		}
		return maxv;
	}

	//Get min of a matrix
	float min(cv::Mat &m){
		float minv = m.at<float>(0, 0);
		for (int i = 0; i < m.rows; i++){
			for (int j = 0; j < m.cols; j++){
				if (m.at<float>(i, j) < minv)
					minv = m.at<float>(i, j);
			}
		}
		return minv;
	}

	//Get max of int matrix
	int maxi(cv::Mat &m){
		int maxv = m.at<int>(0, 0);
		for (int i = 0; i < m.rows; i++){
			for (int j = 0; j < m.cols; j++){
				if (m.at<int>(i, j) > maxv)
					maxv = m.at<int>(i, j);
			}
		}
		return maxv;
	}

	//Get min of int matrix
	int mini(cv::Mat &m){
		int minv = m.at<int>(0, 0);
		for (int i = 0; i < m.rows; i++){
			for (int j = 0; j < m.cols; j++){
				if (m.at<int>(i, j) < minv)
					minv = m.at<int>(i, j);
			}
		}
		return minv;
	}

	double cotan(double i){
		return(1 / tan(i));
	}

	//Simple inear regression, finding the solution to the minimization problem
	//   y_i = A + B*x_i + e_i with min(\sum_i (e_i)^2)
	//where res[0] = A and res[1] = B
	std::vector<float> linearFitting(std::vector<float> &x, std::vector<float> &y){
		float mx, my, mxy, mxx;
		mx = my = mxx = mxy = 0;

		int N = 0;

		for (int i = 0; i < x.size(); i++){
			if (x[i] != -1 && y[i] != -1){
				mx += x[i];
				my += y[i];
				N++;
			}
		}
		mx /= N;
		my /= N;

		for (int i = 0; i < x.size(); i++){
			if (x[i] != -1 && y[i] != -1){
				mxy += (x[i] - mx)*(y[i] - my);
				mxx += (x[i] - mx)*(x[i] - mx);
			}
		}
		std::vector<float> res(2);

		res[1] = mxy / mxx;
		res[0] = my - res[1] * mx;
		return res;
	}

	//Get the mean of the vector
	float getMean(std::vector<float> &x){
		float s = 0;
		for (int i = 0; i < x.size(); i++) if (x[i] != -1){
			s += x[i];
		}
		return s / x.size();
	}

	//Get the mean of the matrix
	float getMean(cv::Mat &im){
		float s = 0;
		for (int i = 0; i < im.rows; i++){
			for (int j = 0; j < im.cols; j++){
				s += im.at<float>(i, j);
			}
		}
		return s / im.rows / im.cols;
	}

	//Get the median of the matrix
	float getMedian(cv::Mat &im){
		std::vector<float> v(im.rows*im.cols);
		int cnt = 0;
		for (int i = 0; i < im.rows; i++){
			for (int j = 0; j < im.cols; j++){
				v[cnt] = im.at<float>(i, j);
				cnt++;
			}
		}
		return getMedian(v);
	}

	//Get the median of the vector
	float getMedian(std::vector<float> &v){
		int s = v.size();
		std::vector<float> mv(s);

		int ts = 0;
		for (int i = 0; i < s; i++){
			if (v[i] != -1){
				mv[ts] = v[i];
				ts++;
			}
		}
		if (ts == 0)
			return -1;

		mv.resize(ts);

		std::sort(mv.begin(), mv.end());
		s = ts;
		if (s == 0)
			return -1;
		if (s % 2 == 1){
			if (s == 1)
				return mv[0];
			else
				return (mv[(s / 2)] + mv[(s / 2) + 1]) / 2;
		}
		else
			return mv[(s / 2)];
	}

	//Get the variance of the matrix
	float getVariance(cv::Mat v){
		float m = getMean(v);
		float sum = 0;
		for (int i = 0; i < v.rows; i++){
			for (int j = 0; j < v.cols; j++){
				float tmp = v.at<float>(i, j) - m;
				sum += tmp * tmp;
			}
		}
		return sum / v.rows / v.cols;
	}
	
	//Get the median of the vector
	float getVariance(std::vector<float> &v){
		int s = v.size();
		std::vector<float> mv(s);

		int ts = 0;
		for (int i = 0; i < s; i++){
			if (v[i] != -1){
				mv[ts] = v[i];
				ts++;
			}
		}
		if (ts == 0)
			return -1;

		mv.resize(ts);
		std::sort(mv.begin(), mv.end());

		float mean = 0;
		int cnt = 0;
		for (int i = ts / 4; i < ts - ts / 4; i++){
			mean += mv[i];
			cnt++;
		}
		mean /= cnt;

		float var = 0;
		for (int i = ts / 4; i < ts - ts / 4; i++){
			var += (mv[i] - mean) * (mv[i] - mean);
		}

		return var / cnt;
	}

	//Save out MarkedSegments structure 'ms' to a file 'name'
	void writeMarkedSegmentsFile(std::string name, MarkedSegments ms){
		std::ofstream output(name.c_str());
		output << ms.pieces << std::endl;	//Nr pieces
		
		//pieceIDh
		output << ms.pieceIDh.size() << std::endl;	
		for (int i = 0; i < ms.pieceIDh.size(); i++){
			output << ms.pieceIDh[i].size() << std::endl;
			for (int j = 0; j < ms.pieceIDh[i].size(); j++)
				output << ms.pieceIDh[i][j] << " ";
			output << std::endl;
		}

		//pieceIDh
		output << ms.pieceIDv.size() << std::endl;
		for (int i = 0; i < ms.pieceIDv.size(); i++){
			output << ms.pieceIDv[i].size() << std::endl;
			for (int j = 0; j < ms.pieceIDv[i].size(); j++)
				output << ms.pieceIDv[i][j] << " ";
			output << std::endl;
		}

		//piece_middle
		output << ms.piece_middle.size() << std::endl;
		for (int i = 0; i < ms.piece_middle.size(); i++){
			output << ms.piece_middle[i].x << " " << ms.piece_middle[i].y << std::endl;
		}

		//write out segment mask file
		output << ms.piece_mask.rows << " " << ms.piece_mask.cols << std::endl;
		for (int i = 0; i < ms.piece_mask.rows; i++){
			int j = 0;
			while (j < ms.piece_mask.cols){
				int v = ms.piece_mask.at<ushort>(i, j);
				int ref = j;
				j++;
				while (j < ms.piece_mask.cols && ms.piece_mask.at<ushort>(i, j) == v){
					j++;
				}
				output << (j - ref) << " " << v << " ";
			}
			output << std::endl;
		}
		output.close();
	}

	//Read MarkedSegments structure 'ms' from file 'name'
	MarkedSegments readMarkedSegmentsFile(std::string name){
		MarkedSegments ms;

		std::string line;
		std::ifstream infile(name);
		int tmp1, tmp2;

		//Nr pieces
		infile >> ms.pieces;

		//pieceIDh
		infile >> tmp1;

		ms.pieceIDh = std::vector<std::vector<int>>(tmp1);
		for (int i = 0; i < tmp1; i++){
			infile >> tmp2;
			ms.pieceIDh[i] = std::vector<int>(tmp2);
			for (int j = 0; j < tmp2; j++){
				infile >> ms.pieceIDh[i][j];
			}
		}

		//pieceIDv
		infile >> tmp1;
		ms.pieceIDv = std::vector<std::vector<int>>(tmp1);
		for (int i = 0; i < tmp1; i++){
			infile >> tmp2;
			ms.pieceIDv[i] = std::vector<int>(tmp2);
			for (int j = 0; j < tmp2; j++){
				infile >> ms.pieceIDv[i][j];
			}
		}

		//piece_middle
		infile >> tmp1;
		ms.piece_middle = std::vector<cv::Point2i>(tmp1);
		for (int i = 0; i < tmp1; i++){
			infile >> ms.piece_middle[i].x >> ms.piece_middle[i].y;
		}

		//mask segment file
		infile >> tmp1 >> tmp2;
		ms.piece_mask = cv::Mat(tmp1, tmp2, CV_16U);
		int ref = 0;
		int i = 0;
		while (infile >> tmp1){
			infile >> tmp2;
			for (int j = 0; j < tmp1; j++){
				ms.piece_mask.at<ushort>(i, ref + j) = tmp2;
			}
			ref += tmp1;
			if (ref >= ms.piece_mask.cols){
				ref -= ms.piece_mask.cols;
				i++;
			}
		}
		infile.close();
		return ms;
	}

	/**
	 * Interface callback functions, used to give feedback on progress during execution
	 **/
	void setCallbacks(const Callbacks *callbacks) {
		s_callbacks = callbacks;
	}

	const Callbacks *callbacks() {
		return s_callbacks;
	}

	bool progress(int value, int total)
	{
		if (s_callbacks)
			return s_callbacks->progress(value, total);
		return true;
	}
}
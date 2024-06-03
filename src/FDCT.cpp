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
#include <platypus/FDCT.h>
#include <opencv2/opencv.hpp>
#include <vector>

/**
* Fast discrete curvelet tranform implementation following the Matlab implementation
* See reference Matlab code available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
*
* For a more detailed explanation on the functions in this file, check out the Matlab
* implementation.
**/

namespace FDCT{

	cv::Mat ifdct_wrapping(std::vector<std::vector<cv::Mat>> &C, int M, int N){
		cv::Mat res;

		int nbscales = C.size();
		int nbangles_coarse = C[1].size();
		std::vector<int> nbangles;
		nbangles.push_back(1);
		for (int i = nbscales; i >= 2; i--){
			nbangles.push_back(nbangles_coarse * std::exp2((ceil((nbscales - i)*1.0 / 2))));
		}
		int finest = 2;

		int N1 = M;
		int N2 = N;

		float M1 = N1 * 1.0 / 3;
		float M2 = N2 * 1.0 / 3;

		int bigN1 = 2 * floor(2 * M1) + 1;
		int bigN2 = 2 * floor(2 * M2) + 1;

		cv::Mat X_r(bigN1, bigN2, CV_32F, cv::Scalar(0));
		cv::Mat X_c(bigN1, bigN2, CV_32F, cv::Scalar(0));

		//Initialization: preparing the lowpass filter at finest scale
		int window_length_1 = floor(2 * M1) - floor(M1) - 1;
		if (N1 % 3 == 0)
			window_length_1--;
		int window_length_2 = floor(2 * M2) - floor(M2) - 1;
		if (N2 % 3 == 0)
			window_length_2--;

		std::vector<float> coord1(window_length_1 + 1);
		std::vector<float> coord2(window_length_2 + 1);

		for (int i = 0; i < coord1.size(); i++){
			coord1[i] = i * (1.0 / window_length_1);
			coord2[i] = i * (1.0 / window_length_2);
		}

		std::vector<float> wl_1, wr_1, wl_2, wr_2;
		fdct_wrapping_window(coord1, wl_1, wr_1);
		fdct_wrapping_window(coord2, wl_2, wr_2);

		//Put filter together
		int cntr = 0;
		std::vector<float> lowpass_1, lowpass_2;
		if (N1 % 3 == 0){
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * std::floor(M1) + 1 + wr_1.size() + 2);
			cntr = 1;
			lowpass_1[0] = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}
			lowpass_1[cntr] = 0;
		}
		else{
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * std::floor(M1) + 1 + wr_1.size());
			cntr = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}
		}

		if (N2 % 3 == 0){
			lowpass_2 = std::vector<float>(wl_2.size() + 2 * std::floor(M2) + 1 + wr_2.size() + 2);
			cntr = 1;
			lowpass_2[0] = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}
			lowpass_2[cntr] = 0;
		}
		else{
			lowpass_2 = std::vector<float>(wl_2.size() + 2 * std::floor(M2) + 1 + wr_2.size());
			cntr = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}
		}

		//Get lowpass filter
		cv::Mat lowpass(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0)), highpass;
		for (int i = 0; i < lowpass_1.size(); i++){
			for (int j = 0; j < lowpass_2.size(); j++){
				lowpass.at<float>(i, j) = lowpass_1[i] * lowpass_2[j];
			}
		}

		// Loop: pyramidal reconstruction
		int Xj_topleft_1 = 1;
		int Xj_topleft_2 = 1;

		for (int z = nbscales; z >= 2; z--){
			M1 /= 2;
			M2 /= 2;

			window_length_1 = floor(2 * M1) - floor(M1) - 1;
			window_length_2 = floor(2 * M2) - floor(M2) - 1;

			coord1 = std::vector<float>(window_length_1 + 1);
			coord2 = std::vector<float>(window_length_2 + 1);

			for (int i = 0; i < coord1.size(); i++){
				coord1[i] = i * (1.0 / window_length_1);
				coord2[i] = i * (1.0 / window_length_2);
			}

			fdct_wrapping_window(coord1, wl_1, wr_1);
			fdct_wrapping_window(coord2, wl_2, wr_2);

			//Put filter together
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * std::floor(M1) + 1 + wr_1.size());
			cntr = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}

			lowpass_2 = std::vector<float>(wl_2.size() + 2 * std::floor(M2) + 1 + wr_2.size());
			cntr = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}

			cv::Mat lowpass_next = cv::Mat(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0));
			highpass = cv::Mat(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0));
			for (int i = 0; i < lowpass_1.size(); i++){
				for (int j = 0; j < lowpass_2.size(); j++){
					lowpass_next.at<float>(i, j) = lowpass_1[i] * lowpass_2[j];
					highpass.at<float>(i, j) = std::sqrt(1 - lowpass_next.at<float>(i, j) * lowpass_next.at<float>(i, j));

				}
			}

			cv::Mat Xj_r(2 * floor(4 * M1) + 1, 2 * floor(4 * M2) + 1, CV_32F, cv::Scalar(0));
			cv::Mat Xj_c(2 * floor(4 * M1) + 1, 2 * floor(4 * M2) + 1, CV_32F, cv::Scalar(0));

			//Loop : angles
			int l = 0;
			int nbquadrants = 2;
			int nbangles_perquad = nbangles[z - 1] / 4;

			for (int q = 1; q <= nbquadrants; q++){
				float M_horiz = M2 * (q % 2) + M1 * ((q + 1) % 2);
				float M_vert = M1 * (q % 2) + M2 * ((q + 1) % 2);

				std::vector<float> wedge_ticks, wedge_endpoints, wedge_midpoints;
				float stepy = 1.0 / (2 * nbangles_perquad);
				for (float i = 0; i <= 0.5; i += stepy){
					wedge_ticks.push_back(round(i * 2 * floor(4 * M_horiz)) + 1);
				}
				for (float i = 0.5 - stepy; i >= 0; i -= stepy){
					wedge_ticks.push_back(2 * floor(4 * M_horiz) + 2 - round(i * 2 * floor(4 * M_horiz) + 1));
				}

				for (int i = 1; i < wedge_ticks.size() - 1; i += 2){
					wedge_endpoints.push_back(wedge_ticks[i]);
				}
				for (int i = 0; i < wedge_endpoints.size() - 1; i++){
					wedge_midpoints.push_back((wedge_endpoints[i] + wedge_endpoints[i + 1])*1.0 / 2);
				}

				//Left corner wedge
				l++;
				int first_wedge_endpoint_vert = round(2 * floor(4 * M_vert) / (2 * nbangles_perquad) + 1);
				int length_corner_wedge = floor(4 * M_vert) - floor(M_vert) + ceil(first_wedge_endpoint_vert * 1.0 / 4);
				cv::Mat XX, YY;

				meshgrid(1, 2 * floor(4 * M_horiz) + 1, 1, length_corner_wedge, XX, YY);

				int width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1;
				float slope_wedge = (floor(4 * M_horiz) + 1 - wedge_endpoints[0])*1.0 / floor(4 * M_vert);
				std::vector<int> left_line(length_corner_wedge + 1);
				for (int i = 0; i < left_line.size(); i++){
					left_line[i] = round(2 - wedge_endpoints[0] + slope_wedge * i);
				}

				// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
				cv::Mat wrapped_XX(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				cv::Mat wrapped_YY(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));

				int first_row = floor(4 * M_vert) + 2 - ceil((length_corner_wedge + 1)*1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_corner_wedge + 1) % 2);
				}
				int first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
				if (((q - 3 + 256) % 2) == (q - 3)){
					first_col += ((width_wedge + 1) % 2);
				}

				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + (value % width_wedge);
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 1 + abs(cols[i] - 1)));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;


					for (int i = 0; i < cols.size(); i++){
						wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, admissible_cols[i] - 1);
						wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, admissible_cols[i] - 1);
					}
				}

				float slope_wedge_right = (floor(4 * M_horiz) + 1 - wedge_midpoints[0]) * 1.0 / floor(4 * M_vert);
				cv::Mat mid_line_right = wedge_midpoints[0] + slope_wedge_right * (wrapped_YY - 1);
				cv::Mat coord_right(mid_line_right.rows, mid_line_right.cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < coord_right.rows; i++){
					for (int j = 0; j < coord_right.cols; j++){
						coord_right.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[1] - wedge_endpoints[0]) *
							(wrapped_XX.at<float>(i, j) - mid_line_right.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
					}
				}

				float C2, C1;

				C2 = 1.0 / (1.0 / (2 * (floor(4 * M_horiz)) *1.0 / (wedge_endpoints[0] - 1) - 1) + 1.0 / (2.0 * (floor(4 * M_vert)) * 1.0 / (first_wedge_endpoint_vert - 1) - 1));
				C1 = C2 / (2.0 * (floor(4 * M_vert)) *1.0 / (first_wedge_endpoint_vert - 1) - 1);

				//Do some funky checks
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						if (((wrapped_XX.at<float>(i, j) - 1)*1.0 / floor(4 * M_horiz) + (wrapped_YY.at<float>(i, j) - 1) * 1.0 / floor(4 * M_vert)) == 2){
							wrapped_XX.at<float>(i, j) += 1;
						}
					}
				}
				cv::Mat coord_corner(wrapped_XX.rows, wrapped_XX.cols, CV_32F, cv::Scalar(0));
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						coord_corner.at<float>(i, j) = C1 + C2 * ((wrapped_XX.at<float>(i, j) - 1) * 1.0 / (floor(4 * M_horiz)) -
							(wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert))) /
							(2 - ((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) + (wrapped_YY.at<float>(i, j) - 1) * 1.0 / (floor(4 * M_vert))));
					}
				}

				cv::Mat wl_left, wl_right, wr_left, wr_right;
				cv::Mat wrapped_data_r, wrapped_data_c;
				fdct_wrapping_window(coord_corner, wl_left, wr_left);
				fdct_wrapping_window(coord_right, wl_right, wr_right);

				cv::Mat x_r(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));
				cv::Mat x_c(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < x_r.rows; i++){
					for (int j = 0; j < x_r.cols; j++){
						x_r.at<float>(i, j) = C[z - 1][l - 1].at<float>(i, j);
						x_c.at<float>(i, j) = C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j);
					}
				}
				x_r = ifftshift(x_r);
				x_c = ifftshift(x_c);
				
				//Put it all together for forward transform
				cv::Mat xx(x_r.rows, x_r.cols, CV_32FC2, cv::Scalar(0));
				for (int i = 0; i < x_r.rows; i++){
					for (int j = 0; j < x_r.cols; j++){
						xx.at<cv::Point2f>(i, j).x = x_r.at<float>(i, j);	//Real part
						xx.at<cv::Point2f>(i, j).y = x_c.at<float>(i, j);	//Imaginary part
					}
				}

				cv::dft(xx, xx, cv::DFT_COMPLEX_OUTPUT);
				wrapped_data_r = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				wrapped_data_c = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				float corr = sqrt(2) * sqrt(xx.rows * xx.cols);
				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) = xx.at<cv::Point2f>(i, j).x / corr;	//Real part
						wrapped_data_c.at<float>(i, j) = xx.at<cv::Point2f>(i, j).y / corr;	//Imaginary part
					}
				}

				wrapped_data_r = fftshift(wrapped_data_r);
				wrapped_data_c = fftshift(wrapped_data_c);

				wrapped_data_r = rot90(wrapped_data_r, (q - 1));
				wrapped_data_c = rot90(wrapped_data_c, (q - 1));

				//Filtering
				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
					}
				}

				//Unwrapping data
				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + (value % width_wedge);
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 1 + abs(cols[i] - 1)));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;


					for (int i = 0; i < cols.size(); i++){
						Xj_r.at<float>(row - 1, admissible_cols[i] - 1) += wrapped_data_r.at<float>(new_row - 1, i);
						Xj_c.at<float>(row - 1, admissible_cols[i] - 1) += wrapped_data_c.at<float>(new_row - 1, i);
					}
				}
				//TextureRemoval::writeMatToFile("writes/picr_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_r);
				//TextureRemoval::writeMatToFile("writes/pici_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_c);

				//Regular wedges
				int length_wedge = floor(4 * M_vert) - floor(M_vert);
				first_row = floor(4 * M_vert) + 2 - ceil((length_wedge + 1) *1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_wedge + 1) % 2);
				}
				for (int subl = 2; subl <= (nbangles_perquad - 1); subl++){
					l++;
					width_wedge = wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1;
					slope_wedge = ((floor(4 * M_horiz) + 1) - wedge_endpoints[subl - 1]) * 1.0 / floor(4 * M_vert);

					left_line = std::vector<int>(length_wedge);
					for (int i = 0; i < left_line.size(); i++){
						left_line[i] = round(wedge_endpoints[subl - 2] + slope_wedge * i);
					}

					// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
					cv::Mat wrapped_data_r(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_data_c(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_XX(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_YY(length_wedge, width_wedge, CV_32F, cv::Scalar(0));

					first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
					if (((q - 3 + 256) % 2) == (q - 3)){
						first_col += ((width_wedge + 1) % 2);
					}

					for (int row = 1; row <= length_wedge; row++){
						std::vector<int> cols(width_wedge);
						for (int i = 0; i < width_wedge; i++){
							int value = i - (left_line[row - 1] - first_col);
							while (value < 0)
								value += width_wedge;
							cols[i] = left_line[row - 1] + (value % width_wedge);
						}
						int new_row = 1 + (row - first_row + length_wedge) % length_wedge;

						for (int i = 0; i < cols.size(); i++){
							wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, cols[i] - 1);
							wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, cols[i] - 1);
						}
					}

					float slope_wedge_left = (floor(4 * M_horiz) + 1 - wedge_midpoints[subl - 2]) * 1.0 / floor(4 * M_vert);
					cv::Mat mid_line_left = wedge_midpoints[subl - 2] + slope_wedge_left * (wrapped_YY - 1);
					cv::Mat coord_left(mid_line_left.rows, mid_line_left.cols, CV_32F, cv::Scalar(0));

					for (int i = 0; i < coord_left.rows; i++){
						for (int j = 0; j < coord_left.cols; j++){
							coord_left.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) *
								(wrapped_XX.at<float>(i, j) - mid_line_left.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
						}
					}

					float slope_wedge_right = (floor(4 * M_horiz) + 1 - wedge_midpoints[subl - 1]) * 1.0 / floor(4 * M_vert);
					cv::Mat mid_line_right = wedge_midpoints[subl - 1] + slope_wedge_right * (wrapped_YY - 1);
					cv::Mat coord_right(mid_line_right.rows, mid_line_right.cols, CV_32F, cv::Scalar(0));

					for (int i = 0; i < coord_right.rows; i++){
						for (int j = 0; j < coord_right.cols; j++){
							coord_right.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) *
								(wrapped_XX.at<float>(i, j) - mid_line_right.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
						}
					}

					cv::Mat wl_left, wl_right, wr_left, wr_right;
					fdct_wrapping_window(coord_left, wl_left, wr_left);
					fdct_wrapping_window(coord_right, wl_right, wr_right);

					cv::Mat x_r(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));
					cv::Mat x_c(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));

					for (int i = 0; i < x_r.rows; i++){
						for (int j = 0; j < x_r.cols; j++){
							x_r.at<float>(i, j) = C[z - 1][l - 1].at<float>(i, j);
							x_c.at<float>(i, j) = C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j);
						}
					}

					x_r = ifftshift(x_r);
					x_c = ifftshift(x_c);

					//Put it all together for forward transform
					cv::Mat xx(x_r.rows, x_r.cols, CV_32FC2, cv::Scalar(0));
					for (int i = 0; i < x_r.rows; i++){
						for (int j = 0; j < x_r.cols; j++){
							xx.at<cv::Point2f>(i, j).x = x_r.at<float>(i, j);	//Real part
							xx.at<cv::Point2f>(i, j).y = x_c.at<float>(i, j);	//Imaginary part
						}
					}

					cv::dft(xx, xx, cv::DFT_COMPLEX_OUTPUT);

					wrapped_data_r = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
					wrapped_data_c = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
					float corr = sqrt(2) * sqrt(xx.rows * xx.cols);
					for (int i = 0; i < wrapped_data_r.rows; i++){
						for (int j = 0; j < wrapped_data_r.cols; j++){
							wrapped_data_r.at<float>(i, j) = xx.at<cv::Point2f>(i, j).x / corr;	//Real part
							wrapped_data_c.at<float>(i, j) = xx.at<cv::Point2f>(i, j).y / corr;	//Imaginary part
						}
					}
					wrapped_data_r = fftshift(wrapped_data_r);
					wrapped_data_c = fftshift(wrapped_data_c);

					wrapped_data_r = rot90(wrapped_data_r, (q - 1));
					wrapped_data_c = rot90(wrapped_data_c, (q - 1));

					//Filtering
					for (int i = 0; i < wrapped_data_r.rows; i++){
						for (int j = 0; j < wrapped_data_r.cols; j++){
							wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
							wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						}
					}

					//Unwrapping data
					for (int row = 1; row <= length_wedge; row++){
						std::vector<int> cols(width_wedge);

						for (int i = 0; i < width_wedge; i++){
							int value = i - (left_line[row - 1] - first_col);
							while (value < 0)
								value += width_wedge;
							cols[i] = left_line[row - 1] + (value % width_wedge);
						}
						int new_row = 1 + (row - first_row + length_wedge) % length_wedge;


						for (int i = 0; i < cols.size(); i++){
							Xj_r.at<float>(row - 1, cols[i] - 1) += wrapped_data_r.at<float>(new_row - 1, i);
							Xj_c.at<float>(row - 1, cols[i] - 1) += wrapped_data_c.at<float>(new_row - 1, i);
						}
					}
					//TextureRemoval::writeMatToFile("writes/picr_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_r);
					//TextureRemoval::writeMatToFile("writes/pici_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_c);
				}

				//Right wedge
				l++;

				width_wedge = 4 * floor(4 * M_horiz) + 3 - wedge_endpoints[wedge_endpoints.size() - 1] - wedge_endpoints[wedge_endpoints.size() - 2];
				slope_wedge = (floor(4 * M_horiz) + 1 - wedge_endpoints[wedge_endpoints.size() - 1])*1.0 / floor(4 * M_vert);
				left_line = std::vector<int>(length_corner_wedge + 1);
				for (int i = 0; i < left_line.size(); i++){
					left_line[i] = round(wedge_endpoints[wedge_endpoints.size() - 2] + slope_wedge * i);
				}

				// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
				wrapped_XX = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				wrapped_YY = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				
				first_row = floor(4 * M_vert) + 2 - ceil((length_corner_wedge + 1)*1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_corner_wedge + 1) % 2);
				}
				first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
				if (((q - 3 + 256) % 2) == (q - 3)){
					first_col += ((width_wedge + 1) % 2);
				}

				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + value % width_wedge;
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 2 * floor(4 * M_horiz) + 1 - abs(cols[i] - (2 * floor(4 * M_horiz) + 1))));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;

					for (int i = 0; i < cols.size(); i++){
						wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, admissible_cols[i] - 1);
						wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, admissible_cols[i] - 1);
					}
				}

				float slope_wedge_left = (floor(4 * M_horiz) + 1 - wedge_midpoints[wedge_midpoints.size() - 1]) * 1.0 / floor(4 * M_vert);
				cv::Mat mid_line_left = wedge_midpoints[wedge_midpoints.size() - 1] + slope_wedge_left * (wrapped_YY - 1);
				cv::Mat coord_left(mid_line_left.rows, mid_line_left.cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < coord_left.rows; i++){
					for (int j = 0; j < coord_left.cols; j++){
						coord_left.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - wedge_endpoints[wedge_endpoints.size() - 2]) *
							(wrapped_XX.at<float>(i, j) - mid_line_left.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
					}
				}

				C2 = -1.0 / ((2.0 * (floor(4 * M_horiz)) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - 1) - 1) + 1.0 / (2.0 * (floor(4 * M_vert)) * 1.0 / (first_wedge_endpoint_vert - 1) - 1));
				C1 = -C2 * (2.0 * (floor(4 * M_horiz)) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - 1) - 1);

				//Do some funky checks
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						if (((wrapped_XX.at<float>(i, j) - 1)*1.0 / floor(4 * M_horiz) == (wrapped_YY.at<float>(i, j) - 1)*1.0 / floor(4 * M_vert))){
							wrapped_XX.at<float>(i, j) -= 1;
						}
					}
				}
				coord_corner = cv::Mat(wrapped_XX.rows, wrapped_XX.cols, CV_32F, cv::Scalar(0));
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						coord_corner.at<float>(i, j) = C1 + C2 * (2 - ((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) +
							(wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert)))) /
							((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) - (wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert)));
					}
				}

				fdct_wrapping_window(coord_left, wl_left, wr_left);
				fdct_wrapping_window(coord_corner, wl_right, wr_right);


				x_r = cv::Mat(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));
				x_c = cv::Mat(C[z - 1][l - 1].rows, C[z - 1][l - 1].cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < x_r.rows; i++){
					for (int j = 0; j < x_r.cols; j++){
						x_r.at<float>(i, j) = C[z - 1][l - 1].at<float>(i, j);
						x_c.at<float>(i, j) = C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j);
					}
				}
				x_r = ifftshift(x_r);
				x_c = ifftshift(x_c);

				//Put it all together for forward transform
				xx = cv::Mat(x_r.rows, x_r.cols, CV_32FC2, cv::Scalar(0));
				for (int i = 0; i < x_r.rows; i++){
					for (int j = 0; j < x_r.cols; j++){
						xx.at<cv::Point2f>(i, j).x = x_r.at<float>(i, j);	//Real part
						xx.at<cv::Point2f>(i, j).y = x_c.at<float>(i, j);	//Imaginary part
					}
				}

				cv::dft(xx, xx, cv::DFT_COMPLEX_OUTPUT);
				wrapped_data_r = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				wrapped_data_c = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				corr = sqrt(2) * sqrt(xx.rows * xx.cols);
				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) = xx.at<cv::Point2f>(i, j).x / corr;	//Real part
						wrapped_data_c.at<float>(i, j) = xx.at<cv::Point2f>(i, j).y / corr;	//Imaginary part
					}
				}
				wrapped_data_r = fftshift(wrapped_data_r);
				wrapped_data_c = fftshift(wrapped_data_c);

				wrapped_data_r = rot90(wrapped_data_r, (q - 1));
				wrapped_data_c = rot90(wrapped_data_c, (q - 1));

				//Filtering
				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
					}
				}

				//Unwrapping data
				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + value % width_wedge;
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 2 * floor(4 * M_horiz) + 1 - abs(cols[i] - (2 * floor(4 * M_horiz) + 1))));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;
					
					for (int i = 0; i < cols.size(); i++){
						Xj_r.at<float>(row - 1, admissible_cols[i] - 1) += wrapped_data_r.at<float>(new_row - 1, i);
						Xj_c.at<float>(row - 1, admissible_cols[i] - 1) += wrapped_data_c.at<float>(new_row - 1, i);
					}
				}
				//TextureRemoval::writeMatToFile("writes/picr_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_r);
				//TextureRemoval::writeMatToFile("writes/pici_" + std::to_string(z) + "_" + std::to_string(l) + ".txt", Xj_c);

				Xj_r = rot90(Xj_r, 1);
				Xj_c = rot90(Xj_c, 1);
			}

			for (int i = 0; i < lowpass.rows; i++){
				for (int j = 0; j < lowpass.cols; j++){
					Xj_r.at<float>(i, j) *= lowpass.at<float>(i, j);
					Xj_c.at<float>(i, j) *= lowpass.at<float>(i, j);
				}
			}
			
			for (int i = -floor(2 * M1) + floor(4 * M1) + 1; i <= floor(2 * M1) + floor(4 * M1) + 1; i++){
				for (int j = -floor(2 * M2) + floor(4 * M2) + 1; j <= floor(2 * M2) + floor(4 * M2) + 1; j++){
					Xj_r.at<float>(i - 1, j - 1) *= highpass.at<float>(i - (-floor(2 * M1) + floor(4 * M1) + 1), j - (-floor(2 * M2) + floor(4 * M2) + 1));
					Xj_c.at<float>(i - 1, j - 1) *= highpass.at<float>(i - (-floor(2 * M1) + floor(4 * M1) + 1), j - (-floor(2 * M2) + floor(4 * M2) + 1));
				}
			}

			for (int i = 0; i <= 2 * floor(4 * M1); i++){
				for (int j = 0; j <= 2 * floor(4 * M2); j++){
					X_r.at<float>(Xj_topleft_1 + i - 1, Xj_topleft_2 + j - 1) += Xj_r.at<float>(i, j);
					X_c.at<float>(Xj_topleft_1 + i - 1, Xj_topleft_2 + j - 1) += Xj_c.at<float>(i, j);
				}
			}

			// Preparing for loop reentry or exit
			Xj_topleft_1 = Xj_topleft_1 + floor(4 * M1) - floor(2 * M1);
			Xj_topleft_2 = Xj_topleft_2 + floor(4 * M2) - floor(2 * M2);

			lowpass = lowpass_next.clone();
		}

		cv::Mat Y_r, Y_c;
		Y_r = X_r.clone();
		Y_c = X_c.clone();
		X_r = rot90(X_r, 2);
		X_c = rot90(X_c, 2);
		X_r += Y_r;
		X_c -= Y_c;

		M1 = M1 / 2;
		M2 = M2 / 2;

		cv::Mat xx = ifftshift(C[0][0]);
		cv::dft(xx, xx, cv::DFT_COMPLEX_OUTPUT);

		cv::Mat Xj_r, Xj_c;

		Xj_r = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
		Xj_c = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
		float corr = sqrt(xx.rows * xx.cols);
		for (int i = 0; i < Xj_r.rows; i++){
			for (int j = 0; j < Xj_r.cols; j++){
				Xj_r.at<float>(i, j) = xx.at<cv::Point2f>(i, j).x / corr;	//Real part
				Xj_c.at<float>(i, j) = xx.at<cv::Point2f>(i, j).y / corr;	//Imaginary part
			}
		}

		Xj_r = fftshift(Xj_r);
		Xj_c = fftshift(Xj_c);
	
		for (int i = 0; i <= 2 * floor(4 * M1); i++){
			for (int j = 0; j <= 2 * floor(4 * M2); j++){
				X_r.at<float>(Xj_topleft_1 + i - 1, Xj_topleft_2 + j - 1) += Xj_r.at<float>(i, j) * lowpass.at<float>(i, j);
				X_c.at<float>(Xj_topleft_1 + i - 1, Xj_topleft_2 + j - 1) += Xj_c.at<float>(i, j) * lowpass.at<float>(i, j);
			}
		}

		M1 = N1 * 1.0 / 3;
		M2 = N2 * 1.0 / 3;

		//Folding back onto N1 - by - N2 matrix
		int shift_1 = floor(2 * M1) - floor(N1 *1.0 / 2);
		int shift_2 = floor(2 * M2) - floor(N2 *1.0 / 2);

		Y_r = X_r(cv::Range(0, X_r.rows), cv::Range(shift_2, N2 + shift_2)).clone();
		Y_c = X_c(cv::Range(0, X_r.rows), cv::Range(shift_2, N2 + shift_2)).clone();

		Y_r(cv::Range(0, Y_r.rows), cv::Range(N2 - shift_2, N2)) += X_r(cv::Range(0, X_r.rows), cv::Range(0, shift_2));
		Y_c(cv::Range(0, Y_r.rows), cv::Range(N2 - shift_2, N2)) += X_c(cv::Range(0, X_r.rows), cv::Range(0, shift_2));

		Y_r(cv::Range(0, Y_r.rows), cv::Range(0, shift_2)) += X_r(cv::Range(0, X_r.rows), cv::Range(N2 + shift_2, N2 + 2 * shift_2));
		Y_c(cv::Range(0, Y_r.rows), cv::Range(0, shift_2)) += X_c(cv::Range(0, X_r.rows), cv::Range(N2 + shift_2, N2 + 2 * shift_2));

		X_r = Y_r(cv::Range(shift_1, N1 + shift_1), cv::Range(0, Y_r.cols)).clone();
		X_c = Y_c(cv::Range(shift_1, N1 + shift_1), cv::Range(0, Y_r.cols)).clone();

		X_r(cv::Range(N1 - shift_1, N1), cv::Range(0, X_r.cols)) += Y_r(cv::Range(0, shift_1), cv::Range(0, Y_r.cols));
		X_c(cv::Range(N1 - shift_1, N1), cv::Range(0, X_r.cols)) += Y_c(cv::Range(0, shift_1), cv::Range(0, Y_r.cols));

		X_r(cv::Range(0, shift_1), cv::Range(0, X_r.cols)) += Y_r(cv::Range(N1 + shift_1, N1 + 2 * shift_1), cv::Range(0, Y_r.cols));
		X_c(cv::Range(0, shift_1), cv::Range(0, X_r.cols)) += Y_c(cv::Range(N1 + shift_1, N1 + 2 * shift_1), cv::Range(0, Y_r.cols));
		
		X_r = ifftshift(X_r);
		X_c = ifftshift(X_c);

		//Put it all together for inverse transform
		xx = cv::Mat(X_r.rows, X_r.cols, CV_32FC2);
		for (int i = 0; i < X_r.rows; i++){
			for (int j = 0; j < X_r.cols; j++){
				xx.at<cv::Point2f>(i, j).x = X_r.at<float>(i, j);	//Real part
				xx.at<cv::Point2f>(i, j).y = X_c.at<float>(i, j);	//Imaginary part
			}
		}
		cv::dft(xx, xx, cv::DFT_INVERSE | cv::DFT_SCALE);

		corr = sqrt(xx.rows * xx.cols);
		res = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
		for (int i = 0; i < xx.rows; i++){
			for (int j = 0; j < xx.cols; j++){
				res.at<float>(i, j) = xx.at<cv::Point2f>(i, j).x * corr;
			}
		}
		res = fftshift(res);

		return res;
	}

	std::vector<std::vector<cv::Mat>> fdct_wrapping(cv::Mat &in, int scale){
		std::vector<std::vector<cv::Mat>> C;

		cv::Mat dftout, cc, rc;
		cv::dft(ifftshift(in), dftout, cv::DFT_COMPLEX_OUTPUT);
		
		//Separate complex and real channel
		cc = cv::Mat(in.rows, in.cols, CV_32F, cv::Scalar(0));
		rc = cv::Mat(in.rows, in.cols, CV_32F, cv::Scalar(0));

		for (int i = 0; i < in.rows; i++){
			for (int j = 0; j < in.cols; j++){
				rc.at<float>(i, j) = dftout.at<float>(i, j * 2);
				cc.at<float>(i, j) = dftout.at<float>(i, j * 2 + 1);
			}
		}
		rc = fftshift(rc);
		cc = fftshift(cc);

		cc /= std::sqrt(in.rows*in.cols);
		rc /= std::sqrt(in.rows*in.cols);

		int N1 = in.rows, N2 = in.cols, nbangles_coarse = 16;

		std::vector<int> nbangles(scale);
		nbangles[0] = 1;
		for (int i = 0; i < scale-1; i++){
			nbangles[i+1] = nbangles_coarse * std::pow(2, ceil(i * 1.0 / 2));
		}

		//Initialize result data structure
		C = std::vector<std::vector<cv::Mat>>(scale);
		for (int i = 0; i < C.size(); i++){
			C[i] = std::vector<cv::Mat>(nbangles[i]);
		}

		//Loop: pyramidal scale decomposition
		float M1 = N1 * 1.0 / 3, M2 = N2 * 1.0 / 3;

		// Initialization: smooth periodic extension of high frequencies
		int bigN1 = 2 * std::floor(2 * M1) + 1;
		int bigN2 = 2 * std::floor(2 * M2) + 1;

		cv::Mat rc_e, cc_e;
		rc_e = cv::Mat(bigN1, bigN2, CV_32F, cv::Scalar(0));
		cc_e = cv::Mat(bigN1, bigN2, CV_32F, cv::Scalar(0));

		for (int i = 0; i < bigN1; i++){
			for (int j = 0; j < bigN2; j++){
				int x = ((int)(std::floor(N1 * 1.0 / 2) - std::floor(2 * M1) + i + N1)) % N1;
				int y = ((int)(std::floor(N2 * 1.0 / 2) - std::floor(2 * M2) + j + N2)) % N2;
				rc_e.at<float>(i, j) = rc.at<float>(x, y);
				cc_e.at<float>(i, j) = cc.at<float>(x, y);
			}
		}
		int mn1 = 0, mn2 = 0;
		if (N1 % 3 == 0) mn1 = 1;
		if (N2 % 3 == 0) mn2 = 1;

		int window_length_1 = std::floor(2 * M1) - std::floor(M1) - 1 - mn1;
		int window_length_2 = std::floor(2 * M2) - std::floor(M2) - 1 - mn2;

		std::vector<float> coord1(window_length_1 + 1);
		std::vector<float> coord2(window_length_2 + 1);

		for (int i = 0; i < coord1.size(); i++){
			coord1[i] = i * (1.0 / window_length_1);
			coord2[i] = i * (1.0 / window_length_2);
		}

		int cntr;
		std::vector<float> wl_1, wr_1, wl_2, wr_2;
		fdct_wrapping_window(coord1, wl_1, wr_1);
		fdct_wrapping_window(coord2, wl_2, wr_2);

		//Put filter together
		std::vector<float> lowpass_1, lowpass_2;
		if (N1 % 3 == 0){
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * std::floor(M1) + 1 + wr_1.size() + 2);
			cntr = 1;
			lowpass_1[0] = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}
			lowpass_1[cntr] = 0;
		}
		else{
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * std::floor(M1) + 1 + wr_1.size());
			cntr = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}
		}

		if (N2 % 3 == 0){
			lowpass_2 = std::vector<float>(wl_2.size() + 2 * std::floor(M2) + 1 + wr_2.size() + 2);
			cntr = 1;
			lowpass_2[0] = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}
			lowpass_2[cntr] = 0;
		}
		else{
			lowpass_2 = std::vector<float>(wl_2.size() + 2 * std::floor(M2) + 1 + wr_2.size());
			cntr = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * std::floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}
		}

		//Get lowpass filter
		cv::Mat lowpass(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0)), highpass;
		for (int i = 0; i < lowpass_1.size(); i++){
			for (int j = 0; j < lowpass_2.size(); j++){
				lowpass.at<float>(i, j) = lowpass_1[i] * lowpass_2[j];
			}
		}

		cv::Mat Xlow_r(rc_e.rows, rc_e.cols, CV_32F, cv::Scalar(0)), Xlow_c(cc_e.rows, cc_e.cols, CV_32F, cv::Scalar(0));
		cv::Mat Xhi_r, Xhi_c;
		for (int i = 0; i < rc_e.rows; i++){
			for (int j = 0; j < rc_e.cols; j++){
				Xlow_r.at<float>(i, j) = lowpass.at<float>(i, j) * rc_e.at<float>(i, j);
				Xlow_c.at<float>(i, j) = lowpass.at<float>(i, j) * cc_e.at<float>(i, j);
			}
		}

		for (int z = scale; z >= 2; z--){
			M1 /= 2;
			M2 /= 2;

			window_length_1 = floor(2 * M1) - floor(M1) - 1;
			window_length_2 = floor(2 * M2) - floor(M2) - 1;


			coord1 = std::vector<float>(window_length_1 + 1);
			coord2 = std::vector<float>(window_length_2 + 1);

			for (int i = 0; i < coord1.size(); i++){
				coord1[i] = i * (1.0 / window_length_1);
				coord2[i] = i * (1.0 / window_length_2);
			}

			fdct_wrapping_window(coord1, wl_1, wr_1);
			fdct_wrapping_window(coord2, wl_2, wr_2);

			//Put filter together
			lowpass_1 = std::vector<float>(wl_1.size() + 2 * floor(M1) + 1 + wr_1.size());
			cntr = 0;
			for (int i = 0; i < wl_1.size(); i++){
				lowpass_1[cntr] = wl_1[i];
				cntr++;
			}
			for (int i = 0; i < 2 * floor(M1) + 1; i++){
				lowpass_1[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_1.size(); i++){
				lowpass_1[cntr] = wr_1[i];
				cntr++;
			}

			lowpass_2 = std::vector<float>(wl_2.size() + 2 * floor(M2) + 1 + wr_2.size());
			cntr = 0;
			for (int i = 0; i < wl_2.size(); i++){
				lowpass_2[cntr] = wl_2[i];
				cntr++;
			}
			for (int i = 0; i < 2 * floor(M2) + 1; i++){
				lowpass_2[cntr] = 1;
				cntr++;
			}
			for (int i = 0; i < wr_2.size(); i++){
				lowpass_2[cntr] = wr_2[i];
				cntr++;
			}

			lowpass = cv::Mat(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0));
			highpass = cv::Mat(lowpass_1.size(), lowpass_2.size(), CV_32F, cv::Scalar(0));
			for (int i = 0; i < lowpass_1.size(); i++){
				for (int j = 0; j < lowpass_2.size(); j++){
					lowpass.at<float>(i, j) = lowpass_1[i] * lowpass_2[j];
					highpass.at<float>(i, j) = std::sqrt(1 - lowpass.at<float>(i, j) * lowpass.at<float>(i, j));

				}
			}

			Xhi_r = Xlow_r.clone();
			Xhi_c = Xlow_c.clone();

			cv::Mat Xlow_tmp_r(2 * floor(2 * M1) + 1, 2 * floor(2 * M2) + 1, CV_32F, cv::Scalar(0));
			cv::Mat Xlow_tmp_c(2 * floor(2 * M1) + 1, 2 * floor(2 * M2) + 1, CV_32F, cv::Scalar(0));
			for (int i = (-floor(2 * M1)); i <= floor(2 * M1); i++){
				for (int j = (-floor(2 * M2)); j <= floor(2 * M2); j++){
					int x = i + floor(4 * M1);
					int y = j + floor(4 * M2);
					Xlow_tmp_r.at<float>(i + floor(2 * M1), j + floor(2 * M2)) = Xlow_r.at<float>(x, y);
					Xlow_tmp_c.at<float>(i + floor(2 * M1), j + floor(2 * M2)) = Xlow_c.at<float>(x, y);
				}
			}
			Xlow_r = Xlow_tmp_r.clone();
			Xlow_c = Xlow_tmp_c.clone();

			//Xhi(Xlow_index_1, Xlow_index_2) = Xlow .* hipass;
			for (int i = (-floor(2 * M1)); i <= floor(2 * M1); i++){
				for (int j = (-floor(2 * M2)); j <= floor(2 * M2); j++){
					int x = i + floor(4 * M1);
					int y = j + floor(4 * M2);
					Xhi_r.at<float>(x, y) = Xlow_r.at<float>(i + floor(2 * M1), j + floor(2 * M2)) * highpass.at<float>(i + floor(2 * M1), j + floor(2 * M2));
					Xhi_c.at<float>(x, y) = Xlow_c.at<float>(i + floor(2 * M1), j + floor(2 * M2)) * highpass.at<float>(i + floor(2 * M1), j + floor(2 * M2));
				}
			}

			//Xlow = Xlow .* lowpass;
			for (int i = 0; i < Xlow_r.rows; i++){
				for (int j = 0; j < Xlow_r.cols; j++){
					Xlow_r.at<float>(i, j) *= lowpass.at<float>(i, j);
					Xlow_c.at<float>(i, j) *= lowpass.at<float>(i, j);
				}
			}

			int l = 0;
			int nbquadrants = 2;
			int nbangles_perquad = nbangles[z-1] / 4;

			for (int q = 1; q <= nbquadrants; q++){
				float M_horiz = M2 * (q % 2) + M1 * ((q + 1) % 2);
				float M_vert = M1 * (q % 2) + M2 * ((q + 1) % 2);

				std::vector<float> wedge_ticks, wedge_endpoints, wedge_midpoints;
				float stepy = 1.0 / (2 * nbangles_perquad);
				for (float i = 0; i <= 0.5; i += stepy){
					wedge_ticks.push_back(round(i * 2 * floor(4 * M_horiz)) + 1);
				}
				for (float i = 0.5 - stepy; i >= 0; i -= stepy){
					wedge_ticks.push_back(2 * floor(4 * M_horiz) + 2 - round(i * 2 * floor(4 * M_horiz) + 1));
				}

				for (int i = 1; i < wedge_ticks.size() - 1; i += 2){
					wedge_endpoints.push_back(wedge_ticks[i]);
				}
				for (int i = 0; i < wedge_endpoints.size() - 1; i++){
					wedge_midpoints.push_back((wedge_endpoints[i] + wedge_endpoints[i + 1])*1.0 / 2);
				}

				//Left corner wedge
				l++;

				int first_wedge_endpoint_vert = round(2 * floor(4 * M_vert) * 1.0 / (2 * nbangles_perquad) + 1);
				int length_corner_wedge = floor(4 * M_vert) - floor(M_vert) + ceil(first_wedge_endpoint_vert * 1.0 / 4);
				cv::Mat XX, YY;

				meshgrid(1, 2 * floor(4 * M_horiz) + 1, 1, length_corner_wedge, XX, YY);

				int width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1;
				float slope_wedge = (floor(4 * M_horiz) + 1 - wedge_endpoints[0])*1.0 / floor(4 * M_vert);
				std::vector<int> left_line(length_corner_wedge + 1);
				for (int i = 0; i < left_line.size(); i++){
					left_line[i] = round(2 - wedge_endpoints[0] + slope_wedge * i);
				}

				// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
				cv::Mat wrapped_data_r(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				cv::Mat wrapped_data_c(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				cv::Mat wrapped_XX(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				cv::Mat wrapped_YY(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));

				int first_row = floor(4 * M_vert) + 2 - ceil((length_corner_wedge + 1)*1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_corner_wedge + 1) % 2);
				}
				int first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
				if (((q - 3 + 256) % 2) == (q - 3)){
					first_col += ((width_wedge + 1) % 2);
				}

				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + (value % width_wedge);
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 1 + abs(cols[i] - 1)));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;


					for (int i = 0; i < cols.size(); i++){
						if (cols[i] > 0){
							wrapped_data_r.at<float>(new_row - 1, i) = Xhi_r.at<float>(row - 1, admissible_cols[i] - 1);
							wrapped_data_c.at<float>(new_row - 1, i) = Xhi_c.at<float>(row - 1, admissible_cols[i] - 1);
						}
						else{
							wrapped_data_r.at<float>(new_row - 1, i) = 0;
							wrapped_data_c.at<float>(new_row - 1, i) = 0;
						}
						wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, admissible_cols[i] - 1);
						wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, admissible_cols[i] - 1);
					}
				}

				float slope_wedge_right = (floor(4 * M_horiz) + 1 - wedge_midpoints[0]) * 1.0 / floor(4 * M_vert);
				cv::Mat mid_line_right = wedge_midpoints[0] + slope_wedge_right * (wrapped_YY - 1);
				cv::Mat coord_right(mid_line_right.rows, mid_line_right.cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < coord_right.rows; i++){
					for (int j = 0; j < coord_right.cols; j++){
						coord_right.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[1] - wedge_endpoints[0]) *
							(wrapped_XX.at<float>(i, j) - mid_line_right.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
					}
				}

				float C2, C1;

				C2 = 1.0 / (1.0 / (2 * (floor(4 * M_horiz)) *1.0 / (wedge_endpoints[0] - 1) - 1) + 1.0 / (2.0 * (floor(4 * M_vert)) * 1.0 / (first_wedge_endpoint_vert - 1) - 1));
				C1 = C2 / (2.0 * (floor(4 * M_vert)) *1.0 / (first_wedge_endpoint_vert - 1) - 1);

				//Do some funky checks
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						if (((wrapped_XX.at<float>(i, j) - 1)*1.0 / floor(4 * M_horiz) + (wrapped_YY.at<float>(i, j) - 1) * 1.0 / floor(4 * M_vert)) == 2){
							wrapped_XX.at<float>(i, j) += 1;
						}
					}
				}
				cv::Mat coord_corner(wrapped_XX.rows, wrapped_XX.cols, CV_32F, cv::Scalar(0));
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						coord_corner.at<float>(i, j) = C1 + C2 * ((wrapped_XX.at<float>(i, j) - 1) * 1.0 / (floor(4 * M_horiz)) - 
							(wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert))) /
							(2 - ((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) + (wrapped_YY.at<float>(i, j) - 1) * 1.0 / (floor(4 * M_vert))));
					}
				}

				cv::Mat wl_left, wl_right, wr_left, wr_right;
				fdct_wrapping_window(coord_corner, wl_left, wr_left);
				fdct_wrapping_window(coord_right, wl_right, wr_right);

				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
					}
				}

				wrapped_data_r = rot90(wrapped_data_r, -(q - 1));
				wrapped_data_c = rot90(wrapped_data_c, -(q - 1));

				wrapped_data_r = ifftshift(wrapped_data_r);
				wrapped_data_c = ifftshift(wrapped_data_c);

				//Put it all together for inverse transform
				cv::Mat xx(wrapped_data_r.rows, wrapped_data_r.cols, CV_32FC2, cv::Scalar(0));

				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						xx.at<cv::Point2f>(i, j).x = wrapped_data_r.at<float>(i, j);	//Real part
						xx.at<cv::Point2f>(i, j).y = wrapped_data_c.at<float>(i, j);	//Imaginary part
					}
				}
				cv::dft(xx, xx, cv::DFT_INVERSE | cv::DFT_SCALE);
				
				//x = fftshift(ifft2(ifftshift(wrapped_data)))*sqrt(prod(size(wrapped_data)));
				//Save out result to C
				C[z - 1][l - 1] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				C[z - 1][l - 1 + nbangles[z - 1] / 2] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				float corrf = std::sqrt(2) * sqrt(xx.rows * xx.cols);
				for (int i = 0; i < xx.rows; i++){
					for (int j = 0; j < xx.cols; j++){
						C[z - 1][l - 1].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).x;
						C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).y;
					}
				}

				C[z - 1][l - 1] = fftshift(C[z - 1][l - 1]);
				C[z - 1][l - 1 + nbangles[z - 1] / 2] = fftshift(C[z - 1][l - 1 + nbangles[z - 1] / 2]);

				//Regular wedges
				int length_wedge = floor(4 * M_vert) - floor(M_vert);
				first_row = floor(4 * M_vert) + 2 - ceil((length_wedge + 1) *1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_wedge + 1) % 2);
				}
				for (int subl = 2; subl <= (nbangles_perquad - 1); subl++){
					l++;
					width_wedge = wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1;
					slope_wedge = ((floor(4 * M_horiz) + 1) - wedge_endpoints[subl - 1]) * 1.0 / floor(4 * M_vert);

					left_line = std::vector<int>(length_wedge);
					for (int i = 0; i < left_line.size(); i++){
							left_line[i] = round(wedge_endpoints[subl - 2] + slope_wedge * i);
					}

					// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
					cv::Mat wrapped_data_r(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_data_c(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_XX(length_wedge, width_wedge, CV_32F, cv::Scalar(0));
					cv::Mat wrapped_YY(length_wedge, width_wedge, CV_32F, cv::Scalar(0));

					first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
					if (((q - 3 + 256) % 2) == (q - 3)){
						first_col += ((width_wedge + 1) % 2);
					}

					for (int row = 1; row <= length_wedge; row++){
						std::vector<int> cols(width_wedge);
						for (int i = 0; i < width_wedge; i++){
							int value = i - (left_line[row - 1] - first_col);
							while (value < 0)
								value += width_wedge;
							cols[i] = left_line[row - 1] + (value % width_wedge);
						}
						int new_row = 1 + (row - first_row + length_wedge) % length_wedge;

						for (int i = 0; i < cols.size(); i++){
							if (cols[i] > 0){
								wrapped_data_r.at<float>(new_row - 1, i) = Xhi_r.at<float>(row - 1, cols[i] - 1);
								wrapped_data_c.at<float>(new_row - 1, i) = Xhi_c.at<float>(row - 1, cols[i] - 1);
							}
							else{
								wrapped_data_r.at<float>(new_row - 1, i) = 0;
								wrapped_data_c.at<float>(new_row - 1, i) = 0;
							}
							wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, cols[i] - 1);
							wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, cols[i] - 1);
						}
					}

					float slope_wedge_left = (floor(4 * M_horiz) + 1 - wedge_midpoints[subl - 2]) * 1.0 / floor(4 * M_vert);
					cv::Mat mid_line_left = wedge_midpoints[subl - 2] + slope_wedge_left * (wrapped_YY - 1);
					cv::Mat coord_left(mid_line_left.rows, mid_line_left.cols, CV_32F, cv::Scalar(0));

					for (int i = 0; i < coord_left.rows; i++){
						for (int j = 0; j < coord_left.cols; j++){
							coord_left.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) *
								(wrapped_XX.at<float>(i, j) - mid_line_left.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
						}
					}

					float slope_wedge_right = (floor(4 * M_horiz) + 1 - wedge_midpoints[subl - 1]) * 1.0 / floor(4 * M_vert);
					cv::Mat mid_line_right = wedge_midpoints[subl - 1] + slope_wedge_right * (wrapped_YY - 1);
					cv::Mat coord_right(mid_line_right.rows, mid_line_right.cols, CV_32F, cv::Scalar(0));

					for (int i = 0; i < coord_right.rows; i++){
						for (int j = 0; j < coord_right.cols; j++){
							coord_right.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) *
								(wrapped_XX.at<float>(i, j) - mid_line_right.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
						}
					}

					cv::Mat wl_left, wl_right, wr_left, wr_right;
					fdct_wrapping_window(coord_left, wl_left, wr_left);
					fdct_wrapping_window(coord_right, wl_right, wr_right);

					for (int i = 0; i < wrapped_data_r.rows; i++){
						for (int j = 0; j < wrapped_data_r.cols; j++){
							wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
							wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						}
					}

					wrapped_data_r = rot90(wrapped_data_r, -(q - 1));
					wrapped_data_c = rot90(wrapped_data_c, -(q - 1));

					wrapped_data_r = ifftshift(wrapped_data_r);
					wrapped_data_c = ifftshift(wrapped_data_c);

					//Put it all together for inverse transform
					cv::Mat xx(wrapped_data_r.rows, wrapped_data_r.cols, CV_32FC2, cv::Scalar(0));

					for (int i = 0; i < wrapped_data_r.rows; i++){
						for (int j = 0; j < wrapped_data_r.cols; j++){
							xx.at<cv::Point2f>(i, j).x = wrapped_data_r.at<float>(i, j);	//Real part
							xx.at<cv::Point2f>(i, j).y = wrapped_data_c.at<float>(i, j);	//Imaginary part
						}
					}
					cv::dft(xx, xx, cv::DFT_INVERSE | cv::DFT_SCALE);

					//x = fftshift(ifft2(ifftshift(wrapped_data)))*sqrt(prod(size(wrapped_data)));
					//Save out result to C
					C[z - 1][l - 1] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
					C[z - 1][l - 1 + nbangles[z - 1] / 2] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
					float corrf = std::sqrt(2) * sqrt(xx.rows * xx.cols);
					for (int i = 0; i < xx.rows; i++){
						for (int j = 0; j < xx.cols; j++){
							C[z - 1][l - 1].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).x;
							C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).y;
						}
					}

					C[z - 1][l - 1] = fftshift(C[z - 1][l - 1]);
					C[z - 1][l - 1 + nbangles[z - 1] / 2] = fftshift(C[z - 1][l - 1 + nbangles[z - 1] / 2]);
				}

				//Right wedge
				l++;

				width_wedge = 4 * floor(4 * M_horiz) + 3 - wedge_endpoints[wedge_endpoints.size() - 1] - wedge_endpoints[wedge_endpoints.size() - 2];
				slope_wedge = (floor(4 * M_horiz) + 1 - wedge_endpoints[wedge_endpoints.size() - 1])*1.0 / floor(4 * M_vert);
				left_line = std::vector<int>(length_corner_wedge + 1);
				for (int i = 0; i < left_line.size(); i++){
					left_line[i] = round(wedge_endpoints[wedge_endpoints.size() - 2] + slope_wedge * i);
				}

				// [wrapped_data, wrapped_XX, wrapped_YY] = deal(zeros(length_corner_wedge,width_wedge));
				wrapped_data_r = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				wrapped_data_c = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				wrapped_XX = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));
				wrapped_YY = cv::Mat(length_corner_wedge, width_wedge, CV_32F, cv::Scalar(0));

				first_row = floor(4 * M_vert) + 2 - ceil((length_corner_wedge + 1)*1.0 / 2);
				if (((q - 2 + 256) % 2) == (q - 2)){
					first_row += ((length_corner_wedge + 1) % 2);
				}
				first_col = floor(4 * M_horiz) + 2 - ceil((width_wedge + 1)*1.0 / 2);
				if (((q - 3 + 256) % 2) == (q - 3)){
					first_col += ((width_wedge + 1) % 2);
				}

				for (int row = 1; row <= length_corner_wedge; row++){
					std::vector<int> cols(width_wedge);
					std::vector<int> admissible_cols(width_wedge);

					for (int i = 0; i < width_wedge; i++){
						int value = i - (left_line[row - 1] - first_col);
						while (value < 0)
							value += width_wedge;
						cols[i] = left_line[row - 1] + value % width_wedge;
						admissible_cols[i] = round(1.0 / 2 * (cols[i] + 2 * floor(4 * M_horiz) + 1 - abs(cols[i] - (2 * floor(4 * M_horiz) + 1))));
					}
					int new_row = 1 + (row - first_row + length_corner_wedge) % length_corner_wedge;

					for (int i = 0; i < cols.size(); i++){
						if (cols[i] <= 2 * floor(4 * M_horiz) + 1){
							wrapped_data_r.at<float>(new_row - 1, i) = Xhi_r.at<float>(row - 1, admissible_cols[i] - 1);
							wrapped_data_c.at<float>(new_row - 1, i) = Xhi_c.at<float>(row - 1, admissible_cols[i] - 1);
						}
						else{
							wrapped_data_r.at<float>(new_row - 1, i) = 0;
							wrapped_data_c.at<float>(new_row - 1, i) = 0;
						}
						wrapped_XX.at<float>(new_row - 1, i) = XX.at<float>(row - 1, admissible_cols[i] - 1);
						wrapped_YY.at<float>(new_row - 1, i) = YY.at<float>(row - 1, admissible_cols[i] - 1);
					}
				}

				float slope_wedge_left = (floor(4 * M_horiz) + 1 - wedge_midpoints[wedge_midpoints.size() - 1]) * 1.0 / floor(4 * M_vert);
				cv::Mat mid_line_left = wedge_midpoints[wedge_midpoints.size() - 1] + slope_wedge_left * (wrapped_YY - 1);
				cv::Mat coord_left(mid_line_left.rows, mid_line_left.cols, CV_32F, cv::Scalar(0));

				for (int i = 0; i < coord_left.rows; i++){
					for (int j = 0; j < coord_left.cols; j++){
						coord_left.at<float>(i, j) = 0.5 + floor(4 * M_vert) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - wedge_endpoints[wedge_endpoints.size() - 2]) *
							(wrapped_XX.at<float>(i, j) - mid_line_left.at<float>(i, j)) * 1.0 / (floor(4 * M_vert) + 1 - wrapped_YY.at<float>(i, j));
					}
				}

				C2 = -1.0 / ((2.0 * (floor(4 * M_horiz)) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - 1) - 1) + 1.0 / (2.0 * (floor(4 * M_vert)) * 1.0 / (first_wedge_endpoint_vert - 1) - 1));
				C1 = -C2 * (2.0 * (floor(4 * M_horiz)) * 1.0 / (wedge_endpoints[wedge_endpoints.size() - 1] - 1) - 1);

				//Do some funky checks
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						if (((wrapped_XX.at<float>(i, j) - 1)*1.0 / floor(4 * M_horiz) == (wrapped_YY.at<float>(i, j) - 1)*1.0 / floor(4 * M_vert))){
							wrapped_XX.at<float>(i, j) -= 1;
						}
					}
				}
				coord_corner = cv::Mat(wrapped_XX.rows, wrapped_XX.cols, CV_32F, cv::Scalar(0));
				for (int i = 0; i < wrapped_XX.rows; i++){
					for (int j = 0; j < wrapped_XX.cols; j++){
						coord_corner.at<float>(i, j) = C1 + C2 * (2 - ((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) +
							(wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert)))) /
							((wrapped_XX.at<float>(i, j) - 1)*1.0 / (floor(4 * M_horiz)) - (wrapped_YY.at<float>(i, j) - 1)*1.0 / (floor(4 * M_vert)));
					}
				}

				fdct_wrapping_window(coord_left, wl_left, wr_left);
				fdct_wrapping_window(coord_corner, wl_right, wr_right);

				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						wrapped_data_r.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
						wrapped_data_c.at<float>(i, j) *= wl_left.at<float>(i, j) * wr_right.at<float>(i, j);
					}
				}

				wrapped_data_r = rot90(wrapped_data_r, -(q - 1));
				wrapped_data_c = rot90(wrapped_data_c, -(q - 1));

				wrapped_data_r = ifftshift(wrapped_data_r);
				wrapped_data_c = ifftshift(wrapped_data_c);

				//Put it all together for inverse transform
				xx = cv::Mat(wrapped_data_r.rows, wrapped_data_r.cols, CV_32FC2, cv::Scalar(0));

				for (int i = 0; i < wrapped_data_r.rows; i++){
					for (int j = 0; j < wrapped_data_r.cols; j++){
						xx.at<cv::Point2f>(i, j).x = wrapped_data_r.at<float>(i, j);	//Real part
						xx.at<cv::Point2f>(i, j).y = wrapped_data_c.at<float>(i, j);	//Imaginary part
					}
				}
				cv::dft(xx, xx, cv::DFT_INVERSE | cv::DFT_SCALE);

				//x = fftshift(ifft2(ifftshift(wrapped_data)))*sqrt(prod(size(wrapped_data)));
				//Save out result to C
				C[z - 1][l - 1] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				C[z - 1][l - 1 + nbangles[z - 1] / 2] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
				corrf = std::sqrt(2) * sqrt(xx.rows * xx.cols);
				for (int i = 0; i < xx.rows; i++){
					for (int j = 0; j < xx.cols; j++){
						C[z - 1][l - 1].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).x;
						C[z - 1][l - 1 + nbangles[z - 1] / 2].at<float>(i, j) = corrf * xx.at<cv::Point2f>(i, j).y;
					}
				}

				C[z - 1][l - 1] = fftshift(C[z - 1][l - 1]);
				C[z - 1][l - 1 + nbangles[z - 1] / 2] = fftshift(C[z - 1][l - 1 + nbangles[z - 1] / 2]);

				if (q < nbquadrants){
					Xhi_r = rot90(Xhi_r, 1);
					Xhi_c = rot90(Xhi_c, 1);
				}
			}
		}

		// Coarsest wavelet level
		Xlow_r = ifftshift(Xlow_r);
		Xlow_c = ifftshift(Xlow_c);

		//Put it all together for inverse transform
		cv::Mat xx = cv::Mat(Xlow_r.rows, Xlow_r.cols, CV_32FC2, cv::Scalar(0));
		for (int i = 0; i < Xlow_r.rows; i++){
			for (int j = 0; j < Xlow_c.cols; j++){
				xx.at<cv::Point2f>(i, j).x = Xlow_r.at<float>(i, j);	//Real part
				xx.at<cv::Point2f>(i, j).y = Xlow_c.at<float>(i, j);	//Imaginary part
			}
		}

		cv::idft(xx, xx, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
		xx = fftshift(xx);

		//Save out result to C
		C[0][0] = cv::Mat(xx.rows, xx.cols, CV_32F, cv::Scalar(0));
		float corrf = sqrt(xx.rows * xx.cols);
		for (int i = 0; i < xx.rows; i++){
			for (int j = 0; j < xx.cols; j++){
				C[0][0].at<float>(i, j) = corrf * xx.at<float>(i, j);
			}
		}

		return C;
	}

	cv::Mat rot90(cv::Mat &x, int k){
		k += 256;

		if (k % 4 == 0)
			return x;

		if (k % 4 == 1){
			cv::Mat y(x.cols, x.rows, CV_32F, cv::Scalar(0));
			for (int i = 0; i < x.rows; i++){
				for (int j = 0; j < x.cols; j++){
					y.at<float>(x.cols - j - 1, i) = x.at<float>(i, j);
				}
			}
			return y;
		}
		if (k % 4 == 2){
			cv::Mat y(x.rows, x.cols, CV_32F, cv::Scalar(0));
			for (int i = 0; i < x.rows; i++){
				for (int j = 0; j < x.cols; j++){
					y.at<float>(x.rows - i - 1, x.cols - j - 1) = x.at<float>(i, j);
				}
			}
			return y;
		}
		if (k % 4 == 3){
			cv::Mat y(x.cols, x.rows, CV_32F, cv::Scalar(0));
			for (int i = 0; i < x.rows; i++){
				for (int j = 0; j < x.cols; j++){
					y.at<float>(j, x.rows - i - 1) = x.at<float>(i, j);
				}
			}
			return y;
		}
		return x;
	}

	void meshgrid(int sy, int ey, int sx, int ex, cv::Mat &xx, cv::Mat &yy){
		xx = cv::Mat(ex - sx + 1, ey - sy + 1, CV_32F, cv::Scalar(0));
		yy = cv::Mat(ex - sx + 1, ey - sy + 1, CV_32F, cv::Scalar(0));

		for (int i = 0; i < ex - sx + 1; i++){
			for (int j = 0; j < ey - sy + 1; j++){
				xx.at<float>(i, j) = j + sy;
				yy.at<float>(i, j) = i + sx;
			}
		}
	}

	void fdct_wrapping_window(cv::Mat &x, cv::Mat &wl, cv::Mat &wr){

		wl = cv::Mat(x.rows, x.cols, CV_32F, cv::Scalar(0));
		wr = cv::Mat(x.rows, x.cols, CV_32F, cv::Scalar(0));
		cv::Mat norm(x.rows, x.cols, CV_32F, cv::Scalar(0));

		for (int i = 0; i < x.rows; i++){
			for (int j = 0; j < x.cols; j++){
				if (x.at<float>(i, j) > 1){
					wr.at<float>(i, j) = 0;
					wl.at<float>(i, j) = 1;
				}
				else if (x.at<float>(i, j) < 0){
					wr.at<float>(i, j) = 1;
					wl.at<float>(i, j) = 0;
				}
				else{
					wr.at<float>(i, j) = std::exp(1 - 1.0 / (1 - std::exp(1 - 1.0 / x.at<float>(i, j))));
					wl.at<float>(i, j) = std::exp(1 - 1.0 / (1 - std::exp(1 - 1.0 / (1 - x.at<float>(i, j)))));
				}

				norm.at<float>(i, j) = std::sqrt(wr.at<float>(i, j) * wr.at<float>(i, j) + wl.at<float>(i, j) * wl.at<float>(i, j));

				wr.at<float>(i, j) /= norm.at<float>(i, j);
				wl.at<float>(i, j) /= norm.at<float>(i, j);
			}
		}
	}

	void fdct_wrapping_window(std::vector<float> &x, std::vector<float> &wl, std::vector<float> &wr){

		wl = std::vector<float>(x.size());
		wr = std::vector<float>(x.size());
		std::vector<float> norm(x.size());

		for (int i = 0; i < x.size(); i++){
			wr[i] = std::exp(1 - 1.0 / (1 - std::exp(1 - 1.0 / x[i])));
			wl[i] = std::exp(1 - 1.0 / (1 - std::exp(1 - 1.0 / (1 - x[i]))));

			norm[i] = std::sqrt(wr[i] * wr[i] + wl[i] * wl[i]);

			wr[i] /= norm[i];
			wl[i] /= norm[i];
		}
	}

	cv::Mat fftshift(cv::Mat &x){
		//Inverse FFT shift
		// FFTSHIFT(X) swaps the first and third
		// quadrants and the second and fourth quadrants.For N - D
		// arrays, FFTSHIFT(X) swaps "half-spaces" of X along each
		// dimension.

		return circshift(x, std::floor(x.rows / 2), std::floor(x.cols / 2));
	}

	cv::Mat ifftshift(cv::Mat &x){
		//Inverse FFT shift
		// IFFTSHIFT(X) swaps the first and third
		// quadrants and the second and fourth quadrants.For N - D
		// arrays, IFFTSHIFT(X) swaps "half-spaces" of X along each
		// dimension.

		return circshift(x, std::ceil(x.rows * 1.0 / 2), std::ceil(x.cols * 1.0 / 2));
	}

	cv::Mat circshift(cv::Mat &M, int x, int y){
		// Circularly shifts the values in matrix M
		// by x along the first dimension and by y along the second one

		if ((x == 0) && (y == 0))
			return M;

		int X = M.rows;
		int Y = M.cols;

		cv::Mat res(X, Y, CV_32F, cv::Scalar(0));
		for (int i = 0; i < M.rows; i++){
			for (int j = 0; j < M.cols; j++){
				res.at<float>((i + x + X) % X, (j + y + Y) % Y) = M.at<float>(i, j);
			}
		}
		return res;
	}
}

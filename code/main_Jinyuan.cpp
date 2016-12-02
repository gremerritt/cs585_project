//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
//#include <unistd.h>

#define IMAGE_NAME "house_2.jpg"
#define DEBUG 0

using namespace cv;
using namespace std;

void ShrinkImage(Mat &src, Mat &output, double scale, char direction);
void EnlargeImage(Mat &src, Mat &output, double scale, char direction);
void process(Mat &src);
void getEnergy(Mat &src, Mat &energy);
void getSeams(Mat &src, Mat &energy, vector<vector<Point> > &seams);
void getSeamMap(Mat &src, Mat &energy, Mat &seam_map, char direction);
void getSeamFromMap(Mat &seam_map, vector<Point> &seam, char direction);
void getSeamFromMap2(Mat &seam_map, vector<vector<Point>> &seams, char direction, int num_seams);
int getMinVal(vector<int> &vals);
int getMinIndex(vector<int> &vals);
int main()
{
    Mat img = imread(IMAGE_NAME, IMREAD_COLOR);
    resize(img, img, Size(960,540));
    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    namedWindow("TMP", CV_WINDOW_AUTOSIZE);
    namedWindow("Final output", CV_WINDOW_AUTOSIZE);
	imshow("Image", img);
	
	Mat tmp, output;
    double x_scale = 0.9, y_scale = 1.1;
	if(x_scale > 1) EnlargeImage(img, tmp, x_scale, 1);
	else ShrinkImage(img, tmp, x_scale, 1);

	if(y_scale > 1) EnlargeImage(tmp, output, y_scale, 0);
	else ShrinkImage(tmp, output, y_scale, 0);
	imwrite("house_2_output.jpg", output);

    cv::waitKey(0);
    return 0;
}

void ShrinkImage(Mat &src, Mat &output, double scale, char direction) {
	if (scale <= 0 || scale > 1){
		std::cout << "Scale must range between 0 and 1." << endl;
		return;
	}

	Mat src_bw, energy;
	cvtColor(src, src_bw, CV_BGR2GRAY);
	output = src.clone();
	vector<Point> seam;
	
	int num_seams;
	if(direction == 0){
		num_seams = (int)((double)src_bw.cols * (1 - scale));
		seam = vector<Point>(src_bw.rows, Point(0,0));
	}
	else {
		num_seams = (int)((double)src_bw.rows * (1 - scale));
		seam = vector<Point>(src_bw.cols, Point(0,0));
	}
	
	for(int i = 0; i < num_seams; i++) {	
		int rows = src_bw.rows;
		int cols = src_bw.cols;
		Mat blurred, energy;
		// GaussianBlur(src_bw, blurred, Size(17, 17), 0, 0);
		Mat tmp_src_bw_shrink;
		Mat tmp_output_shrink; 
		if (direction == 0){
			tmp_src_bw_shrink = Mat::zeros(rows, cols - 1, CV_8UC1);
			tmp_output_shrink = Mat::zeros(rows, cols - 1, CV_8UC3);
		}
		else {
			tmp_src_bw_shrink = Mat::zeros(rows - 1, cols, CV_8UC1);
			tmp_output_shrink = Mat::zeros(rows - 1, cols, CV_8UC3);
		}
		Mat seam_map(src_bw.rows, src_bw.cols, DataType<int>::type);

		getEnergy(src_bw, energy);
		getSeamMap(src_bw, energy, seam_map, direction);
		getSeamFromMap(seam_map, seam, direction);

		for(int j = 0; j < seam.size(); j++) {
			output.at<Vec3b>(seam[j].x, seam[j].y) = Vec3b(0, 0, 255);
		}

		imshow("TMP", output);
		cv::waitKey(1);

		if (direction == 0){
			for(int k = 0; k < rows; k++) {
				bool shift = false;
				for(int j = 0; j < cols; j++) {
					if (seam[k].y == j) shift = true;
					else if (!shift) {
						tmp_src_bw_shrink.at<uchar>(k,j) = src_bw.at<uchar>(k,j);
						tmp_output_shrink.at<Vec3b>(k,j) = output.at<Vec3b>(k,j);
					}
					else {
						tmp_src_bw_shrink.at<uchar>(k,j-1) = src_bw.at<uchar>(k,j);
						tmp_output_shrink.at<Vec3b>(k,j-1) = output.at<Vec3b>(k,j);
					}
				}			
			}
			std::printf("Deleting vertical seams: %d/%d\n", i + 1, num_seams);
		}
		else {
			for(int j = 0; j < cols; j++) {
				bool shift = false;
				for(int k = 0; k < rows; k++) {
					if (seam[j].x == k) shift = true;
					else if (!shift) {
						tmp_src_bw_shrink.at<uchar>(k,j) = src_bw.at<uchar>(k,j);
						tmp_output_shrink.at<Vec3b>(k,j) = output.at<Vec3b>(k,j);
					}
					else {
						tmp_src_bw_shrink.at<uchar>(k-1,j) = src_bw.at<uchar>(k,j);
						tmp_output_shrink.at<Vec3b>(k-1,j) = output.at<Vec3b>(k,j);
					}
				}			
			}
			std::printf("Deleting horizontal seams: %d/%d\n", i + 1, num_seams);
		}
		src_bw = tmp_src_bw_shrink;
		output = tmp_output_shrink;		
	}
	imshow("Final output", output);
}

void EnlargeImage(Mat &src, Mat &output, double scale, char direction) {
	if (scale <= 1 || scale > 2){
		std::cout << "Scale must range between 1 and 2." << endl;
		return;
	}

	Mat src_bw, energy;
	cvtColor(src, src_bw, CV_BGR2GRAY);
	vector<vector<Point>> seams;
	
	int num_seams;
	if(direction == 0){
		num_seams = (int)((double)src_bw.cols * (scale - 1));
	}
	else {
		num_seams = (int)((double)src_bw.rows * (scale - 1));
	}
	
	Mat seam_map(src_bw.rows, src_bw.cols, DataType<int>::type);
	getEnergy(src_bw, energy);
	getSeamMap(src_bw, energy, seam_map, direction);
	getSeamFromMap2(seam_map, seams, direction, num_seams);

	Mat copy_times(src_bw.rows, src_bw.cols, DataType<int>::type);
	for(int i = 0; i < src_bw.rows; i++) {
		for(int j = 0; j < src_bw.cols; j++) {
			copy_times.at<int>(i, j) = 1;
		}
	}
	Mat src2 = src.clone();
	for(int i = 0; i < seams.size(); i++) {
		for(int j = 0; j < seams[i].size(); j++) {
			int x = seams[i][j].x;
			int y = seams[i][j].y;
			copy_times.at<int>(x, y) = copy_times.at<int>(x, y) + 1;
			src2.at<Vec3b>(x,y) = Vec3b(0, 0, 255);
		}
	}
	imshow("TMP", src2);

	if(direction == 0){
		output = Mat(src_bw.rows, src_bw.cols + num_seams, CV_8UC3);
		for(int i = 0; i < src_bw.rows; i++){
			int index = 0;
			for(int j = 0; j < src_bw.cols; j++) {
				for(int k = 0; k < copy_times.at<int>(i, j); k++){
					if(k >= 1) index++;
					output.at<Vec3b>(i, j + index) = src.at<Vec3b>(i, j);					
				}
			}
		}
	}
	else {
		output = Mat(src_bw.rows + num_seams, src_bw.cols, CV_8UC3);
		for(int i = 0; i < src_bw.cols; i++){
			int index = 0;
			for(int j = 0; j < src_bw.rows; j++) {
				for(int k = 0; k < copy_times.at<int>(i, j); k++){
					output.at<Vec3b>(j + index, i) = src.at<Vec3b>(j, i);
					if(k >= 1) index++;
				}
			}
			printf("*%d*\n",i+1);
		}
	}
	imshow("Final output", output);
}

void process(Mat &src) {
	Mat src_bw, energy;
	cvtColor(src, src_bw, CV_BGR2GRAY);

	for(int i = 0; i < 300; i++) {
		int rows = src_bw.rows;
		int cols = src_bw.cols;
		Mat blurred, energy;
		vector<vector<Point> > seams;
		// GaussianBlur(src_bw, blurred, Size(17, 17), 0, 0);
		Mat tmp_src_bw_shrink = Mat::zeros(rows, cols - 1, CV_8UC1);

		getEnergy(src_bw, energy);

		//getSeams(src_bw, energy, seams);
		// for(int i = 0; i < seams.size(); i++) {
		vector<Point> seam = seams[0];
		for(int j = 0; j < seam.size(); j++) {
			src_bw.at<uchar>(seam[j].x, seam[j].y) = 0;
		}
		// }
		imshow("TMP", src_bw);
		waitKey(1);

		for(int k = 0; k < rows; k++) {
			bool shift = false;
			for(int j = 0; j < cols; j++) {
				if (seams[0][k].y == j) shift = true;
				else if (!shift) tmp_src_bw_shrink.at<uchar>(k,j) = src_bw.at<uchar>(k,j);
				else tmp_src_bw_shrink.at<uchar>(k,j-1) = src_bw.at<uchar>(k,j);
			}			
		}
		src_bw = tmp_src_bw_shrink;
		printf("%d\n", i + 1);
	}

	//
	// Vec3b red = Vec3b(0,0,255);
	// for(int i = 0; i < seams.size(); i++) {
	//   vector<Point> seam = seams[i];
	//   for(int j = 0; j < seam.size(); j++) {
	//     src.at<Vec3b>(seam[j].x, seam[j].y) = red;
	//   }
	// }
}

void getEnergy(Mat &src, Mat &energy) {
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Gradient X
	// Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Scharr(src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	// Gradient Y
	// Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	Scharr(src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	add(abs_grad_x, abs_grad_y, energy);

	//
	if (DEBUG) {
		imshow("energy", energy);
		cout << "\nenergy:\n";
		for (int i = 0; i < src.rows; i++) {
		cout << i << ((i<10) ? "  |  " : " |  ");
		for (int j = 0; j < src.cols; j++) {
			cout << (int)(energy.at<uchar>(i,j)) << " ";
		}
		cout << endl;
		}
	}
}

void getSeams(Mat &src, Mat &energy, vector<vector<Point> > &seams) {
	Mat seam_map(src.rows, src.cols, DataType<int>::type);
	vector<Point> vert(src.rows, Point(0,0));
	vector<Point> horz(src.cols, Point(0,0));

	getSeamMap(src, energy, seam_map, 0);
	if (DEBUG) {
		cout << "\nseam map:\n";
		for (int i = 0; i < src.rows; i++) {
			cout << i << ((i<10) ? "  |  " : " |  ");
			for (int j = 0; j < src.cols; j++) {
			cout << (int)(seam_map.at<int>(i,j)) << " ";
		}
		cout << endl;
		}
	}
	getSeamFromMap(seam_map, vert, 0);
	if (DEBUG) {
		for(int i = 0; i < src.rows; i++) cout << vert[i] << endl;
	}

	getSeamMap(src, energy, seam_map, 1);
	getSeamFromMap(seam_map, horz, 1);

	seams.push_back(vert);
	seams.push_back(horz);
}

void getSeamMap(Mat &src, Mat &energy, Mat &seam_map, char direction) {
	unsigned int rows = src.rows;
	unsigned int cols = src.cols;
	int min;
	vector<int> previous_vals(3, 0);

	// vertical seam map
	if (direction == 0) {
		for(int i = 0; i < cols; i++) seam_map.at<int>(0,i) = energy.at<uchar>(0,i);
		for(int i = 1; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				previous_vals[0] = (j > 0) ? seam_map.at<int>(i-1,j-1) : INT_MAX;
				previous_vals[1] = seam_map.at<int>(i-1,j);
				previous_vals[2] = (j < cols-1) ? seam_map.at<int>(i-1,j+1) : INT_MAX;
				min = getMinVal(previous_vals);
				seam_map.at<int>(i,j) = min + energy.at<uchar>(i,j);
			}
		}
	}
	// horizontal seam map
	else if (direction == 1) {
		for(int i = 0; i < rows; i++) seam_map.at<int>(i,0) = energy.at<uchar>(i,0);
		for(int j = 1; j < cols; j++) {
			for(int i = 0; i < rows; i++) {
				previous_vals[0] = (i > 0) ? seam_map.at<int>(i-1,j-1) : INT_MAX;
				previous_vals[1] = seam_map.at<int>(i,j-1);
				previous_vals[2] = (i < rows-1) ? seam_map.at<int>(i+1,j-1) : INT_MAX;
				min = getMinVal(previous_vals);
				seam_map.at<int>(i,j) = min + energy.at<uchar>(i,j);
			}
		}
	}
}

void getSeamFromMap(Mat &seam_map, vector<Point> &seam, char direction) {
	unsigned int rows = seam_map.rows;
	unsigned int cols = seam_map.cols;
	vector<int> vals(3, 0);

	// get the vertical seam
	if (direction == 0) {
		for(int i = rows - 1; i >= 0; i--) {
			// if we're at the bottom row, find the minimum val
			if (i == rows - 1) {
				int min = INT_MAX;
				int min_index;
				for(int j = 0; j < cols; j++) {
					// cout << "(" << i << "," << j << "): " << seam_map.at<int>(i, j) << endl;
					if (seam_map.at<int>(i, j) < min) {
						min = seam_map.at<int>(i, j);
						min_index = j;
					}
				}
				seam[i].x = i;
				seam[i].y = min_index;
			}
			else {
				int last_index = seam[i+1].y;
				vals[0] = (last_index > 0) ? seam_map.at<int>(i, last_index-1) : INT_MAX;
				vals[1] = seam_map.at<int>(i, last_index);
				vals[2] = (last_index < cols-1) ? seam_map.at<int>(i, last_index+1) : INT_MAX;
				int relative_index = getMinIndex(vals);
				seam[i].x = i;
				seam[i].y = last_index + relative_index - 1;
			}
		}
	}
	else if (direction == 1) {
		// get the horizontal seam
		for(int i = cols - 1; i >= 0; i--) {
			// if we're at the rightmost col, find the minimum val
			if (i == cols - 1) {
				int min = INT_MAX;
				int min_index;
				for(int j = 0; j < rows; j++) {
				// cout << "(" << j << "," << i << "): " << seam_map.at<int>(j, i) << endl;
					if (seam_map.at<int>(j, i) < min) {
						min = seam_map.at<int>(j, i);
						min_index = j;
					}
				}
				seam[i].x = min_index;
				seam[i].y = i;
			}
			else {
				int last_index = seam[i+1].x;
				vals[0] = (last_index > 0) ? seam_map.at<int>(last_index-1, i) : INT_MAX;
				vals[1] = seam_map.at<int>(last_index, i);
				vals[2] = (last_index < rows-1) ? seam_map.at<int>(last_index+1, i) : INT_MAX;
				int relative_index = getMinIndex(vals);
				seam[i].x = last_index + relative_index - 1;
				seam[i].y = i;
			}
		}
	}
}

void getSeamFromMap2(Mat &seam_map, vector<vector<Point>> &seams, char direction, int num_seams) {
	unsigned int rows = seam_map.rows;
	unsigned int cols = seam_map.cols;
	vector<int> vals(3, 0);
	vector<bool> flags;
	Mat seam_pos(seam_map.rows, seam_map.cols, DataType<bool>::type);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			seam_pos.at<bool>(i, j) = 0;
		}
	}

	// get the vertical seams
	if (direction == 0) {
		flags = vector<bool>(cols, 0);
		vector<Point> seam(rows, Point(0,0));
		for(int i = 0; i < num_seams; i++) {
			int min = INT_MAX;
			int min_index;
			for(int j = 0; j < cols; j++) {
				if (seam_map.at<int>(rows - 1, j) < min && flags[j] == 0) {
					min = seam_map.at<int>(rows - 1, j);
					min_index = j;
				}
			}
			printf("*%d*\n",i+1);
			flags[min_index] = 1;
		}

		int num_seam = 0;
		for(int ind = 0; ind < cols; ind++){
			if(flags[ind] == 1){
				num_seam++;
				std::printf("Getting vertical seam: %d/%d\n", num_seam, num_seams);
				for(int i = rows - 1; i >= 0; i--) {
					if (i == rows - 1) {
						seam[i].x = i;
						seam[i].y = ind;
						seam_pos.at<bool>(i, ind) = 1;
					}
					else {
						int last_index = seam[i+1].y;
						vals[0] = (last_index > 0 && seam_pos.at<bool>(i, last_index-1) == 0) ? seam_map.at<int>(i, last_index-1) : INT_MAX;
						vals[1] = (seam_pos.at<bool>(i, last_index) == 0) ? seam_map.at<int>(i, last_index) : INT_MAX;
						vals[2] = (last_index < cols-1 && seam_pos.at<bool>(i, last_index+1) == 0) ? seam_map.at<int>(i, last_index+1) : INT_MAX;
						int relative_index = getMinIndex(vals);
						seam[i].x = i;
						seam[i].y = last_index + relative_index - 1;
						seam_pos.at<bool>(i, last_index + relative_index - 1) = 1;
					}
				}
				seams.push_back(seam);
			}
		}
	}
	else if (direction == 1) {
		// get the horizontal seams
		flags = vector<bool>(rows, 0);
		vector<Point> seam(cols, Point(0,0));
		for(int i = 0; i < num_seams; i++) {
			int min = INT_MAX;
			int min_index;
			for(int j = 0; j < rows; j++) {
				if (seam_map.at<int>(j, cols - 1) < min && flags[j] == 0) {
					min = seam_map.at<int>(j, cols - 1);
					min_index = j;
				}
			}
			flags[min_index] = 1;
		}

		int num_seam = 0;
		for(int ind = 0; ind < rows; ind++){
			if(flags[ind] == 1){
				num_seam++;
				std::printf("Getting horizontal seam: %d/%d\n", num_seam, num_seams);
				for(int i = cols - 1; i >= 0; i--) {
					if (i == cols - 1) {
						seam[i].x = ind;
						seam[i].y = i;
						seam_pos.at<bool>(ind, i) = 1;
					}
					else {
						int last_index = seam[i+1].x;
						vals[0] = (last_index > 0 && seam_pos.at<bool>(last_index-1, i) == 0) ? seam_map.at<int>(last_index-1, i) : INT_MAX;
						vals[1] = (seam_pos.at<bool>(last_index, i) == 0) ? seam_map.at<int>(last_index, i) : INT_MAX;
						vals[2] = (last_index < rows-1 && seam_pos.at<bool>(last_index+1, i) == 0) ? seam_map.at<int>(last_index+1, i) : INT_MAX;
						int relative_index = getMinIndex(vals);
						seam[i].x = last_index + relative_index - 1;
						seam[i].y = i;
						seam_pos.at<bool>(last_index + relative_index - 1, i) = 1;
					}
				}
				seams.push_back(seam);
			}
		}
	}
}

int getMinVal(vector<int> &vals) {
	int m1 = min(vals[0], vals[1]);
	int m2 = min(m1, vals[2]);
	return m2;
}

int getMinIndex(vector<int> &vals) {
	int min = INT_MAX;
	int min_index = 1;
	for(int i = 0; i < 3; i++) {
		if (vals[i] < min) {
			min = vals[i];
			min_index = i;
		}
	}
	return min_index;
}
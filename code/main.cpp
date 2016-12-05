//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <unistd.h>

#define IMAGE_NAME "tree.jpg"
#define DEBUG 1
#define RED Scalar(0,0,255)

using namespace cv;
using namespace std;

void ContentAwareResizing(Mat &src, Mat &output, int outWidth, int outHeight);
void ShrinkImage(Mat &src, Mat &output, double scale, char direction);
void EnlargeImage(Mat &src, Mat &output, double scale, char direction);
void resizeByHistogram(Mat &src, Mat &output, int outWidth, int outHeight);
Point getCenter(Mat &src, int buckets = 15);
void process(Mat &src,int outWidth,int outHeight);
void getEnergy(Mat &src, Mat &energy);
void getSeams(Mat &src, Mat &energy, vector<vector<Point> > &seams);
void getSeamMap(Mat &src, Mat &energy, Mat &seam_map, char direction);
void getSeamFromMap(Mat &seam_map, vector<Point> &seam, char direction);
void getSeamFromMap2(Mat &seam_map, vector<vector<Point> > &seams, char direction, int num_seams);
int getMinVal(vector<int> &vals);
int getMinIndex(vector<int> &vals);
int main(int argc,char* argv[])
{
	if (argc != 5){
		cout << "Usage: " << argv[0] << " [inputImage] [outputImage] [desiredWidth] [desiredHeight]" << endl;
		return 0;
	}

	//Mat img = imread(IMAGE_NAME, IMREAD_COLOR);
	Mat img = imread(argv[1], IMREAD_COLOR);
	//resize(img, img, Size(1000,520));
	// resize(img, img, Size(20,10));
	namedWindow("Finished Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Image", CV_WINDOW_AUTOSIZE);
  namedWindow("Energy", CV_WINDOW_AUTOSIZE);
	namedWindow("TMP", CV_WINDOW_AUTOSIZE);
  namedWindow("Crop", CV_WINDOW_AUTOSIZE);
	//namedWindow("energy", CV_WINDOW_AUTOSIZE);
	imshow("Image",img);

	Mat output;
	//process(img,stoi(argv[3]),stoi(argv[4]));
	// ContentAwareResizing(img, output, stoi(argv[3]), stoi(argv[4]));
  resizeByHistogram(img, output, stoi(argv[3]), stoi(argv[4]));
  imshow("Image",img);
	imshow("Finished Image", output);
	imwrite(argv[2], output);
	waitKey(0);
	return 0;
}

void resizeByHistogram(Mat &src, Mat &output, int outWidth, int outHeight) {
  int rows = src.rows;
	int cols = src.cols;
  if (DEBUG) {
    cout << "Height: " << rows << endl;
    cout << "Width:  " << cols << endl;
  }
  Mat src_bw, energy;
  Point upper_left, lower_right;
  int dstWidth, dstHeight;
  float srcAR = ((float)cols) / ((float)rows);
  float dstAR = ((float)outWidth) / ((float)outHeight);
  if (srcAR == dstAR) {
    output = src.clone();
    return;
  }

  cvtColor(src, src_bw, CV_BGR2GRAY);
  getEnergy(src_bw, energy);
  // imshow("Energy", energy);
  Point center = getCenter(energy);
  if (DEBUG) cout << "Center " << center << endl;

  if (srcAR > dstAR) {
    if (DEBUG) cout << "Shrinking horizontally" << endl;
    // shrinking horizontally
    dstWidth  = dstAR * rows;
    dstHeight = rows;

    // center frame horizontally at this point
    int left_pixel = center.x - (dstWidth / 2);
    int rght_pixel = center.x + (dstWidth / 2) - ((dstWidth % 2 == 0) ? 1 : 0);

    // shift the frame if either sides are out of bounds
    if (left_pixel < 0) {
      rght_pixel -= left_pixel;
      left_pixel = 0;
    }
    else if (rght_pixel >= cols) {
      left_pixel -= rght_pixel - cols - 1;
      rght_pixel = cols - 1;
    }

    upper_left = Point(left_pixel, 0);
    lower_right = Point(rght_pixel, rows-1);
  }
  else {
    if (DEBUG) cout << "Shrinking vertically" << endl;
    // shrinking vertically
    dstWidth  = cols;
    dstHeight = cols / dstAR;

    // center frame vertically at this point
    int top_pixel = center.y - (dstHeight / 2);
    int bot_pixel = center.y + (dstHeight / 2) - ((dstHeight % 2 == 0) ? 1 : 0);

    // shift the frame if either sides are out of bounds
    if (top_pixel < 0) {
      bot_pixel -= top_pixel;
      top_pixel = 0;
    }
    else if (bot_pixel >= rows) {
      top_pixel -= bot_pixel - rows - 1;
      bot_pixel = rows - 1;
    }

    upper_left = Point(0, top_pixel);
    lower_right = Point(cols-1, bot_pixel);
  }

  if (DEBUG) {
    cout << "Upper Left:  " << upper_left << endl;
    cout << "Lower Right: " << lower_right << endl;
  }

  // crop the image with the calculated size
  // this cropping method isn't inclusive of the lower right point, hence the shift
  Rect roi(upper_left, Point(lower_right.x + 1, lower_right.y + 1));
  output = src(roi).clone();
  if (DEBUG) {
    imshow("Crop", output);
    cout << "New Height: " << output.rows << endl;
    cout << "New Width:  " << output.cols << endl;
    rectangle(src, upper_left, lower_right, RED);
    circle(src, center, 1, RED, 4);
  }

  // now we have an image with the right aspect ratio
  // so simple resize it
  resize(output, output, Size(outWidth, outHeight));
}

Point getCenter(Mat &src, int buckets) {
  assert (buckets > 0);

  int rows = src.rows;
	int cols = src.cols;
  int horz_bucket_size = cols / buckets;
  int vert_bucket_size = rows / buckets;
  int effectiveWidth   = horz_bucket_size * buckets;
  int effectiveHeight  = vert_bucket_size * buckets;
  vector<int> horz_hist(buckets, 0);
  vector<int> vert_hist(buckets, 0);

  for(int i = 0; i < effectiveHeight; i++) {
    for(int j = 0; j < effectiveWidth; j++) {
      horz_hist[j / horz_bucket_size] += (int)(src.at<uchar>(i,j));
      vert_hist[i / vert_bucket_size] += (int)(src.at<uchar>(i,j));
    }
  }

  int horz_max = distance(horz_hist.begin(), max_element(horz_hist.begin(), horz_hist.end()));
  int vert_max = distance(vert_hist.begin(), max_element(vert_hist.begin(), vert_hist.end()));
  int horz_pixel = (horz_max * horz_bucket_size) + (horz_bucket_size / 2);
  int vert_pixel = (vert_max * vert_bucket_size) + (vert_bucket_size / 2);
  return Point(horz_pixel, vert_pixel);
}

void ContentAwareResizing(Mat &src, Mat &output, int outWidth, int outHeight) {
	int rows = src.rows;
	int cols = src.cols;
	double x_scale = (double)outHeight / (double)rows;
	double y_scale = (double)outWidth / (double)cols;
	Mat tmp;

	if(x_scale > 1) EnlargeImage(src, tmp, x_scale, 1);
	else ShrinkImage(src, tmp, x_scale, 1);

	if(y_scale > 1) EnlargeImage(tmp, output, y_scale, 0);
	else ShrinkImage(tmp, output, y_scale, 0);
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
}

void EnlargeImage(Mat &src, Mat &output, double scale, char direction) {
	if (scale <= 1){
		std::cout << "Scale must be larger than 1." << endl;
		return;
	}

	if (scale > 1.4){
		Mat tmp;
		EnlargeImage(src, tmp, 1.4, direction);
		double scale2 = scale / 1.4;
		EnlargeImage(tmp, output, scale2, direction);
		return;
	}

	Mat src_bw, energy;
	cvtColor(src, src_bw, CV_BGR2GRAY);
	vector<vector<Point> > seams;

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
				for(int k = 0; k < copy_times.at<int>(j, i); k++){
					if(k >= 1) index++;
					output.at<Vec3b>(j + index, i) = src.at<Vec3b>(j, i);
				}
			}
		}
	}
}

void process(Mat &src,int outWidth,int outHeight) {
  Mat src_bw, energy;
  cvtColor(src, src_bw, CV_BGR2GRAY);
  int rows = src_bw.rows;
  int cols = src_bw.cols;

  // iterate until we get to the desired size
  Mat tmp_src_bw_shrink;
  int direction;
  int skewDim;
  int idx = 0;
  while (rows!=outHeight && cols != outWidth){
    //cout << idx++ << "Rows: " << rows << " Cols:" << cols << endl;

    // decide which dimension to skew, and in which direction
    // right now, just brute force, which is larger
    if (abs(rows-outHeight) > abs(cols-outWidth)){
        // skew in y
        if (rows > outHeight){
            // shrink
            direction = -1;
        } else {
            // grow
            direction = 1;
        }
        tmp_src_bw_shrink = Mat::zeros(rows+direction, cols, CV_8UC1);
        skewDim = 0;
    } else {
        // skew in x
        if (cols > outWidth){
            // shrink
            direction = -1;
        } else {
            // grow
            direction = 1;
        }
        tmp_src_bw_shrink = Mat::zeros(rows, cols +direction, CV_8UC1);
        skewDim = 1;
    }
    Mat blurred, energy;
    vector<vector<Point> > seams;
    // GaussianBlur(src_bw, blurred, Size(17, 17), 0, 0);

    getEnergy(src_bw, energy);
    getSeams(src_bw, energy, seams);
    // for(int i = 0; i < seams.size(); i++) {
    // Draw the seams (0 is rows, 1 is cols)
    vector<Point> seam = seams[1-skewDim];
    for(int j = 0; j < seam.size(); j++) {
      src_bw.at<uchar>(seam[j].x, seam[j].y) = 0;
    }
    // }
    //imshow("TMP", src_bw);
    //waitKey(0);

    // shrink col
    if (skewDim){

        for(int i = 0; i < rows; i++) {
          bool shift = false;
          for(int j = 0; j < cols; j++) {
            // horizontal
            if (seams[0][i].y == j) shift = true;
            else if (!shift) tmp_src_bw_shrink.at<uchar>(i,j) = src_bw.at<uchar>(i,j);
            else tmp_src_bw_shrink.at<uchar>(i,j+direction) = src_bw.at<uchar>(i,j);
          }
        }
    }
    // shrink row
    else{
        for(int j = 0; j < cols; j++) {
          bool shift = false;
          for(int i = 0; i < rows; i++) {
            if (seams[1][j].x == i) shift = true;
            else if (!shift) tmp_src_bw_shrink.at<uchar>(i,j) = src_bw.at<uchar>(i,j);
            else tmp_src_bw_shrink.at<uchar>(i+direction,j) = src_bw.at<uchar>(i,j);
        }
      }
    }
    src_bw = tmp_src_bw_shrink;
    rows = src_bw.rows;
    cols = src_bw.cols;

  }
  src = src_bw;

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

  if (DEBUG) imshow("energy", energy);
}

void getSeams(Mat &src, Mat &energy, vector<vector<Point> > &seams) {
  Mat seam_map(src.rows, src.cols, DataType<int>::type);
  vector<Point> vert(src.rows, Point(0,0));
  vector<Point> horz(src.cols, Point(0,0));

  getSeamMap(src, energy, seam_map, 0);
  getSeamFromMap(seam_map, vert, 0);

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

void getSeamFromMap2(Mat &seam_map, vector<vector<Point> > &seams, char direction, int num_seams) {
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
			int min_index_vert;
			for(int j = 0; j < cols; j++) {
				if (seam_map.at<int>(rows - 1, j) < min && flags[j] == 0) {
					min = seam_map.at<int>(rows - 1, j);
					min_index_vert = j;
				}
			}
			//printf("*%d*\n",i+1);
			flags[min_index_vert] = 1;
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
			int min_index_horz;
			for(int j = 0; j < rows; j++) {
				if (seam_map.at<int>(j, cols - 1) < min && flags[j] == 0) {
					min = seam_map.at<int>(j, cols - 1);
					min_index_horz = j;
				}
			}
			flags[min_index_horz] = 1;
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

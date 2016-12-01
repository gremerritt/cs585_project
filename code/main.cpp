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
#define DEBUG 0

using namespace cv;
using namespace std;

void process(Mat &src,int outWidth,int outHeight);
void getEnergy(Mat &src, Mat &energy);
void getSeams(Mat &src, Mat &energy, vector<vector<Point> > &seams);
void getSeamMap(Mat &src, Mat &energy, Mat &seam_map, char direction);
void getSeamFromMap(Mat &seam_map, vector<Point> &seam, char direction);
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
    namedWindow("TMP", CV_WINDOW_AUTOSIZE);
    namedWindow("energy", CV_WINDOW_AUTOSIZE);
    imshow("Image",img);
    process(img,stoi(argv[3]),stoi(argv[4]));
    imshow("Finished Image", img);
    waitKey(0);
    return 0;
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

int getMinVal(vector<int> &vals) {
  int m1 = min(vals[0], vals[1]);
  int m2 = min(m1, vals[2]);
  return m2;
}

int getMinIndex(vector<int> &vals) {
  int min = INT_MAX;
  int min_index;
  for(int i = 0; i < 3; i++) {
    if (vals[i] < min) {
      min = vals[i];
      min_index = i;
    }
  }
  return min_index;
}

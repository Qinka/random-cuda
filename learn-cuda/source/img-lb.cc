

// OpenCV Headers
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <linear-combination.h>

using std::cout;
using std::cerr;
using std::endl;
using std::istringstream;
using std::setw;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " <img-in1> <img-in2> <img-out> [c1] [c2]" << endl;
    return -1;
  }
  cv::Mat image1 = cv::imread(argv[1],cv::IMREAD_UNCHANGED);
  cv::Mat image2 = cv::imread(argv[2],cv::IMREAD_UNCHANGED);
  cv::resize(image2,image2,cv::Size(image1.rows,image1.cols));
  cv::Mat out = image1.clone();
  float c1 = 1;
  float c2 = 2;
  if (argc >= 6) {
    istringstream sc1(argv[4]);
    istringstream sc2(argv[4]);
    sc1 >> c1;
    sc2 >> c2;
  }

  int rt = linearCombination(c1,image1.data,c2,image2.data,image1.cols * image1.rows,out.data);

  int row = image1.rows;
  int col = image1.cols;

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << image1.data[i*col+j];
    cout << endl;
  }
  cout << " + " << endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << image2.data[i*col+j];
    cout << endl;
  }

  cout << " + " << endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << out.data[i*col+j];
    cout << endl;
  }

  cerr << "rt code: " << rt <<endl;
  cv::imwrite(argv[3],out);
}

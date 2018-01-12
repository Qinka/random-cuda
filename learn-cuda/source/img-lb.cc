

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
using std::dec;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " <img-in1> <img-in2> <img-out> [c1] [c2]" << endl;
    return -1;
  }
  cv::Mat image1 = cv::imread(argv[1],cv::IMREAD_UNCHANGED);
  cv::Mat image2 = cv::imread(argv[2],cv::IMREAD_UNCHANGED);
  if (image1.channels() != image2.channels()) {
    cerr << "The number of channals should be same!" << endl;
    return -1;
  }
  if (image1.rows != image2.rows || image1.cols != image2.cols)
    cv::resize(image2,image2,cv::Size(image1.rows,image1.cols));


  int row = image1.rows;
  int col = image1.cols * image1.channels();

  cerr << row << " " << col << " " << image1.channels() << endl;

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << (int)image1.data[i*col+j];
    cout << endl;
  }

  cout << " + " << endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << (int)image2.data[i*col+j];
    cout << endl;
  }

  cv::Mat out = image1.clone();
  float c1 = 1;
  float c2 = 1;
  if (argc >= 6) {
    istringstream sc1(argv[4]);
    istringstream sc2(argv[5]);
    sc1 >> c1;
    sc2 >> c2;
    cerr << c1 << " " << c2 << endl;
  }

  int rt = linearCombination(c1,image1.data,c2,image2.data,row * col,out.data);


  cout << " + " << endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      cout << setw(4) << dec << (int)out.data[i*col+j];
    cout << endl;
  }

  cerr << "rt code: " << rt <<endl;
  cv::imwrite(argv[3],out);
}

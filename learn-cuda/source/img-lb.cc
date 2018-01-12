

// OpenCV Headers
#inlcude <opencv2/opencv.hpp>

#include <iostream>
#include <cstring>

using std::cout;
using std::cerr;
using std::endl;


int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "usage: " << argv[0] << " <img-in1> <img-in2> <img-out> [c1] [c2]" << endl;
    return -1;
  }
  cv::Mat image1 = cv::imread(argv[1],cv::IMREAD_UNCHANGED);
  cv::Mat image2 = cv::imread(argv[2],cv::IMREAD_UNCHANGED);
  cv::resize(image2,image2,cv::Size(image1.row(),image1.col()x));
  cv::Mat out = image1.clone();
  float c1 = 1;
  float c2 = 2;
  if (argc >= 6) {
    std::sscanf(argv[4],"%f",&c1);
    std::sscanf(argv[5],"%f",&c2);
  }
  linearCombination(c1,image1.data,c2,image2.data,image1.cow() * image1.row(),out.data);
  cv::imwrite(argv[3],out)
}

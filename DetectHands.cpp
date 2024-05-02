#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int startx = 0;
int starty = 280;
int endx = 640;
int endy = 480;
int main() {
    _putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");
    VideoCapture cap(0);//640x480を想定
    Mat frame;
    int linesWidth = endx - startx;
    int lineWidth = linesWidth / 16;
    int lineHeight = endy - starty;
    int lineArea = lineWidth * lineHeight;
    double threshold = 0.5;
    while (1) {
        cap >> frame;
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat hue;
        extractChannel(hsv, hue, 0);
        inRange(hue, 0, 20, hue);
        medianBlur(hue, hue, 15);
        for (int line = 0; line < 16; line++) {
            int count = 0;
            for (int x = lineWidth*line; x < lineWidth * (line+1); x++) {
                for (int y = starty; y < endy; y++) {
                    if (hue.at<uchar>(y, x))count++;
                }
            }
            if ((double)(count) / lineArea > threshold) cout << line << ',';
                
        }
        cout << "-1" << endl;
        imshow("hue", hue);
        int key = waitKey(1);
        if (key == 27) { //escが押されたら終了
            break;
        }
    }

    return 0;
}

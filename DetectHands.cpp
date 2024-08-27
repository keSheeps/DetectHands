#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <iostream>
#include <array>
#include <string>

using namespace std;
using namespace cv;

std::string GetConfigString(const std::string& filePath, const char* pSectionName, const char* pKeyName)
{
    if (filePath.empty()) {
        return "";
    }
    std::array<char, MAX_PATH> buf = {};
    GetPrivateProfileStringA(pSectionName, pKeyName, "", &buf.front(), static_cast<DWORD>(buf.size()), filePath.c_str());
    return &buf.front();
}

int main() {
    std::string filePath = ".\\config.ini";
    cout << GetConfigString(filePath, "System", "startx") << endl;
    int startx = stoi(GetConfigString(filePath, "System", "startx"));
    int starty = stoi(GetConfigString(filePath, "System", "starty"));
    int endx = stoi(GetConfigString(filePath, "System", "endx"));
    int endy = stoi(GetConfigString(filePath, "System", "endy"));

    double threshold_ratio = stod(GetConfigString(filePath, "System", "threshold_ratio")) /100;

    int hue_min = stoi(GetConfigString(filePath, "System", "hue_min"));
    int hue_max = stoi(GetConfigString(filePath, "System", "hue_max"));
    int sat_min = stoi(GetConfigString(filePath, "System", "sat_min"));
    int sat_max = stoi(GetConfigString(filePath, "System", "sat_max"));

    int wait = stoi(GetConfigString(filePath, "System", "wait"));

    int camera_id = stoi(GetConfigString(filePath, "System", "camera_id"));

    _putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");
    VideoCapture cap(camera_id);//640x480を想定
    Mat frame;
    namedWindow("hue", WindowFlags::WINDOW_AUTOSIZE | WindowFlags::WINDOW_FREERATIO);
    namedWindow("sat", WindowFlags::WINDOW_AUTOSIZE | WindowFlags::WINDOW_FREERATIO);
    int linesWidth = endx - startx;
    int lineWidth = linesWidth / 16;
    int lineHeight = endy - starty;
    int lineArea = lineWidth * lineHeight;

    while (1) {
        cap >> frame;
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat hue,sat;
        extractChannel(hsv, hue, 0);//色相
        extractChannel(hsv, sat, 1);//彩度 彩度(白黒からどれだけ離れているか)が低いところは除く
        inRange(hue, hue_min, hue_max, hue);
        inRange(sat, sat_min, sat_max, sat);
        medianBlur(hue, hue, 15);
        medianBlur(sat, sat, 15);
        for (int line = 0; line < 16; line++) {
            int count = 0;
            for (int x = lineWidth*line; x < lineWidth * (line+1); x++) {
                for (int y = starty; y < endy; y++) {
                    if (hue.at<uchar>(y, x) && sat.at<uchar>(y, x))count++;
                }
            }
            if ((double)(count) / lineArea > threshold_ratio) cout << line << ',';
                
        }
        cout << "-1" << endl;
        imshow("hue", hue);
        imshow("sat", sat);
        int key = waitKey(wait);
        if (key == 27) { //escが押されたら終了
            break;
        }
    }

    return 0;
}
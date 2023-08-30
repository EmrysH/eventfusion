
//RealsenseLibrary
#include <librealsense2/rs.hpp>


//celex5library
#include "celex5.h"
#include "celex5datamanager.h"
#define FPN_PATH    "/home/emrys/CLionProjects/Fusion/FPN_2.txt"
#include<unistd.h>
#include <signal.h>


//optrisLibrary
#include <memory>
#include "libirimager/direct_binding.h"


//GeneralLibrary
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <thread>
#include "eigen3/Eigen/Dense"

using namespace std;
using namespace cv;
using namespace Eigen;





CeleX5 *pCeleX5 = new CeleX5;


class SensorDataObserver : public CeleX5DataManager
{
public:
    SensorDataObserver(CX5SensorDataServer* pServer)
    {
        m_pServer = pServer;
        m_pServer->registerData(this, CeleX5DataManager::CeleX_Frame_Data);
    }
    ~SensorDataObserver()
    {
        m_pServer->unregisterData(this, CeleX5DataManager::CeleX_Frame_Data);
    }
    virtual void onFrameDataUpdated(CeleX5ProcessedData* pSensorData);//overrides Observer operation

    CX5SensorDataServer* m_pServer;
};

void exit_handler(int sig_num)
{
    printf("SIGNAL received: num =%d\n", sig_num);
    if (sig_num == 1 || sig_num == 2 || sig_num == 3 || sig_num == 9 || sig_num == 15)
    {
        delete pCeleX5;
        pCeleX5 = NULL;
        exit(0);
    }
}


void SensorDataObserver::onFrameDataUpdated(CeleX5ProcessedData* pSensorData)
{
    if (NULL == pSensorData)
        return;
}




rs2::pipeline p;
Mat realsense_frame()
{

    rs2::frameset frames = p.wait_for_frames();

    rs2::frame color = frames.get_color_frame();

    rs2::depth_frame depth = frames.get_depth_frame();

    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();

    Mat image(Size(w, h), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);

    cvtColor(image, image, CV_BGR2RGB);//转换
    return image;

}




Mat event_frame()
{
    std::vector<EventData> vecEvent;
    pCeleX5->getEventDataVector(vecEvent);
    cv::Mat sae= cv::Mat::zeros(cv::Size(1280, 800), CV_64F);

    int dataSize = vecEvent.size();

    vector<int> p(dataSize);
    vector<int> x(dataSize);
    vector<int> y(dataSize);
    vector<double> ts(dataSize);

    double tau = 50e-3 ;

    for (int i = 0; i < dataSize; i++) {
        p[i] = vecEvent[i].polarity;
        x[i] = vecEvent[i].col;
        y[i] = vecEvent[i].row;
        ts[i] = vecEvent[i].tOffPixelIncreasing * 0.000001;
    }

    double t_ref = ts.back();

    for (int i = 0; i < ts.size(); i++)
    {


        if (p[i] > 0) {
            sae.at<double>(800-y[i]-1,1280-x[i]-1) = exp(-(t_ref - ts[i]) / tau);
        }
        if (p[i] < 0 ){
            sae.at<double>(800-y[i]-1,1280-x[i]-1) = -exp(-(t_ref - ts[i]) / tau);
        }

    }

    cv::normalize(sae, sae, 0, 255, cv::NORM_MINMAX, CV_8U);

    flip(sae, sae, 1);
    return sae;

}

int p_w = 384;
int p_h =288;
int t_w = 382;
int t_h = 288;

std::vector<unsigned char> palette_image(p_w * p_h * 3);
std::vector<unsigned short> thermal_data(t_w * t_h);

Mat optris_frame()
{

    evo_irimager_get_thermal_palette_image(t_w, t_h, &thermal_data[0], p_w, p_h, &palette_image[0]);
    cv::Mat cv_img(cv::Size(p_w, p_h), CV_8UC3, &palette_image[0], cv::Mat::AUTO_STEP);
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
    return cv_img;

}



int main(int argc, char * argv[]) {

    cv::FileStorage fs0("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_RGB_intrinsic.xml",
                        cv::FileStorage::READ);
    cv::FileStorage fs1("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_RGB_distortion.xml",
                        cv::FileStorage::READ);
    cv::FileStorage fs2("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Thermal_intrinsic.xml",
                        cv::FileStorage::READ);
    cv::FileStorage fs3("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Thermal_distortion.xml",
                        cv::FileStorage::READ);
    cv::FileStorage fs4("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_rotation_matrix.xml",
                        cv::FileStorage::READ);
    cv::FileStorage fs5("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_translation_matrix.xml",
                        cv::FileStorage::READ);

    cv::Mat rgbIntrinsic, rgbDistortion, thermalIntrinsic, thermalDistortion, relative_R, relative_T,trans_mat,relative_Rvct;
    fs0["rgbIntrinsic"] >> rgbIntrinsic;
    fs1["rgbDistortion"] >> rgbDistortion;
    fs2["thermalIntrinsic"] >> thermalIntrinsic;
    fs3["thermalDistortion"] >> thermalDistortion;
    fs4["relative_R"] >> relative_R;
    fs5["relative_T"] >> relative_T;
//    std::cout << relative_T << std::endl;
    fs0.release();
    fs1.release();
    fs2.release();
    fs3.release();
    fs4.release();
    fs5.release();

    hconcat(relative_R, relative_T, trans_mat);
    Rodrigues(relative_R, relative_Rvct);

//    std::cout << rgbIntrinsic << std::endl;



    Mat rgbImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/0.png");

    Mat rgbImage_g;

    cvtColor(rgbImage,rgbImage_g,COLOR_BGR2GRAY);

    Mat thermalImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/0.png");



    // Output the shape of the matrix
//    std::cout << "Shape: (" << rows << " rows, " << cols << " columns)" << std::endl;
//
//
//
//
//    // Calculate fx, fy, cx, and cy
    double fx = rgbIntrinsic.at<double>(0,0);
    double fy = rgbIntrinsic.at<double>(1,1);
    double cx = rgbIntrinsic.at<double>(0,2);
    double cy = rgbIntrinsic.at<double>(1,2);
//    cout<<fx<<endl;


    int width = 640;
    int height = 480;

    VectorXd x = VectorXd::LinSpaced(width, 0, width - 1);
    Eigen::MatrixXd x_c = Eigen::MatrixXd::Constant(480, 640, x(0));

    // Replicate the values of x in x_c
    for (int row = 0; row < 480; ++row) {
        for (int col = 0; col < 640; ++col) {
            x_c(row, col) = x(col);
        }
    }


    VectorXd y = VectorXd::LinSpaced(height, 0, height - 1);

    Eigen::MatrixXd y_c = Eigen::MatrixXd::Constant(640, 480, y(0));

    // Replicate the values of y in y_c
    for (int row = 0; row < 640; ++row) {
        for (int col = 0; col < 480; ++col) {
            y_c(row, col) = y(col);
        }
    }

    y_c = y.replicate(1, width);

//    cout<<y_c<<endl;
//    std::cout << "Size: (" << y_c.rows() << " rows, " << y_c.cols() << " columns)" << std::endl;

    MatrixXd x_real_imgplane = (x_c.array() - cx) / fx;
    MatrixXd y_real_imgplane = (y_c.array() - cy) / fy;
//
//    cout<<x_real_imgplane<<endl;


    Mat RGB_IMG, Thermal_IMG,Event_IMG;

////        Initializing Celex5

    if (NULL == pCeleX5)
        return 0;

    pCeleX5->openSensor(CeleX5::CeleX5_MIPI);
    pCeleX5->setFpnFile(FPN_PATH);
//    pCeleX5->setSensorFixedMode(CeleX5::Event_Off_Pixel_Timestamp_Mode);
    pCeleX5->setSensorFixedMode(CeleX5::Event_Intensity_Mode);
    pCeleX5->disableFrameModule();
    pCeleX5->disableIMUModule();
    pCeleX5->disableEventCountSlice();
    SensorDataObserver* pSensorData = new SensorDataObserver(pCeleX5->getSensorDataServer());


    // install signal use sigaction
    struct sigaction sig_action;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sig_action.sa_handler = exit_handler;
    sigaction(SIGHUP, &sig_action, NULL);  // 1
    sigaction(SIGINT, &sig_action, NULL);  // 2
    sigaction(SIGQUIT, &sig_action, NULL); // 3
    sigaction(SIGKILL, &sig_action, NULL); // 9
    sigaction(SIGTERM, &sig_action, NULL); // 15

//    sleep(3);


////    Initializing RealSense
    p.start();
    namedWindow("RGB_IMG", WINDOW_AUTOSIZE);

////    Initializing Optris
    evo_irimager_usb_init("/home/emrys/CLionProjects/Fusion/22092003.xml",0,0);
    int err;

    if((err = ::evo_irimager_get_palette_image_size(&p_w, &p_h)) != 0)
    {
        std::cerr << "error on evo_irimager_get_palette_image_size: " << err << std::endl;
        exit(-1);
    }

    //!width of thermal and palette image can be different due to stride of 4 alignment
    if((err = ::evo_irimager_get_thermal_image_size(&t_w, &t_h)) != 0)
    {
        std::cerr << "error on evo_irimager_get_palette_image_size: " << err << std::endl;
        exit(-1);
    }


    cout<<"p_w,p_h"<<endl;
    cout<<p_w<<","<<p_h<<endl;

    sleep(3);

    while (true)
    {

        Thermal_IMG = optris_frame();
        RGB_IMG = realsense_frame();
        Event_IMG = event_frame();


        imshow("palette image daemon", Thermal_IMG);
        imshow("RGB_IMG",RGB_IMG);
        imshow("TS Pic", Event_IMG);
        waitKey(1);

        if (cv::waitKey(1) == 'q') {
            break;
        }

    }
}




//
// Created by emrys on 8/24/23.
//


//RealsenseLibrary
#include <librealsense2/rs.hpp>
#include "rsd455.h"


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
//#include "calcpcd.h"
#include "eigen3/Eigen/Dense"


#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <chrono>
#include <thread>

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


//vector<Point2i> Pointf2i(const vector<Point2d>& src) {
//    vector<Point2i> dst;
//    dst.reserve(src.size());
//
//    // Convert each Point2d to Point2i using std::transform and lambda function
//    std::transform(src.begin(), src.end(), std::back_inserter(dst),
//                   [](const Point2d& point) {
//                       return Point2i(static_cast<int>(std::round(point.x)),
//                                      static_cast<int>(std::round(point.y)));
//                   });
//
//    return dst;
//}



vector<Point2i> Pointf2i(vector<Point2d> &src){
    vector<Point2i> dst(src.size());
    for (int i=0; i<dst.size();i++){

        dst[i].x = round(src[i].x);
        dst[i].y = round(src[i].y);

    }
    return dst;
}


int* gen_bad_points(vector<Point3d> real_point3d){
    static int bad_obj_points[307200];
    for (int i = 0; i < 307200; i++) {
        if (real_point3d[i].z == 0) {
            bad_obj_points[i] = 1;
        }
        else bad_obj_points[i] = 0;
    }
    return bad_obj_points;
}


Mat testfunc(vector<Point2i> imagePoints, Mat thermal_data){

    Mat projected_image = Mat(Size(1,307200),CV_8UC1);
    int row[307200] = {0};
    int col[307200] = {0};


    for (int i = 0; i < 307200; i++){

        row[i] = std::max(0, std::min(imagePoints[i].x, 479));
        col[i] = std::max(0, std::min(imagePoints[i].y, 639));
        projected_image.at<uchar>(i,0) = thermal_data.at<uchar>(col[i],row[i]);

    }

    projected_image = projected_image.reshape(1, 480);

    return projected_image;
}

//Mat testfunc(Mat imagePoints, Mat thermal_data){
//
//    Mat projected_image = Mat(Size(1,307200),CV_8UC1);
//    int row[307200] = {0};
//    int col[307200] = {0};
//
//
//    for (int i = 0; i < 307200; i++){
//
//        row[i] = std::max(0, std::min(imagePoints.at<int>(col[i],row[0]), 479));
//        col[i] = std::max(0, std::min(imagePoints.at<int>(col[i],row[1]), 639));
//        projected_image.at<uchar>(i,0) = thermal_data.at<uchar>(col[i],row[i]);
//
//    }
//
//    projected_image = projected_image.reshape(1, 480);
//
//    return projected_image;
//}





vector<Point3d> Mat2Point(Mat &src){
    vector<Point3d> dst;
    for (int i=0; i<307200; i++){
        Point3f p;
        //p.x = src.at<double>(0,i);
        p.x = (double)src.at<Vec3d>(i,0).val[0];
        //p.y = src.at<double>(1,i);
        p.y = (double)src.at<Vec3d>(i,0).val[1];
        //p.z = src.at<double>(2,i);
        p.z = (double)src.at<Vec3d>(i,0).val[2];
        dst.push_back(p);
    }
    return dst;
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


    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
    resize(cv_img,cv_img,Size(640,480));
    return cv_img;

}



cv::Mat rgbIntrinsic, rgbDistortion, thermalIntrinsic, thermalDistortion, relative_R, relative_T,trans_mat,relative_Rvct;

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

//    std::cout << thermalDistortion << std::endl;



    Mat rgbImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/0.png");

    Mat rgbImage_g;

    cvtColor(rgbImage,rgbImage_g,COLOR_BGR2GRAY);

    Mat thermalImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/0.png");




//    std::cout << "Shape: (" << rows << " rows, " << cols << " columns)" << std::endl;
//    // Calculate fx, fy, cx, and cy
    double fx = rgbIntrinsic.at<double>(0,0);
    double fy = rgbIntrinsic.at<double>(1,1);
    double cx = rgbIntrinsic.at<double>(0,2);
    double cy = rgbIntrinsic.at<double>(1,2);
//    cout<<fx<<endl;


    int width = 640;
    int height = 480;

    Eigen::VectorXd x = VectorXd::LinSpaced(width, 0, width - 1);
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


    MatrixXd x_real_imgplane = (x_c.array() - cx) / fx;

    Mat x_real,y_real;
    eigen2cv(x_real_imgplane,x_real);

//    cout<<x_real_imgplane<<endl;
    MatrixXd y_real_imgplane = (y_c.array() - cy) / fy;
    eigen2cv(y_real_imgplane,y_real);

    x_real.convertTo(x_real,CV_64FC1);
    y_real.convertTo(y_real,CV_64FC1);

//
//    cout<<x_real_imgplane<<endl;




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
//    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
//    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
//    pip.start(cfg);
//
//    namedWindow("RGB_IMG", WINDOW_AUTOSIZE);


    rs2::colorizer c;
    // 创建数据管道
    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    // start() 函数返回数据管道的profile
    rs2::pipeline_profile profile = pipe.start(pipe_config);
    // rs2::pipeline_profile profile = pipe.start();

    // 使用数据管道的 profile 获取深度图像像素对应于长度单位（米）的转换比例

    float depth_scale = get_depth_scale(profile.get_device());

    // 选择彩色图像数据流来作为对齐对象
    rs2_stream align_to = RS2_STREAM_COLOR; // 对齐的是彩色图，所以彩色图是不变的
    // // 将深度图对齐到RGB图
    rs2::align align(align_to);

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

    ::evo_irimager_set_palette(1);


    cout<<"p_w,p_h"<<endl;
    cout<<p_w<<","<<p_h<<endl;


    cout<<"Initializing>>>"<<endl;
    sleep(3);


    Mat w_x,w_y;

    Mat RGB_IMG, Depth_IMG,Thermal_IMG,Event_IMG,depthdata,RGBImage,jacobian,projected_image,real_point,real_point_depth;
    vector<Point2d> imagePoints;

    vector<Point3d> real_point3d;

    vector<Point2i> imagePoints_int;






    int count = 0;
    auto start = std::chrono::steady_clock::now();

    while (true)
    {

        Thermal_IMG = optris_frame();

        //////////realsense
        rs2::frameset frameset = pipe.wait_for_frames();

//        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
//        {
//            // 如果profile发生改变，则更新align对象，重新获取深度图像像素到长度单位的转换比例
//            profile = pipe.get_active_profile();
//            align = rs2::align(align_to);
//            depth_scale = get_depth_scale(profile.get_device());
//        }
        auto processed = align.process(frameset);
        rs2::frame aligned_color_frame = processed.get_color_frame();
        rs2::frame aligned_depth_frame = processed.get_depth_frame();

        rs2::frame before_depth_frame = frameset.get_depth_frame().apply_filter(c);
        const int depth_w = aligned_depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = aligned_depth_frame.as<rs2::video_frame>().get_height();
        const int color_w = aligned_color_frame.as<rs2::video_frame>().get_width();
        const int color_h = aligned_color_frame.as<rs2::video_frame>().get_height();


        if (!aligned_depth_frame || !aligned_color_frame)
        {
            continue;
        }


        Mat aligned_depth_image(Size(depth_w, depth_h), CV_16UC1, (void *)aligned_depth_frame.get_data(), Mat::AUTO_STEP);
        Mat aligned_color_image(Size(color_w, color_h), CV_8UC3, (void *)aligned_color_frame.get_data(), Mat::AUTO_STEP);

        cvtColor(aligned_color_image, aligned_color_image, COLOR_RGB2BGR);


        depthdata = aligned_depth_image.clone();
//        depthdata = aligned_depth_image;
//        rgbImage = aligned_color_image.clone();
        RGBImage = aligned_color_image.clone();

        depthdata.convertTo(depthdata, CV_64FC1);


        w_x = x_real.mul(depthdata);
        w_y = y_real.mul(depthdata);
//
        cv::merge(std::vector<cv::Mat>{w_x, w_y, depthdata}, real_point_depth);
        real_point = real_point_depth.reshape(1, 307200);



//        real_point3d = Mat2Point(real_point);
//        bad_obj_points = gen_bad_points(real_point3d);


        cv::projectPoints(real_point, relative_Rvct, relative_T, thermalIntrinsic, thermalDistortion, imagePoints);


        imagePoints_int = Pointf2i(imagePoints);

//        convertScaleAbs(imagePoints, imagePoints);
////
//        flip(imagePoints, imagePoints, 1);

        projected_image = testfunc(imagePoints_int,Thermal_IMG);


        imshow("Fusion",projected_image);
//        imshow("Thermal", Thermal_IMG);
//        imshow("RGB_IMG",RGB_IMG);
//        imshow("TS Pic", Event_IMG);

        waitKey(1);

        count++;


        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        if (elapsed >= 1) {
            std::cout << "FPS is: " << count << std::endl;
            count = 0;
            start = std::chrono::steady_clock::now();
        }





    }
}




//
// Created by emrys on 8/24/23.
//

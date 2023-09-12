
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



// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>












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





vector<cv::Point2i> Pointf2i(vector<cv::Point2d> &src){
    vector<cv::Point2i> dst(src.size());
    for (int i=0; i<dst.size();i++){

        dst[i].x = round(src[i].x);
        dst[i].y = round(src[i].y);

    }
    return dst;
}


int* gen_bad_points(vector<cv::Point3d> real_point3d){
    static int bad_obj_points[307200];
    for (int i = 0; i < 307200; i++) {
        if (real_point3d[i].z == 0) {
            bad_obj_points[i] = 1;
        }
        else bad_obj_points[i] = 0;
    }
    return bad_obj_points;
}

cv::Mat projected_image;
cv::Mat Fusion(vector<cv::Point2i> imagePoints, cv::Mat thermal_data,int bad_obj_points[307200]){

    projected_image = cv::Mat(cv::Size(1,307200),CV_8UC3);
    int row[307200] = {0};
    int col[307200] = {0};


    for (int i = 0; i < 307200; i++){

        row[i] = std::max(0, std::min(imagePoints[i].x, 479));
        col[i] = std::max(0, std::min(imagePoints[i].y, 639));
        projected_image.at<cv::Vec3b>(i,0) = thermal_data.at<cv::Vec3b>(col[i],row[i]);

        if (bad_obj_points[i] == 1)
//        {projected_image_E.at<int>(i,0) = 128;  //for TS
        {projected_image.at<int>(i,0) = 0;  //for Eventoff
        }

    }

    projected_image = projected_image.reshape(3, 480);

    return projected_image;
}


cv::Mat Fusion_E(vector<cv::Point2i> imagePoints, cv::Mat event_data, int bad_obj_points[307200]){

//    Mat projected_image_E = Mat(Size(1,307200),CV_8UC1);


//for Eventoff
    cv::Mat projected_image_E= cv::Mat::zeros(cv::Size(1, 307200), CV_8UC3);


//for TS
//    Mat projected_image_E(cv::Size(1, 307200), CV_8UC1,Scalar(128));

    int row[307200] = {0};
    int col[307200] = {0};



    for (int i = 0; i < 307200; i++){


        row[i] = std::max(0, std::min(imagePoints[i].x, 479));
        col[i] = std::max(0, std::min(imagePoints[i].y, 639));
//        projected_image_E.at<uchar>(i,0) = event_data.at<uchar>(col[i],row[i]);
        projected_image_E.at<cv::Vec3b>(i,0) = event_data.at<cv::Vec3b>(col[i],row[i]);

        if (bad_obj_points[i] == 1)
//        {projected_image_E.at<int>(i,0) = 128;  //for TS
        {projected_image_E.at<int>(i,0) = 0;  //for Eventoff
        }

    }




    projected_image_E = projected_image_E.reshape(3, 480);

    return projected_image_E;
}




vector<cv::Point3d> Mat2Point(cv::Mat &src){
    vector<cv::Point3d> dst;
    for (int i=0; i<307200; i++){
        cv::Point3f p;
        //p.x = src.at<double>(0,i);
        p.x = (double)src.at<cv::Vec3d>(i,0).val[0];
        //p.y = src.at<double>(1,i);
        p.y = (double)src.at<cv::Vec3d>(i,0).val[1];
        //p.z = src.at<double>(2,i);
        p.z = (double)src.at<cv::Vec3d>(i,0).val[2];
        dst.push_back(p);
    }
    return dst;
}





int true_tem(int x){

    int temp = (x*10)+1000;
    return temp;
}


int p_w = 384;
int p_h =288;
int t_w = 382;
int t_h = 288;

std::vector<unsigned char> palette_image(p_w * p_h * 3);
std::vector<unsigned short> thermal_data(t_w * t_h);

cv::Mat optris_frame()
{

    evo_irimager_get_thermal_palette_image(t_w, t_h, &thermal_data[0], p_w, p_h, &palette_image[0]);
    cv::Mat cv_img(cv::Size(p_w, p_h), CV_8UC3, &palette_image[0], cv::Mat::AUTO_STEP);



    for (int y = 0; y < t_h; y++)
    {
        for (int x = 0; x < t_w; x++)
        {
            if (thermal_data[y*t_w + x]< true_tem(25))
            {
                cv_img.at<cv::Vec3b>(y,x) = 0;
            }
        }
    }



//    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
    resize(cv_img,cv_img,cv::Size(640,480));
    return cv_img;

}

string dir_path = "../calib/";
cv::Mat readxml(string name,string index){
    std::string path = dir_path + name + ".xml";
    cv::FileStorage file(path,cv::FileStorage::READ);
    if(!file.isOpened()){
        std::cout<<"Failed to open the xml file "<<path<<std::endl;
    }
    cv::Mat rst;
    file[index]>>rst;
    file.release();
    //std::cout << name << "is:" << rst << std::endl;
    return rst;
}


void real_plane(cv::Mat rgbIntrinsic,cv::Mat& x_real , cv::Mat& y_real ){


    double fx = rgbIntrinsic.at<double>(0,0);
    double fy = rgbIntrinsic.at<double>(1,1);
    double cx = rgbIntrinsic.at<double>(0,2);
    double cy = rgbIntrinsic.at<double>(1,2);
//    cout<<fx<<endl;


    int width = 640;
    int height = 480;

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(width, 0, width - 1);
    Eigen::MatrixXd x_c = Eigen::MatrixXd::Constant(480, 640, x(0));

    // Replicate the values of x in x_c
    for (int row = 0; row < 480; ++row) {
        for (int col = 0; col < 640; ++col) {
            x_c(row, col) = x(col);
        }
    }


    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(height, 0, height - 1);

    Eigen::MatrixXd y_c = Eigen::MatrixXd::Constant(640, 480, y(0));

    // Replicate the values of y in y_c
    for (int row = 0; row < 640; ++row) {
        for (int col = 0; col < 480; ++col) {
            y_c(row, col) = y(col);
        }
    }

    y_c = y.replicate(1, width);


    Eigen::MatrixXd x_real_imgplane = (x_c.array() - cx) / fx;


    eigen2cv(x_real_imgplane,x_real);


    Eigen::MatrixXd y_real_imgplane = (y_c.array() - cy) / fy;
    eigen2cv(y_real_imgplane,y_real);

    x_real.convertTo(x_real,CV_64FC1);
    y_real.convertTo(y_real,CV_64FC1);



}






cv::Mat cut_img(cv::Size(280,340),CV_8UC3,cv::Scalar(0,0,0));
cv::Mat cut_depth(cv::Size(280,340),CV_16UC1,cv::Scalar(0));


void create_new_Mat(cv::Mat src,cv::Mat &dst,int h,int l,int j,int k){
    for (int row = 0;row<(l-h);row++){
        for(int col = 0;col<(k-j);col++){
            dst.at<cv::Vec3b>(row,col) = src.at<cv::Vec3b>(row+h,col+j);
        }
    }
}
void create_new_Mat_depth(cv::Mat src,cv::Mat &dst,int h,int l,int j,int k){
    for (int row = 0;row<(l-h);row++){
        for(int col = 0;col<(k-j);col++){
            dst.at<ushort>(row,col) = src.at<ushort>(row+h,col+j);
        }
    }
}





pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));




void pcl_generator (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,cv::Mat &rgb, cv::Mat &depth) {

    double cx = 381.0;
    double cy = 379.2;
    double fx = 316.7;
    double fy = 233.7;
    double depthScale = 1000.0;


    typedef pcl::PointXYZRGB PointT;
//    typedef pcl::PointCloud<PointT> PointCloud;

    cloud->clear();



//    PointCloud::Ptr pointCloud(new PointCloud);
    for (int v = 0; v < rgb.rows; v++)
        for (int u = 0; u < rgb.cols; u++) {


            unsigned int d = depth.ptr<unsigned short>(v)[u];
            if (d == 0)
                continue;
            PointT p;
            p.z = double(d) / depthScale;
            p.x = (u - cx) * p.z / fx;
            p.y = (v - cy) * p.z / fy;
            p.b = rgb.data[v * rgb.step + u * rgb.channels()];
            p.g = rgb.data[v * rgb.step + u * rgb.channels() + 1];
            p.r = rgb.data[v * rgb.step + u * rgb.channels() + 2];
            cloud->points.push_back(p);
        }


}

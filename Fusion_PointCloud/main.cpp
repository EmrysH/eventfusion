#include "Fusion_Pointcloud.h"



using namespace std;
using namespace cv;
using namespace Eigen;





uint8_t * pSensorBuffer = new uint8_t[CELEX5_PIXELS_NUMBER];

cv::Mat rgbImage;
Mat event_frame()
{

    //Time Surface
//    std::vector<EventData> vecEvent;
//    pCeleX5->getEventDataVector(vecEvent);
//
//    cv::Mat sae= cv::Mat::zeros(cv::Size(1280, 800), CV_64F);
//
//    int dataSize = vecEvent.size();
//
//    vector<int> p(dataSize);
//    vector<int> x(dataSize);
//    vector<int> y(dataSize);
//    vector<double> ts(dataSize);
//
//    double tau = 50e-3 ;
//
//    for (int i = 0; i < dataSize; i++) {
//        p[i] = vecEvent[i].polarity;
//        x[i] = vecEvent[i].col;
//        y[i] = vecEvent[i].row;
//        ts[i] = vecEvent[i].tOffPixelIncreasing * 0.000001;
//    }
//
//    double t_ref = ts.back();
//
//    for (int i = 0; i < ts.size(); i++)
//    {
//
//
//        if (p[i] > 0) {
//            sae.at<double>(800-y[i]-1,1280-x[i]-1) = exp(-(t_ref - ts[i]) / tau);
//        }
//        if (p[i] < 0 ){
//            sae.at<double>(800-y[i]-1,1280-x[i]-1) = -exp(-(t_ref - ts[i]) / tau);
//        }
//
//    }
//
//    cv::normalize(sae, sae, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    flip(sae, sae, 1);
//    resize(sae,sae,Size(640,480));
//
//    return sae;




//    pCeleX5->getFullPicBuffer(pSensorBuffer); //full pic
//    cv::Mat matFullPic(800, 1280, CV_8UC1, pSensorBuffer);
//
//
//    resize(matFullPic,matFullPic,Size(640,480));
//    return matFullPic;

    pCeleX5->getEventPicBuffer(pSensorBuffer, CeleX5::EventBinaryPic);
    cv::Mat event(800, 1280, CV_8UC1, pSensorBuffer);
//
    cv::Mat mask;
//    cv::threshold(event, mask, 254, 255, cv::THRESH_BINARY_INV);
    mask = ( event ==255 );
    // Create a RGB version of the gray image

    cv::cvtColor(event, rgbImage, cv::COLOR_GRAY2RGB);

    // Set the white pixels to green in the RGB image using the mask
    rgbImage.setTo(cv::Scalar(0, 255, 0), mask);

//    event = rgbImage;

//    resize(event,event,Size(640,480));
    resize(rgbImage,rgbImage,Size(640,480));
    return rgbImage;



}





cv::Mat rgbIntrinsic_E, rgbDistortion_E, rgbIntrinsic_T,rgbDistortion_T,thermalIntrinsic, thermalDistortion, eventIntrinsic,eventDistortion,relative_R, relative_T,relative_R_Event, relative_T_Event,trans_mat,relative_Rvct,trans_mat_Event,relative_Rvct_Event;

int main(int argc, char * argv[]) {



    rgbIntrinsic_T = readxml("optris_rs_RGB_intrinsic","rgbIntrinsic");
//    rgbDistortion_T = readxml("optris_rs_RGB_distortion","rgbDistortion");



    rgbIntrinsic_E = readxml("event_realsense_RGB_intrinsic","rgbIntrinsic");
//    rgbDistortion_E = readxml("event_realsense_RGB_distortion","rgbDistortion");

    thermalIntrinsic = readxml("optris_rs_Thermal_intrinsic","thermalIntrinsic");
    thermalDistortion = readxml("optris_rs_Thermal_distortion","thermalDistortion");

    eventIntrinsic = readxml("event_realsense_event_intrinsic","eventIntrinsic");
    thermalDistortion = readxml("event_realsense_event_distortion","eventDistortion");


    relative_R_Event = readxml("event_rgb_Relative_rotation_matrix","relative_R");
    relative_T_Event = readxml("event_rgb_Relative_translation_matrix","relative_T");

    hconcat(relative_R_Event, relative_T_Event, trans_mat_Event);
    Rodrigues(relative_R_Event, relative_Rvct_Event);


    relative_R = readxml("optris_rs_Ralative_rotation_matrix","relative_R");
    relative_T = readxml("optris_rs_Ralative_translation_matrix","relative_T");

    hconcat(relative_R, relative_T, trans_mat);
    Rodrigues(relative_R, relative_Rvct);

//    std::cout << thermalDistortion << std::endl;

    Mat rgbImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/0.png");

    Mat rgbImage_g;

    cvtColor(rgbImage,rgbImage_g,COLOR_BGR2GRAY);

    Mat thermalImage = imread("/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/0.png");

    Mat eventImage = imread("/home/emrys/Project/Calibration_Pics/RGB_E/Event/RAW/0.png");


//    std::cout << "Shape: (" << rows << " rows, " << cols << " columns)" << std::endl;
//    // Calculate fx, fy, cx, and cy

    Mat x_real_T, y_real_T,x_real_E,y_real_E;
    real_plane(rgbIntrinsic_T,x_real_T,y_real_T);
    real_plane(rgbIntrinsic_E,x_real_E,y_real_E);





////        Initializing Celex5

    if (NULL == pCeleX5)
        return 0;

    pCeleX5->openSensor(CeleX5::CeleX5_MIPI);
    pCeleX5->setFpnFile(FPN_PATH);


//    pCeleX5->setSensorFixedMode(CeleX5::Event_Intensity_Mode);
    pCeleX5->setSensorFixedMode(CeleX5::Event_Off_Pixel_Timestamp_Mode);
//    pCeleX5->setSensorFixedMode(CeleX5::Full_Picture_Mode);
//    pCeleX5->disableFrameModule();
//    pCeleX5->disableIMUModule();
//    pCeleX5->disableEventCountSlice();
//    pCeleX5->setEventFrameTime(8333);
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
    pCeleX5->setRotateType(2);

//    sleep(3);


////    Initializing RealSense

    rs2::colorizer c;

    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);

    rs2::pipeline_profile profile = pipe.start(pipe_config);
    // rs2::pipeline_profile profile = pipe.start();


    float depth_scale = get_depth_scale(profile.get_device());
    rs2_stream align_to = RS2_STREAM_COLOR;

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


    //define pointcloud data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//    cloud->height = 480;
//    cloud->width = 640;
//    cloud->resize(640*480);
    // define point cloud filter
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    cloud->height = 280;
    cloud->width = 390;
    cloud->resize(390*280);

    // define filter
    pcl::PassThrough<pcl::PointXYZRGB> pass;



    // set viewer parameter
    viewer->setCameraPosition(0, 0, -3.0, 0, -1, 0);
    viewer->addCoordinateSystem(1);
    bool showFPS = true;
    viewer->setShowFPS(showFPS);





    Mat w_x_T,w_y_T,w_x_E,w_y_E;

    int* bad_obj_points;
    int* bad_obj_points_E;

    Mat RGB_IMG, Depth_IMG,Thermal_IMG,Event_IMG,depthdata,RGBImage,jacobian,projected_image,projected_image_Event,real_point_T,real_point_E,real_point_depth_T,real_point_depth_E;

    cv::Mat RGB_T_image,RGB_D_T_E_Image;

    vector<Point2d> imagePoints;
    vector<Point2d> imagePoints_Event;

    vector<Point3d> real_point3d;
    vector<Point3d> real_point3d_Event;

    vector<Point2i> imagePoints_int;
    vector<Point2i> imagePoints_int_Event;


    cout<<"Initializing>>>"<<endl;
    sleep(3);


    int pcd_count = 0;
//    int count = 0;
//    auto start = std::chrono::steady_clock::now();

    while (true)
    {

        Thermal_IMG = optris_frame();
        Event_IMG = event_frame();

        //////////realsense
        rs2::frameset frameset = pipe.wait_for_frames();

        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            profile = pipe.get_active_profile();
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }
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


        w_x_T = x_real_T.mul(depthdata);
        w_y_T = y_real_T.mul(depthdata);
        cv::merge(std::vector<cv::Mat>{w_x_T, w_y_T, depthdata}, real_point_depth_T);
        real_point_T = real_point_depth_T.reshape(1, 307200);
        real_point3d = Mat2Point(real_point_T);
        bad_obj_points = gen_bad_points(real_point3d);

        w_x_E = x_real_E.mul(depthdata);
        w_y_E = y_real_E.mul(depthdata);
        cv::merge(std::vector<cv::Mat>{w_x_E, w_y_E, depthdata}, real_point_depth_E);
        real_point_E = real_point_depth_E.reshape(1, 307200);
        real_point3d_Event = Mat2Point(real_point_E);
        bad_obj_points_E = gen_bad_points(real_point3d_Event);


/////////////////RGB_T_FUSION//////////////////


        cv::projectPoints(real_point_T, relative_Rvct, relative_T, thermalIntrinsic, thermalDistortion, imagePoints);


        imagePoints_int = Pointf2i(imagePoints);

        projected_image = Fusion(imagePoints_int,Thermal_IMG,bad_obj_points);

        cv::Mat mask;
        cv::cvtColor(projected_image, mask, cv::COLOR_BGR2GRAY);

        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY_INV);

        cv::Mat foreground;
        cv::bitwise_and(RGBImage, RGBImage, foreground, mask);


        cv::Mat mask_inv;
        cv::bitwise_not(mask, mask_inv);

        cv::Mat background;
        cv::bitwise_and(projected_image, projected_image, background, mask_inv);
        cv::add(foreground, background, RGB_T_image);





        ///////////////////END//////////////////////////////////



        cv::projectPoints(real_point_E, relative_Rvct_Event, relative_T_Event, eventIntrinsic, eventDistortion, imagePoints_Event);

        imagePoints_int_Event = Pointf2i((imagePoints_Event));

//        projected_image_Event = Fusion_E(imagePoints_int_Event,Event_IMG,bad_obj_points);
        projected_image_Event = Fusion_E(imagePoints_int_Event,Event_IMG,bad_obj_points_E);

        cv::Mat maskE, Event_BGR;


//        cv::cvtColor(projected_image_Event, Event_BGR, cv::COLOR_GRAY2BGR);
        cv::cvtColor(projected_image_Event, Event_BGR, cv::COLOR_RGB2BGR);

        cv::cvtColor(Event_BGR, maskE, cv::COLOR_BGR2GRAY);



        //mask for Eventoff
        cv::threshold(maskE, maskE, 1, 255, cv::THRESH_BINARY_INV);


        //mask for TS
//        maskE = (maskE>100 & maskE<130);

        cv::Mat foregroundE;
//        cv::bitwise_and(RGBImage, RGBImage, foregroundE, maskE);
        cv::bitwise_and(RGB_T_image, RGB_T_image, foregroundE, maskE);


        cv::Mat mask_invE;
        cv::bitwise_not(maskE, mask_invE);
//
        cv::Mat backgroundE;


        cv::bitwise_and(Event_BGR, Event_BGR, backgroundE, mask_invE);
        cv::add(foregroundE, backgroundE, RGB_D_T_E_Image);



        create_new_Mat(RGB_D_T_E_Image,cut_img,100,380,150,490);
        create_new_Mat_depth(aligned_depth_image,cut_depth,100,380,150,490);


        pcl_generator(cloud,cut_img,cut_depth);


        viewer->removeAllPointClouds();
        viewer->addPointCloud(cloud,"Cloud Viewer");
        viewer->updatePointCloud(cloud,"Cloud Viewer");
        viewer->spinOnce(0.001);




    }
}




//
// Created by emrys on 8/24/23.
//

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char * argv[])
{
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;

    // Configure and start the pipeline
    p.start();

    Mat RGB_IMG;
    while (true)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        rs2::frame color = frames.get_color_frame();
        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get the depth frame's dimensions
        auto width = depth.get_width();
        auto height = depth.get_height();

        const int w = color.as<rs2::video_frame>().get_width();
        const int h = color.as<rs2::video_frame>().get_height();

        Mat image(Size(w, h), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
        //Dontla 20210827

        cvtColor(image, RGB_IMG, CV_BGR2RGB);//转换


        const auto window_name = "RGB_IMG";
        namedWindow(window_name, WINDOW_AUTOSIZE);

        imshow(window_name,RGB_IMG);
        waitKey(1);

//        // Query the distance from the camera to the object in the center of the image
//        float dist_to_center = depth.get_distance(width / 2, height / 2);
//
//        // Print the distance
//        std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
    }

    return EXIT_SUCCESS;
}

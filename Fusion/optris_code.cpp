
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui_c.h>

//--Code for displaying image -----------------
#include <opencv2/opencv.hpp>

#include "libirimager/direct_binding.h"

//---------------------------------------------

using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{

    ::evo_irimager_usb_init("/home/emrys/CLionProjects/Fusion/22092003.xml",0,0);


    int err;
    int p_w;
    int p_h;
    if((err = ::evo_irimager_get_palette_image_size(&p_w, &p_h)) != 0)
    {
        std::cerr << "error on evo_irimager_get_palette_image_size: " << err << std::endl;
        exit(-1);
    }

    int t_w;
    int t_h;

    //!width of thermal and palette image can be different due to stride of 4 alignment
    if((err = ::evo_irimager_get_thermal_image_size(&t_w, &t_h)) != 0)
    {
        std::cerr << "error on evo_irimager_get_palette_image_size: " << err << std::endl;
        exit(-1);
    }

    std::vector<unsigned char> palette_image(p_w * p_h * 3);
    std::vector<unsigned short> thermal_data(t_w * t_h);

    while (true)
    {
        if((err = ::evo_irimager_get_thermal_palette_image(t_w, t_h, &thermal_data[0], p_w, p_h, &palette_image[0]))==0)
        {
//            unsigned long int mean = 0;
//            //--Code for calculation mean temperature of image -----------------
//            for (int y = 0; y < t_h; y++)
//            {
//                for (int x = 0; x < t_w; x++)
//                {
//                    mean += thermal_data[y*t_w + x];
//                }
//            }
//            std::cout << (mean / (t_h * t_w)) / 10.0 - 100 << std::endl;
            //---------------------------------------------

            //--Code for displaying image -----------------
            cv::Mat cv_img(cv::Size(p_w, p_h), CV_8UC3, &palette_image[0], cv::Mat::AUTO_STEP);
            cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
            cv::imshow("palette image daemon", cv_img);
            cv::waitKey(1);
            //---------------------------------------------
        }
        else
        {
            std::cerr << "failed evo_irimager_get_thermal_palette_image: " << err << std::endl;
        }

    } while(cv::waitKey(1) != 'q');//cvGetWindowHandle("palette image daemon"));

    ::evo_irimager_terminate();

    return 0;
}

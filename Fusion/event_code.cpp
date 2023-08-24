
#include <opencv2/opencv.hpp>
#include "celex5.h"
#include "celex5datamanager.h"

#define MAT_ROWS 800
#define MAT_COLS 1280
#define FPN_PATH    "../Samples/config/FPN_2.txt"
#include<unistd.h>
#include <signal.h>



CeleX5 *pCeleX5 = new CeleX5;

using namespace std;
using namespace cv;
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



void SensorDataObserver::onFrameDataUpdated(CeleX5ProcessedData* pSensorData)
{
    if (NULL == pSensorData)
        return;

        std::vector<EventData> vecEvent;
        pCeleX5->getEventDataVector(vecEvent);
        cv::Mat matPolarity(800, 1280, CV_8UC1, cv::Scalar::all(128));
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
        int row = 0, col = 0;
        for (int i = 0; i < ts.size(); i++)
        {

            row = 799 - vecEvent[i].row;
            col = 1279 - vecEvent[i].col;
            if (vecEvent[i].polarity == 1)
            {
                matPolarity.at<uchar>(row, col) = 255;
            }
            else if (vecEvent[i].polarity == -1)
            {
                matPolarity.at<uchar>(row, col) = 0;
            }
//            else
//            {
//                matPolarity.at<uchar>(row, col) = 128;
//            }

            if (p[i] > 0) {
                sae.at<double>(800-y[i]-1,1280-x[i]-1) = exp(-(t_ref - ts[i]) / tau);
            }
            if (p[i] < 0 ){
                sae.at<double>(800-y[i]-1,1280-x[i]-1) = -exp(-(t_ref - ts[i]) / tau);
            }

//            mat.at<uchar>(800 - vecEvent[i].row - 1, 1280 - vecEvent[i].col - 1) = 255;
//            sae.at<uchar>(800 - y[i] - 1, 1280 - x[i] - 1) = 255;
        }

//            cout<<ts.back()<<endl;
            cv::normalize(sae, sae, 0, 255, cv::NORM_MINMAX, CV_8U);

            cv::imshow("TS Pic", sae);
//            cv::imshow("Event Binary Pic", matPolarity);
            cv::waitKey(1);

}



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


int main()
{
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


    while (true)
    {

        usleep(1000);

    }
    return 1;
}




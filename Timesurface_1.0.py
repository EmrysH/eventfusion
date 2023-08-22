
import time

import cv2

from pylibcelex import PyCeleX5

import numpy as np

from matplotlib import pyplot as plt

import matplotlib.animation as animation

import threading

import pandas as pd

BIN_FILE = "/home/event/Desktop/A0001_P0001_S00.bin"

celex5 = PyCeleX5.PyCeleX5(debug=True)


def read_bin_file():
    # 必须先打开翻录模式
    celex5.startRippingBinFile()
    # 设置图片保存路径
    celex5.enableImageFileOutput("/tmp/test/images/")
    # 设置Event数据保存路径
    celex5.enableEventDataOutput("/tmp/test/event.csv")
    # 然后打开BinFile
    celex5.openBinFile(BIN_FILE)
    # 循环读取到结束
    while not celex5.readBinFileData() or not celex5.rippingBinFileFinished():
        pass
    time.sleep(1)
    # 用好后要结束
    celex5.stopRippingBinFile()

def event_config():
    print("{} * {} = {}".format(PyCeleX5.WIDTH, PyCeleX5.HEIGHT, PyCeleX5.RESOLUTION))
    celex5.openSensor(PyCeleX5.DeviceType.CeleX5_MIPI)
    celex5.isSensorReady()
    celex5.getRotateType()

    # sensorMode = PyCeleX5.CeleX5Mode.Full_Picture_Mode
    # celex5.setSensorFixedMode(sensorMode)
    # sensorMode = PyCeleX5.CeleX5Mode.Event_Off_Pixel_Timestamp_Mode
    # celex5.setSensorFixedMode(sensorMode)

    sensorMode = PyCeleX5.CeleX5Mode.Event_Intensity_Mode
    celex5.setSensorFixedMode(sensorMode)

    celex5.setFpnFile( "/home/emrys/PycharmProjects/Event_RGB_IMG/FPN_2.txt")
    celex5.getSensorFixedMode()
    celex5.getSensorLoopMode(1)
    celex5.getSensorLoopMode(2)
    celex5.getSensorLoopMode(3)
    celex5.isLoopModeEnabled()
    celex5.getEventFrameTime()
    celex5.getOpticalFlowFrameTime()
    celex5.getThreshold()
    celex5.getBrightness()
    # celex5.getContrast()
    celex5.getClockRate()
    celex5.getEventDataFormat()
    celex5.disableFrameModule()
    celex5.isFrameModuleEnabled()
    celex5.isEventStreamEnabled()
    celex5.isIMUModuleEnabled()
    celex5.isEventDenoisingEnabled()
    celex5.isFrameDenoisingEnabled()
    celex5.isEventCountSliceEnabled()
    celex5.isEventOpticalFlowEnabled()

    # celex5.setEventFrameTime(8333)
    # celex5.setRotateType(2)


if __name__ == "__main__":

    event_config()
    print("Initializing in 3 Seconds >>>>>>>>")
    time.sleep(3)
    tau = 50e-3


while True:

        Event_data = celex5.getEventDataVector()

        sae = np.full((800, 1280), 0, dtype=np.float32)
        # mat = np.zeros((800, 1280), dtype=np.uint8)

        ts = [None] * len(Event_data)
        x = [None] * len(Event_data)
        y = [None] * len(Event_data)
        p = [None] * len(Event_data)

        i = 0
        for e in Event_data:
            # mat[800 - e.row - 1, 1280 - e.col - 1] = 255

            ts[i] = e.tOffPixelIncreasing *0.000001
            x[i] = e.col
            y[i] = e.row
            p[i] = e.polarity
            i += 1

        t_ref = ts[-1]
        # print(ts)
        # print(t_ref)
        # print(ts[0],x[0],y[0],p[0])
        # print(ts[-1],x[-1],y[-1],p[-1])
        for j in range(len(ts)):
            if (p[j] > 0):
                sae[y[j], x[j]] = np.exp(-(t_ref - ts[j]) / tau)
                # print(sae[y[j],x[j]])
                # print(t_ref-ts[j])
            if(p[j] < 0 ):
                sae[y[j], x[j]] = -np.exp(-(t_ref - ts[j]) / tau)

            # sae[y[j], x[j]] = np.exp(-(t_ref - ts[j]) / tau)


        sae = cv2.flip(sae, 1)
        sae = cv2.flip(sae, 0)
        # print(sae)

        sae = cv2.normalize(sae, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        sae = cv2.resize(sae,(640,400))


        cv2.imshow("TimeSurface", sae)
        # cv2.imshow("Event Binary Pic", mat)
        cv2.waitKey(10)


        # img = np.zeros(shape=(800,1280), dtype=int)
        #
        # for i in range(len(Event_data)):
        #     img[y[i], x[i]] = (2 * p[i] - 1)
        #
        # img = cv2.flip(img, 1)
        # img = cv2.flip(img, 0)
        #
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(sae,cmap="gray")
        # axes[0].set_title('time surface')
        #
        # # draw image
        # axes[1].imshow(img,cmap="gray")
        # axes[1].set_title('Image 2')
        #
        # plt.tight_layout()
        # plt.show()



import time

import cv2

from pylibcelex import PyCeleX5

import numpy as np

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


    celex5.disableFrameModule()

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
    celex5.isFrameModuleEnabled()
    celex5.isEventStreamEnabled()
    celex5.isIMUModuleEnabled()
    celex5.isEventDenoisingEnabled()
    celex5.isFrameDenoisingEnabled()
    celex5.isEventCountSliceEnabled()
    celex5.isEventOpticalFlowEnabled()

    # celex5.setEventFrameTime(8333)
    celex5.setRotateType(2)









if __name__ == "__main__":



    event_config()

    while True:
        event = celex5.getEventPicBuffer()

        Event_data = celex5.getEventDataVector()

        # dataSize = len(Event_data)
        #
        # print(len(Event_data))

        mat = np.zeros((800, 1280), dtype=np.uint8)

        for i, event in enumerate(Event_data):
            mat[800 - event.row - 1, 1280 - event.col - 1] = 255
            # print(Event_data[i].tOffPixelIncreasing)
        cv2.imshow("Event Binary Pic", mat);
        cv2.waitKey(1);







#! /usr/bin/env python3
from ctypes.util import find_library
import numpy as np
import ctypes as ct
import cv2
import os
import pyrealsense2 as rs
import pyOptris as optris
import time



## setup realsense
# Configure depth and color streams


# Vector for 2D pixels points for RGB camera
rgbTwoDPoints = []

# Vector for 2D pixels points for Thermal camera
thermalTwoDPoints = []

# Define the dimensions of checkerboard
CHECKERBOARD = (7, 5)

index_p = 0

def detectCircle(colorImg) :
    colorImgCopy = cv2.cvtColor(colorImg, cv2.COLOR_BGR2BGRA)
    grayColor = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

    #bolb detect
    params = cv2.SimpleBlobDetector_Params()
    # params.maxArea = 500
    # params.minArea = 40
    # params.minDistBetweenBlobs = 20
    # detector = cv2.SimpleBlobDetector_create(params)
    detector = cv2.SimpleBlobDetector_create()
    try :
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findCirclesGrid(grayColor, CHECKERBOARD, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector)    #For symmetric circle calibration board
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board

        #if find the circle calibration board, return the x,y of point
        if ret == True:
            # Refining pixel coordinates
            # for given 2d points.
            # Draw and display the corners
            circle_det = cv2.drawChessboardCorners(colorImg, CHECKERBOARD, corners, ret)
            return circle_det, corners ,colorImgCopy     #two parameter return, one is image, the other is pixel coordinate

        # if not find the circle calibration board, return coordinate is none
        else :
            return  colorImg ,[],colorImgCopy
    except :
        return  colorImg ,[],colorImgCopy

def detectCircle_T(colorImg):
    colorImgCopy = cv2.cvtColor(colorImg, cv2.COLOR_BGR2BGRA)
    grayColor = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

    # bolb detect
    params = cv2.SimpleBlobDetector_Params()
    # params.maxArea = 500
    # params.minArea = 40
    # params.minDistBetweenBlobs = 20
    # detector = cv2.SimpleBlobDetector_create(params)
    detector = cv2.SimpleBlobDetector_create()
    try:
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findCirclesGrid(grayColor, CHECKERBOARD, cv2.CALIB_CB_SYMMETRIC_GRID,
                                           blobDetector=detector)  # For symmetric circle calibration board
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board

        # if find the circle calibration board, return the x,y of point
        if ret == True:
            # Refining pixel coordinates
            # for given 2d points.
            # Draw and display the corners
            circle_det = cv2.drawChessboardCorners(colorImg, CHECKERBOARD, corners, ret)
            return circle_det, corners, colorImgCopy  # two parameter return, one is image, the other is pixel coordinate

        # if not find the circle calibration board, return coordinate is none
        else:
            return colorImg, [], colorImgCopy
    except:
        return colorImg, [], colorImgCopy

if __name__ == "__main__":


        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)
        point_coordinate = []

        DLL_path = "/usr/lib/libirdirectsdk.so"
        optris.load_DLL(DLL_path)

        # USB connection initialisation
        optris.usb_init("22092003.xml")

        optris.set_palette(3)

        w, h = optris.get_palette_image_size()



        while True:


            frames = pipeline.wait_for_frames()
            color_rgb_frame = frames.get_color_frame()

            if not color_rgb_frame:
                continue


            color_rgb_image = np.asanyarray(color_rgb_frame.get_data())
            circle_rgb, corner_rgb, raw_rgb = detectCircle(color_rgb_image)

            cv2.imshow('Realsense D455', circle_rgb)


            Thermal = optris.get_palette_image(w, h)
            Thermal = cv2.resize(Thermal,(640,480))

            circle_thermal, Corner_thermal, raw_thermal = detectCircle_T(Thermal)

            cv2.imshow('image',circle_thermal)

            # cv2.imshow('image',circle_thermal)



            if cv2.waitKey(1) & 0xFF == ord('s'):


                if len(corner_rgb) != 0 and len(Corner_thermal) != 0:
                    rgbTwoDPoints.append(corner_rgb)
                    thermalTwoDPoints.append(Corner_thermal)

                    cv2.imwrite('/home/emrys/Project/Calibration_Pics/RGB_T/RGB/' + str(index_p) + '.png',
                                circle_rgb)
                    cv2.imwrite('/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/' + str(index_p) + '.png',
                                circle_thermal)
                    cv2.imwrite('/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/' + str(index_p) + '.png',
                                raw_rgb)
                    cv2.imwrite('/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/' + str(index_p) + '.png',
                                raw_thermal)
                    print("take picture %s" % index_p)
                    index_p += 1

                else:  # have not detect the feature point
                    print("Undetect feature point...")




            elif index_p == 55:  # ESC and save coordinate datas if press Esc
                rgbTwoDPoints = np.array(rgbTwoDPoints)
                np.save("/home/emrys/Project/Calibration_Pics/RGB_T/Data/RGB_Coordinate.npy", rgbTwoDPoints)
                thermalTwoDPoints = np.array(thermalTwoDPoints)
                np.save("/home/emrys/Project/Calibration_Pics/RGB_T/Data/Thermal_coordinate.npy", thermalTwoDPoints)
                break

        pipeline.stop()
        optris.terminate()
        cv2.destroyAllWindows()

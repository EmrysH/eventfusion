
## import necessary module
import cv2
import numpy as np
from numpy.lib.format import _filter_header

## import realsense module
import pyrealsense2 as rs

## import IR module
import pyOptris as optris

import threading
import time

# cv2.namedWindow('fusion image', cv2.WINDOW_NORMAL)

img_num = 0

DLL_path = "/usr/lib/libirdirectsdk.so"
optris.load_DLL(DLL_path)
optris.usb_init("22092003.xml")


## read the datas ##
# RGBD data
rgbIntrinsic = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_RGB_intrinsic.npy")
rgbDistortion = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_RGB_distortion.npy")

# thermal data
thermalIntrinsic = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Thermal_intrinsic.npy")
thermalDistortion = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Thermal_distortion.npy")

# rotation and translation data
relative_R = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_rotation_matrix.npy")
relative_T = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_translation_matrix.npy")
trans_mat = np.c_[relative_R, relative_T]  # creat relative translation matrix
relative_Rvct = (cv2.Rodrigues(relative_R))[0]
## read image of RGB and thermal camera and to get size of images
rgbImage = cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/0.png')
rgbImage_g = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)  # graylize
thermalImage = cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/0.png')

# thermalImage_g = cv2.cvtColor(thermalImage, cv2.COLOR_BGR2GRAY)   #graylize
# rgbShape = rgbImage_g.shape[::-1]
# thermalShape = thermalImage.shape[::-1]





optris.set_palette(1)
w_t, h_t = optris.get_thermal_image_size()
w_p, h_p = optris.get_palette_image_size()

# Getting the depth sensor's depth scale (see rs-align example for explanation)



thermal_img = []
thermal_data = []
depthdata = []
RGBImage = []


def get_rgbd_stream():  # with depth and color aligned
    global depthdata, RGBImage
    while True:
        pc = rs.pointcloud()
        points = rs.points()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # depth_image = np.multiply(depth_scale,depth_image)
        color_image = np.asanyarray(color_frame.get_data())

        pc.map_to(color_frame)
        points = pc.calculate(aligned_depth_frame)
        depthdata = depth_image
        rgbImage = color_image
        RGBImage = np.asanyarray(rgbImage)







# Palette:384*288 Thermal: 382*288
def get_thermal_stream():
    global thermal_img, thermal_data
    while True:
        thermal_img_raw = optris.get_thermal_image(w_t, h_t) # get raw thermal data ,16bit, Kelvin temperature scale
        thermal_img_gray = optris.get_palette_image(w_p, h_p)
        # thermal_img_gray = cv2.cvtColor(thermal_img_gray, cv2.COLOR_BGR2GRAY)
        thermal_img = cv2.resize(thermal_img_gray, (640, 480))
        thermal_data = cv2.resize(thermal_img_raw, (640, 480))
        code_terms = np.where(thermal_data < true_temp(25))

        rows = code_terms[0]
        cols = code_terms[1]
        thermal_img[rows,cols] = 0

def img_process():
    while True:
        w_x = x_real_imgplane * depthdata
        w_y = y_real_imgplane * depthdata
        real_point = np.stack((w_x, w_y, depthdata), axis=-1)
        real_point = real_point.reshape(307200, 3)
        bad_obj_points = np.where(real_point[:, 2] == 0)
        # thermal_1D = thermalImage.reshape(307200)

        imagePoints, jacobian = cv2.projectPoints(real_point, relative_Rvct, relative_T, thermalIntrinsic,
                                                  thermalDistortion)
        imagePoints = imagePoints[:, 0, :]
        imagePoints = np.round((imagePoints).astype(int))
        imagePoints = np.flip(imagePoints, axis=1)
        row = np.clip(imagePoints[:, 0], 0, 479)
        col = np.clip(imagePoints[:, 1], 0, 639)

        projected_image = thermal_img[row, col]

        # print([row,col])

        projected_image[bad_obj_points] = 0

        projected_image = projected_image.reshape(480, 640, 3)

        mask = cv2.cvtColor(projected_image, cv2.COLOR_BGR2GRAY)
        # mask = thermalImage
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

        # Apply the mask to imageA
        foreground = cv2.bitwise_and(RGBImage, RGBImage, mask=mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Get the background by removing the foreground from imageB
        background = cv2.bitwise_and(projected_image, projected_image, mask=mask_inv)

        # print(foreground.shape)
        # print(background.shape)

        # Combine the foreground and background
        RGB_T_image = cv2.add(foreground, background)

        RGB_T_image = RGB_T_image[102:378, 135:503]
        cv2.namedWindow("Fusion")
        cv2.imshow("Fusion", RGB_T_image)
        key = cv2.waitKey(10) & 0xFF
        if key & 0xFF == 27:
            break



def true_temp(x):

    temprature = (x*10)+1000

    return temprature

## from pixel coordinate to image coordinate
fx = rgbIntrinsic[0, 0]
fy = rgbIntrinsic[1, 1]
cx = rgbIntrinsic[0, 2]
cy = rgbIntrinsic[1, 2]
x = np.arange(0, 640)
y = np.arange(0, 480)
x_c = np.tile(x, (480, 1))
y_c = np.tile(y, (640, 1))
y_c = np.transpose(y_c)
x_real_imgplane = (x_c - cx) / fx
y_real_imgplane = (y_c - cy) / fy







thermal_stream_thread = threading.Thread(target=get_thermal_stream)
rgbd_stream_thread = threading.Thread(target=get_rgbd_stream)
img_process_thread = threading.Thread(target=img_process)


if __name__ == "__main__":
    # try :
    ## setup realsense
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    rgbd_stream_thread.start()
    thermal_stream_thread.start()

    print("initallizing....")
    time.sleep(5)

    img_process_thread.start()





    # # finally :
    # # Stop streaming
    # pipeline.stop()
    # img_num = 0
    # # shutdown thermal camera
    # optris.terminate()
    # # close all windows
    # cv2.destroyAllWindows()
    # print("close all successfully, ending......")

import sys

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



from pylibcelex import PyCeleX5
celex5 = PyCeleX5.PyCeleX5(debug=True)
import open3d as o3d


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

EventIntrinsic = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_event_intrinsic.npy")
EventDistortion = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_event_distortion.npy")

# rotation and translation data
relative_R_Thermal = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_rotation_matrix.npy")
relative_T_Thermal = np.load("/home/emrys/Project/Calibration_Pics/RGB_T/Data/optris_rs_Ralative_translation_matrix.npy")
trans_mat = np.c_[relative_R_Thermal, relative_T_Thermal]  # creat relative translation matrix
relative_Rvct_Thermal = (cv2.Rodrigues(relative_R_Thermal))[0]


relative_R_Event = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_rgb_Relative_rotation_matrix.npy")
relative_T_Event = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_rgb_Relative_translation_matrix.npy")
trans_mat_Event = np.c_[relative_R_Event, relative_T_Event]  # creat relative translation matrix
relative_Rvct_Event = (cv2.Rodrigues(relative_R_Event))[0]



## read image of RGB and thermal camera and to get size of images
rgbImage = cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_T/RGB/RAW/0.png')
rgbImage_g = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)  # graylize
thermalImage = cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_T/Thermal/RAW/0.png')
EventImage= cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_E/Event/RAW/0.png')

# Getting the depth sensor's depth scale (see rs-align example for explanation)

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

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)
test_frames = pipeline.wait_for_frames()
aligned_frames = align.process(test_frames)
intrinsics_profile = aligned_frames.get_profile()
intrinsics = intrinsics_profile.as_video_stream_profile().get_intrinsics()





depthdata = []
RGBImage = []
real_point = []
def get_rgbd_stream():  # with depth and color aligned
    global depthdata, RGBImage, real_point
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
        w_x = x_real_imgplane * depthdata
        w_y = y_real_imgplane * depthdata
        real_point = np.stack((w_x, w_y, depthdata), axis=-1)
        real_point = real_point.reshape(307200, 3)
        # bad_obj_points = np.where(real_point[:, 2] == 0)

        # cv2.imwrite('T_depthdata.jpg', depthdata)
        cv2.imwrite('T_RGB.jpg', RGBImage)
        np.save("T_depthdata", depthdata)




def true_temp(x):

    temprature = (x*10)+1000

    return temprature


thermal_img = []
T_img = [ ]
# Palette:384*288 Thermal: 382*288
def get_thermal_stream():
    global thermal_img,T_img
    optris.set_palette(1)
    w_t, h_t = optris.get_thermal_image_size()
    w_p, h_p = optris.get_palette_image_size()
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
            T_img = thermal_img.copy()

            cv2.imwrite('Thermal.jpg', T_img)



def Celex_config():
    print("{} * {} = {}".format(PyCeleX5.WIDTH, PyCeleX5.HEIGHT, PyCeleX5.RESOLUTION))
    celex5.openSensor(PyCeleX5.DeviceType.CeleX5_MIPI)
    celex5.isSensorReady()
    celex5.getRotateType()

    # sensorMode = PyCeleX5.CeleX5Mode.Full_Picture_Mode
    # celex5.setSensorFixedMode(sensorMode)
    sensorMode = PyCeleX5.CeleX5Mode.Event_Off_Pixel_Timestamp_Mode
    celex5.setSensorFixedMode(sensorMode)
    celex5.setEventFrameTime(33)

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

    celex5.setEventFrameTime(8333)
    celex5.setRotateType(2)

Event_Frame = [ ]
fullpic = [ ]
def get_Event_stream():
    global Event_Frame, fullpic
    Celex_config()
    while True:
        # fullpic = celex5.getFullPicBuffer()
        # Event_Frame = fullpic.copy( )
        event = celex5.getEventPicBuffer()
        Event_Frame = event.copy()






RGB_T_image = []
def img_process_thermal():
    global RGB_T_image

    time.sleep(2)

    while True:

        # thermal_1D = thermalImage.reshape(307200)

        imagePoints, jacobian = cv2.projectPoints(real_point, relative_Rvct_Thermal, relative_T_Thermal, thermalIntrinsic,
                                                  thermalDistortion)
        imagePoints = imagePoints[:, 0, :]
        imagePoints = np.round((imagePoints).astype(int))
        imagePoints = np.flip(imagePoints, axis=1)
        row = np.clip(imagePoints[:, 0], 0, 479)
        col = np.clip(imagePoints[:, 1], 0, 639)
        # projected_image = thermal_img[row, col]
        projected_image = T_img[row, col]
        # projected_image[bad_obj_points] = 0
        projected_image = projected_image.reshape(480, 640, 3)
        # cv2.imshow("1",projected_image)
        mask = cv2.cvtColor(projected_image, cv2.COLOR_BGR2GRAY)


        # mask = thermalImage
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

        # # Only Dispaly the warm area
        mask = cv2.bitwise_not(mask)


        # Apply the mask to imageA
        foreground = cv2.bitwise_and(RGBImage, RGBImage, mask=mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Get the background by removing the foreground from imageB
        background = cv2.bitwise_and(projected_image, projected_image, mask=mask_inv)

        # Combine the foreground and background
        RGB_T_image = cv2.add(foreground, background)

        # RGB_T_image = RGB_T_image[102:378, 135:503]


projected_image = []
Fusion_img = []
def img_process_event():
    global projected_image, Fusion_img

    time.sleep(3)
    while True:
        imagePoints, jacobian = cv2.projectPoints(real_point, relative_Rvct_Event, relative_T_Event, EventIntrinsic,
                                                  EventDistortion)

        imagePoints = imagePoints[:, 0, :]
        imagePoints = np.round((imagePoints).astype(int))
        imagePoints = np.flip(imagePoints, axis=1)

        row = np.clip(imagePoints[:, 0], 0, 479)
        col = np.clip(imagePoints[:, 1], 0, 639)

        projected_image = Event_Frame[row, col]

        projected_image = projected_image.reshape(480, 640)


        projected_image = cv2.cvtColor(projected_image, cv2.COLOR_GRAY2BGR)

        mask = cv2.cvtColor(projected_image, cv2.COLOR_BGR2GRAY)
        # mask = thermalImage
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

        # # Only Dispaly the warm area
        mask = cv2.bitwise_not(mask)



        # Apply the mask to imageA
        foreground = cv2.bitwise_and(RGB_T_image, RGB_T_image, mask=mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Get the background by removing the foreground from imageB
        background = cv2.bitwise_and(projected_image, projected_image, mask=mask_inv)
        # Combine the foreground and background
        Fusion_img = cv2.add(foreground, background)




def get_pcd():


    open3dintrinsics = o3d.camera.PinholeCameraIntrinsic(

    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)



    o3ddepthimg = np.asanyarray(depthdata)
    # o3dcolorimg = np.asanyarray(temp_colormap)


    o3dcolorimg = np.asanyarray(Fusion_img)


    # cv2.imshow("1",Fusion_img)
    # cv2.waitKey(10)
    # change opencv's bgr to rgb
    o3dcolorimg = o3dcolorimg[..., ::-1].copy()

    o3ddepthimg = o3d.geometry.Image(o3ddepthimg)
    o3dcolorimg = o3d.geometry.Image(o3dcolorimg)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3dcolorimg, o3ddepthimg, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, open3dintrinsics)
    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # return pcd



    # only display non-black pixels
    points = np.asarray(pcd.points)

    # Get the colors of the point cloud
    colors = np.asarray(pcd.colors)

    # Filter out black pixels
    non_black_indices = np.sum(colors, axis=1) > 0  # Filter non-black pixels
    filtered_points = points[non_black_indices, :]
    filtered_colors = colors[non_black_indices, :]

    # Create a new point cloud with the filtered points and colors
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    return filtered_point_cloud











event_stream_thread = threading.Thread(target=get_Event_stream)
thermal_stream_thread = threading.Thread(target=get_thermal_stream)
rgbd_stream_thread = threading.Thread(target=get_rgbd_stream)
img_process_thermal_thread = threading.Thread(target=img_process_thermal)
img_process_thread_event = threading.Thread(target=img_process_event)


if __name__ == "__main__":
    # try :
    ## setup realsense
    # Configure depth and color streams






    event_stream_thread.start()
    rgbd_stream_thread.start()
    thermal_stream_thread.start()
    img_process_thermal_thread.start()
    img_process_thread_event.start()



    print("Initializing>>>>")
    time.sleep(5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.register_animation_callback(callback_function)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    while True:

        pcd_init = get_pcd()
        vis.clear_geometries()
        vis.add_geometry(pcd_init)
        vis.update_renderer()
        # o3d.io.write_point_cloud("/home/emrys/Desktop/1.pcd", pcd_init)
        vis.poll_events()









## import necessary module
import cv2
import numpy as np
from numpy.lib.format import _filter_header

## import realsense module
import pyrealsense2 as rs

from pylibcelex import PyCeleX5
celex5 = PyCeleX5.PyCeleX5(debug=True)

import threading
import time

import open3d as o3d


img_num = 0


rgbIntrinsic = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_RGB_intrinsic.npy")
rgbDistortion = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_RGB_distortion.npy")


EventIntrinsic = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_event_intrinsic.npy")
EventDistortion = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_realsense_event_distortion.npy")

# rotation and translation data
relative_R = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_rgb_Relative_rotation_matrix.npy")
relative_T = np.load("/home/emrys/Project/Calibration_Pics/RGB_E/Data/event_rgb_Relative_translation_matrix.npy")
trans_mat = np.c_[relative_R, relative_T]  # creat relative translation matrix
relative_Rvct = (cv2.Rodrigues(relative_R))[0]
## read image of RGB and thermal camera and to get size of images
rgbImage = cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_E/RGB/RAW/0.png')
rgbImage_g = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)  # graylize
EventImage= cv2.imread('/home/emrys/Project/Calibration_Pics/RGB_E/Event/RAW/0.png')






pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)


# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)


align_to = rs.stream.color
align = rs.align(align_to)


test_frames = pipeline.wait_for_frames()
aligned_frames = align.process(test_frames)
intrinsics_profile = aligned_frames.get_profile()
intrinsics = intrinsics_profile.as_video_stream_profile().get_intrinsics()


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



# thermal_stream_thread = threading.Thread(target=get_thermal_stream)



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
        Event_Frame = event


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


projected_image = []
Fusion_img = []

def img_process():
    global projected_image, Fusion_img

    time.sleep(3)
    while True:
        imagePoints, jacobian = cv2.projectPoints(real_point, relative_Rvct, relative_T, EventIntrinsic,
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

        # Apply the mask to imageA
        foreground = cv2.bitwise_and(RGBImage, RGBImage, mask=mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Get the background by removing the foreground from imageB
        background = cv2.bitwise_and(projected_image, projected_image, mask=mask_inv)
        # Combine the foreground and background
        Fusion = cv2.add(foreground, background)
        Fusion_img = Fusion.copy( )

def Fusion_img_show():

    print("Programme initializing>>>>>")
    time.sleep(5)
    while True:
        cv2.imshow("1",Fusion_img)
        cv2.waitKey(10)



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

    return pcd


rgbd_stream_thread = threading.Thread(target=get_rgbd_stream)
event_stream_thread = threading.Thread(target=get_Event_stream)
img_process_thread = threading.Thread(target=img_process)
Fusion_img_show_thread = threading.Thread(target=Fusion_img_show)


if __name__ == "__main__":

    event_stream_thread.start()

    rgbd_stream_thread.start()

    img_process_thread.start()


    print("Initializing>>>>>>>")
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
        vis.poll_events()







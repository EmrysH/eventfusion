
## import necessary module
import cv2
import numpy as np
from numpy.lib.format import _filter_header

## import realsense module
import pyrealsense2 as rs

## import IR module
import pyOptris as optris

# cv2.namedWindow('fusion image', cv2.WINDOW_NORMAL)
import time

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

## setup realsense
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)


# Start streaming
profile = pipeline.start(config)
## boson camera


optris.set_palette(3)
w_t, h_t = optris.get_thermal_image_size()
w_p, h_p = optris.get_palette_image_size()


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)


align_to = rs.stream.color
align = rs.align(align_to)

test_frames = pipeline.wait_for_frames()
aligned_frames = align.process(test_frames)
intrinsics_profile = aligned_frames.get_profile()
intrinsics = intrinsics_profile.as_video_stream_profile().get_intrinsics()


def get_rgbd_stream():  # with depth and color aligned
    # Wait for a coherent pair of frames: color
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
    return depth_image, color_image


def true_temp(x):

    temprature = (x*10)+1000

    return temprature
# Palette:384*288 Thermal: 382*288
def get_thermal_stream():
    thermal_img_raw = optris.get_thermal_image(w_t, h_t) # get raw thermal data ,16bit, Kelvin temperature scale
    thermal_img_gray = optris.get_palette_image(w_p, h_p)
    thermal_img_gray = cv2.cvtColor(thermal_img_gray, cv2.COLOR_BGR2GRAY)


    thermal_img = cv2.resize(thermal_img_gray, (640, 480))
    thermal_data = cv2.resize(thermal_img_raw, (640, 480))

    code_terms = np.where(thermal_data < true_temp(25))
    rows = code_terms[0]
    cols = code_terms[1]
    thermal_img[rows, cols] = 0


    return thermal_img, thermal_data


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


def get_pcd():
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    thermal_img_raw, thermal_data = get_thermal_stream()
    # 2.1 filter at this step

    thermalImage = thermal_img_raw

    depthdata, rgbImage = get_rgbd_stream()
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthdata, alpha=0.03), cv2.COLORMAP_HOT)
    RGBImage = rgbImage.copy()



    thermalImage = 255 - thermalImage
    w_x = np.multiply(x_real_imgplane, depthdata)
    w_y = np.multiply(y_real_imgplane, depthdata)
    real_point_depth = np.stack((w_x, w_y, depthdata), axis=-1)

    real_point = real_point_depth.reshape(307200, 3)
    bad_obj_points = np.where(real_point[:, 2] == 0)
    # print(len(bad_obj_points[0]))
    imagePoints, jacobian = cv2.projectPoints(real_point, relative_Rvct, relative_T, thermalIntrinsic,
                                              thermalDistortion)
    imagePoints = imagePoints[:, 0, :]
    imagePoints = np.round((imagePoints).astype(int))
    imagePoints = np.flip(imagePoints, axis=1)

    row = np.clip(imagePoints[:, 0], 0, 479)
    col = np.clip(imagePoints[:, 1], 0, 639)
    projected_image = thermalImage[row, col]

    projected_image[bad_obj_points] = 0

    projected_image = projected_image.reshape(480, 640)
    # Fusion_img = projected_image[102:378, 135:503]
    # print(projected_image.min(),projected_image.max())


    temp_colormap = cv2.applyColorMap(projected_image, cv2.COLORMAP_HOT)




    open3dintrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    o3ddepthimg = np.asanyarray(depthdata)
    o3dcolorimg = np.asanyarray(temp_colormap)
    # o3dcolorimg = np.asanyarray(projected_image)
    # change opencv's bgr to rgb
    o3dcolorimg = o3dcolorimg[..., ::-1].copy()
    # set pcd's color
    pcd_colors = o3dcolorimg.reshape(-1, 3) / 255

    # print(pcd_colors[2000:2050])

    o3ddepthimg = o3d.geometry.Image(o3ddepthimg)
    o3dcolorimg = o3d.geometry.Image(o3dcolorimg)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3dcolorimg, o3ddepthimg, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, open3dintrinsics)
    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd



def callback_function(vis):
    # Get the current view control object
    view_ctrl = vis.get_view_control()

    # Check if the mouse is being pressed
    if vis.poll_events() and vis.update_renderer():
        if vis.get_window_event_status().mouse_event:
            # Get the current mouse state
            mouse_event = vis.get_window_event_status().mouse_event

            # Check if the left mouse button is pressed
            if mouse_event.button == o3d.visualization.gui.MouseButton.Left:
                # Update the view control based on the mouse motion
                view_ctrl.rotate(mouse_event.last_x, mouse_event.last_y, mouse_event.x, mouse_event.y)

            # Check if the right mouse button is pressed
            elif mouse_event.button == o3d.visualization.gui.MouseButton.Right:
                # Update the view control position based on the mouse motion
                view_ctrl.translate(mouse_event.x - mouse_event.last_x, mouse_event.y - mouse_event.last_y)

    return False  # Return False to indicate the callback should not be unregistered

if __name__ == "__main__":
    # try :

    pcd_init = get_pcd()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.register_animation_callback(callback_function)
    vis.add_geometry(pcd_init)



    while True:

        vis.poll_events()
        pcd_init = get_pcd()
        # vis.clear_geometries()
        # vis.add_geometry(pcd_init)
        vis.update_renderer()
        vis.poll_events()



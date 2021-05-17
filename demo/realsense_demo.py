# detectron2で使うモジュールのインポート
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2


import pyrealsense2 as rs
import numpy as np
import cv2
import time
import filter
import math
import datetime
import os


#outputs   = predictor(im)

# 結果の表示
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Results', v.get_image()[:, :, ::-1])
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

def save_dir():

    #?????
       #????????????
    nowtime = datetime.now()
    savedir = os.getcwd()


    if not os.path.exists(os.path.join(savedir)):
        os.mkdir(os.path.join(savedir))
        print("MAKE_DIR: " + savedir)
    #?????????
    savedir += datetime.now().strftime("/%Y")
    if not os.path.exists(os.path.join(savedir)):
        os.mkdir(os.path.join(savedir))
        print("MAKE_DIR: " + savedir)
    #????????
    savedir += nowtime.strftime("/%m")
    if not os.path.exists(os.path.join(savedir)):
        os.mkdir(os.path.join(savedir))
        print("MAKE_DIR: " + savedir)

    #?????????
    savedir += nowtime.strftime("/%d")
    if not os.path.exists(os.path.join(savedir)):
        os.mkdir(os.path.join(savedir))
        print("MAKE_DIR: " + savedir)
    #?????????
    savedir += nowtime.strftime("/%H")
    if not os.path.exists(os.path.join(savedir)):
        os.mkdir(os.path.join(savedir))
        print("MAKE_DIR: " + savedir)


    # ??_?_?????????
    print(str(savedir) +"??????")

    return savedir

def main():
    # ?????(Depth/Color)???
    config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

    # ?????????
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Align????????
    align_to = rs.stream.color
    align = rs.align(align_to)

    time.sleep(5)

    # ネットワークの設定
    cfg = get_cfg()
    cfg.MODEL.DEVICE="cpu"
    cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    # 推論器の作成
    predictor = DefaultPredictor(cfg)

    while(1):

        try:
            print("start")
            # ??????(Color & Depth)
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # ???????
            #filter??
            # state = AppState()
            # decimate = rs.decimation_filter()
            # decimate.set_option(rs.option.filter_magnitude, 1 ** state.decimate)
            # depth_to_disparity = rs.disparity_transform(True)
            # disparity_to_depth = rs.disparity_transform(False)
            # spatial = rs.spatial_filter()
            # spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
            # spatial.set_option(rs.option.filter_smooth_delta, 8)
            # temporal = rs.temporal_filter()
            # temporal.set_option(rs.option.filter_smooth_alpha, 0.5)
            # temporal.set_option(rs.option.filter_smooth_delta, 20)
            # hole_filling = rs.hole_filling_filter()

            # depth_frame = decimate.process(depth_frame)
            # depth_frame = depth_to_disparity.process(depth_frame)
            # depth_frame = spatial.process(depth_frame)
            # depth_frame = temporal.process(depth_frame)
            # depth_frame = disparity_to_depth.process(depth_frame)
            # depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            #+np.savetxt(save_dir + '/image/Leaf.csv', depth_image)

            color_image_s = cv2.resize(color_image, (1280, 720))
            #+cv2.imwrite(save_dir + "/image/Leaf.png", color_image)
            im = cv2.resize(color_image, (1280, 720))
            outputs   = predictor(im)
            # 結果の表示

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('Results', v.get_image()[:, :, ::-1])
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

        except Exception as e:
            print(str(e))
            cv2.destroyAllWindows()
            #?????????
            #pipeline.stop()

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


if __name__ == '__main__':
    main()


import torch
import argparse
import cv2
import re
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import argparse
import cv2
import time
import re
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from modules.yolact import Yolact
from config import get_config, COLORS
from utils.coco import COCODetection, detect_collate
from utils import timer
from utils.box_utils import box_iou, crop
from utils.output_utils import nms, after_nms, draw_img
from utils.common_utils import ProgressBar
from utils.augmentations import val_aug
# ... (Other imports and functions)
import mediapipe as mp
import os
import re
import argparse
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from math import sqrt
from collections import namedtuple


CM2MM = 10

class Detection:
    """
    Class wrapping relevant information about detections together.
    """

    def __init__(
        self,
        crop_border_px: int = 50,
    ):
        """
        Detection object constructor.

        Args:
            crop_border_px (int): Size of the border in pixels around the detection that is included in the depth crop.
        """

        # NAMED TUPLES
        ##############
        self.PointTuple = namedtuple("Point", ["x", "y"])

        # NEW detection PARAMETERS
        #######################
        # ID of the detection for tracking between frames
        self.id = None

        # Type of the detection
        self.type = None

        # X Y centroid value in frame pixels,
        # of the position where the detection last last detected by the camera.
        # X increases from left to right.
        # Y increases from top to bottom.
        self.centroid_px = None

        # Width of horizontally aligned bounding box in pixels
        self.bounding_width_px = None
        # Height of horizontally aligned bounding box in pixels
        self.bounding_height_px = None

        # A 3x3 homography matrix, which corresponds to the transformation
        # from the pixels coordinates of the image frame where the detection was detected
        # into real-world coordinates
        self.homography_matrix = None

        # Time in milliseconds, when the detection data was last updated
        # It should correspond with the data capture timestamp,
        # that is the time when the data used for detection parameters was captured.
        # TODO: Implement detection timestamps
        self.timestamp_ms = None

        # Binary mask for the depth detection
        # TODO: Find out how mask is used
        self.mask = None

        # Ammount of extra pixels around each depth crop
        self.crop_border_px = crop_border_px
        # Number of averaged depth crops
        self.num_avg_depths = 0
        # Numpy array with average depth value for the detection
        self.avg_depth_crop = None

        # The angle gives the rotation of detection contours from horizontal line in the frame
        # Number of averaged angles
        self.num_avg_angles = 0
        # Number with average angle of the detection
        self.avg_angle_deg = None

        # Last detected position of detection centroid in pixels
        self.centroid_base_px = None

        # Last detected position of encoder in pixels
        self.encoder_base_position = None

        # Number of frames the detection has been tracked for
        self.track_frame = 0

        # Indicates if the detection is marked to be the next detection to be sorted
        self.in_pick_list = False

        # Number of frames the detection has disappeared for
        # TODO: Find out how "disappeared" counter is used
        self.disappeared = 0

        # OBSOLETE detection PARAMETERS
        ############################

        # Width and height of detection bounding box
        self.width = 0
        self.height = 0

    def set_id(self, detection_id: int) -> None:
        """
        Sets detection ID.

        Args:
            detection_id (int): Unique integer greater then or equal to 0.
        """

        if detection_id < 0:
            print(
                f"[WARNING] Tried to set detection ID to {detection_id} (Should be greater than or equal to 0)"
            )
            return

        self.id = detection_id

    def set_type(self, detection_type: int) -> None:
        """
        Sets detection type.

        Args:
            detection_type (int): Integer representing type of the detection.
        """

        self.type = detection_type

    def set_centroid(self, x: int, y: int) -> None:
        """
        Sets detection centroid.

        Args:
            x (int): X coordinate of centroid in pixels.
            y (int): Y coordinate of centroid in pixels.
        """

        self.centroid_px = self.PointTuple(x, y)

    def set_homography_matrix(self, homography_matrix: np.ndarray) -> None:
        """
        Sets a homography matrix corresponding to the transformation between pixels and centimeters.

        Args:
            homography_matrix (np.ndarray): 3x3 homography matrix.
        """

        self.homography_matrix = homography_matrix

    def set_base_encoder_position(self, encoder_position: float) -> None:
        """
        Sets detection base encoder position and associated centroid position.

        Args:
            encoder_pos (float): Position of encoder.
        """

        self.centroid_base_px = self.centroid_px
        self.encoder_base_position = encoder_position

    def update_timestamp(self, timestamp: float) -> None:
        """
        Updates detection timestamp variable.
        The time should correspond to the time when the data used for detection parameter computation was captured.

        Args:
            timestamp (float): Time in milliseconds.
        """

        self.timestamp_ms = timestamp

    def set_bounding_size(self, width: int, height: int) -> None:
        """
        Sets width and height of detection bounding box.

        Args:
            width (int): Width of the detection in pixels.
            height (int): Height of the detection in pixels.
        """

        if width <= 0 or height <= 0:
            print(
                "[WARNING] Tried to set detection WIDTH and HEIGHT to ({}, {}) (Should be greater than 0)".format(
                    width, height
                )
            )
            return

        self.bounding_width_px = width
        self.bounding_height_px = height

        # OBSOLETE PARAMS
        # TODO: Replace detection width and height with bounding_width_px and bounding_height_px
        self.width = width
        self.height = height

    def set_mask(self, mask) -> None:
        """
        Sets the inner rectangle (params mask of the detection).

        Args:
            mask (tuple): Center(x, y), (width, height), angle of rotation.
        """

        if not isinstance(mask, np.ndarray):
            print(
                f"[WARN]: Tried to crop detection mask of type {type(mask)} (Not a np.ndarray)"
            )
            return

        # Update the mask
        if self.mask is None:
            self.mask = mask
        else:
            if mask.shape != self.mask.shape:
                print(f"[WARN]: Tried to average two uncompatible sizes")
                return
            self.mask = np.logical_and(mask, self.mask)

    def add_angle_to_average(self, angle: float) -> None:
        """
        Adds new angle value into the average angle of the detection.

        Args:
            angle (int | float): Angle of detection contours from horizontal line in the frame.
        """

        if not -90 <= angle <= 90:
            print(
                "[WARNING] Tried to add detection ANGLE with value {} (Should be between -90 and 90)".format(
                    angle
                )
            )
            return

        self.avg_angle_deg = angle

        # Update average
        # if self.avg_angle_deg is None:
        #     self.avg_angle_deg = angle
        # else:
        #     self.avg_angle_deg = (self.num_avg_angles * self.avg_angle_deg + angle) / (
        #         self.num_avg_angles + 1
        #     )

        # self.num_avg_angles += 1

    def add_depth_crop_to_average(self, depth_crop: np.ndarray) -> None:
        """
        Adds new depth crop into the average depth image of the detection.

        Args:
            depth_crop (np.ndarray): Frame containing depth values, has to have same size as previous depth frames
        """

        if self.avg_depth_crop is None:
            self.avg_depth_crop = depth_crop
        else:
            if not self.avg_depth_crop.shape == depth_crop.shape:
                print(
                    "[WARNING] Tried to average two depth maps with incompatible shape together: {} VS {}".format(
                        self.avg_depth_crop.shape, depth_crop.shape
                    )
                )
                return
            self.avg_depth_crop = (
                self.num_avg_depths * self.avg_depth_crop + depth_crop
            ) / (self.num_avg_depths + 1)

        self.num_avg_depths += 1

    def get_crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops detection with some small border around it (border given by self.crop_border_px) and returns the cropped array.
        If it is the first cropped depth map, detection width and height are used, otherwise the previous size is used.

        Args:
            frame (np.ndarray): Image frame with the detection values.

        Returns:
            np.ndarray: Numpy array containing the cropped detection.
        """

        if self.num_avg_depths == 0:
            crop = frame[
                int(
                    self.centroid_px.y
                    - self.bounding_height_px // 2
                    - self.crop_border_px
                ) : int(
                    self.centroid_px.y
                    + self.bounding_height_px // 2
                    + self.crop_border_px
                ),
                int(
                    self.centroid_px.x
                    - self.bounding_width_px // 2
                    - self.crop_border_px
                ) : int(
                    self.centroid_px.x
                    + self.bounding_width_px // 2
                    + self.crop_border_px
                ),
            ]
        else:
            crop = frame[
                int(self.centroid_px.y - self.avg_depth_crop.shape[0] // 2) : int(
                    self.centroid_px.y + self.avg_depth_crop.shape[0] // 2
                ),
                int(self.centroid_px.x - self.avg_depth_crop.shape[1] // 2) : int(
                    self.centroid_px.x + self.avg_depth_crop.shape[1] // 2
                ),
            ]

        return crop

    def get_centroid_in_px(self):
        return self.centroid_px

    def get_centroid_in_mm(self):
        # Transform centroid from pixels to centimeters using a homography matrix
        centroid_cm = np.matmul(
            self.homography_matrix,
            np.array([self.centroid_px.x, self.centroid_px.y, 1]),
        )
        # Transform centroid from centimeters to millimeters
        centroid_mm = self.PointTuple(centroid_cm[0] * CM2MM, centroid_cm[1] * CM2MM)
        return centroid_mm

    def get_centroid_from_encoder_in_px(
        self, encoder_position: float
    ):
        # k is a constant for translating
        # from distance in mm computed from encoder data
        # into frame pixels for a specific resolution
        # k = 0.8299  # 640 x 480
        # k = 1.2365  # 1280 x 720
        k = 1.8672  # 1440 x 1080
        # k = 1.2365  # 1080 x 720

        centroid_encoder_px = self.PointTuple(
            int(
                k * (encoder_position - self.encoder_base_position)
                + self.centroid_base_px.x
            ),
            self.centroid_px.y,
        )

        return centroid_encoder_px

    def get_centroid_from_encoder_in_mm(
        self, encoder_position: float
    ):
        centroid_robot_frame = np.matmul(
            self.homography_matrix,
            np.array([self.centroid_px.x, self.centroid_px.y, 1]),
        )

        detection_x = centroid_robot_frame[0] * CM2MM
        detection_y = centroid_robot_frame[1] * CM2MM
        return detection_x, detection_y

    def get_angle(self) -> float:
        return self.avg_angle_deg

    def get_width_in_px(self) -> int:
        return self.bounding_width_px

    def get_height_in_px(self) -> int:
        return self.bounding_height_px

    def get_width_in_mm(self) -> float:
        bounding_width_mm = (
            self.bounding_width_px
            * sqrt(
                self.homography_matrix[0, 0] ** 2 + self.homography_matrix[1, 0] ** 2
            )
            * CM2MM
        )
        return bounding_width_mm

    def get_height_in_mm(self) -> float:
        bounding_height_mm = (
            self.bounding_height_px
            * sqrt(
                self.homography_matrix[0, 1] ** 2 + self.homography_matrix[1, 1] ** 2
            )
            * CM2MM
        )
        return bounding_height_mm

class ItemsDetector:
    """
    Class for detecting Detections using neural network.
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        detect_classes: dict = {46: 0, 39: 1},
        max_detect: int = 1,
        detect_thres: float = 0.7,
        ignore_vertical_px: int = 60,
        ignore_horizontal_px: int = 10,
        cnt_area_up_thresh: int = 110000,
        cnt_area_low_thresh: int = 25000,
    ):
        """
        Detector object constructor.

        Args:
            paths (dict): Dictionary with annotation and checkpoint paths.
            files (dict): Dictionary with pipeline and config paths.
            checkpt (str): Name of training checkpoint to be restored.
            max_detect (int): Maximal ammount of concurrent detections in an image.
            detect_thres (float): Minimal confidence for detected object to be labeled as a item.
            ignore_vertical_px (int): Number of rows of pixels ignored from top and bottom of the image frame.
            ignore_horizontal_px (int): Number of columns of pixels ignored from left and right of the image frame.
            cnt_area_up_thresh (int): Upper contour area threshold for item mask.
            cnt_area_low_thresh (int): Lower contour area threshold for item mask.

        """

        self.detect_classes = detect_classes
        self.detected_objects = []
        self.homography_matrix = None
        self.homography_determinant = None

        self.ignore_vertical_px = ignore_vertical_px
        self.ignore_horizontal_px = ignore_horizontal_px

        self.cnt_area_up_thresh = cnt_area_up_thresh
        self.cnt_area_low_thresh = cnt_area_low_thresh

        args = parser.parse_args()
        prefix = re.findall(r"best_\d+\.\d+_", args.weight)[0]
        suffix = re.findall(r"_\d+\.pth", args.weight)[0]
        args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
        cfg = get_config(args, mode="detect")
        self.cfg = cfg
        self.net = Yolact(cfg)
        self.net.load_weights(cfg.weight, cfg.cuda)
        self.net.eval()

        if cfg.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.net = self.net.cuda()

    def set_homography(self, homography_matrix: np.ndarray) -> None:
        """
        Sets the homography matrix and calculates its determinant.

        Args:
            homography_matrix(np.ndarray): Homography matrix.
        """

        self.homography_matrix = homography_matrix
        self.homography_determinant = np.linalg.det(homography_matrix[0:2, 0:2])

    def fast_nms(self, box_thre, coef_thre, class_thre, cfg):
        class_thre, idx = class_thre.sort(
            1, descending=True
        )  # [80, 64 (the number of kept boxes)]

        idx = idx[:, : cfg.top_k]
        class_thre = class_thre[:, : cfg.top_k]

        num_classes, num_dets = idx.size()
        box_thre = box_thre[idx.reshape(-1), :].reshape(
            num_classes, num_dets, 4
        )  # [80, 64, 4]
        coef_thre = coef_thre[idx.reshape(-1), :].reshape(
            num_classes, num_dets, -1
        )  # [80, 64, 32]

        iou = box_iou(box_thre, box_thre)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = iou_max <= cfg.nms_iou_thre

        # Assign each kept detection to its corresponding class
        class_ids = torch.arange(num_classes, device=box_thre.device)[
            :, None
        ].expand_as(keep)

        class_ids, box_nms, coef_nms, class_nms = (
            class_ids[keep],
            box_thre[keep],
            coef_thre[keep],
            class_thre[keep],
        )

        # Only keep the top cfg.max_num_detections highest scores across all classes
        class_nms, idx = class_nms.sort(0, descending=True)

        idx = idx[: cfg.max_detections]
        class_nms = class_nms[: cfg.max_detections]

        class_ids = class_ids[idx]
        box_nms = box_nms[idx]
        coef_nms = coef_nms[idx]

        return box_nms, coef_nms, class_ids, class_nms

    def nms(self, class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
        class_p = class_pred.squeeze()  # [19248, 81]
        box_p = box_pred.squeeze()  # [19248, 4]
        coef_p = coef_pred.squeeze()  # [19248, 32]
        proto_p = proto_out.squeeze()  # [138, 138, 32]

        if isinstance(anchors, list):
            anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

        class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

        # exclude the background class
        class_p = class_p[1:, :]
        # get the max score class of 19248 predicted boxes
        class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

        # filter predicted boxes according the class score
        keep = class_p_max > cfg.nms_score_thre
        class_thre = class_p[:, keep]
        box_thre, anchor_thre, coef_thre = (
            box_p[keep, :],
            anchors[keep, :],
            coef_p[keep, :],
        )

        # decode boxes
        box_thre = torch.cat(
            (
                anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2),
            ),
            1,
        )
        box_thre[:, :2] -= box_thre[:, 2:] / 2
        box_thre[:, 2:] += box_thre[:, :2]

        box_thre = torch.clip(box_thre, min=0.0, max=1.0)

        if class_thre.shape[1] == 0:
            return None, None, None, None, None
        else:
            box_thre, coef_thre, class_ids, class_thre = self.fast_nms(
                box_thre, coef_thre, class_thre, cfg
            )
            return class_ids, class_thre, box_thre, coef_thre, proto_p

    def after_nms(
        self,
        ids_p,
        class_p,
        box_p,
        coef_p,
        proto_p,
        img_h,
        img_w,
        cfg=None,
        img_name=None,
    ):
        if ids_p is None:
            return None, None, None, None

        if cfg and cfg.visual_thre > 0:
            keep = class_p >= cfg.visual_thre
            if not keep.any():
                return None, None, None, None

            ids_p = ids_p[keep]
            class_p = class_p[keep]
            box_p = box_p[keep]
            coef_p = coef_p[keep]

        masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))

        if not cfg or not cfg.no_crop:  # Crop masks by box_p
            masks = crop(masks, box_p)

        masks = masks.permute(2, 0, 1).contiguous()

        ori_size = max(img_h, img_w)
        # in OpenCV, cv2.resize is `align_corners=False`.
        masks = F.interpolate(
            masks.unsqueeze(0),
            (ori_size, ori_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        masks.gt_(0.5)  # Binarize the masks because of interpolation.
        masks = masks[:, 0:img_h, :] if img_h < img_w else masks[:, :, 0:img_w]

        box_p *= ori_size
        box_p = box_p.int()

        return ids_p, class_p, box_p, masks

    def draw_img(
        self, ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None
    ):

        if ids_p is None:
            return img_origin

        if isinstance(ids_p, torch.Tensor):
            ids_p = ids_p.cpu().numpy()
            class_p = class_p.cpu().numpy()
            box_p = box_p.cpu().numpy()
            mask_p = mask_p.cpu().numpy()

        num_detected = ids_p.shape[0]

        img_fused = img_origin
        if not cfg.hide_mask:
            masks_semantic = mask_p * (
                ids_p[:, None, None] + 1
            )  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype("int").sum(axis=0) % (
                cfg.num_classes - 1
            )
            color_masks = COLORS[masks_semantic].astype("uint8")
            img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX

        if not cfg.hide_bbox:
            for i in reversed(range(num_detected)):
                x1, y1, x2, y2 = box_p[i, :]

                color = COLORS[ids_p[i] + 1].tolist()
                cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

                class_name = cfg.class_names[ids_p[i]]
                text_str = (
                    f"{class_name}: {class_p[i]:.2f}"
                    if not cfg.hide_score
                    else class_name
                )

                text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
                cv2.rectangle(
                    img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1
                )
                cv2.putText(
                    img_fused,
                    text_str,
                    (x1, y1 + 15),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return img_fused

    def get_item_from_mask(
        self, img: np.array, bbox: dict, mask: np.array, type: int, encoder_pos: float
    ):
        """
        Creates object inner rectangle given the mask from neural net.

        Args:
            img (np.array): image for drawing detections.
            bbox (dict): bounding box parameters.
            mask (np.array): mask produced by neural net.
            type (int): Type of the item.
            encoder_pos (float): Position of the encoder.

        Returns:
            Detection: Created Detection object
        """
        centroid = bbox["centroid"]
        ymin = bbox["ymin"]
        ymax = bbox["ymax"]
        xmin = bbox["xmin"]
        xmax = bbox["xmax"]
        angle = 0
        box = np.int64(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > self.cnt_area_low_thresh and area < self.cnt_area_up_thresh:
        #         rectangle = cv2.minAreaRect(cnt)
        #         centroid = (int(rectangle[0][0]), int(rectangle[0][1]))
        #         box = np.int0(cv2.boxPoints(rectangle))
        #         angle = int(rectangle[2])

        # cv2.polylines(img, [box], True, (255, 0, 0), 3)
        item = Detection()

        item.set_type(type)
        item.set_centroid(centroid[0], centroid[1])
        item.set_base_encoder_position(encoder_pos)
        item.set_bounding_size(bbox["w"], bbox["h"])
        item.add_angle_to_average(angle)

        return item

    def deep_item_detector(
        self,
        rgb_frame: np.ndarray,
        encoder_pos: float,
        draw_box: bool = True,
        image_frame: np.ndarray = None,
    ):
        """
        Detects Detections using Yolact model.

        Args:
            rgb_frame (np.ndarray): RGB frame in which Detections should be detected.
            encoder_position (float): Position of the encoder.
            draw_box (bool): If bounding and min area boxes should be drawn.
            image_frame (np.ndarray): Image frame into which information should be drawn.

        Returns:
            np.ndarray: Image frame with information drawn into it.
            list[Detection]: List of detected Detections.
        """
        scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        self.detected_objects = []
        img_h, img_w = rgb_frame.shape[0:2]
        frame_trans = val_aug(rgb_frame, self.cfg.img_size)

        frame_tensor = torch.tensor(frame_trans).float()
        if self.cfg.cuda:
            frame_tensor = frame_tensor.cuda()

        with torch.no_grad():
            class_p, box_p, coef_p, proto_p = self.net.forward(
                frame_tensor.unsqueeze(0)
            )

        ids_p, class_p, box_p, coef_p, proto_p = self.nms(
            class_p, box_p, coef_p, proto_p, self.net.anchors, self.cfg
        )
        ids_p, class_p, boxes_p, masks_p = self.after_nms(
            ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, self.cfg
        )
        # img_np_detect = self.draw_img(ids_p, class_p, boxes_p, masks_p, rgb_frame, self.cfg)

        if ids_p is None:
            return rgb_frame, self.detected_objects

        if isinstance(ids_p, torch.Tensor):
            ids_p = ids_p.cpu().numpy()
            class_p = class_p.cpu().numpy()
            boxes_p = boxes_p.cpu().numpy()
            masks_p = masks_p.cpu().numpy()

        num_detected = ids_p.shape[0]

        img_np_detect = rgb_frame
        if not self.cfg.hide_mask:
            masks_semantic = masks_p * (
                ids_p[:, None, None] + 1
            )  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype("int").sum(axis=0) % (
                self.cfg.num_classes - 1
            )
            color_masks = COLORS[masks_semantic].astype("uint8")
            img_np_detect = cv2.addWeighted(color_masks, 0.4, rgb_frame, 0.6, gamma=0)

        for i in reversed(range(num_detected)):
            # if int(ids_p[i]) in self.detect_classes:
                xmin, ymin, xmax, ymax = boxes_p[i, :]
                w = int(xmax - xmin)
                h = int(ymax - ymin)
                cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
                centroid = (int(cx), int(cy))

                bbox = {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "centroid": centroid,
                    "w": w,
                    "h": h,
                }
                if not self.cfg.hide_bbox:
                    color = COLORS[ids_p[i] + 1].tolist()
                    cv2.rectangle(
                        img_np_detect, (xmin, ymin), (xmax, ymax), color, thickness
                    )

                    class_name = self.cfg.class_names[ids_p[i]]
                    text_str = (
                        f"{class_name}: {class_p[i]:.2f}"
                        if not self.cfg.hide_score
                        else class_name
                    )

                    text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[
                        0
                    ]
                    cv2.rectangle(
                        img_np_detect,
                        (xmin, ymin),
                        (xmin + text_w, ymin + text_h + 5),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img_np_detect,
                        text_str,
                        (xmin, ymin + 15),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                item = self.get_item_from_mask(
                    img_np_detect, bbox, masks_p[i], int(ids_p[i]), encoder_pos
                )
                is_cx_low_ok = (
                    item.centroid_px.x - item.width / 2 < self.ignore_horizontal_px
                )
                is_cx_up_ok = item.centroid_px.x + item.width / 2 > (
                    img_w - self.ignore_horizontal_px
                )
                is_cx_out_range = is_cx_low_ok or is_cx_up_ok
                if is_cx_out_range:
                    continue
                self.detected_objects.append(item)
        return img_np_detect, self.detected_objects
    
def detect_body_pose(frame, mp_pose, pose, mp_drawing):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose
    results = pose.process(frame_rgb)
    # Draw landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Body Pose Detection', frame)

def image_callback(msg):
    frame_origin = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    detect_body_pose(frame_origin, mp_pose, pose, mp_drawing)
    img_np_detect, detected_objects = detector.deep_item_detector(frame_origin,0.0)
    cv2.imshow('detections', img_np_detect)
    key = cv2.waitKey(10)
    if key == 27:
        rospy.signal_shutdown('User requested shutdown')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLACT Detection.')
    parser.add_argument('--weight', default='weights/best_30.4_res101_coco_340000.pth', type=str)
    parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
    parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
    parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop the output masks with the predicted bounding box.')
    parser.add_argument('--real_time', default=True, action='store_true', help='Show the detection results real-timely.')
    parser.add_argument('--visual_thre', default=0.5, type=float,
                        help='Detections with a score under this threshold will be removed.')

    # Initialize MediaPipe Drawing module
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    detector = ItemsDetector(parser)
    rospy.init_node('yolact_detector')
    rospy.Subscriber("/xtion/rgb/image_raw", Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

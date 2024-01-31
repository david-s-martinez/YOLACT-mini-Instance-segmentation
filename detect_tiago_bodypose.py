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
from config import get_config
from utils.coco import COCODetection, detect_collate
from utils import timer
from utils.output_utils import nms, after_nms, draw_img
from utils.common_utils import ProgressBar
from utils.augmentations import val_aug
# ... (Other imports and functions)
import mediapipe as mp

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

    # # Break the loop when 'q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

def image_callback(msg):
    frame_origin = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    detect_body_pose(frame_origin, mp_pose, pose, mp_drawing)
    img_h, img_w = frame_origin.shape[0:2]
    frame_trans = val_aug(frame_origin, cfg.img_size)
    t_fps = 0
    frame_tensor = torch.tensor(frame_trans).float()
    if cfg.cuda:
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad(), timer.counter('forward'):
        class_p, box_p, coef_p, proto_p = net.forward(frame_tensor.unsqueeze(0))

    with timer.counter('nms'):
        ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg)

    with timer.counter('after_nms'):
        ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg)

    with timer.counter('save_img'):
        frame_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, frame_origin, cfg, fps=t_fps)

    if cfg.real_time:
        cv2.imshow('Detection', cv2.resize(frame_numpy, (frame_numpy.shape[1] * 2, frame_numpy.shape[0] * 2)))
        key = cv2.waitKey(10)
        if key == 27:
            rospy.signal_shutdown('User requested shutdown')

    # Additional processing or publishing can be done here

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
    parser.add_argument('--visual_thre', default=0.6, type=float,
                        help='Detections with a score under this threshold will be removed.')

    args = parser.parse_args()
    prefix = re.findall(r'best_\d+\.\d+_', args.weight)[0]
    suffix = re.findall(r'_\d+\.pth', args.weight)[0]
    args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
    cfg = get_config(args, mode='detect')

    net = Yolact(cfg)
    net.load_weights(cfg.weight, cfg.cuda)
    net.eval()
    # Initialize MediaPipe Drawing module
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    if cfg.cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        net = net.cuda()

    rospy.init_node('yolact_detector')
    rospy.Subscriber("/xtion/rgb/image_raw", Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

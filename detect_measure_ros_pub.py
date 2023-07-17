import torch
import argparse
import cv2
import time
import re
import numpy as np
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from config import COLORS
from modules.yolact import Yolact
from config import get_config
from utils.coco import COCODetection, detect_collate
from utils import timer
from utils.output_utils import nms, after_nms
from utils.common_utils import ProgressBar
from utils.augmentations import val_aug
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pipe
import rospy
from std_msgs.msg import String

class processing_apriltag:
    def __init__(self,intrinsic,color_image,depth_frame):
        self.color_image = color_image
        self.intrinsic = intrinsic
        self.depth_frame = depth_frame
        self.radius = 20
        self.axis = 0
        self.image_points = {}
        self.world_points = {}
        self.world_points_detect = []
        self.image_points_detect = []
        self.homography= None
                
    def detect_tags(self):
            rect1 = []
            pixel_cm_ratio = None
            image = self.color_image.copy()
            # self.load_original_points()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (corners, ids, rejected) = cv2.aruco.detectMarkers( image, cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11),parameters=cv2.aruco.DetectorParameters_create())
            ids = ids.flatten()
            int_corners = np.int0(corners)
            cv2.polylines(image, int_corners, True, (0, 255, 0), 2)
            
            for (tag_corner, tag_id) in zip(corners, ids):
                # get (x,y) corners of tag
                aruco_perimeter = cv2.arcLength(corners[0], True)
                pixel_cm_ratio = aruco_perimeter / 24
                corners = tag_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
                top_right, bottom_right = (int(top_right[0]), int(top_right[1])),(int(bottom_right[0]), int(bottom_right[1]))
                bottom_left, top_left = (int(bottom_left[0]), int(bottom_left[1])),(int(top_left[0]), int(top_left[1]))
                # compute centroid
                cX = int((top_left[0] + bottom_right[0]) / 2.0)
                cY = int((top_left[1] + bottom_right[1]) / 2.0)
                self.image_points[str(int(tag_id))] = [cX,cY]
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw ID on frame
                cv2.putText(image, str(tag_id),(top_left[0], top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            return image,pixel_cm_ratio

def measure_size(img, mask, ymin, ymax, xmin, xmax,pixel_cm_ratio):
        # print(ymin, ymax, xmin, xmax,pixel_cm_ratio)
        box = np.int64(np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]))
        w = xmax - xmin
        h = ymax - ymin
        centroid = None
        angle = 0
        object_dims = [0,0]
        # box = None
        crop = img[int(ymin):int(ymax),int(xmin):int(xmax),:]
        mask_crop = mask[int(ymin):int(ymax),int(xmin):int(xmax)].astype(np.uint8)
        # cv2.imshow('mask', mask_crop)

        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
            # if area > 25000 and area < 110000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                cx , cy = x+xmin, y+ymin
                centroid = (int(cx), int(cy))
                box = cv2.boxPoints(((cx,cy),(w, h), angle))
                box = np.int0(box)
            # crop = cv2.drawContours(crop, contours, -1, (0,255,0), 3,lineType = cv2.LINE_AA)

        if (pixel_cm_ratio is not 0) and (centroid is not None):
            object_width = int(w / pixel_cm_ratio)
            object_height = int(h / pixel_cm_ratio)
            object_dims = sorted([object_width,object_height])

        return object_dims, abs(angle)

def draw_img(pixel_cm_ratio, ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None):
    if ids_p is None:
        return img_origin

    if isinstance(ids_p, torch.Tensor):
        ids_p = ids_p.cpu().numpy()
        class_p = class_p.cpu().numpy()
        box_p = box_p.cpu().numpy()
        mask_p = mask_p.cpu().numpy()
    idxs = np.in1d(ids_p, [39,64,67])
    idxs = np.nonzero(idxs)
    # print(idxs)
    ids_p = ids_p[idxs]
    class_p = class_p[idxs]
    box_p =  box_p[idxs]
    mask_p = mask_p[idxs]

    num_detected = ids_p.shape[0]
    img_fused = img_origin
    if not cfg.hide_mask:
        masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if cfg.cutout:
            total_obj = (masks_semantic != 0)[:, :, None].repeat(3, 2)
            total_obj = total_obj * img_origin
            new_mask = ((masks_semantic == 0) * 255)[:, :, None].repeat(3, 2)
            img_matting = (total_obj + new_mask).astype('uint8')
            cv2.imwrite(f'results/images/{img_name}_total_obj.jpg', img_matting)

            for i in range(num_detected):
                one_obj = (mask_p[i])[:, :, None].repeat(3, 2)
                one_obj = one_obj * img_origin
                new_mask = ((mask_p[i] == 0) * 255)[:, :, None].repeat(3, 2)
                x1, y1, x2, y2 = box_p[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)
    scale = 0.45
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX
    hand_opening = 'n'

    if not cfg.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = box_p[i, :]

            color = COLORS[ids_p[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)
            # print(y1, y2, x1, x2,pixel_cm_ratio)
            mask = mask_p[i]
            object_dims, angle = measure_size(img_fused, mask, y1, y2, x1, x2, pixel_cm_ratio)
            
            if 15 < object_dims[0] < 20:
                hand_opening = 'xl'
            elif 10 < object_dims[0] <= 15:
                hand_opening = 'l'
            elif 5 < object_dims[0] <= 10:
                hand_opening = 'm'
            elif 3 < object_dims[0] <= 5:
                hand_opening = 's'
            else:
                hand_opening = 'n'
            print(hand_opening)

            dims_str = f',{object_dims[1]}x{object_dims[0]}cm,{int(angle):.1f}deg'
            class_name = cfg.class_names[ids_p[i]]
            text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name
            text_str = text_str + dims_str if 0 not in object_dims else text_str
            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if cfg.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        # img_fused = img_fused.astype(np.float32)
        # img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        # img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused, hand_opening

def main(net_in_conn, net_out_conn):
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

    if cfg.cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        net = net.cuda()
        print('Model running in gpu: ',next(net.parameters()).is_cuda)

    num_frames = net_in_conn.recv()['num_frames']
    progress_bar = ProgressBar(40, num_frames)
    timer.reset()
    t_fps = 0
    i = 0
    pixel_cm_ratio = 0
    # while cap.isOpened():
    while True:
        if i == 1:
            timer.start()

        # frame_origin = cap.read()[1]
        frame_origin = net_in_conn.recv()['frame']
        img_h, img_w = frame_origin.shape[0:2]
        frame_trans = val_aug(frame_origin, cfg.img_size)

        frame_tensor = torch.tensor(frame_trans).float()
        if cfg.cuda:
            frame_tensor = frame_tensor.cuda()

        with torch.no_grad(), timer.counter('forward'):
            class_p, box_p, coef_p, proto_p = net.forward(frame_tensor.unsqueeze(0))

        with timer.counter('nms'):
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg)

        with timer.counter('after_nms'):
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg)
            # print(ids_p)
        apriltag = processing_apriltag(None, frame_origin, None)
        try:
            frame_origin, pixel_cm_ratio = apriltag.detect_tags()
                
        except:
		#Triggered when no markers are in the frame:
            pass
        with timer.counter('save_img'):
            frame_numpy, hand_opening = draw_img(pixel_cm_ratio, ids_p, class_p, boxes_p, masks_p, frame_origin, cfg, fps=t_fps)
            net_out_conn.send(hand_opening)
        pixel_cm_ratio = 0

        if cfg.real_time:
            cv2.imshow('Detection', cv2.resize(frame_numpy,(frame_numpy.shape[1]*2,frame_numpy.shape[0]*2)) )
            key = cv2.waitKey(10)
            if key == 27:
                break
        aa = time.perf_counter()
        if i > 0:
            batch_time = aa - temp
            timer.add_batch_time(batch_time)
        temp = aa

        if i > 0:
            t_t, t_d, t_f, t_nms, t_an, t_si = timer.get_times(['batch', 'data', 'forward',
                                                                'nms', 'after_nms', 'save_img'])
            fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
            bar_str = progress_bar.get_bar(i + 1)
            print(f'\rDetecting: {bar_str} {i + 1}/{num_frames}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                f't_after_nms: {t_an:.3f} | t_save_img: {t_si:.3f}', end='')
        i+=1
 

def cam_reader(cam_out_conn, cam_source):
    cap = cv2.VideoCapture(cam_source)
    num_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cam_source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                height, width, channels = frame.shape
                # frame = cv2.imread("C:/Users/David/Pictures/PXL_20230627_112438768.jpg") 
                # frame = cv2.resize(frame, (width, height))
                cam_out_conn.send({'frame':frame, 'num_frames':num_frames})
            except:
                print("CAMERA COULD NOT BE OPEN")
                break
    else:
        del cap
        while True:
            try:
                cap = cv2.VideoCapture(cam_source)
                ret, frame = cap.read()
                height, width, channels = frame.shape
                # frame = cv2.imread("C:/Users/David/Pictures/PXL_20230627_112438768.jpg") 
                # frame = cv2.resize(frame, (width, height))
                cam_out_conn.send({'frame':frame, 'num_frames':num_frames})
                del cap
            except:
                print("CAMERA COULD NOT BE OPEN")
                break

def post_detections(send_detect_in_conn):
    pub = rospy.Publisher('hand_opening', String, queue_size=10)
    rospy.init_node('hand_opening_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    old_opening = 'n'
    while not rospy.is_shutdown():
        new_opening = send_detect_in_conn.recv()
        if new_opening != old_opening :
            rospy.loginfo(new_opening)
            old_opening = new_opening
            pub.publish(new_opening)
            rate.sleep()

if __name__ == '__main__':
    cam_source = 0
    # cam_source = 'http://192.168.178.41:8000/video'
    multiprocessing.set_start_method('spawn')
    net_in_conn, cam_out_conn = Pipe()
    send_detect_in_conn, net_out_conn = Pipe()

    stream_reader_process = Process(target=cam_reader, 
                                    args=(cam_out_conn, cam_source))
    rob_percept_process = Process(target=main, 
                                    args=(net_in_conn, net_out_conn))
    post_detect_process = Process(target=post_detections, 
                                    args=(send_detect_in_conn,))
    # start the receiver
    stream_reader_process.start()
    rob_percept_process.start()
    post_detect_process.start()
    
    # wait for all processes to finish
    rob_percept_process.join()
    stream_reader_process.kill()
    post_detect_process.kill()
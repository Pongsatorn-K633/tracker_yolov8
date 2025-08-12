# pylint: disable=import-error,no-member,too-many-locals,too-many-statements,too-many-branches
"""YOLOv11 + StrongSORT tracking with ReID"""
from typing import Optional, Union, List
import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.checks import check_file, check_imshow, print_args, check_requirements
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.ops import Profile
from ultralytics.utils.plotting import Annotator, colors

from trackers.multi_tracker_zoo import create_tracker  # type: ignore

@torch.no_grad()
def run(
        source: Union[str, int] = '0',
        yolo_weights: Union[str, Path, List] = WEIGHTS / 'yolo11n.pt',
        reid_weights: Union[str, Path] = WEIGHTS / 'osnet_x0_25_msmt17.pt',
        tracking_method: str = 'strongsort',
        tracking_config: Optional[Union[str, Path]] = None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=[2],  # filter by class: --class 0, or --class 0 2 3 (2 is car in COCO)
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):

    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    if isinstance(yolo_weights, list):
        yolo_weights = yolo_weights[0]
    is_seg = '-seg' in str(yolo_weights)
    model = YOLO(yolo_weights)
    names = model.names

    # Setup video capture
    import cv2
    if source.isnumeric():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    # Check if webcam
    webcam = source == 0 or str(source).startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if show_vid:
        show_vid = check_imshow(warn=True)
    
    bs = 1
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs
    

    # Run tracking
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    frame_idx = 0
    
    while True:
        ret, im0 = cap.read()
        if not ret:
            break
            
        path = source if isinstance(source, str) else f"frame_{frame_idx}"
        s = f'{frame_idx}: '
        
        curr_frames[0] = im0
        
        with dt[0]:
            # Resize image to model input size
            im = cv2.resize(im0, (imgsz[1], imgsz[0]))
        
        # Inference using YOLO predict
        with dt[1]:
            results = model.predict(im0, conf=conf_thres, iou=iou_thres, classes=classes, 
                                   agnostic_nms=agnostic_nms, max_det=max_det, verbose=False)
        
        # Process results
        with dt[2]:
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    # Convert to the expected format
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    
                    # Create detection array in expected format [x1, y1, x2, y2, conf, cls]
                    det = np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1)
                    det = torch.from_numpy(det)
                else:
                    det = torch.empty((0, 6))
            else:
                det = torch.empty((0, 6))
            
        # Process detections
        i = 0  # Single camera/source
        if True:  # Always process
            seen += 1
            if webcam:
                p = Path(f"webcam_{frame_idx}")
                txt_file_name = f"webcam_{frame_idx}"
                save_path = str(save_dir / f"webcam_{frame_idx}.jpg")
            else:
                p = Path(path) if isinstance(path, str) else Path(f"frame_{frame_idx}")
                txt_file_name = p.stem if isinstance(path, str) else f"frame_{frame_idx}"
                save_path = str(save_dir / f"{txt_file_name}.jpg")
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % (im0.shape[1], im0.shape[0])  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det) > 0:
                # Boxes are already in the correct format from YOLO predict

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # Create a mapping of track_id to current detection confidence
                track_to_current_conf = {}
                if len(det) > 0:
                    # Map detections to tracks by finding closest bbox matches
                    for det_idx in range(len(det)):
                        det_bbox = det[det_idx, :4].cpu().numpy()
                        det_conf = float(det[det_idx, 4].cpu().numpy())
                        
                        # Find the closest tracked object
                        min_dist = float('inf')
                        closest_track_id = None
                        for output in outputs[i]:
                            track_bbox = output[0:4]
                            if not isinstance(track_bbox, list):
                                track_bbox = track_bbox.tolist() if hasattr(track_bbox, 'tolist') else list(track_bbox)
                            
                            # Calculate IoU or distance to match detection to track
                            center_dist = ((det_bbox[0] + det_bbox[2])/2 - (track_bbox[0] + track_bbox[2])/2)**2 + \
                                         ((det_bbox[1] + det_bbox[3])/2 - (track_bbox[1] + track_bbox[3])/2)**2
                            if center_dist < min_dist:
                                min_dist = center_dist
                                closest_track_id = int(output[4])
                        
                        if closest_track_id is not None and min_dist < 10000:  # Threshold for matching
                            track_to_current_conf[closest_track_id] = det_conf

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    # Note: Segmentation masks not implemented in this version
                    
                    for j, output in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        track_id = output[4]
                        cls = output[5]
                        conf = output[6]  # Original tracking confidence
                        
                        # Use current detection confidence if available
                        current_conf = track_to_current_conf.get(int(track_id), float(conf))
                        
                        if not isinstance(bbox, list):
                            bbox = bbox.tolist()
                        # Log detection data showing current frame confidence
                        print(f"Track ID: {int(track_id)}, Class: {names[int(cls)]}, Current_Conf: {current_conf:.3f}, Track_Conf: {float(conf):.3f}, Frame: {frame_idx}")

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a', encoding='utf-8') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, track_id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            display_id = int(track_id)  # integer id
                            # Use current confidence for display
                            display_conf = current_conf
                            label = None if hide_labels else (f'{display_id} {names[c]}' if hide_conf else \
                                (f'{display_id} {display_conf:.2f}' if hide_class else f'{display_id} {names[c]} {display_conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                            
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                _ = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                # Note: save_one_box not available in current setup
                            
            else:
                # No detections in this frame
                pass
                
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if cap:  # video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
        print([d.time() for d in dt])
        print(79*"*")
        
        frame_idx += 1
        
    # Close video capture
    cap.release()
    if show_vid:
        cv2.destroyAllWindows()

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt) if seen > 0 else (0, 0, 0, 0)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolo11n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=[2], help='filter by class: --classes 0, or --classes 0 2 3 (default: 2=car)')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.base.txt', exclude=('tensorboard', 'thop', 'ultralytics'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

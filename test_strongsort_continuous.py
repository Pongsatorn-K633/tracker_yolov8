import cv2
import torch
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
from trackers import StrongSORT

def main():
    video_path = "sample2.mp4"
    yolo_model = "bangchakv2/yolov8n.pt"
    reid_weights = "bangchakv2/osnet_x0_25_msmt17.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading YOLO model...")
    model = YOLO(yolo_model)
    
    print("Loading StrongSORT tracker...")
    tracker = StrongSORT(
        model_weights=reid_weights,
        device=device,
        fp16=False,
        max_dist=0.3,          # More lenient for continuous tracking
        max_iou_dist=0.8,      # More lenient IoU
        max_age=30,            # Moderate memory
        max_unmatched_preds=5, # Moderate tolerance
        n_init=3,              # Require confirmation for new tracks
        nn_budget=150          # Moderate budget
    )
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # CONTINUOUS TRACKING: Process every 118 frames (~2.0s intervals)
    frame_skip = 118
    print(f"🎯 CONTINUOUS MODE: Processing every {frame_skip} frames ({frame_skip/fps:.2f}s intervals)")
    print("Requires 4 consecutive detections for verification")
    print("Press 'q' to quit")
    
    # Setup video writer for output (adjust fps for interval processing)
    output_path = "tracking_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = fps / frame_skip  # Adjust fps based on frame intervals
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Track car IDs and their consecutive detections
    car_id_counts = defaultdict(int)
    successful_cars = set()
    last_positions = {}
    processed_count = 0
    
    frame_idx = 0
    
    while True:
        # Skip frames to maintain interval
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                print("\nNo more frames to read")
                cap.release()
                cv2.destroyAllWindows()
                return
            frame_idx += 1
        
        processed_count += 1
        current_time = frame_idx / fps
        
        print(f"\n🎬 Frame {frame_idx} at {current_time:.2f}s (processed #{processed_count})")
        
        results = model(frame, verbose=False)
        tracks = []
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            scores = results[0].boxes.conf.cpu()
            classes = results[0].boxes.cls.cpu()
            
            # Filter for cars only (class 2) with confidence > 0.7 (HIGH threshold)
            car_mask = (classes == 2) & (scores > 0.7)
            
            if car_mask.any():
                boxes = boxes[car_mask]
                scores = scores[car_mask]
                classes = classes[car_mask]
                
                print(f"  🚗 Detected {len(boxes)} high-confidence cars")
                
                detections = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
                tracks = tracker.update(detections, frame)
        
        # Process tracking results with continuous validation
        detected_car_ids = set()
        if len(tracks) > 0:
            print(f"  📊 Tracker returned {len(tracks)} tracks")
            
            # Process all active tracks
            active_tracks = [track for track in tracks if track[7] is not None]
            
            for track in active_tracks:
                x1, y1, x2, y2, track_id, class_id, conf, _ = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                
                current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Validate track continuity
                is_valid = True
                
                # Check for suspicious jumps (smaller threshold for continuous tracking)
                if track_id in last_positions:
                    last_center = last_positions[track_id]
                    distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                                     (current_center[1] - last_center[1])**2)
                    
                    # Adjusted threshold for 2.0s intervals
                    if distance > 400:  # pixels in ~2.0s
                        is_valid = False
                        print(f"  ⚠️ Car ID {track_id}: suspicious jump {distance:.0f}px")
                
                # Skip already successful cars
                if track_id in successful_cars:
                    is_valid = False
                    print(f"  ✅ Car ID {track_id}: already successful, skipping")
                
                # Only process valid, high-confidence tracks
                if is_valid and conf > 0.7:
                    detected_car_ids.add(track_id)
                    car_id_counts[track_id] += 1
                    last_positions[track_id] = current_center
                    
                    # Draw tracking results
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID:{track_id} #{car_id_counts[track_id]} {conf:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f"  ✅ Car ID {track_id}: detection #{car_id_counts[track_id]} (conf: {conf:.2f})")
                    
                    # Check for success (4 consecutive detections)
                    if car_id_counts[track_id] == 4:
                        print(f"🏆 SUCCESS: Car ID {track_id} achieved 4 continuous detections - TRIGGER NEXT MODEL!")
                        successful_cars.add(track_id)
                        
                        # Add success indicator to frame
                        cv2.putText(frame, f"SUCCESS: Car {track_id}!", 
                                  (50, height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Display results
        if detected_car_ids:
            print(f"  📋 Active Car IDs: {sorted(detected_car_ids)}")
        else:
            print("  📋 No valid cars this frame")
        
        # Add annotations to original frame for output video
        cv2.putText(frame, f"Continuous Tracking | Frame: {frame_idx} | {current_time:.2f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Interval: {frame_skip/fps:.2f}s | Processed: {processed_count}", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Successful Cars: {len(successful_cars)}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write annotated frame to output video
        out.write(frame)
        
        # Show video with continuous tracking info
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Continuous Car Tracking', display_frame)
        
        # Quick check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Small delay to see results
        time.sleep(0.1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n🎯 Continuous tracking completed!")
    print(f"📊 Processed {processed_count} frames with {frame_skip/fps:.2f}s intervals")
    print(f"🏆 Successfully tracked {len(successful_cars)} unique cars")
    print(f"💾 Annotated video saved to: {output_path}")

if __name__ == "__main__":
    main()
    

# After trigger the next pipeline --> stop the tracking session for 2 mins
# After trigger the next pipeline --> the camera stil feed, but the worker is not activated
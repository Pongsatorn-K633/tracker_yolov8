import cv2
import torch
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def draw_zone(frame, zone_polygon, color=(255, 0, 0), thickness=3):
    """Draw tracking zone on frame"""
    pts = np.array(zone_polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)
    
    # Add semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

def setup_video_writer(output_path, fps, width, height):
    """Setup video writer for output"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def write_frame_to_video(video_writer, frame, repeat_count):
    """Write frame to video with specified repeat count"""
    for _ in range(repeat_count):
        video_writer.write(frame)

def finalize_video(video_writer):
    """Release video writer"""
    video_writer.release()

def main():
    video_path = "sample2.mp4"
    yolo_model = "bangchakv2/yolov8n.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading YOLO model...")
    model = YOLO(yolo_model)
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Define tracking zone - Gas station floor area (trapezoidal shape)
    # Based on the perspective of the gas station floor from your image
    # width 2560, height 1440
    
    tracking_zone = [
        (423, 974),   # Point 1
        (1540, 1407), # Point 2
        (1976, 806),  # Point 3
        (1364, 749)   # Point 4
    ]
    
    print(f"🎯 Tracking zone defined: {tracking_zone}")
    
    # CONTINUOUS TRACKING: Process every 118 frames (~2.0s intervals)
    frame_skip = 118
    
    print(f"🎯 CONTINUOUS MODE: Processing every {frame_skip} frames ({frame_skip/fps:.2f}s intervals)")
    print(f"🎬 Output video will have same duration as input (each processed frame shown for 2 seconds)")
    print("🔥 ZONE-FIRST TRACKING: Only cars entering the zone will be tracked!")
    print("Requires 5 consecutive detections IN ZONE for verification")
    print("🕐 24/7 MODE: Memory reset every hour to prevent overflow")
    print("Press 'q' to quit")
    
    # Setup video writer for output (same fps as input for normal playback speed)
    output_path = "tracking_output_botsort_zone_track.mp4"
    output_fps = fps  # Use same fps as input video
    out = setup_video_writer(output_path, output_fps, width, height)
    
    # Track car IDs and their consecutive detections
    car_id_counts = defaultdict(int)
    successful_cars = set()
    last_positions = {}
    processed_count = 0
    
    # ID remapping for clean sequential zone IDs
    tracker_to_zone_id = {}  # Maps tracker IDs to clean zone IDs
    next_zone_id = 1  # Next clean zone ID to assign
    
    # Store previous frame detections to filter tracking inputs
    previous_zone_cars = set()
    
    # 24/7 operation: Reset every hour (1800 snapshots at 2-sec intervals = 1 hour)
    RESET_INTERVAL = 1800  # Reset every 1800 processed frames (1 hour)
    
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
        
        # 24/7 Memory Management: Reset every hour
        if processed_count % RESET_INTERVAL == 0:
            print(f"🕐 HOURLY RESET: Clearing all tracking data (processed {processed_count} frames)")
            print(f"   📊 Before reset: {len(tracker_to_zone_id)} tracked cars, next Zone ID was {next_zone_id}")
            
            # Clear all tracking data
            tracker_to_zone_id.clear()
            car_id_counts.clear()
            successful_cars.clear()
            last_positions.clear()
            next_zone_id = 1  # Reset to 1
            
            # Reset BoT-SORT tracker state
            try:
                model.reset()
                print(f"   ✅ BoT-SORT tracker reset successfully")
            except:
                print(f"   ⚠️ BoT-SORT reset not available (continuing without reset)")
            
            print(f"   🆕 Zone IDs will start from 1 again")
        
        # Draw tracking zone on frame
        draw_zone(frame, tracking_zone, color=(0, 255, 255), thickness=3)  # Yellow zone
        
        # First run YOLO detection (without tracking) to find cars in zone
        detection_results = model(frame, verbose=False, conf=0.7, classes=[2])
        
        # Find cars currently in the tracking zone
        current_zone_cars = []
        total_detections = 0
        
        if detection_results[0].boxes is not None:
            boxes = detection_results[0].boxes.xyxy.cpu()
            scores = detection_results[0].boxes.conf.cpu()
            
            total_detections = len(boxes)
            print(f"  🔍 Total car detections: {total_detections}")
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(scores[i])
                
                # Check if detection is in zone (using bottom center)
                box_bottom = ((x1 + x2) / 2, y2)
                if point_in_polygon(box_bottom, tracking_zone):
                    current_zone_cars.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'conf': conf,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'bottom': box_bottom
                    })
        
        print(f"  🎯 Cars in zone: {len(current_zone_cars)}")
        
        # Only run tracking if there are cars in the zone
        detected_car_ids = set()
        
        if current_zone_cars:
            # Run tracking on the full frame (let tracker handle associations)
            # But we'll filter results to only zone cars afterward
            results = model.track(
                frame, 
                persist=True,
                verbose=False,
                conf=0.7,
                classes=[2],
                tracker="botsort_reid.yaml"
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                scores = results[0].boxes.conf.cpu()
                track_ids = results[0].boxes.id.cpu().int()
                
                print(f"  📊 Total tracked objects: {len(track_ids)}")
                
                # Filter tracked objects to only those in zone
                zone_tracks = []
                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = boxes[i]
                    conf = float(scores[i])
                    
                    # Check if this tracked object is in our zone
                    box_bottom = ((x1 + x2) / 2, y2)
                    if point_in_polygon(box_bottom, tracking_zone):
                        zone_tracks.append({
                            'id': int(track_id),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf,
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'bottom': box_bottom
                        })
                
                print(f"  ✅ Zone tracks: {len(zone_tracks)}")
                
                # Process each zone track
                for track in zone_tracks:
                    tracker_id = track['id']  # Original tracker ID
                    x1, y1, x2, y2 = track['bbox']
                    conf = track['conf']
                    box_center = track['center']
                    
                    # Map tracker ID to clean zone ID
                    if tracker_id not in tracker_to_zone_id:
                        tracker_to_zone_id[tracker_id] = next_zone_id
                        print(f"  🆕 New car: Tracker ID {tracker_id} → Zone ID {next_zone_id}")
                        next_zone_id += 1
                    
                    zone_id = tracker_to_zone_id[tracker_id]  # Clean sequential ID
                    
                    # Validate track continuity (use tracker_id for internal logic)
                    is_valid = True
                    
                    # Check for suspicious jumps
                    if tracker_id in last_positions:
                        last_center = last_positions[tracker_id]
                        distance = np.sqrt((box_center[0] - last_center[0])**2 + 
                                         (box_center[1] - last_center[1])**2)
                        
                        if distance > 400:  # pixels in ~2.0s
                            is_valid = False
                            print(f"  ⚠️ Zone ID {zone_id} (Tracker {tracker_id}): suspicious jump {distance:.0f}px")
                    
                    # Skip already successful cars (use zone_id for user logic)
                    if zone_id in successful_cars:
                        is_valid = False
                        print(f"  ✅ Zone ID {zone_id}: already successful, skipping")
                    
                    # Only process valid, high-confidence zone tracks
                    if is_valid and conf > 0.7:
                        detected_car_ids.add(zone_id)  # Use zone_id for display
                        car_id_counts[zone_id] += 1
                        last_positions[tracker_id] = box_center  # Track by tracker_id internally
                        
                        # Draw tracking results with clean zone ID
                        zone_color = (0, 255, 0)  # Green for zone cars
                        cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 2)
                        cv2.putText(frame, f'ZONE ID:{zone_id}', 
                                  (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                        cv2.putText(frame, f'#{car_id_counts[zone_id]} {conf:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                        
                        # Draw center point
                        cv2.circle(frame, (int(track['bottom'][0]), int(track['bottom'][1])), 5, zone_color, -1)
                        
                        print(f"  ✅ Zone ID {zone_id} (Tracker {tracker_id}): ZONE detection #{car_id_counts[zone_id]} (conf: {conf:.2f})")
                        
                        # Check for success (5 consecutive detections IN ZONE)
                        if car_id_counts[zone_id] == 5:
                            print(f"🏆 SUCCESS: Zone ID {zone_id} achieved 5 continuous ZONE detections - TRIGGER NEXT MODEL!")
                            successful_cars.add(zone_id)
                            
                            # Add success indicator to frame
                            cv2.putText(frame, f"SUCCESS: Zone Car {zone_id}!", 
                                      (50, height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            print("  📋 No cars in zone - no tracking performed")
            
            # Draw any cars outside the zone in red (for reference)
            if detection_results[0].boxes is not None:
                boxes = detection_results[0].boxes.xyxy.cpu()
                scores = detection_results[0].boxes.conf.cpu()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = float(scores[i])
                    
                    box_bottom = ((x1 + x2) / 2, y2)
                    if not point_in_polygon(box_bottom, tracking_zone):
                        # Draw cars outside zone in red (not tracked)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(frame, f'OUT {conf:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Display results
        if detected_car_ids:
            print(f"  📋 Active Zone IDs: {sorted(detected_car_ids)} (Clean sequential IDs)")
        
        # Show ID mapping for debugging
        if tracker_to_zone_id:
            mapping_str = ", ".join([f"Tracker{k}→Zone{v}" for k, v in tracker_to_zone_id.items()])
            print(f"  🔄 ID Mapping: {mapping_str}")
        
        # Add annotations to frame
        cv2.putText(frame, f"BoT-SORT Zone-First Tracking | Frame: {frame_idx} | {current_time:.2f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Zone Cars: {len(current_zone_cars)} | Active Tracks: {len(detected_car_ids)}", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Successful Cars: {len(successful_cars)}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "TRACKING ZONE", 
                   (tracking_zone[0][0], tracking_zone[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Write annotated frame to output video (repeat for 2 seconds duration)
        write_frame_to_video(out, frame, frame_skip)
        
        # Show video with zone tracking info
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('BoT-SORT Zone-First Tracking', display_frame)
        
        # Quick check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Small delay to see results
        time.sleep(0.1)
    
    cap.release()
    finalize_video(out)
    cv2.destroyAllWindows()
    print(f"\n🎯 BoT-SORT zone-first tracking completed!")
    print(f"📊 Processed {processed_count} frames with {frame_skip/fps:.2f}s intervals")
    print(f"🏆 Successfully tracked {len(successful_cars)} unique cars IN ZONE")
    print(f"💾 Annotated video saved to: {output_path}")

if __name__ == "__main__":
    main()
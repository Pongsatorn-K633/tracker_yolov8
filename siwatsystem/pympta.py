import os
import json
import logging
import torch
import cv2
import zipfile
import shutil
import traceback
import redis
import time
import uuid
import concurrent.futures
from ultralytics import YOLO
from urllib.parse import urlparse
from .database import DatabaseManager
from .strongsort_integration import create_pipeline_tracker, integrate_tracking_with_ultralytics

# Create a logger specifically for this module
logger = logging.getLogger("detector_worker.pympta")

def validate_redis_config(redis_config: dict) -> bool:
    """Validate Redis configuration parameters."""
    required_fields = ["host", "port"]
    for field in required_fields:
        if field not in redis_config:
            logger.error(f"Missing required Redis config field: {field}")
            return False
    
    if not isinstance(redis_config["port"], int) or redis_config["port"] <= 0:
        logger.error(f"Invalid Redis port: {redis_config['port']}")
        return False
    
    return True

def validate_postgresql_config(pg_config: dict) -> bool:
    """Validate PostgreSQL configuration parameters."""
    required_fields = ["host", "port", "database", "username", "password"]
    for field in required_fields:
        if field not in pg_config:
            logger.error(f"Missing required PostgreSQL config field: {field}")
            return False
    
    if not isinstance(pg_config["port"], int) or pg_config["port"] <= 0:
        logger.error(f"Invalid PostgreSQL port: {pg_config['port']}")
        return False
    
    return True

def crop_region_by_class(frame, regions_dict, class_name):
    """Crop a specific region from frame based on detected class."""
    # Try exact match first
    if class_name in regions_dict:
        target_class = class_name
    else:
        # Try case-insensitive match
        class_name_lower = class_name.lower()
        matching_classes = [k for k in regions_dict.keys() if k.lower() == class_name_lower]
        
        if matching_classes:
            target_class = matching_classes[0]
            logger.debug(f"Found case-insensitive match: '{class_name}' -> '{target_class}'")
        else:
            logger.warning(f"Class '{class_name}' not found in detected regions: {list(regions_dict.keys())}")
            return None
    
    bbox = regions_dict[target_class]['bbox']
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    
    if cropped.size == 0:
        logger.warning(f"Empty crop for class '{target_class}' with bbox {bbox}")
        return None
    
    logger.debug(f"Successfully cropped region for '{target_class}': {bbox}")
    return cropped

def format_action_context(base_context, additional_context=None):
    """Format action context with dynamic values."""
    context = {**base_context}
    if additional_context:
        context.update(additional_context)
    return context

def load_pipeline_node(node_config: dict, mpta_dir: str, redis_client, db_manager=None) -> dict:
    # Recursively load a model node from configuration.
    model_path = os.path.join(mpta_dir, node_config["modelFile"])
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found. Current directory: {os.getcwd()}")
        logger.error(f"Directory content: {os.listdir(os.path.dirname(model_path))}")
        raise FileNotFoundError(f"Model file {model_path} not found.")
    logger.info(f"Loading model for node {node_config['modelId']} from {model_path}")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Moving model {node_config['modelId']} to GPU")
        model.to("cuda")
    else:
        logger.info(f"CUDA not available. Using CPU for model {node_config['modelId']}")

    # Configure tracking if enabled
    tracking_config = node_config.get("tracking", {})
    pipeline_tracker = None
    ultralytics_tracking_params = {}
    
    if tracking_config.get("enabled", False):
        # Create pipeline tracker instance
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_tracker = create_pipeline_tracker(tracking_config, mpta_dir, device_str)
        
        # Get ultralytics tracking parameters
        ultralytics_tracking_params = integrate_tracking_with_ultralytics(model, tracking_config, mpta_dir)
        
        logger.info(f"Tracking configured for {node_config['modelId']}: method={tracking_config.get('method', 'default')}")

    # Prepare trigger class indices for optimization
    trigger_classes = node_config.get("triggerClasses", [])
    trigger_class_indices = None
    if trigger_classes and hasattr(model, "names"):
        # Convert class names to indices for the model
        trigger_class_indices = [i for i, name in model.names.items() 
                                if name in trigger_classes]
        logger.debug(f"Converted trigger classes to indices: {trigger_class_indices}")

    node = {
        "modelId": node_config["modelId"],
        "modelFile": node_config["modelFile"],
        "triggerClasses": trigger_classes,
        "triggerClassIndices": trigger_class_indices,
        "crop": node_config.get("crop", False),
        "cropClass": node_config.get("cropClass"),
        "minConfidence": node_config.get("minConfidence", None),
        "multiClass": node_config.get("multiClass", False),
        "expectedClasses": node_config.get("expectedClasses", []),
        "parallel": node_config.get("parallel", False),
        "actions": node_config.get("actions", []),
        "parallelActions": node_config.get("parallelActions", []),
        "model": model,
        "pipeline_tracker": pipeline_tracker,
        "tracking": tracking_config,
        "ultralytics_tracking_params": ultralytics_tracking_params,
        "branches": [],
        "redis_client": redis_client,
        "db_manager": db_manager
    }
    logger.debug(f"Configured node {node_config['modelId']} with trigger classes: {node['triggerClasses']}")
    for child in node_config.get("branches", []):
        logger.debug(f"Loading branch for parent node {node_config['modelId']}")
        node["branches"].append(load_pipeline_node(child, mpta_dir, redis_client, db_manager))
    return node

def load_pipeline_from_zip(zip_source: str, target_dir: str) -> dict:
    logger.info(f"Attempting to load pipeline from {zip_source} to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "pipeline.mpta")
    
    # Check if it's a local file path (works for both Unix and Windows paths)
    if os.path.exists(zip_source) or os.path.isabs(zip_source):
        # It's a local file path
        local_path = zip_source
        logger.debug(f"Using local file path: {local_path}")
        
        if os.path.exists(local_path):
            try:
                shutil.copy(local_path, zip_path)
                logger.info(f"Copied local .mpta file from {local_path} to {zip_path}")
            except Exception as e:
                logger.error(f"Failed to copy local .mpta file from {local_path}: {str(e)}", exc_info=True)
                return None
        else:
            logger.error(f"Local file {local_path} does not exist. Current directory: {os.getcwd()}")
            return None
    else:
        # Try to parse as URL for backward compatibility
        parsed = urlparse(zip_source)
        if parsed.scheme in ("http", "https"):
            logger.error(f"HTTP download functionality has been moved. Use a local file path here. Received: {zip_source}")
            return None
        elif parsed.scheme == "file":
            local_path = parsed.path
            logger.debug(f"Using file:// URL path: {local_path}")
            if os.path.exists(local_path):
                try:
                    shutil.copy(local_path, zip_path)
                    logger.info(f"Copied local .mpta file from {local_path} to {zip_path}")
                except Exception as e:
                    logger.error(f"Failed to copy local .mpta file from {local_path}: {str(e)}", exc_info=True)
                    return None
            else:
                logger.error(f"File URL path {local_path} does not exist")
                return None
        else:
            logger.error(f"Unsupported path format: {zip_source}")
            return None

    try:
        if not os.path.exists(zip_path):
            logger.error(f"Zip file not found at expected location: {zip_path}")
            return None
            
        logger.debug(f"Extracting .mpta file from {zip_path} to {target_dir}")
        # Extract contents and track the directories created
        extracted_dirs = []
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            logger.debug(f"Files in .mpta archive: {file_list}")
            
            # Extract and track the top-level directories
            for file_path in file_list:
                parts = file_path.split('/')
                if len(parts) > 1:
                    top_dir = parts[0]
                    if top_dir and top_dir not in extracted_dirs:
                        extracted_dirs.append(top_dir)
            
            # Now extract the files
            zip_ref.extractall(target_dir)
            
        logger.info(f"Successfully extracted .mpta file to {target_dir}")
        logger.debug(f"Extracted directories: {extracted_dirs}")
        
        # Check what was actually created after extraction
        actual_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        logger.debug(f"Actual directories created: {actual_dirs}")
    except zipfile.BadZipFile as e:
        logger.error(f"Bad zip file {zip_path}: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to extract .mpta file {zip_path}: {str(e)}", exc_info=True)
        return None
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.debug(f"Removed temporary zip file: {zip_path}")

    # Use the first extracted directory if it exists, otherwise use the expected name
    pipeline_name = os.path.basename(zip_source)
    pipeline_name = os.path.splitext(pipeline_name)[0]
    
    # Find the directory with pipeline.json
    mpta_dir = None
    # First try the expected directory name
    expected_dir = os.path.join(target_dir, pipeline_name)
    if os.path.exists(expected_dir) and os.path.exists(os.path.join(expected_dir, "pipeline.json")):
        mpta_dir = expected_dir
        logger.debug(f"Found pipeline.json in the expected directory: {mpta_dir}")
    else:
        # Look through all subdirectories for pipeline.json
        for subdir in actual_dirs:
            potential_dir = os.path.join(target_dir, subdir)
            if os.path.exists(os.path.join(potential_dir, "pipeline.json")):
                mpta_dir = potential_dir
                logger.info(f"Found pipeline.json in directory: {mpta_dir} (different from expected: {expected_dir})")
                break
    
    if not mpta_dir:
        logger.error(f"Could not find pipeline.json in any extracted directory. Directory content: {os.listdir(target_dir)}")
        return None
        
    pipeline_json_path = os.path.join(mpta_dir, "pipeline.json")
    if not os.path.exists(pipeline_json_path):
        logger.error(f"pipeline.json not found in the .mpta file. Files in directory: {os.listdir(mpta_dir)}")
        return None

    try:
        with open(pipeline_json_path, "r") as f:
            pipeline_config = json.load(f)
        logger.info(f"Successfully loaded pipeline configuration from {pipeline_json_path}")
        logger.debug(f"Pipeline config: {json.dumps(pipeline_config, indent=2)}")
        
        # Establish Redis connection if configured
        redis_client = None
        if "redis" in pipeline_config:
            redis_config = pipeline_config["redis"]
            if not validate_redis_config(redis_config):
                logger.error("Invalid Redis configuration, skipping Redis connection")
            else:
                try:
                    redis_client = redis.Redis(
                        host=redis_config["host"],
                        port=redis_config["port"],
                        password=redis_config.get("password"),
                        db=redis_config.get("db", 0),
                        decode_responses=True
                    )
                    redis_client.ping()
                    logger.info(f"Successfully connected to Redis at {redis_config['host']}:{redis_config['port']}")
                except redis.exceptions.ConnectionError as e:
                    logger.error(f"Failed to connect to Redis: {e}")
                    redis_client = None
        
        # Establish PostgreSQL connection if configured
        db_manager = None
        if "postgresql" in pipeline_config:
            pg_config = pipeline_config["postgresql"]
            if not validate_postgresql_config(pg_config):
                logger.error("Invalid PostgreSQL configuration, skipping database connection")
            else:
                try:
                    db_manager = DatabaseManager(pg_config)
                    if db_manager.connect():
                        logger.info(f"Successfully connected to PostgreSQL at {pg_config['host']}:{pg_config['port']}")
                    else:
                        logger.error("Failed to connect to PostgreSQL")
                        db_manager = None
                except Exception as e:
                    logger.error(f"Error initializing PostgreSQL connection: {e}")
                    db_manager = None
        
        return load_pipeline_node(pipeline_config["pipeline"], mpta_dir, redis_client, db_manager)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing pipeline.json: {str(e)}", exc_info=True)
        return None
    except KeyError as e:
        logger.error(f"Missing key in pipeline.json: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading pipeline.json: {str(e)}", exc_info=True)
        return None

def execute_actions(node, frame, detection_result, regions_dict=None):
    if not node["redis_client"] or not node["actions"]:
        return

    # Create a dynamic context for this detection event
    from datetime import datetime
    action_context = {
        **detection_result,
        "timestamp_ms": int(time.time() * 1000),
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        "filename": f"{uuid.uuid4()}.jpg"
    }

    for action in node["actions"]:
        try:
            if action["type"] == "redis_save_image":
                key = action["key"].format(**action_context)
                
                # Check if we need to crop a specific region
                region_name = action.get("region")
                image_to_save = frame
                
                if region_name and regions_dict:
                    cropped_image = crop_region_by_class(frame, regions_dict, region_name)
                    if cropped_image is not None:
                        image_to_save = cropped_image
                        logger.debug(f"Cropped region '{region_name}' for redis_save_image")
                    else:
                        logger.warning(f"Could not crop region '{region_name}', saving full frame instead")
                
                # Encode image with specified format and quality (default to JPEG)
                img_format = action.get("format", "jpeg").lower()
                quality = action.get("quality", 90)
                
                if img_format == "jpeg":
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', image_to_save, encode_params)
                elif img_format == "png":
                    success, buffer = cv2.imencode('.png', image_to_save)
                else:
                    success, buffer = cv2.imencode('.jpg', image_to_save, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                if not success:
                    logger.error(f"Failed to encode image for redis_save_image")
                    continue
                
                expire_seconds = action.get("expire_seconds")
                if expire_seconds:
                    node["redis_client"].setex(key, expire_seconds, buffer.tobytes())
                    logger.info(f"Saved image to Redis with key: {key} (expires in {expire_seconds}s)")
                else:
                    node["redis_client"].set(key, buffer.tobytes())
                    logger.info(f"Saved image to Redis with key: {key}")
                action_context["image_key"] = key
            elif action["type"] == "redis_publish":
                channel = action["channel"]
                try:
                    # Handle JSON message format by creating it programmatically
                    message_template = action["message"]
                    
                    # Check if the message is JSON-like (starts and ends with braces)
                    if message_template.strip().startswith('{') and message_template.strip().endswith('}'):
                        # Create JSON data programmatically to avoid formatting issues
                        json_data = {}
                        
                        # Add common fields
                        json_data["event"] = "frontal_detected"
                        json_data["display_id"] = action_context.get("display_id", "unknown")
                        json_data["session_id"] = action_context.get("session_id")
                        json_data["timestamp"] = action_context.get("timestamp", "")
                        json_data["image_key"] = action_context.get("image_key", "")
                        
                        # Convert to JSON string
                        message = json.dumps(json_data)
                    else:
                        # Use regular string formatting for non-JSON messages
                        message = message_template.format(**action_context)
                    
                    # Publish to Redis
                    if not node["redis_client"]:
                        logger.error("Redis client is None, cannot publish message")
                        continue
                        
                    # Test Redis connection
                    try:
                        node["redis_client"].ping()
                        logger.debug("Redis connection is active")
                    except Exception as ping_error:
                        logger.error(f"Redis connection test failed: {ping_error}")
                        continue
                    
                    result = node["redis_client"].publish(channel, message)
                    logger.info(f"Published message to Redis channel '{channel}': {message}")
                    logger.info(f"Redis publish result (subscribers count): {result}")
                    
                    # Additional debug info
                    if result == 0:
                        logger.warning(f"No subscribers listening to channel '{channel}'")
                    else:
                        logger.info(f"Message delivered to {result} subscriber(s)")
                    
                except KeyError as e:
                    logger.error(f"Missing key in redis_publish message template: {e}")
                    logger.debug(f"Available context keys: {list(action_context.keys())}")
                except Exception as e:
                    logger.error(f"Error in redis_publish action: {e}")
                    logger.debug(f"Message template: {action['message']}")
                    logger.debug(f"Available context keys: {list(action_context.keys())}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error executing action {action['type']}: {e}")

def execute_parallel_actions(node, frame, detection_result, regions_dict):
    """Execute parallel actions after all required branches have completed."""
    if not node.get("parallelActions"):
        return
    
    logger.debug("Executing parallel actions...")
    branch_results = detection_result.get("branch_results", {})
    
    for action in node["parallelActions"]:
        try:
            action_type = action.get("type")
            logger.debug(f"Processing parallel action: {action_type}")
            
            if action_type == "postgresql_update_combined":
                # Check if all required branches have completed
                wait_for_branches = action.get("waitForBranches", [])
                missing_branches = [branch for branch in wait_for_branches if branch not in branch_results]
                
                if missing_branches:
                    logger.warning(f"Cannot execute postgresql_update_combined: missing branch results for {missing_branches}")
                    continue
                
                logger.info(f"All required branches completed: {wait_for_branches}")
                
                # Execute the database update
                execute_postgresql_update_combined(node, action, detection_result, branch_results)
            else:
                logger.warning(f"Unknown parallel action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing parallel action {action.get('type', 'unknown')}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")

def execute_postgresql_update_combined(node, action, detection_result, branch_results):
    """Execute a PostgreSQL update with combined branch results."""
    if not node.get("db_manager"):
        logger.error("No database manager available for postgresql_update_combined action")
        return
        
    try:
        table = action["table"]
        key_field = action["key_field"]
        key_value_template = action["key_value"]
        fields = action["fields"]
        
        # Create context for key value formatting
        action_context = {**detection_result}
        key_value = key_value_template.format(**action_context)
        
        logger.info(f"Executing database update: table={table}, {key_field}={key_value}")
        
        # Process field mappings
        mapped_fields = {}
        for db_field, value_template in fields.items():
            try:
                mapped_value = resolve_field_mapping(value_template, branch_results, action_context)
                if mapped_value is not None:
                    mapped_fields[db_field] = mapped_value
                    logger.debug(f"Mapped field: {db_field} = {mapped_value}")
                else:
                    logger.warning(f"Could not resolve field mapping for {db_field}: {value_template}")
            except Exception as e:
                logger.error(f"Error mapping field {db_field} with template '{value_template}': {e}")
        
        if not mapped_fields:
            logger.warning("No fields mapped successfully, skipping database update")
            return
            
        # Execute the database update
        success = node["db_manager"].execute_update(table, key_field, key_value, mapped_fields)
        
        if success:
            logger.info(f"Successfully updated database: {table} with {len(mapped_fields)} fields")
        else:
            logger.error(f"Failed to update database: {table}")
            
    except KeyError as e:
        logger.error(f"Missing required field in postgresql_update_combined action: {e}")
    except Exception as e:
        logger.error(f"Error in postgresql_update_combined action: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

def resolve_field_mapping(value_template, branch_results, action_context):
    """Resolve field mapping templates like {car_brand_cls_v1.brand}."""
    try:
        # Handle simple context variables first (non-branch references)
        if not '.' in value_template:
            return value_template.format(**action_context)
        
        # Handle branch result references like {model_id.field}
        import re
        branch_refs = re.findall(r'\{([^}]+\.[^}]+)\}', value_template)
        
        resolved_template = value_template
        for ref in branch_refs:
            try:
                model_id, field_name = ref.split('.', 1)
                
                if model_id in branch_results:
                    branch_data = branch_results[model_id]
                    if field_name in branch_data:
                        field_value = branch_data[field_name]
                        resolved_template = resolved_template.replace(f'{{{ref}}}', str(field_value))
                        logger.debug(f"Resolved {ref} to {field_value}")
                    else:
                        logger.warning(f"Field '{field_name}' not found in branch '{model_id}' results. Available fields: {list(branch_data.keys())}")
                        return None
                else:
                    logger.warning(f"Branch '{model_id}' not found in results. Available branches: {list(branch_results.keys())}")
                    return None
            except ValueError as e:
                logger.error(f"Invalid branch reference format: {ref}")
                return None
        
        # Format any remaining simple variables
        try:
            final_value = resolved_template.format(**action_context)
            return final_value
        except KeyError as e:
            logger.warning(f"Could not resolve context variable in template: {e}")
            return resolved_template
            
    except Exception as e:
        logger.error(f"Error resolving field mapping '{value_template}': {e}")
        return None

def validate_pipeline_execution(node, regions_dict):
    """
    Pre-validate that all required branches will execute successfully before 
    committing to Redis actions and database records.
    
    Returns:
        - (True, []) if pipeline can execute completely
        - (False, missing_branches) if some required branches won't execute
    """
    # Get all branches that parallel actions are waiting for
    required_branches = set()
    
    for action in node.get("parallelActions", []):
        if action.get("type") == "postgresql_update_combined":
            wait_for_branches = action.get("waitForBranches", [])
            required_branches.update(wait_for_branches)
    
    if not required_branches:
        # No parallel actions requiring specific branches
        logger.debug("No parallel actions with waitForBranches - validation passes")
        return True, []
    
    logger.debug(f"Pre-validation: checking if required branches {list(required_branches)} will execute")
    
    # Check each required branch
    missing_branches = []
    
    for branch in node.get("branches", []):
        branch_id = branch["modelId"]
        
        if branch_id not in required_branches:
            continue  # This branch is not required by parallel actions
            
        # Check if this branch would be triggered
        trigger_classes = branch.get("triggerClasses", [])
        min_conf = branch.get("minConfidence", 0)
        
        branch_triggered = False
        for det_class in regions_dict:
            det_confidence = regions_dict[det_class]["confidence"]
            
            if (det_class in trigger_classes and det_confidence >= min_conf):
                branch_triggered = True
                logger.debug(f"Pre-validation: branch {branch_id} WILL be triggered by {det_class} (conf={det_confidence:.3f} >= {min_conf})")
                break
        
        if not branch_triggered:
            missing_branches.append(branch_id)
            logger.warning(f"Pre-validation: branch {branch_id} will NOT be triggered - no matching classes or insufficient confidence")
            logger.debug(f"  Required: {trigger_classes} with min_conf={min_conf}")
            logger.debug(f"  Available: {[(cls, regions_dict[cls]['confidence']) for cls in regions_dict]}")
    
    if missing_branches:
        logger.error(f"Pipeline pre-validation FAILED: required branches {missing_branches} will not execute")
        return False, missing_branches
    else:
        logger.info(f"Pipeline pre-validation PASSED: all required branches {list(required_branches)} will execute")
        return True, []

def run_pipeline(frame, node: dict, return_bbox: bool=False, context=None):
    """
    Enhanced pipeline that supports:
    - Multi-class detection (detecting multiple classes simultaneously)
    - Parallel branch processing
    - Region-based actions and cropping
    - Context passing for session/camera information
    """
    try:
        task = getattr(node["model"], "task", None)

        # ─── Classification stage ───────────────────────────────────
        if task == "classify":
            results = node["model"].predict(frame, stream=False)
            if not results:
                return (None, None) if return_bbox else None

            r = results[0]
            probs = r.probs
            if probs is None:
                return (None, None) if return_bbox else None

            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            class_name = node["model"].names[top1_idx]

            det = {
                "class": class_name,
                "confidence": top1_conf,
                "id": None,
                class_name: class_name  # Add class name as key for backward compatibility
            }
            
            # Add specific field mappings for database operations based on model type
            model_id = node.get("modelId", "").lower()
            if "brand" in model_id or "brand_cls" in model_id:
                det["brand"] = class_name
            elif "bodytype" in model_id or "body" in model_id:
                det["body_type"] = class_name
            elif "color" in model_id:
                det["color"] = class_name
            
            execute_actions(node, frame, det)
            return (det, None) if return_bbox else det

        # ─── Detection stage - Multi-class support ──────────────────
        tk = node["triggerClassIndices"]
        logger.debug(f"Running detection for node {node['modelId']} with trigger classes: {node.get('triggerClasses', [])} (indices: {tk})")
        logger.debug(f"Node configuration: minConfidence={node['minConfidence']}, multiClass={node.get('multiClass', False)}")
        
        # Use regular predict instead of track if tracking has issues
        try:
            # Configure tracking parameters
            track_kwargs = {
                "stream": False,
                **({"classes": tk} if tk else {}),
                **node.get("ultralytics_tracking_params", {})
            }
            
            # For now, use predict instead of track to avoid ultralytics tracking issues
            # This will still do detection, just without built-in tracking
            logger.debug(f"Using predict for {node['modelId']} (tracking handled separately)")
            res = node["model"].predict(frame, stream=False, **({"classes": tk} if tk else {}))[0]
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return (None, None) if return_bbox else None

        # Collect all detections above confidence threshold
        all_detections = []
        all_boxes = []
        regions_dict = {}
        
        logger.debug(f"Raw detection results from model: {len(res.boxes) if res.boxes is not None else 0} detections")
        
        for i, box in enumerate(res.boxes):
            conf = float(box.cpu().conf[0])
            cid = int(box.cpu().cls[0])
            name = node["model"].names[cid]
            
            logger.debug(f"Detection {i}: class='{name}' (id={cid}), confidence={conf:.3f}, threshold={node['minConfidence']}")
            
            if conf < node["minConfidence"]:
                logger.debug(f"  -> REJECTED: confidence {conf:.3f} < threshold {node['minConfidence']}")
                continue
                
            xy = box.cpu().xyxy[0]
            x1, y1, x2, y2 = map(int, xy)
            bbox = (x1, y1, x2, y2)
            
            # Extract track ID if available
            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id.item())
            
            detection = {
                "class": name,
                "confidence": conf,
                "id": track_id,
                "track_id": track_id,
                "bbox": bbox
            }
            
            # Apply stability threshold for tracking
            stability_threshold = node.get("tracking", {}).get("stabilityThreshold", 0)
            if stability_threshold > 0 and track_id is not None:
                # Track stability logic would go here
                # For now, we accept all tracked detections
                detection["stable"] = True
            else:
                detection["stable"] = True
            
            all_detections.append(detection)
            all_boxes.append(bbox)
            
            logger.debug(f"  -> ACCEPTED: {name} with confidence {conf:.3f}, bbox={bbox}")
            
            # Store highest confidence detection for each class
            if name not in regions_dict or conf > regions_dict[name]["confidence"]:
                regions_dict[name] = {
                    "bbox": bbox,
                    "confidence": conf,
                    "detection": detection
                }
                logger.debug(f"  -> Updated regions_dict['{name}'] with confidence {conf:.3f}")

        logger.info(f"Detection summary: {len(all_detections)} accepted detections from {len(res.boxes) if res.boxes is not None else 0} total")
        logger.info(f"Detected classes: {list(regions_dict.keys())}")

        if not all_detections:
            logger.warning("No detections above confidence threshold - returning null")
            return (None, None) if return_bbox else None

        # ─── Multi-class validation ─────────────────────────────────
        if node.get("multiClass", False) and node.get("expectedClasses"):
            expected_classes = node["expectedClasses"]
            detected_classes = list(regions_dict.keys())
            
            logger.info(f"Multi-class validation: expected={expected_classes}, detected={detected_classes}")
            
            # Check if at least one expected class is detected (flexible mode)
            matching_classes = [cls for cls in expected_classes if cls in detected_classes]
            missing_classes = [cls for cls in expected_classes if cls not in detected_classes]
            
            logger.debug(f"Matching classes: {matching_classes}, Missing classes: {missing_classes}")
            
            if not matching_classes:
                # No expected classes found at all
                logger.warning(f"PIPELINE REJECTED: No expected classes detected. Expected: {expected_classes}, Detected: {detected_classes}")
                return (None, None) if return_bbox else None
            
            if missing_classes:
                logger.info(f"Partial multi-class detection: {matching_classes} found, {missing_classes} missing")
            else:
                logger.info(f"Complete multi-class detection success: {detected_classes}")
        else:
            logger.debug("No multi-class validation - proceeding with all detections")

        # ─── Pre-validate pipeline execution ────────────────────────
        pipeline_valid, missing_branches = validate_pipeline_execution(node, regions_dict)
        
        if not pipeline_valid:
            logger.error(f"Pipeline execution validation FAILED - required branches {missing_branches} cannot execute")
            logger.error("Aborting pipeline: no Redis actions or database records will be created")
            return (None, None) if return_bbox else None

        # ─── Execute actions with region information ────────────────
        detection_result = {
            "detections": all_detections,
            "regions": regions_dict,
            **(context or {})
        }
        
        # ─── Create initial database record when Car+Frontal detected ────
        if node.get("db_manager") and node.get("multiClass", False):
            # Only create database record if we have both Car and Frontal
            has_car = "Car" in regions_dict
            has_frontal = "Frontal" in regions_dict
            
            if has_car and has_frontal:
                # Generate UUID session_id since client session is None for now
                import uuid as uuid_lib
                from datetime import datetime
                generated_session_id = str(uuid_lib.uuid4())
                
                # Insert initial detection record
                display_id = detection_result.get("display_id", "unknown")
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                
                inserted_session_id = node["db_manager"].insert_initial_detection(
                    display_id=display_id,
                    captured_timestamp=timestamp,
                    session_id=generated_session_id
                )
                
                if inserted_session_id:
                    # Update detection_result with the generated session_id for actions and branches
                    detection_result["session_id"] = inserted_session_id
                    detection_result["timestamp"] = timestamp  # Update with proper timestamp
                    logger.info(f"Created initial database record with session_id: {inserted_session_id}")
            else:
                logger.debug(f"Database record not created - missing required classes. Has Car: {has_car}, Has Frontal: {has_frontal}")
        
        execute_actions(node, frame, detection_result, regions_dict)

        # ─── Parallel branch processing ─────────────────────────────
        if node["branches"]:
            branch_results = {}
            
            # Filter branches that should be triggered
            active_branches = []
            for br in node["branches"]:
                trigger_classes = br.get("triggerClasses", [])
                min_conf = br.get("minConfidence", 0)
                
                logger.debug(f"Evaluating branch {br['modelId']}: trigger_classes={trigger_classes}, min_conf={min_conf}")
                
                # Check if any detected class matches branch trigger
                branch_triggered = False
                for det_class in regions_dict:
                    det_confidence = regions_dict[det_class]["confidence"]
                    logger.debug(f"  Checking detected class '{det_class}' (confidence={det_confidence:.3f}) against triggers {trigger_classes}")
                    
                    if (det_class in trigger_classes and det_confidence >= min_conf):
                        active_branches.append(br)
                        branch_triggered = True
                        logger.info(f"Branch {br['modelId']} activated by class '{det_class}' (conf={det_confidence:.3f} >= {min_conf})")
                        break
                
                if not branch_triggered:
                    logger.debug(f"Branch {br['modelId']} not triggered - no matching classes or insufficient confidence")
            
            if active_branches:
                if node.get("parallel", False) or any(br.get("parallel", False) for br in active_branches):
                    # Run branches in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_branches)) as executor:
                        futures = {}
                        
                        for br in active_branches:
                            crop_class = br.get("cropClass", br.get("triggerClasses", [])[0] if br.get("triggerClasses") else None)
                            sub_frame = frame
                            
                            logger.info(f"Starting parallel branch: {br['modelId']}, crop_class: {crop_class}")
                            
                            if br.get("crop", False) and crop_class:
                                cropped = crop_region_by_class(frame, regions_dict, crop_class)
                                if cropped is not None:
                                    sub_frame = cv2.resize(cropped, (224, 224))
                                    logger.debug(f"Successfully cropped {crop_class} region for {br['modelId']}")
                                else:
                                    logger.warning(f"Failed to crop {crop_class} region for {br['modelId']}, skipping branch")
                                    continue
                            
                            future = executor.submit(run_pipeline, sub_frame, br, True, context)
                            futures[future] = br
                        
                        # Collect results
                        for future in concurrent.futures.as_completed(futures):
                            br = futures[future]
                            try:
                                result, _ = future.result()
                                if result:
                                    branch_results[br["modelId"]] = result
                                    logger.info(f"Branch {br['modelId']} completed: {result}")
                            except Exception as e:
                                logger.error(f"Branch {br['modelId']} failed: {e}")
                else:
                    # Run branches sequentially  
                    for br in active_branches:
                        crop_class = br.get("cropClass", br.get("triggerClasses", [])[0] if br.get("triggerClasses") else None)
                        sub_frame = frame
                        
                        logger.info(f"Starting sequential branch: {br['modelId']}, crop_class: {crop_class}")
                        
                        if br.get("crop", False) and crop_class:
                            cropped = crop_region_by_class(frame, regions_dict, crop_class)
                            if cropped is not None:
                                sub_frame = cv2.resize(cropped, (224, 224))
                                logger.debug(f"Successfully cropped {crop_class} region for {br['modelId']}")
                            else:
                                logger.warning(f"Failed to crop {crop_class} region for {br['modelId']}, skipping branch")
                                continue
                        
                        try:
                            result, _ = run_pipeline(sub_frame, br, True, context)
                            if result:
                                branch_results[br["modelId"]] = result
                                logger.info(f"Branch {br['modelId']} completed: {result}")
                            else:
                                logger.warning(f"Branch {br['modelId']} returned no result")
                        except Exception as e:
                            logger.error(f"Error in sequential branch {br['modelId']}: {e}")
                            import traceback
                            logger.debug(f"Branch error traceback: {traceback.format_exc()}")

            # Store branch results in detection_result for parallel actions
            detection_result["branch_results"] = branch_results

        # ─── Execute Parallel Actions ───────────────────────────────
        if node.get("parallelActions") and "branch_results" in detection_result:
            execute_parallel_actions(node, frame, detection_result, regions_dict)

        # ─── Return detection result ────────────────────────────────
        primary_detection = max(all_detections, key=lambda x: x["confidence"])
        primary_bbox = primary_detection["bbox"]
        
        # Add branch results and session_id to primary detection for compatibility
        if "branch_results" in detection_result:
            primary_detection["branch_results"] = detection_result["branch_results"]
        if "session_id" in detection_result:
            primary_detection["session_id"] = detection_result["session_id"]
        
        return (primary_detection, primary_bbox) if return_bbox else primary_detection

    except Exception as e:
        logger.error(f"Error in node {node.get('modelId')}: {e}")
        import traceback
        traceback.print_exc()
        return (None, None) if return_bbox else None

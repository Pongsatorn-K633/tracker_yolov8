#!/usr/bin/env python3
import os
import json
import zipfile
import argparse
import sys

def validate_pipeline_node(node, directory, path="pipeline"):
    """
    Validates a single node of the pipeline.

    Required keys:
      - modelId (str)
      - modelFile (str) -> must exist in the directory
      - crop (bool)
      - triggerClasses (list of str)
      - branches (list)
    
    Optional key:
      - minConfidence (float) if present, must be a number between 0 and 1

    No extra keys are allowed.
    """
    allowed_keys = {"modelId", "modelFile", "crop", "triggerClasses", "branches", "minConfidence", 
                    "useTracking", "trackingConfig", "tracking", "stabilityThreshold", "multiClass", "expectedClasses", "actions", 
                    "parallelActions", "cropClass", "resizeTarget", "parallel"}
    required_keys = {"modelId", "modelFile", "crop", "triggerClasses", "branches"}
    
    # Check for extra keys
    for key in node.keys():
        if key not in allowed_keys:
            raise ValueError(
                f"Extra key '{key}' found in node at {path}. "
                f"Allowed keys: {sorted(allowed_keys)}. "
                "Please remove any keys not listed in the guide."
            )
    
    # Check for missing required keys
    for key in required_keys:
        if key not in node:
            raise ValueError(
                f"Missing required key '{key}' in node at {path}. "
                "Ensure all required keys are present."
            )

    # Validate data types
    if not isinstance(node['modelId'], str):
        raise ValueError(
            f"'modelId' must be a string at {path}. "
            "Please ensure the value is enclosed in quotes."
        )
    
    if not isinstance(node['modelFile'], str):
        raise ValueError(
            f"'modelFile' must be a string at {path}. "
            "Please ensure the file name is provided as a string."
        )
    
    # Check that modelFile exists in the same directory as pipeline.json
    model_file_path = os.path.join(directory, node['modelFile'])
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(
            f"Model file '{node['modelFile']}' referenced in {path} not found in directory '{directory}'. "
            "Make sure the file exists in the same folder as pipeline.json."
        )
    
    if not isinstance(node['crop'], bool):
        raise ValueError(
            f"'crop' must be a boolean at {path}. "
            "Please use true or false."
        )
    
    if not isinstance(node['triggerClasses'], list):
        raise ValueError(
            f"'triggerClasses' must be a list at {path}. "
            "Ensure that you enclose the classes in square brackets."
        )
    else:
        for i, item in enumerate(node['triggerClasses']):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in 'triggerClasses' must be strings at {path} -> triggerClasses[{i}]. "
                    "Please enclose each class name in quotes."
                )
    
    if not isinstance(node['branches'], list):
        raise ValueError(
            f"'branches' must be a list at {path}. "
            "Ensure that branches are defined in an array."
        )
    
    if 'minConfidence' in node:
        if not isinstance(node['minConfidence'], (float, int)):
            raise ValueError(
                f"'minConfidence' must be a number at {path}. "
                "Provide a numerical value between 0 and 1."
            )
        if not (0 <= node['minConfidence'] <= 1):
            raise ValueError(
                f"'minConfidence' must be between 0 and 1 at {path}. "
                "Please adjust the confidence threshold to a valid value."
            )

    # Validate optional keys
    if 'useTracking' in node:
        if not isinstance(node['useTracking'], bool):
            raise ValueError(
                f"'useTracking' must be a boolean at {path}. "
                "Please use true or false."
            )
    
    if 'trackingConfig' in node:
        if not isinstance(node['trackingConfig'], str):
            raise ValueError(
                f"'trackingConfig' must be a string at {path}. "
                "Please ensure the config file name is provided as a string."
            )
    
    if 'stabilityThreshold' in node:
        if not isinstance(node['stabilityThreshold'], int):
            raise ValueError(
                f"'stabilityThreshold' must be an integer at {path}. "
                "Please provide a valid integer value."
            )
    
    if 'multiClass' in node:
        if not isinstance(node['multiClass'], bool):
            raise ValueError(
                f"'multiClass' must be a boolean at {path}. "
                "Please use true or false."
            )
    
    if 'expectedClasses' in node:
        if not isinstance(node['expectedClasses'], list):
            raise ValueError(
                f"'expectedClasses' must be a list at {path}. "
                "Ensure that you enclose the classes in square brackets."
            )
        for i, item in enumerate(node['expectedClasses']):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in 'expectedClasses' must be strings at {path} -> expectedClasses[{i}]. "
                    "Please enclose each class name in quotes."
                )
    
    if 'cropClass' in node:
        if not isinstance(node['cropClass'], str):
            raise ValueError(
                f"'cropClass' must be a string at {path}. "
                "Please ensure the class name is provided as a string."
            )
    
    if 'resizeTarget' in node:
        if not isinstance(node['resizeTarget'], list) or len(node['resizeTarget']) != 2:
            raise ValueError(
                f"'resizeTarget' must be a list with exactly 2 integers at {path}. "
                "Please provide [width, height] as integers."
            )
        for i, item in enumerate(node['resizeTarget']):
            if not isinstance(item, int):
                raise ValueError(
                    f"'resizeTarget' values must be integers at {path} -> resizeTarget[{i}]. "
                    "Please provide integer values for width and height."
                )
    
    if 'parallel' in node:
        if not isinstance(node['parallel'], bool):
            raise ValueError(
                f"'parallel' must be a boolean at {path}. "
                "Please use true or false."
            )
    
    if 'actions' in node:
        if not isinstance(node['actions'], list):
            raise ValueError(
                f"'actions' must be a list at {path}. "
                "Ensure that actions are defined in an array."
            )
        # Basic validation for actions - could be expanded based on requirements
        for i, action in enumerate(node['actions']):
            if not isinstance(action, dict):
                raise ValueError(
                    f"Each action must be a JSON object at {path} -> actions[{i}]. "
                    "Please ensure action entries are defined as objects."
                )
    
    if 'parallelActions' in node:
        if not isinstance(node['parallelActions'], list):
            raise ValueError(
                f"'parallelActions' must be a list at {path}. "
                "Ensure that parallel actions are defined in an array."
            )
        # Basic validation for parallel actions - could be expanded based on requirements
        for i, action in enumerate(node['parallelActions']):
            if not isinstance(action, dict):
                raise ValueError(
                    f"Each parallel action must be a JSON object at {path} -> parallelActions[{i}]. "
                    "Please ensure parallel action entries are defined as objects."
                )
    
    if 'tracking' in node:
        if not isinstance(node['tracking'], dict):
            raise ValueError(
                f"'tracking' must be a JSON object at {path}. "
                "Please ensure the tracking configuration is defined as an object."
            )
        # Basic validation for tracking object - could be expanded based on requirements
        tracking = node['tracking']
        if 'enabled' in tracking and not isinstance(tracking['enabled'], bool):
            raise ValueError(
                f"'tracking.enabled' must be a boolean at {path}. "
                "Please use true or false."
            )

    # Recursively validate each branch
    for idx, branch in enumerate(node['branches']):
        if not isinstance(branch, dict):
            raise ValueError(
                f"Each branch must be a JSON object (dict) at {path} -> branches[{idx}]. "
                "Please ensure branch entries are defined as objects."
            )
        validate_pipeline_node(branch, directory, path=f"{path} -> branches[{idx}]")

def validate_pipeline_file(pipeline_data, directory):
    """
    Validates the overall pipeline JSON structure.
    It expects the top-level key 'pipeline' with a valid node.
    """
    if 'pipeline' not in pipeline_data:
        raise ValueError(
            "The top-level key 'pipeline' is missing. "
            "Ensure the JSON file has a top-level 'pipeline' object."
        )
    
    pipeline_node = pipeline_data['pipeline']
    
    if not isinstance(pipeline_node, dict):
        raise ValueError(
            "The 'pipeline' key must contain a JSON object. "
            "Ensure that the value of 'pipeline' is properly defined as an object."
        )
    
    validate_pipeline_node(pipeline_node, directory, path="pipeline")

def zip_directory(folder_path, zip_path):
    """
    Zips the contents of folder_path into a zip file specified by zip_path.
    The zip file will maintain the folder structure relative to folder_path.
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                # Calculate relative path to store in the zip archive.
                rel_path = os.path.relpath(full_path, os.path.dirname(folder_path))
                zipf.write(full_path, rel_path)

def process_folder(folder_path, output_root):
    """
    Validates the pipeline.json in folder_path and, if valid, zips the folder.
    The resulting zip file is placed in output_root/dist with the name {folder_name}.mpta.
    """
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    pipeline_path = os.path.join(folder_path, 'pipeline.json')
    if os.path.isfile(pipeline_path):
        print(f"\nValidating pipeline in directory: {folder_path}")
        try:
            with open(pipeline_path, 'r') as f:
                data = json.load(f)
            validate_pipeline_file(data, folder_path)
            print("Validation successful!")
            
            # Create dist directory if it doesn't exist
            dist_dir = os.path.join(output_root, 'dist')
            os.makedirs(dist_dir, exist_ok=True)
            
            # Zip the directory to dist folder
            zip_file_name = f"{folder_name}.mpta"
            zip_file_path = os.path.join(dist_dir, zip_file_name)
            zip_directory(folder_path, zip_file_path)
            print(f"Folder '{folder_name}' has been compiled to 'dist/{zip_file_name}'")
        except (ValueError, FileNotFoundError) as e:
            print(f"Validation error in '{folder_name}': {e}")
            print("Suggestion: Please review the pipeline.json file in this folder and correct the issue above.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{folder_name}': {e}")
            print("Suggestion: Ensure the JSON syntax is valid (e.g., proper commas, quotes, and braces).")
        except Exception as e:
            print(f"An unexpected error occurred in '{folder_name}': {e}")
    else:
        print(f"Skipping folder '{folder_name}' (no pipeline.json found)")

def scan_and_process(root_dir, target_folder=None):
    """
    If target_folder is specified, validate and process that folder.
    Otherwise, scan all subdirectories in root_dir and process those that contain pipeline.json.
    """
    if target_folder:
        folder_path = os.path.abspath(target_folder)
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a valid directory. Please check the folder path and try again.")
            sys.exit(1)
        process_folder(folder_path, root_dir)
    else:
        # Process all subdirectories in the root_dir
        for item in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, item)
            if os.path.isdir(dir_path):
                process_folder(dir_path, root_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO model pipeline configurations and zip valid folders as .mpta archives."
    )
    parser.add_argument(
        "-f", "--folder",
        help="Specify a folder to process. If not provided, all subdirectories will be scanned.",
        type=str
    )
    parser.add_argument(
        "-d", "--directory",
        help="Specify the root directory to scan (default: current directory).",
        type=str,
        default=os.getcwd()
    )
    # Note: The help option (-h/--help) is automatically provided by argparse.
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.directory)
    print(f"Processing root directory: {root_dir}")
    
    if args.folder:
        print(f"Processing specified folder: {args.folder}")
    else:
        print("No specific folder provided; scanning all subdirectories.")
    
    scan_and_process(root_dir, target_folder=args.folder)

if __name__ == "__main__":
    main()

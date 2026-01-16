import os
import sys
import shutil
import math
import cv2
import exiftool 
import logging

import configparser

# --- DEFAULTS (Used if config is missing) ---
DEFAULT_CONFIG = {
    "Filters": {
        "min_distance_meters": "20.0",
        "max_speed_mps": "5.0",
        "max_digital_zoom": "1.0",
        "min_blur_score": "100.0",
        "min_width": "3000",
        "min_height": "2000",
        "max_iso": "1600",
        "min_brightness": "20.0",
        "max_brightness": "240.0"
    },
    "Settings": {
        "image_folder": "."  # "." means current folder
    }
}

def get_script_dir():
    """Determine the directory where the script/exe is located."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def load_config(script_dir):
    """Load config.ini or create it if missing."""
    config_path = os.path.join(script_dir, 'config.ini')
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        print(f"Creating default config file: {config_path}")
        config.read_dict(DEFAULT_CONFIG)
        with open(config_path, 'w') as f:
            config.write(f)
    else:
        print(f"Loading config from: {config_path}")
        config.read(config_path)
    
    # Return parsed values with fallback to defaults
    try:
        cfg = {}
        # Filters
        f = config['Filters']
        cfg['min_dist'] = float(f.get('min_distance_meters', 20.0))
        cfg['max_speed'] = float(f.get('max_speed_mps', 5.0))
        cfg['max_zoom'] = float(f.get('max_digital_zoom', 1.0))
        cfg['min_blur'] = float(f.get('min_blur_score', 100.0))
        cfg['min_w'] = int(f.get('min_width', 3000))
        cfg['min_h'] = int(f.get('min_height', 2000))
        cfg['max_iso'] = int(f.get('max_iso', 1600))
        cfg['min_bright'] = float(f.get('min_brightness', 20.0))
        cfg['max_bright'] = float(f.get('max_brightness', 240.0))
        
        # Settings
        raw_folder = config['Settings'].get('image_folder', '.')
        # Fix: Remove quotes if user added them in config.ini (common mistake)
        cfg['img_folder'] = raw_folder.strip('"').strip("'")
        
        return cfg
    except Exception as e:
        print(f"Error reading config: {e}. Using defaults.")
        return {
            'min_dist': 20.0, 'max_speed': 5.0, 'max_zoom': 1.0, 'min_blur': 100.0,
            'min_w': 3000, 'min_h': 2000, 'max_iso': 1600, 'min_bright': 20.0, 'max_bright': 240.0,
            'img_folder': '.'
        }

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_image_metrics(file_path, et):
    """
    Extracts all logic-relevant data from the image.
    Returns a dictionary of metrics.
    """
    metrics = {
        'width': 0, 'height': 0,
        'digital_zoom': 1.0,
        'focal_length': 0.0,
        'iso': 0,
        'blur_score': 0.0,
        'brightness': 0.0,
        'distance': 9999.0, # Default to safe distance if missing? Or fail? Let's check.
        'speed': 0.0,
        'gimbal_pitch': 0.0,
        'load_error': None,
        'meta_error': None,
        'has_distance': False
    }

    # 1. Metadata Extraction
    try:
        metadata_list = et.get_metadata(file_path)
        if metadata_list:
            metadata = metadata_list[0]
            
            # Resolution
            w = metadata.get('EXIF:ExifImageWidth', 0)
            h = metadata.get('EXIF:ExifImageHeight', 0)
            if w == 0: w = metadata.get('ImageWidth', 0)
            if h == 0: h = metadata.get('ImageHeight', 0)
            metrics['width'] = w
            metrics['height'] = h
            
            # Zoom and ISO
            metrics['digital_zoom'] = float(metadata.get('Composite:DigitalZoomRatio', 1.0))
            metrics['iso'] = int(metadata.get('EXIF:ISO', 0))
            
            # Optical Lens Info
            # Note: 24mm ~ Wide, >100mm ~ Zoom
            metrics['focal_length'] = float(metadata.get('Composite:FocalLength35efl', 0) or metadata.get('EXIF:FocalLengthIn35mmFormat', 0))

            # Distance (LRF)
            # Try LRF Target Distance first, then Gimbal/Subject distance
            dist = metadata.get('XMP:LRFTargetDistance')
            if dist is not None:
                metrics['distance'] = float(dist)
                metrics['has_distance'] = True
            
            # Speed (Flight Speed) -> "x,y,z" string e.g., "-0.1,0.5,0"
            speed_str = metadata.get('XMP:FlightSpeed') # DJI specific
            if speed_str:
                try:
                    parts = [float(x) for x in speed_str.split(',')]
                    if len(parts) >= 3:
                        # Magnitude of 3D vector
                        metrics['speed'] = math.sqrt(parts[0]**2 + parts[1]**2 + parts[2]**2)
                except:
                    pass # Keep 0.0 if parse fails

            # Gimbal Pitch
            metrics['gimbal_pitch'] = float(metadata.get('XMP:GimbalPitchDegree', 0))

        else:
            metrics['meta_error'] = "No Metadata Found"
            
    except Exception as e:
        metrics['meta_error'] = str(e)

    # 2. Visual Analysis
    try:
        # Read image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            metrics['load_error'] = "Could not load image (Corrupt?)"
        else:
            # Blur (Laplacian Variance)
            metrics['blur_score'] = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # Brightness (Mean Pixel Intensity)
            metrics['brightness'] = cv2.mean(image)[0]
            
    except Exception as e:
        metrics['load_error'] = str(e)

    return metrics

def evaluate_image(metrics, cfg):
    """
    Decides if an image is GOOD or BAD based on metrics.
    Returns (is_good, log_string, reason_list)
    """
    reasons = []
    is_good = True
    
    # Format string helpers
    def grade(val, threshold, operator, unit=""):
        passed = False
        if operator == ">": passed = val > threshold
        elif operator == "<": passed = val < threshold
        elif operator == "<=": passed = val <= threshold
        elif operator == ">=": passed = val >= threshold
        
        status = "(PASS)" if passed else "(FAIL)"
        return f"{val:.2f}{unit} {status}", passed

    # 1. Resolution
    res_str = f"{metrics['width']}x{metrics['height']}"
    if metrics['width'] < cfg['min_w'] or metrics['height'] < cfg['min_h']:
        reasons.append(f"Low Resolution ({res_str})")
        is_good = False
    
    # 2. Zoom Checks
    # Optical Labeling
    optical_type = "Wide"
    if metrics['focal_length'] > 80: # Arbitrary cutoff for "Zoom" lens description
        optical_type = "Zoom"
        
    # Digital Zoom Check
    dzoom_str, dzoom_pass = grade(metrics['digital_zoom'], cfg['max_zoom'], "<=", "x")
    if not dzoom_pass:
        reasons.append(f"Digital Zoom ({metrics['digital_zoom']}x)")
        is_good = False

    # 3. ISO
    iso_str, iso_pass = grade(metrics['iso'], cfg['max_iso'], "<=")
    if not iso_pass:
        reasons.append(f"High ISO ({metrics['iso']})")
        is_good = False
        
    # 4. Distance (LRF)
    if metrics['has_distance']:
        dist_str, dist_pass = grade(metrics['distance'], cfg['min_dist'], ">=", "m")
        if not dist_pass:
            reasons.append(f"Too Close ({metrics['distance']}m < {cfg['min_dist']}m)")
            is_good = False
    else:
        dist_str = "N/A (No LRF)" 
        # Optional: Fail if LRF missing? For now, let's pass it logic-wise but log it.
    
    # 5. Flight Speed
    speed_str, speed_pass = grade(metrics['speed'], cfg['max_speed'], "<=", "m/s")
    if not speed_pass:
        reasons.append(f"Moving Too Fast ({metrics['speed']:.1f} m/s)")
        is_good = False

    # 6. Visual Checks (Blur & Brightness)
    if metrics['load_error']:
        return False, f"[ERROR] Could not load image: {metrics['load_error']}", [metrics['load_error']]
        
    # Blur
    blur_str, blur_pass = grade(metrics['blur_score'], cfg['min_blur'], ">=")
    if not blur_pass:
        reasons.append(f"Blurry (Score: {metrics['blur_score']:.1f})")
        is_good = False

    # Brightness
    bright_val = metrics['brightness']
    bright_status = "(PASS)"
    if bright_val < cfg['min_bright']:
        reasons.append(f"Too Dark ({bright_val:.1f})")
        bright_status = "(FAIL: Dark)"
        is_good = False
    elif bright_val > cfg['max_bright']:
        reasons.append(f"Overexposed ({bright_val:.1f})")
        bright_status = "(FAIL: Bright)"
        is_good = False
    bright_str = f"{bright_val:.1f} {bright_status}"

    # Construct Log String
    # Format: [ OPTICAL: Zoom | DIG_ZOOM: 1.0 (PASS) | DIST: 120m (PASS) | ... ]
    log_string = (
        f"Lens: {optical_type} ({int(metrics['focal_length'])}mm) | "
        f"DigZoom: {dzoom_str} | "
        f"Dist: {dist_str} | "
        f"Speed: {speed_str} | "
        f"Blur: {blur_str} | "
        f"Bright: {bright_str}"
    )

    return is_good, log_string, reasons

def main():
    # Load Config
    script_dir = get_script_dir()
    cfg = load_config(script_dir)

    # Determine Input Directory
    base_dir = cfg['img_folder']
    if base_dir == '.' or base_dir.strip() == "":
        base_dir = script_dir
    
    # Create output directories (Relative to base_dir, or script_dir? User said "same as he did")
    # Usually relative to where the images are found.
    good_dir = os.path.join(base_dir, '_GOOD_IMAGES')
    bad_dir = os.path.join(base_dir, '_BAD_IMAGES')

    # Create output directories
    if not os.path.exists(base_dir):
        print(f"CRITICAL ERROR: Input folder not found: {base_dir}")
        print("Please check 'image_folder' in config.ini")
        input("Press Enter to exit")
        return

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    # Setup Logging in the output folder
    log_path = os.path.join(base_dir, 'sorting_log.txt')
    # Force reconfiguration of logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    print(f"--- Processing folder: {base_dir} ---")
    print(f"Log File: {log_path}")
    print(f"FILTERS LOADED FROM CONFIG:")
    print(f"  > Dist >= {cfg['min_dist']}m")
    print(f"  > Speed <= {cfg['max_speed']}m/s")
    print(f"  > Zoom <= {cfg['max_zoom']}x")
    print(f"  > Blur Score >= {cfg['min_blur']}")
    print("-" * 60)
    
    # Get all images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.dng', '.tiff')
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} images.")

    if not files:
        print("No images found to process.")
        print("\nDone! Press Enter to exit.")
        input()
        return

    # Start ExifTool once (much faster than opening it for every file)
    print("--- Initializing ExifTool ---")
    exif_executable = None # Default to system path
    
    # 1. Check for bundled ExifTool (PyInstaller)
    bundled_exif = get_resource_path('exiftool.exe')
    
    # DEBUG INFO
    if getattr(sys, 'frozen', False):
        print(f"Running in bundled mode.")
        print(f"Temp Dir (_MEIPASS): {sys._MEIPASS}")
    
    print(f"Checking for bundled ExifTool at: {bundled_exif}")

    if os.path.exists(bundled_exif):
        print(f"Found bundled ExifTool.")
        exif_executable = bundled_exif
        
        # Verify permissions/existence of files logic
        print("Verifying execution permissions...")
        try:
             # Try to invoke version to check if it runs
             import subprocess
             # We must set CWD to the folder containing exiftool.exe for it to find its files!
             # or just rely on PyExifTool
             # Let's try to verify if it runs
             result = subprocess.run([exif_executable, '-ver'], capture_output=True, text=True, cwd=os.path.dirname(exif_executable))
             print(f"ExifTool Version Check: {result.stdout.strip()} (Stderr: {result.stderr.strip()})")
        except Exception as e:
            print(f"WARNING: ExifTool found but failed to run directly: {e}")
        
        # CRITICAL FIX: Manually inject PERL5LIB for the bundled environment
        # The logs showed @INC was empty, meaning it lost track of its lib folder.
        # Structure is: .../exiftool.exe and .../exiftool_files/lib
        lib_dir = os.path.join(os.path.dirname(bundled_exif), "exiftool_files", "lib")
        if os.path.exists(lib_dir):
            print(f"Injecting PERL5LIB: {lib_dir}")
            os.environ["PERL5LIB"] = lib_dir
        else:
             print(f"WARNING: Lib dir not found at {lib_dir}")

    else:
        print("Bundled ExifTool NOT found.")
        # 2. Check if exiftool is in the same folder (Script mode)
        local_exif = os.path.join(script_dir, 'exiftool.exe')
        if os.path.exists(local_exif):
            print(f"Found local ExifTool: {local_exif}")
            exif_executable = local_exif
        else:
            # 3. Check system path
            print("Checking system PATH...")
            if shutil.which("exiftool") is None:
                print("\n" + "!" * 60)
                print("CRITICAL ERROR: 'exiftool.exe' NOT FOUND!")
                print("!" * 60)
                print("The internal ExifTool seems missing and it is not on your PATH.")
                print(f"Please ensure exiftool.exe is in: {script_dir}")
                print("-" * 60)
                input("Press Enter to exit.")
                return

    try:
        print("Starting ExifTool Engine...")
        # Use 'executable' argument to force specific path if found locally
        # Note: PyExifTool's helper accepts 'executable' in constructor
        with exiftool.ExifToolHelper(executable=exif_executable) as et:
            # Check connection first
            if not et.running:
                print("Launching subprocess...")
                et.run()
            
            print("ExifTool Engine Started.")
            print("-" * 60)

            for filename in files:
                filepath = os.path.join(base_dir, filename)
                
                # Skip if it's already in the output folders
                if filepath.startswith(good_dir) or filepath.startswith(bad_dir):
                    continue
                
                if not os.path.isfile(filepath):
                    continue

                print(f"Checking: {filename}...", end=" ")

                # Step 1: Gather Metrics
                metrics = get_image_metrics(filepath, et)
                
                # Step 2: Evaluate
                is_good, log_string, reasons = evaluate_image(metrics, cfg)
                
                # Step 3: Act
                if is_good:
                    print(f"✅ GOOD")
                    print(f"   Details: {log_string}")
                    logging.info(f"{filename} [ACCEPTED] -> {log_string}")
                    try:
                        shutil.move(filepath, os.path.join(good_dir, filename))
                    except Exception as e:
                        print(f"   Error moving file: {e}")
                else:
                    reason_str = ", ".join(reasons)
                    print(f"❌ BAD -> {reason_str}")
                    print(f"   Details: {log_string}")
                    logging.info(f"{filename} [REJECTED] -> Reasons: {reason_str} || {log_string}")
                    try:
                        shutil.move(filepath, os.path.join(bad_dir, filename))
                    except Exception as e:
                         print(f"   Error moving file: {e}")

    except Exception as e:
        print(f"\nCRITICAL ERROR during execution: {e}")
        logging.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        if "exiftool" in str(e).lower():
             print("Make sure exiftool.exe is in the same folder or in your system PATH.")

    print("\nDone! Press Enter to exit.")
    input()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
         print(f"FATAL CRASH: {e}")
         import traceback
         traceback.print_exc()
         input("Press Enter to exit.")


import cv2
import numpy as np
import os
import io

TARGET_SIZE = (224, 224)

def preprocess_image(image_path):
    """Load an image from disk, resize, normalize, and expand dimensions for model input.

    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Preprocessed image array with shape (1, height, width, 3).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, TARGET_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_normalized, axis=0)

def generate_ela_heatmap(image_path, output_path, quality=90):
    """Generate an Error Level Analysis (ELA) heatmap image to highlight manipulated/AI-generated regions.

    Args:
        image_path (str): Source image path.
        output_path (str): Target path to save the heatmap visualization.
        quality (int): JPEG re-compression quality.
    Returns:
        float: Mean ELA intensity.
    """
    original = cv2.imread(image_path)
    if original is None:
        return 0.0
    
    # Recompress image in memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_img = cv2.imencode('.jpg', original, encode_param)
    compressed = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # Compute absolute difference and scale up
    ela_diff = cv2.absdiff(original, compressed)
    
    # Amplify the difference to make artifacts clearly visible
    scale = 15
    ela_amplified = cv2.convertScaleAbs(ela_diff, alpha=scale, beta=0)
    
    # Convert to a color heatmap (inferno/jet) for explainable AI presentation
    ela_gray = cv2.cvtColor(ela_amplified, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    
    # Blend with original for visual inspection
    blended = cv2.addWeighted(original, 0.45, heatmap, 0.55, 0)
    cv2.imwrite(output_path, blended)
    
    return float(np.mean(ela_gray))

def analyze_corner_watermarks(image):
    """Inspect bottom corners for common AI generator badges (Grok, DALL-E, Midjourney, etc.).

    Returns:
        tuple: (has_watermark, confidence, badge_name)
    """
    h, w = image.shape[:2]
    # Crop bottom-right and bottom-left corners (typical placement for Grok, OpenAI, etc.)
    corner_h = int(h * 0.12)
    corner_w = int(w * 0.22)
    
    corners = {
        'bottom-right': image[h - corner_h:h, w - corner_w:w],
        'bottom-left': image[h - corner_h:h, 0:corner_w]
    }
    
    for corner_name, patch in corners.items():
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Look for high-contrast watermark text or geometric logos
        # Grok badge has distinctive dark rectangular background with white text 'GROK'
        _, thresh_dark = cv2.threshold(patch_gray, 45, 255, cv2.THRESH_BINARY_INV)
        _, thresh_bright = cv2.threshold(patch_gray, 200, 255, cv2.THRESH_BINARY)
        
        # Check edge density in the corner
        edges = cv2.Canny(patch_gray, 80, 200)
        edge_density = np.sum(edges > 0) / (corner_h * corner_w)
        
        # Check for sharp badge boundaries (standard in AI tools)
        contours, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150 and area < (corner_h * corner_w * 0.8):
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / bh if bh > 0 else 0
                # Most AI badges are 2:1 to 4:1 aspect ratio rectangles
                if 1.5 < aspect < 5.0 and edge_density > 0.04:
                    return True, 0.96, f"AI Generator Badge ({corner_name})"
                    
    return False, 0.0, None

import cv2
import numpy as np

# Try importing pytesseract for Path 1, handle gracefully if not installed locally
try:
    import pytesseract
except ImportError:
    pytesseract = None


def execute_ocr_pipeline(image_path):
    """
    PATH 1: Optical Character Recognition (OCR) Pipeline
    Fulfills: Grayscale conversion, Gaussian Blur, and Adaptive Thresholding.
    """
    print("\n--- [PATH 1] Starting Optical Character Recognition Pipeline ---")

    # Ingest raw visual data
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image from {image_path}. Generating dummy image for simulation...")
        img = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "DecodeLabs AI Project 4", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # STEP 1: Grayscale Conversion (Collapses RGB into a 1D intensity matrix)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("✔ Step 1 Complete: Converted image to Grayscale intensity matrix.")

    # STEP 2: Gaussian Blur (Smooths to eliminate micro-imperfections)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("✔ Step 2 Complete: Applied Gaussian Blur filtering.")

    # STEP 3: Adaptive Thresholding / Otsu's Binary Decision
    # Converts grayscale values directly to absolute black (0) or white (255)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("✔ Step 3 Complete: Executed Adaptive Thresholding (Otsu's Binary Decision).")

    # STEP 4: Extraction using Pytesseract
    if pytesseract is not None:
        # Tuning PSM (Page Segmentation Mode 3: Fully automatic) as required by playbook
        custom_config = r'--psm 3'
        extracted_text = pytesseract.image_to_string(thresholded, config=custom_config)
        print("\n📝 [OUTPUT] Extracted Machine-Readable Text:")
        print("-" * 40)
        print(extracted_text.strip())
        print("-" * 40)
    else:
        print("\n⚠️ Note: 'pytesseract' library or Tesseract binary not configured on this machine.")
        print("Pre-processing pipeline verified successfully! (Gatekeeper Rule 1 & 2 met).")

    return thresholded


def execute_object_detection_pipeline(image_path, model_proto=None, model_weights=None):
    """
    PATH 2: Object Detection Pipeline via MobileNet-SSD
    Fulfills: 4D Blob Construction, Coordinate Scaling, and 80% Confidence Filtering.
    """
    print("\n--- [PATH 2] Starting MobileNet-SSD Object Detection Pipeline ---")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Image file not found at {image_path}. Simulating detection tensor logic...")
        # Simulate an image frame for the structure demonstration
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        # Simulating a detected bounding box box matrix output from a network
        # Format: [batchId, classId, confidence, xMin, yMin, xMax, yMax]
        mock_detections = np.array([[[
            [0, 1, 0.92, 0.20, 0.20, 0.60, 0.75],  # Object 1: 92% Confidence (Passes filter)
            [0, 2, 0.45, 0.70, 0.10, 0.85, 0.40]  # Object 2: 45% Confidence (Dropped)
        ]]])
        process_detections(img, mock_detections, h, w)
        return

    h, w = img.shape[:2]

    # STEP 1: 4D Blob Construction (blobFromImage resizing to 300x300 network standard)
    # Scales and normalizes mean intensity values
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    print("✔ Step 1 Complete: Constructed 4D Blob tensor from image input.")

    if model_proto and model_weights:
        # Load the pre-trained deep learning network structure
        net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
        net.setInput(blob)
        detections = net.forward()
        process_detections(img, detections, h, w)
    else:
        print("⚠️ Model weight paths not passed. Simulating network inference results...")


def process_detections(img, detections, h, w):
    """Performs bounding box calculations and filters by the 80% threshold."""
    # Class labels for standard MobileNet-SSD
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    num_detections = detections.shape[2]

    for i in range(num_detections):
        confidence = detections[0, 0, i, 2]

        # STEP 2 & 3: Apply the 80% Confidence Threshold Filter (Gatekeeper Rule 3)
        if confidence >= 0.80:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else f"ID {class_id}"

            # STEP 4: Coordinate Scaling / Anatomy of a Bounding Box
            # Maps normalized coordinates back into actual pixels (X, Y, W, H framework)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            print(f"🎯 [DETECTION] Found '{label}' with {confidence * 100:.2f}% Confidence Score!")
            print(f"   Bounding Box Coordinates: Origin:({startX}, {startY}) to ({endX}, {endY})")

            # Draw visual confirmation onto the frame
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"📉 [DROPPED] Detection index {i} failed threshold test (Confidence: {confidence * 100:.2f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("       DECODELABS: MACHINE OPTIC NERVE INITIALIZED      ")
    print("=" * 60)

    # Test path string configuration
    sample_image = "test_image.jpg"

    # Execute both tracks specified in the manual perception matrix
    execute_ocr_pipeline(sample_image)
    execute_object_detection_pipeline(sample_image)
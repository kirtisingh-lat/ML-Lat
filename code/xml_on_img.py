import xml.etree.ElementTree as ET
import cv2
import math
import numpy as np
from pathlib import Path

def plot_xml_on_image(img_path, xml_path):
    # Load the image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        # Filter out deleted objects
        deleted = obj.find('deleted')
        if deleted is not None and deleted.text == '1':
            continue
        
        bbox = obj.find('bndbox')
        if bbox is None or bbox.find('cx') is None:
            continue

        # Geometric Reconstruction Logic
        # We use cx, cy, w, h, and a to find the 4 ACTUAL corners
        cx = float(bbox.find('cx').text)
        cy = float(bbox.find('cy').text)
        w  = float(bbox.find('w').text)
        h  = float(bbox.find('h').text)
        a  = float(bbox.find('a').text)

        # In this dataset, h is the long axis (Length) and w is the short axis (Width)
        # OpenCV's rotatated rectangle takes: (center), (width, height), angle
        # Note: If the box is rotated 90 degrees wrong, swap (h, w) to (w, h)
        rect = ((cx, cy), (w, h), a)
        
        # Calculate the 4 corner points
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.intp)

        # Draw the solid rectangle in Green
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        
        # Optional: Label the object ID near the box
        obj_id = obj.find('id').text if obj.find('id') is not None else ""
        cv2.putText(img, f"ID:{obj_id}", (int(cx), int(cy)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save the mapped image
    output_path = f"mapped_{xml_path.stem}.jpg"
    cv2.imwrite(output_path, img)
    print(f"Mapping complete. View the result at: {output_path}")

# --- EXECUTION ---
# Path to your image and corresponding XML
image_path = Path("/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/images/2006-05-03-Allianz-links-yr7e0006.jpg")
xml_path = Path("/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/label_xmls/2006-05-03-Allianz-links-yr7e0006.xml")

plot_xml_on_image(image_path, xml_path)
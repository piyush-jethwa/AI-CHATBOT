from brain_of_the_doctor import analyze_image, analyze_text_query
from image_analysis import detect_edges
import cv2

def run_demo(image_path):
    print("\n=== IMAGE ANALYSIS DEMO ===")
    
    # 1. Color Analysis
    print("\n[1] Analyzing dominant colors...")
    colors = analyze_image(image_path)
    print(f"Result: {colors}")
    
    # 2. Poetic Description
    print("\n[2] Generating poetic description...")
    poetry = analyze_text_query(f"Describe these colors creatively: {colors}")
    print(poetry)
    
    # 3. Edge Detection
    print("\n[3] Performing edge detection...")
    edges = detect_edges(image_path)
    cv2.imwrite('edges.jpg', edges)
    print("Edge detection complete - saved to edges.jpg")
    
    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    run_demo("test_image.jpg")

#!/usr/bin/env python3
"""
Webcam-integrated vision pipeline for pick-and-place tasks
"""
import cv2
import torch
import numpy as np
from PIL import Image
import time
import os
from typing import Optional, Dict
from dotenv import load_dotenv

# For this example, I'll include a simplified version
import json
import random
from typing import List, Tuple
from dataclasses import dataclass
from google import genai
from google.genai import types

load_dotenv()   # Loads .env into os.environ

@dataclass
class BoundingBox:
    """Represents a 2D bounding box with label"""
    label: str
    coords: List[float]  # [ymin, xmin, ymax, xmax] in 0-1000 scale
    
    @property
    def normalized(self) -> List[float]:
        """Return coordinates normalized to 0-1 scale"""
        return [c / 1000.0 for c in self.coords]


class WebcamVisionPipeline:
    """Webcam-integrated vision pipeline"""
    
    def __init__(self, camera_index: int = 4, api_key: Optional[str] = None):
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize Gemini API
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = "gemini-2.0-flash-lite-001"
            self.vision_enabled = True
        else:
            print("Warning: No Google API key found. Vision analysis disabled.")
            self.vision_enabled = False
        
        self.system_prompt = """
        You are an expert at analyzing images for robotic pick-and-place tasks.
        Detect LEGO bricks, small toys, and containers (bins, boxes) on the desk.
        Return bounding boxes as JSON: [{"label": "object_name", "box_2d": [ymin,xmin,ymax,xmax]}]
        Coordinates should be normalized to 0-1000 scale. Return ONLY the JSON array.
        """
    
    def initialize_camera(self) -> bool:
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Camera {self.camera_index} opened successfully!")
        return True
    
    def cv2_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert OpenCV frame to PyTorch tensor"""
        # OpenCV uses BGR, convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to 0-1
        tensor = torch.from_numpy(rgb_frame).float() / 255.0
        
        # Convert from HWC to CHW format
        tensor = tensor.permute(2, 0, 1)
        
        return tensor
    
    def cv2_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV frame to PIL Image"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for objects"""
        if not self.vision_enabled:
            return {"error": "Vision analysis not available (no API key)"}
        
        try:
            # Convert frame to PIL image
            pil_image = self.cv2_to_pil(frame)
            
            # Call Gemini vision API
            prompt = """
            Analyze this image for robotic pick-and-place. Detect all LEGO bricks, 
            small toys, and containers/bins. Return as JSON array with label and box_2d coordinates.
            """
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[pil_image, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.3
                ),
            )
            
            # Parse response
            json_text = self._clean_json_response(response.text)
            # print('json reply from Gemini : ')
            # print(json_text)
            objects_data = json.loads(json_text)
            
            # only keep boxes with exactly 4 coords
            valid_objects = []
            for obj in objects_data:
                coords = obj.get("box_2d", [])
                if len(coords) != 4:
                    print(f"Warning: skipping invalid box_2d={coords!r}")
                    continue
                valid_objects.append(BoundingBox(obj["label"], coords))

            objects = valid_objects
            
            # Categorize objects
            pick_objects, place_objects = self._categorize_objects(objects)
                        
            return {
                "success": True,
                "pick_objects": pick_objects,
                "place_objects": place_objects,
                "total_objects": len(objects)
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from markdown formatting"""
        lines = response.strip().splitlines()
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if "```json" in line.lower():
                start_idx = i + 1
            elif "```" in line and i > start_idx:
                end_idx = i
                break
        
        return "\n".join(lines[start_idx:end_idx])
    
    def _categorize_objects(self, objects: List[BoundingBox]) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """Separate objects into pick targets and place locations"""
        place_keywords = ["bin", "box", "container", "basket"]
        pick_objects = []
        place_objects = []
        
        for obj in objects:
            if any(keyword in obj.label.lower() for keyword in place_keywords):
                place_objects.append(obj)
            else:
                pick_objects.append(obj)
        
        return pick_objects, place_objects
    
    def draw_bounding_boxes(self, frame: np.ndarray, 
                           pick_targets: List[BoundingBox],
                           place_targets: List[BoundingBox]) -> np.ndarray:
        """Draw bounding boxes on OpenCV frame"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        def draw_bbox(bbox: BoundingBox, color: tuple, thickness: int = 3):
            # Convert normalized coordinates to absolute
            y1, x1, y2, x2 = bbox.coords
            abs_coords = [
                int(y1/1000 * height), int(x1/1000 * width),
                int(y2/1000 * height), int(x2/1000 * width)
            ]
            
            # Ensure coordinates are in correct order
            y1, x1, y2, x2 = abs_coords
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label_size = cv2.getTextSize(bbox.label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, bbox.label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        for pick_tgt in pick_targets:
            draw_bbox(pick_tgt, (0, 0, 255))

        # Draw all place objects in blue
        for place_tgt in place_targets:
            draw_bbox(place_tgt, (255, 0, 0))
        
        return annotated_frame
    
    def run_interactive_session(self):
        """Run interactive webcam session with vision analysis"""
        if not self.initialize_camera():
            return
        
        print("Interactive Vision Session Started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        
        last_analysis_time = 0
        analysis_interval = 1.0  # Analyze every X seconds in continuous mode

        result = {"pick_objects": [], "place_objects": []}
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # display_frame = frame.copy()
                
                # Only hit the API every `analysis_interval` seconds
                if time.time() - last_analysis_time > analysis_interval: 
                    print("Analyzing frame...")
                    result = self.analyze_frame(frame)
                    
                    if "success" in result:
                        print(f"Found {result['total_objects']} objects")

                        for pick_obj in result.get('pick_objects', []):
                            print(f"Pick: {pick_obj.label}")

                        for place_obj in result.get('place_objects', []):
                            print(f"Place: {place_obj.label}")
                        
                    else:
                        print("Analysis failed:", result.get("error"))
                    
                    last_analysis_time = time.time()

                # Always draw the **latest** targets on each frame
                display_frame = self.draw_bounding_boxes(
                    frame,
                    result.get('pick_objects', []),
                    result.get('place_objects', [])
                )
                    
                # Add status text
                status_text = "CONTINUOUS"
                cv2.putText(display_frame, f"Mode: {status_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Vision Pipeline - Webcam Feed', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the webcam vision pipeline"""
    
    # Initialize with your camera index (4 for Tolulu webcam)
    pipeline = WebcamVisionPipeline(camera_index=4)
            
    pipeline.run_interactive_session()

if __name__ == "__main__":
    main()


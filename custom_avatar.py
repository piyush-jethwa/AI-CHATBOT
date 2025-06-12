import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

class SpeakingAvatar:
    def __init__(self, image_path="portrait-3d-female-doctor[1].jpg"):
        self.avatar_cache = {}
        try:
            # Try to load custom image from current directory
            full_path = os.path.join(os.path.dirname(__file__), image_path)
            if os.path.exists(full_path):
                # Open and ensure proper format
                print(f"Attempting to load avatar image from: {full_path}")
                try:
                    self.base_image = Image.open(full_path).convert("RGB")
                    # Resize to maintain aspect ratio while fitting 300x300
                    # Resize to fill 400x300 container while maintaining aspect ratio
                    target_width = 300
                    target_height = 400
                    width, height = self.base_image.size
                    ratio = max(target_width/width, target_height/height)
                    new_size = (int(width*ratio), int(height*ratio))
                    self.base_image = self.base_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Create canvas and center the image
                    canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                    canvas.paste(
                        self.base_image,
                        ((target_width - self.base_image.width) // 2,
                         (target_height - self.base_image.height) // 2)
                    )
                    self.base_image = canvas
                    print(f"Successfully loaded avatar image from {full_path}")
                    print(f"Image format: {self.base_image.format}, mode: {self.base_image.mode}, size: {self.base_image.size}")
                except Exception as e:
                    print(f"Error loading image: {str(e)}")
                    raise
            else:
                # Create a default doctor avatar
                self.base_image = self._create_default_avatar()
                print("Using default avatar image")
                
        except Exception as e:
            print(f"Error loading avatar image: {str(e)}")
            self.base_image = self._create_default_avatar()
            
    def _create_default_avatar(self):
        """Create a simple default doctor avatar"""
        img = Image.new('RGB', (300, 300), (255, 255, 255))  # White background
        d = ImageDraw.Draw(img)
        
        # Draw a simple face
        face_color = (200, 200, 255)  # Light blue
        d.ellipse((50, 50, 250, 250), fill=face_color, outline=(0,0,0))
        
        # Eyes
        d.ellipse((100, 100, 120, 120), fill=(0,0,0))  # Left eye
        d.ellipse((180, 100, 200, 120), fill=(0,0,0))  # Right eye
        
        # Smile
        d.arc((100, 150, 200, 200), 0, 180, fill=(0,0,0), width=2)
        
        # Doctor symbol (red cross)
        cross_color = (255, 0, 0)
        d.line((150, 50, 150, 100), fill=cross_color, width=3)
        d.line((120, 80, 180, 80), fill=cross_color, width=3)
        
        # Verify image has non-zero pixels
        if np.array(img).sum() == 0:
            raise ValueError("Default avatar image is completely black!")
            
        return img
        
    def get_avatar(self, text=""):
        """Return avatar image with optional speech text and animation"""
        try:
            print(f"Starting avatar generation with text: {text[:100]}")
            
            # Use cached version if available
            cache_key = hash(text[:100])  # First 100 chars as key
            if cache_key in self.avatar_cache:
                print("Returning cached avatar")
                return self.avatar_cache[cache_key]
                
            print("Creating new avatar image")
            img = self.base_image.copy()
            d = ImageDraw.Draw(img)
            
            if text:
                print(f"Adding text to avatar: {text[:100]}")
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                    print("Using arial.ttf font")
                except:
                    font = ImageFont.load_default()
                    print("Using default font")
                
                # Add speech bubble
                d.rectangle([(5,5),(295,50)], fill=(255,255,255), outline=(0,0,0))
                d.text((10, 10), text[:100], fill=(0,0,0), font=font)
                
                # Create talking animation (mouth movement)
                mouth_y = 180 if len(text) % 2 else 190  # Alternate mouth position
                d.arc((100, 150, 200, mouth_y), 0, 180, fill=(0,0,0), width=2)
                
            print("Converting to numpy array")
            # Ensure image is in RGB mode and convert to numpy array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            result = np.array(img)
            
            print("Verifying array format")
            if not isinstance(result, np.ndarray):
                raise ValueError("Failed to convert image to numpy array")
            if len(result.shape) != 3 or result.shape[2] != 3:
                print(f"Converting array shape from {result.shape} to (300, 300, 3)")
                result = np.stack((result,)*3, axis=-1)[:,:,:3]  # Ensure 3 channels
                result = result[:300, :300]  # Ensure correct dimensions
                
            self.avatar_cache[cache_key] = result
            print("Avatar generated successfully")
            return result
            
        except Exception as e:
            print(f"Error in get_avatar: {str(e)}")
            # Return a default error image
            error_img = Image.new('RGB', (300, 300), (255, 200, 200))
            d = ImageDraw.Draw(error_img)
            d.text((10, 10), "Avatar Error", fill=(255,0,0))
            return np.array(error_img)

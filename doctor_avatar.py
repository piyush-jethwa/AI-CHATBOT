import gradio as gr
from PIL import Image
import numpy as np
import os

class DoctorAvatar:
    def __init__(self):
        # Create a simple doctor avatar image (replace with your actual image)
        self.avatar_image = self._create_default_avatar()
        
    def _create_default_avatar(self):
        """Create a simple placeholder avatar if no image is provided"""
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 200), color=(220, 220, 220))
        d = ImageDraw.Draw(img)
        d.ellipse((50, 50, 150, 150), fill=(255, 255, 255))
        d.ellipse((80, 80, 100, 100), fill=(0, 0, 0))
        d.ellipse((120, 80, 140, 100), fill=(0, 0, 0))
        d.arc((80, 100, 140, 140), 0, 180, fill=(0, 0, 0), width=5)
        return img
        
    def get_avatar(self):
        """Return the avatar image"""
        return np.array(self.avatar_image)
        
    def speak(self, text, language="English"):
        """Make the avatar appear to speak by animating"""
        # This would be enhanced with actual animation in a real implementation
        return self.get_avatar()

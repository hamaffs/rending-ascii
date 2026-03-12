"""
Video Assembler Module
Combines visuals with audio to create final MP4 video.

2-Step Process:
- Step 1: Generate clear video with mood-aware visuals (3D enhanced available)
- Step 2: Convert the clear video to ASCII art style
"""

import numpy as np
import cv2
import os
import shutil
from typing import Optional, Callable, List, Tuple
from moviepy.editor import AudioFileClip, ImageSequenceClip
from tqdm import tqdm

from audio_analyzer import AudioAnalyzer
from ascii_renderer import ASCIIRenderer, VisualStyle, render_frame


# Color palette for clear visuals (BGR format for OpenCV)
MOOD_COLORS_CLEAR = {
    'energetic': (50, 100, 255),
    'calm': (255, 200, 100),
    'happy': (50, 220, 255),
    'sad': (180, 100, 100),
    'angry': (50, 50, 255),
    'mysterious': (200, 50, 150),
    'uplifting': (150, 255, 100),
    'melancholic': (180, 130, 120),
    'romantic': (255, 100, 150),
    'dreamy': (150, 150, 255),
}


# ================== 3D VISUALS ==================

def apply_3d_depth_effect(frame, depth_level, frame_num, beat_intensity):
    """Apply 3D depth effect with perspective and parallax."""
    height, width = frame.shape[:2]
    result = frame.copy()
    
    num_layers = 3
    for layer in range(num_layers):
        layer_depth = (layer + 1) / num_layers
        if layer > 0 and depth_level > 0.3:
            kernel_size = int(layer * 3 * depth_level)
            if kernel_size % 2 == 0:
                kernel_size += 1
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    if depth_level > 0.5:
        zoom = 1.0 + beat_intensity * 0.05
        if zoom > 1.0:
            new_width = int(width * zoom)
            new_height = int(height * zoom)
            result = cv2.resize(result, (new_width, new_height))
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            result = result[start_y:start_y+height, start_x:start_x+width]
    
    vignette_strength = depth_level * 0.3
    kernel_x = cv2.getGaussianKernel(width, width/2)
    kernel_y = cv2.getGaussianKernel(height, height/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = cv2.resize(mask, (width, height))
    
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - vignette_strength + vignette_strength * mask)
    
    return result


def draw_3d_object(frame, obj_type, center_x, center_y, size, color, rotation, depth, audio_reaction):
    """Draw a 3D-looking object with depth and shadow."""
    shadow_offset = int(depth * 15)
    shadow_color = tuple(max(0, c - 60) for c in color)
    
    if obj_type == 'sphere':
        cv2.circle(frame, (center_x + shadow_offset, center_y + shadow_offset), 
                   int(size * 1.1), shadow_color, -1)
        cv2.circle(frame, (center_x, center_y), size, color, -1)
        highlight_x = center_x - int(size * 0.3)
        highlight_y = center_y - int(size * 0.3)
        cv2.circle(frame, (highlight_x, highlight_y), int(size * 0.2), 
                   tuple(min(255, c + 80) for c in color), -1)
        cv2.circle(frame, (center_x, center_y), size, tuple(max(0, c - 40) for c in color), 2)
        
    elif obj_type == 'cube':
        offset = int(size * 0.3 * (1 - depth))
        pts_front = np.array([
            [center_x - size, center_y - size],
            [center_x + size, center_y - size],
            [center_x + size, center_y + size],
            [center_x - size, center_y + size]
        ], np.int32)
        cv2.fillPoly(frame, [pts_front], color)
        pts_top = np.array([
            [center_x - size, center_y - size],
            [center_x + size, center_y - size],
            [center_x + size - offset, center_y - size - offset],
            [center_x - size - offset, center_y - size - offset]
        ], np.int32)
        cv2.fillPoly(frame, [pts_top], tuple(min(255, c + 30) for c in color))
        pts_side = np.array([
            [center_x + size, center_y - size],
            [center_x + size, center_y + size],
            [center_x + size - offset, center_y + size - offset],
            [center_x + size - offset, center_y - size - offset]
        ], np.int32)
        cv2.fillPoly(frame, [pts_side], tuple(max(0, c - 30) for c in color))
        
    elif obj_type == 'pyramid':
        pts = np.array([
            [center_x, center_y - size],
            [center_x + size, center_y + size],
            [center_x - size, center_y + size]
        ], np.int32)
        cv2.fillPoly(frame, [pts], color)
        cv2.line(frame, (center_x, center_y - size), (center_x + size, center_y + size),
                 tuple(max(0, c - 40) for c in color), 2)
        cv2.line(frame, (center_x, center_y - size), (center_x - size, center_y + size),
                 tuple(max(0, c - 40) for c in color), 2)
        
    elif obj_type == 'torus':
        cv2.ellipse(frame, (center_x + shadow_offset, center_y + shadow_offset),
                    (int(size*1.2), int(size*0.6)), 0, 0, 360, shadow_color, -1)
        cv2.ellipse(frame, (center_x, center_y),
                    (int(size*1.2), int(size*0.6)), 0, 0, 360, color, 15)
        cv2.ellipse(frame, (center_x, center_y),
                    (int(size*1.2), int(size*0.6)), 0, 0, 360, 
                    tuple(max(0, c - 40) for c in color), 3)
    
    if audio_reaction > 0.5:
        pulse_size = int(size * (1 + audio_reaction * 0.2))
        cv2.circle(frame, (center_x, center_y), pulse_size + 5, color, 1)
    
    return frame


def create_3d_enhanced_visual(width, height, frame_num, audio_features, color, lyrics_mood, objects_config):
    """Create enhanced 3D visual with objects and lyrics awareness."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    bass = audio_features.get('low_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    energy_level = audio_features.get('energy_level', 0.5)
    
    depth_level = 0.3 + energy_level * 0.5
    center_x, center_y = width // 2, height // 2
    
    # Background gradient based on lyrics mood
    if lyrics_mood == 'melodic':
        for y in range(height):
            alpha = y / height
            frame[y, :] = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
    elif lyrics_mood == 'intense':
        wave = int(np.sin(frame_num * 0.05) * 30)
        for y in range(height):
            alpha = (y + wave) / height
            frame[y, :] = tuple(int(c * (0.4 + 0.6 * alpha)) for c in color)
    elif lyrics_mood == 'ethereal':
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        base = np.full((height, width, 3), color, dtype=np.uint8)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    else:
        frame[:, :] = [int(c * 0.2) for c in color]
    
    # Floating particles
    for _ in range(int(50 * depth_level)):
        px = np.random.randint(0, width)
        py = np.random.randint(0, height)
        pz = np.random.random()
        size = int(1 + pz * 3)
        alpha = 0.3 + pz * 0.5
        frame[py, px] = tuple(int(c * alpha) for c in color)
    
    # Draw 3 objects
    beat_intensity = bass + is_beat * 0.5
    
    for i, obj in enumerate(objects_config):
        obj_type = obj.get('type', ['sphere', 'cube', 'pyramid', 'torus'][i % 4])
        angle = (i / len(objects_config)) * 2 * np.pi + frame_num * 0.01
        orbit_radius = 100 + bass * 150 + rms * 100
        
        obj_x = int(center_x + np.cos(angle) * orbit_radius)
        obj_y = int(center_y + np.sin(angle * 0.7) * orbit_radius * 0.6)
        
        base_size = obj.get('size', 40 + i * 10)
        obj_size = int(base_size * (1 + energy_level * 0.5 + is_beat * 0.3))
        depth = 0.5 + np.sin(frame_num * 0.02 + i) * 0.3
        
        obj_color_offset = i * 30
        obj_color = tuple(min(255, max(0, c + obj_color_offset - 45)) for c in color)
        
        frame = draw_3d_object(frame, obj_type, obj_x, obj_y, obj_size,
                               obj_color, 0, depth, bass + rms)
    
    # Energy lines between objects
    if energy_level > 0.4:
        for i in range(len(objects_config)):
            for j in range(i + 1, len(objects_config)):
                angle1 = (i / len(objects_config)) * 2 * np.pi + frame_num * 0.01
                angle2 = (j / len(objects_config)) * 2 * np.pi + frame_num * 0.01
                orbit_radius = 100 + bass * 150 + rms * 100
                x1 = int(center_x + np.cos(angle1) * orbit_radius)
                y1 = int(center_y + np.sin(angle1 * 0.7) * orbit_radius * 0.6)
                x2 = int(center_x + np.cos(angle2) * orbit_radius)
                y2 = int(center_y + np.sin(angle2 * 0.7) * orbit_radius * 0.6)
                line_color = tuple(int(c * energy_level * 0.5) for c in color)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 1)
    
    frame = apply_3d_depth_effect(frame, depth_level, frame_num, beat_intensity)
    
    if is_beat and bass > 0.15:
        glow_radius = int(50 + bass * 100)
        frame = cv2.circle(frame, (center_x, center_y), glow_radius, 
                          tuple(int(c * 0.5) for c in color), -1)
    
    return frame


def detect_lyrics_mood(audio_features):
    """Detect lyrics/vocal mood from audio characteristics."""
    spectral_centroid = audio_features.get('spectral_centroid', 2000)
    rms = audio_features.get('rms', 0.1)
    energy = audio_features.get('energy_level', 0.5)
    
    if spectral_centroid > 4000 and energy > 0.6:
        return 'ethereal'
    elif spectral_centroid > 3000 and energy < 0.4:
        return 'dreamy'
    elif energy > 0.7 and rms > 0.15:
        return 'intense'
    elif spectral_centroid < 2000 and energy < 0.4:
        return 'deep'
    elif spectral_centroid > 2500 and energy > 0.3:
        return 'melodic'
    else:
        return 'narrative'


# ================== STANDARD VISUALS ==================

def create_clear_geometric_pattern(width, height, frame_num, audio_features, color):
    """Create geometric patterns that pulse to the beat."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bass = audio_features.get('low_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    
    pulse = int(50 + bass * 200 + is_beat * 100)
    center_x, center_y = width // 2, height // 2
    
    cv2.circle(frame, (center_x, center_y), pulse, color, 3)
    
    for i in range(4):
        radius = pulse // 2 + i * 60 + int(rms * 100)
        alpha = int(255 * (1 - i / 4) * 0.7)
        color_with_alpha = tuple(min(255, c * alpha // 255 + 50) for c in color)
        cv2.circle(frame, (center_x, center_y), radius, color_with_alpha, 2)
    
    corner_size = int(50 + rms * 150)
    margin = 80
    corners = [(margin, margin), (width - margin - corner_size, margin),
               (margin, height - margin - corner_size), (width - margin - corner_size, height - margin - corner_size)]
    
    for cx, cy in corners:
        cv2.rectangle(frame, (cx, cy), (cx + corner_size, cy + corner_size), color, 2)
    
    return frame


def create_clear_waveform_pattern(width, height, frame_num, audio_features, color):
    """Create scrolling sine wave."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    rms = audio_features.get('rms', 0)
    treble = audio_features.get('high_energy', 0)
    
    for wave_num in range(3):
        offset_y = height // 4 * (wave_num + 1)
        amplitude = 50 + rms * 100 + wave_num * 30
        frequency = 0.005 + treble * 0.01
        
        points = []
        for x in range(0, width, 5):
            y = int(offset_y + np.sin((x + frame_num * 5) * frequency) * amplitude +
                    np.sin((x + frame_num * 2) * frequency * 2) * amplitude * 0.5)
            if 0 <= y < height:
                points.append((x, y))
        
        thickness = max(2, int(2 + rms * 5))
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, thickness)
    
    cv2.line(frame, (0, height // 2), (width, height // 2), tuple(max(0, c - 80) for c in color), 1)
    
    return frame


def create_clear_particle_pattern(width, height, frame_num, audio_features, particles, color):
    """Create particles that explode on beats."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bass = audio_features.get('low_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    
    if is_beat and bass > 0.1:
        for _ in range(int(bass * 15)):
            particles.append({
                'x': width // 2 + np.random.randint(-50, 50),
                'y': height // 2 + np.random.randint(-50, 50),
                'vx': np.random.randint(-8, 8) * (1 + bass * 3),
                'vy': np.random.randint(-8, 8) * (1 + bass * 3),
                'life': 60,
                'size': np.random.randint(3, 10)
            })
    
    new_particles = []
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        p['vy'] += 0.15
        
        if p['life'] > 0 and 0 < p['x'] < width and 0 < p['y'] < height:
            alpha = p['life'] / 60
            size = int(p['size'] * alpha)
            if size > 0:
                frame[int(p['y']), int(p['x'])] = tuple(int(c * alpha) for c in color)
            new_particles.append(p)
    
    if is_beat:
        for i in range(6):
            angle = i * (2 * np.pi / 6)
            length = 30 + bass * 50
            x2 = int(center_x := width // 2 + np.cos(angle + frame_num * 0.2) * length)
            y2 = int(height // 2 + np.sin(angle + frame_num * 0.2) * length)
            cv2.line(frame, (width // 2, height // 2), (x2, y2), color, 3)
    
    return frame, new_particles


def create_clear_mood_pattern(width, height, frame_num, audio_features, color, genre):
    """Create genre-specific patterns."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bass = audio_features.get('low_energy', 0)
    mid = audio_features.get('mid_energy', 0)
    treble = audio_features.get('high_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    energy_level = audio_features.get('energy_level', 0.5)
    
    center_x, center_y = width // 2, height // 2
    
    if genre == 'electronic':
        num_rings = max(3, int(energy_level * 8) + 3)
        for i in range(num_rings):
            radius = i * 50 + int(bass * 100) + int(is_beat * 30)
            thickness = max(2, 4 - i // 2)
            alpha = max(0.4, 1.0 - i / num_rings)
            ring_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, (center_x, center_y), radius, ring_color, thickness)
        
        if is_beat and bass > 0.1:
            cv2.circle(frame, (center_x, center_y), int(30 + bass * 100), color, -1)
            
    elif genre == 'rock':
        num_lines = max(5, int(energy_level * 15))
        spacing = width // (num_lines + 1)
        for i in range(num_lines):
            x = spacing * (i + 1)
            vibration = int(mid * 30 * (1 if i % 2 == 0 else -1))
            cv2.line(frame, (x + vibration, 0), (x - vibration, height), color, 2)
        
        if bass > 0.05:
            cv2.line(frame, (0, height - int(bass * 120)), (width, height - int(bass * 120)), color, 4)
            
    elif genre == 'pop':
        bounce_y = center_y - int(rms * 180)
        radius = max(25, int(60 + rms * 120))
        cv2.circle(frame, (center_x, bounce_y), radius, color, 3)
        
        for trail in range(1, 4):
            trail_y = bounce_y + trail * 25
            trail_color = tuple(int(c * (1.0 - trail * 0.3)) for c in color)
            cv2.circle(frame, (center_x, trail_y), int(radius * (1 - trail * 0.2)), trail_color, 2)
        
        if is_beat:
            cv2.circle(frame, (center_x, bounce_y), radius + 15, color, 4)
            
    elif genre == 'classical':
        for wave_idx in range(5):
            base_y = height // 6 * (wave_idx + 1)
            amplitude = 40 + bass * 60
            frequency = 0.008 + treble * 0.015
            phase = frame_num * 0.04 + wave_idx * 0.5
            
            points = []
            for x in range(0, width, 3):
                y = int(base_y + np.sin((x * frequency) + phase) * amplitude)
                if 0 <= y < height:
                    points.append((x, y))
            
            thickness = max(2, int(2 + rms * 4))
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, thickness)
                
    elif genre == 'hiphop':
        box_size = int(120 + bass * 120)
        cv2.rectangle(frame, (center_x - box_size//2, center_y - box_size//2),
                      (center_x + box_size//2, center_y + box_size//2), color, 3)
        
        if is_beat:
            cv2.rectangle(frame, (center_x - box_size//2 - 8, center_y - box_size//2 - 8),
                          (center_x + box_size//2 + 8, center_y + box_size//2 + 8), color, 2)
        
        for i in range(8):
            y = height // 9 * (i + 1)
            freq = bass if i < 3 else (rms if i < 6 else 0.1)
            bar_len = int(freq * 120)
            cv2.line(frame, (60, y), (60 + bar_len, y), color, 3)
            cv2.line(frame, (width - 60, y), (width - 60 - bar_len, y), color, 3)
            
    else:
        for i in range(5):
            radius = 50 + i * 70 + int(bass * 100)
            alpha = 1.0 - i / 5
            cv2.circle(frame, (center_x, center_y), radius, tuple(int(c * alpha) for c in color), 2)
    
    return frame


# ================== VIDEO ASSEMBLER ==================

class VideoAssembler:
    """Assembles audio-reactive frames into a video with 2-step process."""
    
    def __init__(
        self,
        audio_path: str,
        output_path: str = "output.mp4",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        style: str = "classic",
        mode: str = "geometric",
        objects: List[str] = None
    ):
        self.audio_path = audio_path
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.style = style
        self.mode = mode
        self.objects = objects or ['sphere', 'cube', 'pyramid']
        
        self.analyzer = AudioAnalyzer(audio_path)
        self.renderer = None
        self.progress_callback = None
        self.particles = []
        self._current_section = 'verse'
        self._mood = 'calm'
        self._genre = 'pop'
        self._lyrics_mood = 'narrative'
        
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def analyze_audio(self):
        print("Analyzing audio...")
        self.features = self.analyzer.analyze()
        print(f"BPM: {self.features['bpm']:.1f}")
        print(f"Duration: {self.features['duration']:.2f}s")
        
        self._mood = self.features.get('mood', 'calm')
        self._genre = self.features.get('genre', 'pop')
        print(f"Detected Mood: {self._mood}")
        print(f"Detected Genre: {self._genre}")
        
        sections = self.features.get('sections', [])
        print(f"Song Sections: {len(sections)} detected")
        
        self.renderer = ASCIIRenderer(
            width=self.width,
            height=self.height,
            style=VisualStyle(self.style),
            mood=self._mood,
            genre=self._genre
        )
        
        return self.features
    
    def generate_frames(self) -> List[np.ndarray]:
        """Generate frames based on selected mode."""
        if not hasattr(self, 'features'):
            self.analyze_audio()
        
        duration = self.features['duration']
        total_frames = int(duration * self.fps)
        
        print(f"Generating {total_frames} frames at {self.fps} fps...")
        print(f"Mode: {self.mode}, Objects: {self.objects}")
        
        color = MOOD_COLORS_CLEAR.get(self._mood, (255, 255, 255))
        
        # Objects config for 3D mode
        objects_config = [
            {'type': self.objects[0] if len(self.objects) > 0 else 'sphere', 'size': 50},
            {'type': self.objects[1] if len(self.objects) > 1 else 'cube', 'size': 45},
            {'type': self.objects[2] if len(self.objects) > 2 else 'pyramid', 'size': 40},
        ]
        
        frames = []
        self.particles = []
        
        for frame_num in tqdm(range(total_frames), desc="Creating visuals"):
            audio_features = self.analyzer.get_frame_features(frame_num, self.fps)
            
            # Detect lyrics mood for 3D mode
            if self.mode == 'enhanced_3d':
                self._lyrics_mood = detect_lyrics_mood(audio_features)
            
            current_section = audio_features.get('section', 'verse')
            if current_section != self._current_section:
                print(f"  Section change: {self._current_section} -> {current_section}")
                self._current_section = current_section
            
            # Generate frame based on mode
            if self.mode == "enhanced_3d":
                frame = create_3d_enhanced_visual(
                    self.width, self.height, frame_num, audio_features,
                    color, self._lyrics_mood, objects_config
                )
            elif self.mode == "geometric":
                frame = create_clear_geometric_pattern(
                    self.width, self.height, frame_num, audio_features, color
                )
            elif self.mode == "waveform":
                frame = create_clear_waveform_pattern(
                    self.width, self.height, frame_num, audio_features, color
                )
            elif self.mode == "particles":
                frame, self.particles = create_clear_particle_pattern(
                    self.width, self.height, frame_num, audio_features, self.particles, color
                )
            elif self.mode == "mood":
                frame = create_clear_mood_pattern(
                    self.width, self.height, frame_num, audio_features, color, self._genre
                )
            else:
                frame = create_clear_geometric_pattern(
                    self.width, self.height, frame_num, audio_features, color
                )
            
            frames.append(frame)
            
            # Update progress (0-50% for this step)
            if self.progress_callback:
                progress = (frame_num + 1) / total_frames * 0.5
                self.progress_callback(progress)
        
        return frames
    
    def create_video(self, frames: List[np.ndarray], output_path: str) -> str:
        print(f"Creating video: {output_path}")
        
        clip = ImageSequenceClip(frames, fps=self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        
        clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=self.fps,
            verbose=False,
            logger=None
        )
        
        print(f"Video saved to: {output_path}")
        return output_path
    
    def convert_to_ascii(self, input_video_path: str) -> str:
        print("Step 2: Converting video to ASCII style...")
        
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Converting {total_frames} frames to ASCII...")
        
        ascii_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            resized = cv2.resize(
                gray,
                (self.renderer.char_resolution[0], self.renderer.char_resolution[1]),
                interpolation=cv2.INTER_AREA
            )
            
            char_array = self.renderer.pixel_to_ascii(resized)
            color = self.renderer.get_color()
            ascii_frame = self.renderer.ascii_to_image(char_array, color)
            
            ascii_frames.append(ascii_frame)
            frame_count += 1
            
            if self.progress_callback and total_frames > 0:
                progress = 0.5 + (frame_count / total_frames) * 0.5
                self.progress_callback(progress)
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        
        print("Creating final ASCII video...")
        
        clip = ImageSequenceClip(ascii_frames, fps=self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        
        clip.write_videofile(
            self.output_path,
            codec='libx264',
            audio_codec='aac',
            fps=self.fps,
            verbose=False,
            logger=None
        )
        
        print(f"Final ASCII video saved to: {self.output_path}")
        return self.output_path
    
    def render(self) -> str:
        """Complete rendering pipeline with 2-step process."""
        # STEP 1: Generate clear video
        print("\n" + "="*60)
        print("STEP 1: Generating clear video with mood-aware visuals")
        print("="*60 + "\n")
        
        self.analyze_audio()
        frames = self.generate_frames()
        
        intermediate_path = self.output_path.replace('.mp4', '_clear.mp4')
        self.create_video(frames, intermediate_path)
        
        print(f"Step 1 complete: {intermediate_path}")
        
        # STEP 2: Convert to ASCII
        print("\n" + "="*60)
        print("STEP 2: Converting to ASCII art style")
        print("="*60 + "\n")
        
        final_output = self.convert_to_ascii(intermediate_path)
        
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)
        
        print("\n" + "="*60)
        print("RENDERING COMPLETE!")
        print("="*60 + "\n")
        
        return final_output


def create_video_from_audio(
    audio_path: str,
    output_path: str = "output.mp4",
    fps: int = 30,
    style: str = "classic",
    mode: str = "geometric",
    objects: List[str] = None,
    progress_callback = None
) -> str:
    """Convenience function to create video from audio."""
    assembler = VideoAssembler(
        audio_path=audio_path,
        output_path=output_path,
        fps=fps,
        style=style,
        mode=mode,
        objects=objects
    )
    
    if progress_callback:
        assembler.set_progress_callback(progress_callback)
    
    return assembler.render()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
        
        mode = "enhanced_3d"
        objects = ['sphere', 'cube', 'pyramid']
        
        create_video_from_audio(audio_path, output_path, mode=mode, objects=objects)
    else:
        print("Usage: python video_assembler.py <audio_file> [output_file]")

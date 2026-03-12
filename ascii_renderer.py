"""
ASCII Renderer Module
Converts video frames to ASCII art with audio-reactive and mood-aware effects.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum


class VisualStyle(Enum):
    CLASSIC = "classic"
    MATRIX = "matrix"
    HIGH_CONTRAST = "high_contrast"
    MOOD_AWARE = "mood_aware"  # New: adapts to detected mood


class Mood(Enum):
    """Audio mood categories for visual mapping."""
    ENERGETIC = "energetic"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    MYSTERIOUS = "mysterious"
    UPLIFTING = "uplifting"
    MELANCHOLIC = "melancholic"


class Genre(Enum):
    """Genre categories for visual mapping."""
    ELECTRONIC = "electronic"
    ROCK = "rock"
    POP = "pop"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    HIPHOP = "hiphop"
    ACOUSTIC = "acoustic"
    AMBIENT = "ambient"


# Mood to color mapping
MOOD_COLORS = {
    'energetic': (255, 100, 50),      # Orange-red
    'calm': (100, 200, 255),          # Light blue
    'happy': (255, 220, 50),          # Yellow
    'sad': (100, 100, 180),           # Blue-purple
    'angry': (255, 50, 50),          # Red
    'mysterious': (150, 50, 200),     # Purple
    'uplifting': (100, 255, 150),     # Bright green
    'melancholic': (120, 130, 180),   # Muted blue
}

# Genre to character set mapping
GENRE_CHARS = {
    'electronic': " .:-=+*#%@MNW$",
    'rock': " .:-=+*#%@X#",
    'pop': " .:-=+*#%@O@",
    'classical': " .:-=+*#%@",
    'jazz': " .:-=+*#%@$&",
    'hiphop': " .:-=+*#%@#$",
    'acoustic': " .:-=+*#%@",
    'ambient': " . -:=",
}

# Genre to pattern style mapping
GENRE_PATTERNS = {
    'electronic': 'pulse',
    'rock': 'vibration',
    'pop': 'bounce',
    'classical': 'flow',
    'jazz': 'swing',
    'hiphop': 'groove',
    'acoustic': 'gentle',
    'ambient': 'drift',
}


class ASCIIRenderer:
    """Renders audio-reactive ASCII art visuals with mood awareness."""
    
    # Character sets for different brightness levels
    ASCII_CHARS = " .:-=+*#%@"
    MATRIX_CHARS = " .:-=+*#%@MN$@#"
    HIGH_CONTRAST_CHARS = " .#@"
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        char_resolution: Tuple[int, int] = (160, 90),
        style: VisualStyle = VisualStyle.CLASSIC,
        mood: str = "calm",
        genre: str = "pop"
    ):
        self.width = width
        self.height = height
        self.char_resolution = char_resolution
        self.style = style
        self.mood = mood
        self.genre = genre
        self.char_width = 10  # Approximate width of monospace character
        self.char_height = 18  # Approximate height of monospace character
        
        # Set character set based on style and genre
        self.chars = self._get_characters()
        
        # Visual state
        self.shake_offset = (0, 0)
        self.scale_factor = 1.0
        self.flicker_intensity = 0
        
        # Animation state
        self._pulse_phase = 0.0
        self._wave_offset = 0.0
        self._particles = []
        
    def _get_characters(self) -> str:
        """Get character set based on style."""
        if self.style == VisualStyle.MATRIX:
            return self.MATRIX_CHARS
        elif self.style == VisualStyle.HIGH_CONTRAST:
            return self.HIGH_CONTRAST_CHARS
        else:
            return self.ASCII_CHARS
    
    def set_style(self, style: VisualStyle):
        """Change the visual style."""
        self.style = style
        self.chars = self._get_characters()
    
    def pixel_to_ascii(self, frame: np.ndarray) -> np.ndarray:
        """Convert a frame to ASCII art as a numpy array."""
        # Resize to character resolution
        resized = cv2.resize(
            frame, 
            (self.char_resolution[0], self.char_resolution[1]),
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Normalize to 0-1
        normalized = gray.astype(np.float32) / 255.0
        
        # Map to characters
        char_indices = (normalized * (len(self.chars) - 1)).astype(np.uint8)
        
        return char_indices
    
    def ascii_to_image(
        self, 
        char_array: np.ndarray, 
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Convert ASCII character array back to an image."""
        # Create blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate scaling for shake effect
        offset_x, offset_y = self.shake_offset
        scale = self.scale_factor
        
        # Calculate character sizes
        char_w = self.width / self.char_resolution[0]
        char_h = self.height / self.char_resolution[1]
        
        for y in range(self.char_resolution[1]):
            for x in range(self.char_resolution[0]):
                char_idx = char_array[y, x]
                char = self.chars[char_idx]
                
                # Calculate position with shake
                pos_x = int(x * char_w + offset_x) * scale
                pos_y = int(y * char_h + offset_y) * scale
                
                # Draw character
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4 * scale
                thickness = 1
                
                # Get text size for centering
                (text_w, text_h), baseline = cv2.getTextSize(
                    char, font, font_scale, thickness
                )
                
                # Ensure within bounds
                if pos_x < self.width and pos_y < self.height:
                    cv2.putText(
                        img, 
                        char, 
                        (int(pos_x), int(pos_y + text_h)), 
                        font, 
                        font_scale, 
                        color, 
                        thickness
                    )
        
        return img
    
    def create_blank_frame(self) -> np.ndarray:
        """Create a blank black frame."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def apply_shake(self, intensity: float):
        """Apply shake effect based on bass frequency."""
        if intensity > 0.1:
            offset = int(intensity * 20)
            self.shake_offset = (
                np.random.randint(-offset, offset),
                np.random.randint(-offset, offset)
            )
        else:
            self.shake_offset = (0, 0)
    
    def apply_scale(self, intensity: float):
        """Apply scaling based on bass frequency."""
        self.scale_factor = 1.0 + intensity * 0.3
    
    def apply_flicker(self, intensity: float):
        """Apply flicker effect based on treble frequency."""
        self.flicker_intensity = intensity
        if intensity > 0.3 and np.random.random() < intensity:
            # Skip some characters to create flicker effect
            self._flicker_active = True
        else:
            self._flicker_active = False
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color based on visual style and mood."""
        if self.style == VisualStyle.MATRIX:
            # Matrix green
            return (0, 255, 70)
        elif self.style == VisualStyle.HIGH_CONTRAST:
            # White
            return (255, 255, 255)
        elif self.style == VisualStyle.MOOD_AWARE:
            # Get color based on mood
            return MOOD_COLORS.get(self.mood, (255, 255, 255))
        else:
            # For other modes (geometric, waveform, particles), also use mood-based color
            # This makes ALL modes respond to the song's mood
            return MOOD_COLORS.get(self.mood, (255, 255, 255))
    
    def get_mood_color(self) -> Tuple[int, int, int]:
        """Get color based on detected mood."""
        return MOOD_COLORS.get(self.mood, (255, 255, 255))
    
    def set_mood(self, mood: str):
        """Set the mood for visual adaptation."""
        self.mood = mood
        if self.style == VisualStyle.MOOD_AWARE:
            self.chars = GENRE_CHARS.get(self.genre, self.ASCII_CHARS)
    
    def set_genre(self, genre: str):
        """Set the genre for visual adaptation."""
        self.genre = genre
        if self.style == VisualStyle.MOOD_AWARE:
            self.chars = GENRE_CHARS.get(genre, self.ASCII_CHARS)


# ============================================================================
# MOOD-AWARE VISUAL FUNCTIONS
# ============================================================================

def create_mood_pattern(
    renderer: ASCIIRenderer,
    frame_num: int,
    audio_features: dict
) -> np.ndarray:
    """
    Create patterns that adapt to the detected mood, genre, and section.
    This creates meaningful visuals that match the song's character.
    """
    frame = renderer.create_blank_frame()
    
    # Extract audio features
    bass = audio_features.get('low_energy', 0)
    mid = audio_features.get('mid_energy', 0)
    treble = audio_features.get('high_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    energy_level = audio_features.get('energy_level', 0.5)
    dynamics = audio_features.get('dynamics', 0.5)
    
    # Get mood and genre
    mood = audio_features.get('mood', 'calm')
    genre = audio_features.get('genre', 'pop')
    section = audio_features.get('section', 'verse')
    
    # Get color based on mood
    color = MOOD_COLORS.get(mood, (255, 255, 255))
    
    # Get pattern style based on genre
    pattern_style = GENRE_PATTERNS.get(genre, 'bounce')
    
    # Adjust intensity based on section (climax = more intense)
    section_multiplier = 1.0
    if section == 'climax':
        section_multiplier = 1.5
    elif section == 'dynamic':
        section_multiplier = 1.2
    elif section == 'low':
        section_multiplier = 0.7
    
    # Choose pattern based on genre/style
    if pattern_style == 'pulse':
        frame = _create_pulse_pattern(frame, renderer, frame_num, bass, rms, is_beat, 
                                       energy_level, section_multiplier, color)
    elif pattern_style == 'vibration':
        frame = _create_vibration_pattern(frame, renderer, frame_num, bass, mid, treble,
                                           is_beat, energy_level, section_multiplier, color)
    elif pattern_style == 'bounce':
        frame = _create_bounce_pattern(frame, renderer, frame_num, rms, is_beat,
                                        energy_level, dynamics, section_multiplier, color)
    elif pattern_style == 'flow':
        frame = _create_flow_pattern(frame, renderer, frame_num, bass, mid, treble,
                                     rms, energy_level, section_multiplier, color)
    elif pattern_style == 'swing':
        frame = _create_swing_pattern(frame, renderer, frame_num, rms, is_beat,
                                       energy_level, section_multiplier, color)
    elif pattern_style == 'groove':
        frame = _create_groove_pattern(frame, renderer, frame_num, bass, rms, is_beat,
                                        energy_level, section_multiplier, color)
    elif pattern_style == 'gentle':
        frame = _create_gentle_pattern(frame, renderer, frame_num, rms, is_beat,
                                         energy_level, section_multiplier, color)
    elif pattern_style == 'drift':
        frame = _create_drift_pattern(frame, renderer, frame_num, rms, bass, treble,
                                        energy_level, section_multiplier, color)
    else:
        frame = _create_pulse_pattern(frame, renderer, frame_num, bass, rms, is_beat,
                                       energy_level, section_multiplier, color)
    
    return frame


def _create_pulse_pattern(frame, renderer, frame_num, bass, rms, is_beat, 
                          energy_level, multiplier, color) -> np.ndarray:
    """Electronic/EDM style - concentric pulses on beats."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Number of pulse rings based on energy
    num_rings = max(3, int(energy_level * 8) + 3)
    
    for i in range(num_rings):
        # Each ring pulses outward
        base_radius = i * 40
        pulse_offset = int(bass * 100 * multiplier) if i == 0 else 0
        beat_offset = int(is_beat * 30)
        
        radius = base_radius + pulse_offset + beat_offset
        thickness = max(1, 3 - i // 3)
        
        # Fade color based on ring distance
        alpha = max(0.3, 1.0 - i / num_rings)
        ring_color = tuple(int(c * alpha) for c in color)
        
        cv2.circle(frame, (center_x, center_y), radius, ring_color, thickness)
    
    # Central glow on beat
    if is_beat and bass > 0.1:
        glow_radius = int(20 + bass * 80)
        cv2.circle(frame, (center_x, center_y), glow_radius, color, -1)
    
    return frame


def _create_vibration_pattern(frame, renderer, frame_num, bass, mid, treble,
                              is_beat, energy_level, multiplier, color) -> np.ndarray:
    """Rock style - vibrating lines and chords."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Vertical vibration lines
    num_lines = max(5, int(energy_level * 15))
    spacing = renderer.width // (num_lines + 1)
    
    for i in range(num_lines):
        x = spacing * (i + 1)
        # Vibrate based on mid frequencies
        vibration = int(mid * 30 * multiplier * (1 if i % 2 == 0 else -1))
        
        cv2.line(frame, (x + vibration, 0), (x - vibration, renderer.height), 
                 color, 1)
    
    # Horizontal bass lines at bottom
    if bass > 0.05:
        bass_height = int(bass * 100 * multiplier)
        cv2.line(frame, (0, renderer.height - bass_height), 
                 (renderer.width, renderer.height - bass_height), color, 3)
    
    return frame


def _create_bounce_pattern(frame, renderer, frame_num, rms, is_beat,
                           energy_level, dynamics, multiplier, color) -> np.ndarray:
    """Pop style - bouncing shapes to the beat."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Bouncing ball effect
    bounce_height = int(rms * 150 * multiplier)
    bounce_y = center_y - bounce_height
    
    # Draw bouncing circle
    radius = max(20, int(50 + rms * 100))
    cv2.circle(frame, (center_x, bounce_y), radius, color, 2)
    
    # Add motion trail
    for trail in range(1, 4):
        trail_y = bounce_y + trail * 20
        alpha = 1.0 - trail * 0.3
        trail_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, (center_x, trail_y), int(radius * (1 - trail * 0.2)), 
                   trail_color, 1)
    
    # Bounce on beat
    if is_beat:
        cv2.circle(frame, (center_x, bounce_y), radius + 10, color, 3)
    
    return frame


def _create_flow_pattern(frame, renderer, frame_num, bass, mid, treble,
                         rms, energy_level, multiplier, color) -> np.ndarray:
    """Classical style - flowing waves and elegant curves."""
    # Draw multiple flowing sine waves
    num_waves = 5
    
    for wave_idx in range(num_waves):
        base_y = renderer.height // (num_waves + 1) * (wave_idx + 1)
        amplitude = 30 + bass * 50 * multiplier
        frequency = 0.01 + treble * 0.02
        phase = frame_num * 0.05 + wave_idx * 0.5
        
        points = []
        for x in range(0, renderer.width, 3):
            y = int(base_y + np.sin((x * frequency) + phase) * amplitude)
            if 0 <= y < renderer.height:
                points.append((x, y))
        
        # Draw wave with varying thickness based on position
        for i in range(len(points) - 1):
            thickness = max(1, int(2 + rms * 3))
            cv2.line(frame, points[i], points[i + 1], color, thickness)
    
    return frame


def _create_swing_pattern(frame, renderer, frame_num, rms, is_beat,
                          energy_level, multiplier, color) -> np.ndarray:
    """Jazz style - swing dancing curves and improvisation."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Rotating swing pattern
    num_arms = 6
    swing_speed = frame_num * 0.03
    
    for i in range(num_arms):
        angle = (2 * np.pi / num_arms) * i + swing_speed
        arm_length = 100 + rms * 150 * multiplier
        
        x1, y1 = center_x, center_y
        x2 = int(center_x + np.cos(angle) * arm_length)
        y2 = int(center_y + np.sin(angle) * arm_length)
        
        # Add swing offset
        x2 += int(np.sin(swing_speed * 2 + i) * 20)
        
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw note at end
        if is_beat and i == frame_num % num_arms:
            cv2.circle(frame, (x2, y2), 15, color, -1)
    
    return frame


def _create_groove_pattern(frame, renderer, frame_num, bass, rms, is_beat,
                           energy_level, multiplier, color) -> np.ndarray:
    """Hip-hop style - bass-heavy groove and rhythm."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Beat indicator bar at bottom
    bar_width = int(renderer.width * (0.3 + rms * 0.5))
    bar_height = int(20 + bass * 80 * multiplier)
    
    # Draw center beat box
    box_size = int(100 + bass * 100 * multiplier)
    cv2.rectangle(frame, 
                  (center_x - box_size//2, center_y - box_size//2),
                  (center_x + box_size//2, center_y + box_size//2),
                  color, 2)
    
    # Pulse the box on beat
    if is_beat:
        cv2.rectangle(frame,
                      (center_x - box_size//2 - 5, center_y - box_size//2 - 5),
                      (center_x + box_size//2 + 5, center_y + box_size//2 + 5),
                      color, 1)
    
    # Side equalizer bars
    num_bars = 8
    bar_spacing = renderer.height // (num_bars + 1)
    
    for i in range(num_bars):
        y = bar_spacing * (i + 1)
        # Vary bar heights based on frequency
        freq = bass if i < 3 else (rms if i < 6 else 0.1)
        bar_len = int(freq * 100 * multiplier)
        
        # Left side
        cv2.line(frame, (50, y), (50 + bar_len, y), color, 2)
        # Right side  
        cv2.line(frame, (renderer.width - 50, y), (renderer.width - 50 - bar_len, y), color, 2)
    
    return frame


def _create_gentle_pattern(frame, renderer, frame_num, rms, is_beat,
                           energy_level, multiplier, color) -> np.ndarray:
    """Acoustic style - gentle, natural movements."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Gentle floating circles
    num_circles = 4
    
    for i in range(num_circles):
        offset = i * 50
        x = center_x + int(np.sin(frame_num * 0.02 + i) * 100)
        y = center_y + int(np.cos(frame_num * 0.015 + i * 0.5) * 50)
        
        radius = 30 + i * 20 + int(rms * 30 * multiplier)
        
        # Gentle pulsing
        pulse = int(np.sin(frame_num * 0.05 + i) * 5)
        
        alpha = 0.7 - i * 0.15
        circle_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, (x, y), radius + pulse, circle_color, 1)
    
    return frame


def _create_drift_pattern(frame, renderer, frame_num, rms, bass, treble,
                         energy_level, multiplier, color) -> np.ndarray:
    """Ambient style - slow drifting particles and ethereal feel."""
    center_x, center_y = renderer.width // 2, renderer.height // 2
    
    # Create drifting particles
    num_particles = 20
    
    np.random.seed(42)  # For consistent pattern
    for i in range(num_particles):
        # Organic movement
        t = frame_num * 0.01
        angle = i * 0.5 + t
        radius = 100 + i * 20
        
        x = int(center_x + np.cos(angle + i * 0.3) * radius * 0.8)
        y = int(center_y + np.sin(angle + i * 0.2) * radius * 0.5)
        
        # Fade based on distance
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        alpha = max(0.2, 1.0 - dist / 400)
        
        particle_color = tuple(int(c * alpha) for c in color)
        
        # Draw particle
        size = max(1, int(2 + rms * 5))
        cv2.circle(frame, (x, y), size, particle_color, -1)
    
    # Central glow
    glow_radius = int(50 + bass * 50 * multiplier)
    glow_color = tuple(int(c * 0.3) for c in color)
    cv2.circle(frame, (center_x, center_y), glow_radius, glow_color, -1)
    
    return frame


def create_section_transition(
    renderer: ASCIIRenderer,
    frame_num: int,
    audio_features: dict,
    prev_section: str
) -> np.ndarray:
    """
    Create a transition effect between song sections.
    """
    section = audio_features.get('section', 'verse')
    
    # If section changed, create a transition effect
    if section != prev_section:
        frame = renderer.create_blank_frame()
        
        color = renderer.get_mood_color()
        
        if section == 'climax':
            # Flash effect for climax
            for _ in range(3):
                cv2.rectangle(frame, (0, 0), (renderer.width, renderer.height), color, -1)
        
        elif section == 'dynamic':
            # Ripple effect for dynamic sections
            center = (renderer.width // 2, renderer.height // 2)
            for i in range(5):
                radius = i * 50 + (frame_num % 30) * 10
                alpha = 1.0 - i * 0.2
                ripple_color = tuple(int(c * alpha) for c in color)
                cv2.circle(frame, center, radius, ripple_color, 2)
        
        else:
            # Gentle fade for low sections
            frame = create_mood_pattern(renderer, frame_num, audio_features)
    else:
        frame = create_mood_pattern(renderer, frame_num, audio_features)
    
    return frame, section


def create_geometric_pattern(
    renderer: ASCIIRenderer,
    frame_num: int,
    audio_features: dict
) -> np.ndarray:
    """Create geometric patterns that pulse to the beat."""
    frame = renderer.create_blank_frame()
    
    # Get audio-reactive values
    bass = audio_features.get('low_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    
    # Pulse size based on bass
    pulse = int(50 + bass * 200 + is_beat * 100)
    
    center_x = renderer.width // 2
    center_y = renderer.height // 2
    
    # Draw pulsing circle
    color = renderer.get_color()
    cv2.circle(frame, (center_x, center_y), pulse, color, 2)
    
    # Draw secondary circles
    num_circles = 3
    for i in range(num_circles):
        radius = pulse // 2 + i * 50
        alpha = int(255 * (1 - i / num_circles))
        color_with_alpha = tuple(min(255, c + alpha // 3) for c in color)
        cv2.circle(frame, (center_x, center_y), radius, color_with_alpha, 1)
    
    # Add corner rectangles that pulse
    corner_size = int(50 + rms * 150)
    margin = 50
    corners = [
        (margin, margin),
        (renderer.width - margin - corner_size, margin),
        (margin, renderer.height - margin - corner_size),
        (renderer.width - margin - corner_size, renderer.height - margin - corner_size)
    ]
    
    for cx, cy in corners:
        cv2.rectangle(frame, (cx, cy), (cx + corner_size, cy + corner_size), color, 2)
    
    return frame


def create_waveform_pattern(
    renderer: ASCIIRenderer,
    frame_num: int,
    audio_features: dict
) -> np.ndarray:
    """Create a scrolling sine wave that reacts to amplitude."""
    frame = renderer.create_blank_frame()
    
    # Get audio values
    rms = audio_features.get('rms', 0)
    bass = audio_features.get('low_energy', 0)
    treble = audio_features.get('high_energy', 0)
    
    color = renderer.get_color()
    
    # Draw multiple waveforms
    for wave_num in range(3):
        offset_y = renderer.height // 4 * (wave_num + 1)
        amplitude = 50 + rms * 100 + wave_num * 30
        frequency = 0.02 + treble * 0.05
        
        points = []
        for x in range(0, renderer.width, 5):
            y = int(
                offset_y + 
                np.sin((x + frame_num * 5) * frequency) * amplitude +
                np.sin((x + frame_num * 2) * frequency * 2) * amplitude * 0.5
            )
            if 0 <= y < renderer.height:
                points.append((x, y))
        
        # Draw the wave
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
    
    # Add horizontal center line
    cv2.line(
        frame, 
        (0, renderer.height // 2), 
        (renderer.width, renderer.height // 2), 
        tuple(c // 3 for c in color), 
        1
    )
    
    return frame


def create_particle_pattern(
    renderer: ASCIIRenderer,
    frame_num: int,
    audio_features: dict,
    particles: List[dict] = None
) -> Tuple[np.ndarray, List[dict]]:
    """Create ASCII characters that explode on drum hits."""
    frame = renderer.create_blank_frame()
    
    if particles is None:
        particles = []
    
    # Get audio values
    bass = audio_features.get('low_energy', 0)
    rms = audio_features.get('rms', 0)
    is_beat = audio_features.get('is_beat', 0)
    
    color = renderer.get_color()
    
    # Create new particles on beat
    if is_beat and bass > 0.1:
        num_new = int(bass * 20)
        for _ in range(num_new):
            particles.append({
                'x': renderer.width // 2 + np.random.randint(-50, 50),
                'y': renderer.height // 2 + np.random.randint(-50, 50),
                'vx': np.random.randint(-10, 10) * (1 + bass * 5),
                'vy': np.random.randint(-10, 10) * (1 + bass * 5),
                'life': 60,
                'char': np.random.choice(list(renderer.chars))
            })
    
    # Update and draw particles
    new_particles = []
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        p['vy'] += 0.2  # Gravity
        
        if p['life'] > 0 and 0 < p['x'] < renderer.width and 0 < p['y'] < renderer.height:
            # Draw particle
            font = cv2.FONT_HERSHEY_SIMPLEX
            alpha = p['life'] / 60
            color_faded = tuple(int(c * alpha) for c in color)
            cv2.putText(
                frame, 
                p['char'], 
                (int(p['x']), int(p['y'])), 
                font, 
                0.5, 
                color_faded, 
                1
            )
            new_particles.append(p)
    
    # Draw center "explosion" on beat
    if is_beat:
        for i in range(5):
            angle = i * (2 * np.pi / 5)
            length = 30 + bass * 50
            x1 = renderer.width // 2
            y1 = renderer.height // 2
            x2 = int(x1 + np.cos(angle + frame_num * 0.2) * length)
            y2 = int(y1 + np.sin(angle + frame_num * 0.2) * length)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame, new_particles


def render_frame(
    renderer: ASCIIRenderer,
    mode: str,
    frame_num: int,
    audio_features: dict,
    particles: List[dict] = None,
    prev_section: str = 'verse'
) -> Tuple[np.ndarray, List[dict]]:
    """Render a single frame based on the selected mode."""
    
    # Apply audio-reactive effects
    bass = audio_features.get('low_energy', 0)
    treble = audio_features.get('high_energy', 0)
    
    renderer.apply_shake(bass)
    renderer.apply_scale(bass)
    renderer.apply_flicker(treble)
    
    # Create base pattern - now includes mood-aware mode
    if mode == "geometric":
        base_frame = create_geometric_pattern(renderer, frame_num, audio_features)
        particles_out = particles
    elif mode == "waveform":
        base_frame = create_waveform_pattern(renderer, frame_num, audio_features)
        particles_out = particles
    elif mode == "particles":
        base_frame, particles_out = create_particle_pattern(
            renderer, frame_num, audio_features, particles
        )
    elif mode == "mood":
        # New: Mood-aware rendering that adapts to detected mood/genre
        base_frame = create_mood_pattern(renderer, frame_num, audio_features)
        particles_out = particles
    else:
        base_frame = renderer.create_blank_frame()
        particles_out = particles
    
    # Convert to ASCII
    char_array = renderer.pixel_to_ascii(base_frame)
    
    # Convert back to image with ASCII characters
    # For mood mode, use mood-based color
    if mode == "mood":
        color = renderer.get_mood_color()
    else:
        color = renderer.get_color()
    
    final_frame = renderer.ascii_to_image(char_array, color)
    
    return final_frame, particles_out


if __name__ == "__main__":
    # Test the renderer
    renderer = ASCIIRenderer(style=VisualStyle.CLASSIC)
    frame = renderer.create_blank_frame()
    print(f"Created blank frame: {frame.shape}")
    
    # Test pixel to ASCII
    test_array = renderer.pixel_to_ascii(frame)
    print(f"ASCII array shape: {test_array.shape}")

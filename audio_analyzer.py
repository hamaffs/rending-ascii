"""
Audio Analyzer Module
Extracts BPM, beats, spectral features, mood, and genre from audio files using Librosa.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum


class Mood(Enum):
    """Audio mood/emotion categories."""
    ENERGETIC = "energetic"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    MYSTERIOUS = "mysterious"
    UPLIFTING = "uplifting"
    MELANCHOLIC = "melancholic"


class Genre(Enum):
    """Audio genre categories (detected from audio features)."""
    ELECTRONIC = "electronic"
    ROCK = "rock"
    POP = "pop"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    HIPHOP = "hiphop"
    ACOUSTIC = "acoustic"
    AMBIENT = "ambient"


class AudioAnalyzer:
    """Analyzes audio files to extract features for visual mapping."""
    
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.y = None
        self.sr = None
        self.features = {}
        self._mood = None
        self._genre = None
        self._sections = None
        
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate."""
        self.y, self.sr = librosa.load(self.audio_path, sr=None)
        return self.y, self.sr
    
    def analyze(self) -> Dict[str, Any]:
        """Perform complete audio analysis and return all features."""
        if self.y is None:
            self.load_audio()
            
        # Extract BPM
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.features['bpm'] = float(tempo)
        
        # Extract beat frames for onset detection
        onset_env = librosa.onset.onset_detect(y=self.y, sr=self.sr, backtrack=True)
        self.features['onset_frames'] = onset_env
        self.features['beat_frames'] = beat_frames
        
        # Extract spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        self.features['spectral_centroid'] = spectral_centroid[0]
        self.features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=self.y)
        self.features['rms'] = rms[0]
        self.features['rms_mean'] = float(np.mean(rms))
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=self.y)
        self.features['zcr'] = zcr[0]
        
        # Extract frequency bands
        self._extract_frequency_bands()
        
        # Get duration
        self.features['duration'] = float(librosa.get_duration(y=self.y, sr=self.sr))
        
        # Detect mood
        self._mood = self._detect_mood()
        self.features['mood'] = self._mood.value
        
        # Detect genre
        self._genre = self._detect_genre()
        self.features['genre'] = self._genre.value
        
        # Detect song sections
        self._sections = self._detect_sections()
        self.features['sections'] = self._sections
        
        # Calculate overall energy level (0-1)
        self.features['energy_level'] = self._calculate_energy_level()
        
        # Calculate dynamics (variation in energy)
        self.features['dynamics'] = self._calculate_dynamics()
        
        return self.features
    
    def _detect_mood(self) -> Mood:
        """Detect the mood/emotion of the audio based on spectral features."""
        spectral_mean = self.features.get('spectral_centroid_mean', 2000)
        rms_mean = self.features.get('rms_mean', 0.1)
        zcr_mean = np.mean(self.features.get('zcr', [0]))
        
        # Analyze the overall contour (beginning vs end)
        duration = self.features.get('duration', 60)
        
        # Get feature statistics
        rms_std = float(np.std(self.features['rms']))
        spectral_std = float(np.std(self.features['spectral_centroid']))
        
        # Energy indicators
        high_energy = rms_mean > 0.1
        low_energy = rms_mean < 0.05
        high_treble = spectral_mean > 3000
        low_treble = spectral_mean < 1500
        high_variation = rms_std > 0.05
        steady = rms_std < 0.02
        
        # Mood classification logic
        if high_energy and high_treble and high_variation:
            if spectral_std > 1000:
                return Mood.ANGRY
            return Mood.ENERGETIC
        elif high_energy and not high_variation:
            return Mood.HAPPY
        elif low_energy and low_treble and steady:
            return Mood.SAD
        elif low_energy and high_treble:
            return Mood.MYSTERIOUS
        elif high_energy and spectral_std > 800:
            return Mood.UPLIFTING
        elif low_energy and steady:
            return Mood.CALM
        elif low_energy and spectral_std > 500:
            return Mood.MELANCHOLIC
        else:
            # Default based on energy
            return Mood.ENERGETIC if high_energy else Mood.CALM
    
    def _detect_genre(self) -> Genre:
        """Detect genre based on audio characteristics."""
        bpm = self.features.get('bpm', 120)
        rms_mean = self.features.get('rms_mean', 0.1)
        spectral_mean = self.features.get('spectral_centroid_mean', 2000)
        
        # Get frequency band ratios
        low_sum = np.sum(self.features.get('low_energy', [0]))
        mid_sum = np.sum(self.features.get('mid_energy', [0]))
        high_sum = np.sum(self.features.get('high_energy', [0]))
        total = low_sum + mid_sum + high_sum + 1e-10
        
        bass_ratio = low_sum / total
        mid_ratio = mid_sum / total
        treble_ratio = high_sum / total
        
        # Zero crossing rate for texture
        zcr_mean = np.mean(self.features.get('zcr', [0]))
        
        # Genre detection heuristics
        # Electronic: high BPM, bass-heavy, steady rhythm
        if bpm > 120 and bass_ratio > 0.4 and zcr_mean < 0.1:
            return Genre.ELECTRONIC
        
        # Hip-hop: moderate BPM, bass-heavy, distinctive rhythm
        if bpm > 70 and bpm < 110 and bass_ratio > 0.35:
            return Genre.HIPHOP
        
        # Rock: mid-frequency focused, high energy
        if mid_ratio > 0.5 and rms_mean > 0.08:
            return Genre.ROCK
        
        # Classical: low BPM, balanced frequencies, low energy variation
        if bpm < 90 and bass_ratio < 0.3 and treble_ratio < 0.3:
            return Genre.CLASSICAL
        
        # Jazz: moderate BPM, complex texture
        if bpm > 80 and bpm < 150 and zcr_mean > 0.05:
            return Genre.JAZZ
        
        # Acoustic: low bass, natural sound
        if bass_ratio < 0.25 and zcr_mean > 0.03:
            return Genre.ACOUSTIC
        
        # Ambient: very low energy, high treble
        if rms_mean < 0.03 and treble_ratio > 0.25:
            return Genre.AMBIENT
        
        # Pop: default (balanced, moderate energy)
        return Genre.POP
    
    def _detect_sections(self) -> List[Dict[str, Any]]:
        """Detect song sections (verse, chorus, bridge, etc.) based on energy changes."""
        duration = self.features.get('duration', 60)
        
        # Get energy envelope (smoothed RMS)
        rms = self.features['rms']
        
        # Smooth the energy curve
        window_size = max(10, len(rms) // 20)
        smoothed = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
        
        # Find sections based on energy changes
        sections = []
        
        # Split into ~8 segments
        num_segments = 8
        segment_length = len(smoothed) // num_segments
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(smoothed)
            
            segment_energy = np.mean(smoothed[start_idx:end_idx])
            segment_std = np.std(smoothed[start_idx:end_idx])
            
            # Determine section type based on energy
            if segment_std > 0.03:
                section_type = "dynamic"  # Chorus or build-up
            elif segment_energy > np.mean(smoothed):
                section_type = "high"  # Verse
            else:
                section_type = "low"  # Bridge or outro
            
            sections.append({
                'type': section_type,
                'start_time': (start_idx / len(rms)) * duration,
                'end_time': (end_idx / len(rms)) * duration,
                'energy': float(segment_energy),
                'variation': float(segment_std)
            })
        
        # Refine section detection
        # Identify the highest energy section as "climax"
        max_energy_idx = np.argmax([s['energy'] for s in sections])
        sections[max_energy_idx]['type'] = 'climax'
        
        return sections
    
    def _calculate_energy_level(self) -> float:
        """Calculate overall energy level (0-1 scale)."""
        rms_mean = self.features.get('rms_mean', 0.1)
        # Normalize RMS to 0-1 range (assuming typical range 0.01-0.3)
        energy = min(1.0, max(0.0, (rms_mean - 0.01) / 0.29))
        return float(energy)
    
    def _calculate_dynamics(self) -> float:
        """Calculate dynamic range (variation in energy)."""
        rms_std = np.std(self.features.get('rms', [0]))
        # Normalize to 0-1
        dynamics = min(1.0, rms_std * 10)
        return float(dynamics)
    
    def _extract_frequency_bands(self):
        """Extract low, mid, and high frequency energy."""
        # Compute STFT
        stft = np.abs(librosa.stft(self.y))
        
        # Split into frequency bands (approximate)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Low frequencies (bass) - 20-250 Hz
        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000
        
        self.features['low_energy'] = np.mean(stft[low_mask, :], axis=0)
        self.features['mid_energy'] = np.mean(stft[mid_mask, :], axis=0)
        self.features['high_energy'] = np.mean(stft[high_mask, :], axis=0)
        
    def get_frame_features(self, frame_number: int, fps: int = 30) -> Dict[str, float]:
        """Get audio features for a specific video frame."""
        if not self.features:
            self.analyze()
            
        # Calculate which audio samples correspond to this frame
        samples_per_frame = self.sr / fps
        frame_start = int(frame_number * samples_per_frame)
        frame_end = int((frame_number + 1) * samples_per_frame)
        
        features = {}
        
        # Get features at this frame position
        frame_idx = min(frame_number, len(self.features['rms']) - 1)
        
        features['rms'] = float(self.features['rms'][frame_idx])
        features['spectral_centroid'] = float(self.features['spectral_centroid'][frame_idx])
        
        # Frequency bands
        if frame_idx < len(self.features['low_energy']):
            features['low_energy'] = float(self.features['low_energy'][frame_idx])
            features['mid_energy'] = float(self.features['mid_energy'][frame_idx])
            features['high_energy'] = float(self.features['high_energy'][frame_idx])
        else:
            features['low_energy'] = 0.0
            features['mid_energy'] = 0.0
            features['high_energy'] = 0.0
            
        # Check if this frame is near an onset (beat)
        features['is_beat'] = int(frame_number in self.features['onset_frames'])
        
        # Add mood and genre info
        features['mood'] = self.features.get('mood', 'calm')
        features['genre'] = self.features.get('genre', 'pop')
        
        # Add energy level and dynamics
        features['energy_level'] = self.features.get('energy_level', 0.5)
        features['dynamics'] = self.features.get('dynamics', 0.5)
        
        # Add section info
        if self._sections:
            frame_time = frame_number / fps
            for section in self._sections:
                if section['start_time'] <= frame_time < section['end_time']:
                    features['section'] = section['type']
                    features['section_energy'] = section['energy']
                    break
            else:
                features['section'] = 'verse'
                features['section_energy'] = 0.5
        else:
            features['section'] = 'verse'
            features['section_energy'] = 0.5
        
        return features
    
    def get_mood(self) -> Mood:
        """Get detected mood."""
        if self._mood is None:
            if not self.features:
                self.analyze()
        return self._mood
    
    def get_genre(self) -> Genre:
        """Get detected genre."""
        if self._genre is None:
            if not self.features:
                self.analyze()
        return self._genre
    
    def get_song_info(self) -> Dict[str, Any]:
        """Get comprehensive song information."""
        if not self.features:
            self.analyze()
        return {
            'mood': self._mood.value if self._mood else 'unknown',
            'genre': self._genre.value if self._genre else 'unknown',
            'bpm': self.features.get('bpm', 0),
            'duration': self.features.get('duration', 0),
            'energy_level': self.features.get('energy_level', 0.5),
            'dynamics': self.features.get('dynamics', 0.5),
            'sections': self._sections
        }


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Convenience function to analyze an audio file."""
    analyzer = AudioAnalyzer(audio_path)
    return analyzer.analyze()


if __name__ == "__main__":
    # Test the analyzer
    import sys
    if len(sys.argv) > 1:
        features = analyze_audio(sys.argv[1])
        print(f"BPM: {features['bpm']:.1f}")
        print(f"Duration: {features['duration']:.2f}s")
        print(f"Spectral Centroid: {features['spectral_centroid_mean']:.2f}")
        print(f"RMS Energy: {features['rms_mean']:.4f}")
        print(f"Mood: {features.get('mood', 'unknown')}")
        print(f"Genre: {features.get('genre', 'unknown')}")
        print(f"Energy Level: {features.get('energy_level', 0):.2f}")
        print(f"Dynamics: {features.get('dynamics', 0):.2f}")
    else:
        print("Usage: python audio_analyzer.py <audio_file>")

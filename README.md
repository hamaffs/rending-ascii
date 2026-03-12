# ASCII Video Generator

A web-based application that transforms audio files (MP3/WAV) into high-quality audio-reactive ASCII art videos.

## Features

- **Audio Analysis**: Extracts BPM, beats, and frequency features using Librosa
- **Three Visual Modes**:
  - Geometric: Pulsing shapes that react to the beat
  - Waveform: Scrolling sine waves driven by amplitude
  - Particles: ASCII characters that explode on drum hits
- **Three Visual Styles**:
  - Classic: White ASCII on black background
  - Matrix: Green "Matrix" style
  - High Contrast: Bold black and white
- **Audio-Reactive Effects**:
  - Bass frequencies control shaking/scaling
  - Treble frequencies control flicker
  - BPM sets the pulse rate
- **Modern Web Interface**: Drag & drop upload with real-time progress

## Project Structure

```
long v2/
├── main.py                 # FastAPI backend
├── audio_analyzer.py       # Audio analysis module (Librosa)
├── ascii_renderer.py       # ASCII rendering engine (OpenCV)
├── video_assembler.py      # Video assembly (MoviePy)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Frontend UI
├── uploads/               # Temporary audio uploads
└── outputs/               # Generated video files
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for MoviePy):
   - Windows: Download from ffmpeg.org and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

## Usage

1. Start the server:
```bash
python main.py
```

2. Open your browser to `http://localhost:8000`

3. Upload an audio file and select your preferred style and mode

4. Click "Generate Video" and wait for rendering to complete

5. Download the generated MP4 video

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the main page |
| `/health` | GET | Health check |
| `/api/render` | POST | Upload audio and start rendering |
| `/api/status/{job_id}` | GET | Get rendering job status |
| `/api/download/{filename}` | GET | Download generated video |
| `/api/styles` | GET | Get available styles and modes |

## Technical Details

- **Backend**: FastAPI with async job processing
- **Audio Analysis**: Librosa for feature extraction
- **Visual Rendering**: OpenCV for image manipulation
- **Video Creation**: MoviePy for frame assembly
- **Frontend**: Pure HTML/CSS/JavaScript

## Example Usage (Python)

```python
from video_assembler import create_video_from_audio

# Generate a video from audio
output_path = create_video_from_audio(
    audio_path="song.mp3",
    output_path="output.mp4",
    fps=30,
    style="matrix",
    mode="particles"
)
```

## License

MIT

import os
import time
from pathlib import Path

import cv2
import yt_dlp

from google import genai
from google.genai import types

import config

os.environ['CHROMA_TELEMETRY'] = "false"
import chromadb

def download_video(url:str)->Path:
    """Downloads a Youtube Video and returns a local file path"""
    config.VIDEO_DIR.mkdir(exist_ok = True)

    ydl_opts = {
        "format" : config.VIDEO_FORMAT,
        "outtmpl" : str(config.VIDEO_DIR / "%(id)s.%(ext)s"),
        "quiet" : True,
        "no_warnings" : True,
        "noplaylist" : True,          
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = config.VIDEO_DIR / f"{info['id']}.{info.get('ext','mp4')}"

    return video_path


def extract_frames(video_path: Path) -> list[dict]:
    """Extract one jpeg frame every FRAME_INTERVAL_SECONDS from the video"""

    config.FRAMES_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps =cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames/fps if fps>0 else 0

    print("Video : {duration.0f}s at {fps.1f} fps")

    frame_interval = max(1,int(fps * config.FRAME_INTERVAL_SECONDS))

    extracted = []

    frame_idx = 0

    while True:
        ret , frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx/fps
            frame_path = config.FRAMES_DIR/f"frame_{frame_idx:06d}.jpg"

            # Frame ko image file me save karta hai.
            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY,config.JPEG_QUALITY]
            )

            extracted.append({
                "frame_idx" : frame_idx,
                "timestamp" : timestamp,
                "path" : frame_path
            }
            )

        frame_idx += 1

    cap.release()

    print(f"Extracted : {len(extracted)} frames")  
    return extracted      

def describe_frame(client: genai.client, frame_path : Path) -> str:
    """Send a frame to Gemini Vision and get a detailed text discription """
    with open(frame_path, "rb") as f:
        img_bytes = f.read()

    prompt = (
        "Describe this video frame in detail. Include:" \
        "any text visible on the screen such as whiteboards, slides, terminals or code," \
        "any diagrams, charts, or visual elements" \
        "what is happening in the scene" \
        "and any tools or interfaces visible." \
        "Be specific and thorough. Write 3-5 sentences."
    )    

    response = client.models.generate_content(
        model = config.VISION_MODEL,
        contents = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(img_bytes, mime_type="image/jpeg"),
        ],
    )

    return response.text.strip()

def embed_text(client : genai.Client, text: str)-> list[float]:
    """Convert a text discription into a 3072 dimensional vector"""
    response = client.models.embed_content(
        model = config.EMBED_MODEL,
        contents = text,
        
    )
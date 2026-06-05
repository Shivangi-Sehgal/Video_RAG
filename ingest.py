import os
import time
from pathlib import Path

import cv2
import yt_dlp

import base64
from openai import OpenAI

# from google import genai
# from google.genai import types

import config

os.environ['CHROMA_TELEMETRY'] = "false"
import chromadb

def get_video_id(url: str) -> str:
    """Fetch the YouTube video ID without downloading the video."""
    ydl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["id"]


def download_video(url:str)->tuple[Path,str]:
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

    return video_path, info["id"]


def extract_frames(video_path: Path) -> list[dict]:
    """Extract one jpeg frame every FRAME_INTERVAL_SECONDS from the video"""

    config.FRAMES_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps =cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames/fps if fps>0 else 0

    print(f"Video : {duration:.0f}s at {fps:.1f} fps")

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
                "frame_index" : frame_idx,
                "timestamp" : timestamp,
                "path" : frame_path
            }
            )

        frame_idx += 1

    cap.release()

    print(f"Extracted : {len(extracted)} frames")  
    return extracted      

def describe_frame(client: OpenAI, frame_path : Path) -> str:
    """Send a frame to OpenAI Vision and get a detailed text discription """
    with open(frame_path, "rb") as f:
        # img_bytes = f.read()
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "Describe this video frame in detail. Include:" \
        "any text visible on the screen such as whiteboards, slides, terminals or code," \
        "any diagrams, charts, or visual elements" \
        "what is happening in the scene" \
        "and any tools or interfaces visible." \
        "Be specific and thorough. Write 3-5 sentences."
    )    

    # response = client.models.generate_content(
    #     model = config.VISION_MODEL,
    #     contents = [
    #         types.Part.from_text(text=prompt),
    #         types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    #     ],
    # )
    response = client.chat.completions.create(
        model = config.VISION_MODEL,
        messages = [{
            "role":"user",
            "content": [
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
    )

    # return response.text.strip()
    return response.choices[0].message.content.strip()

# def embed_text(client : genai.Client, text: str)-> list[float]:
#     """Convert a text discription into a 3072 dimensional vector"""
#     response = client.models.embed_content(
#         model = config.EMBED_MODEL,
#         contents = text,
#         config = types.EmbedContentConfig(
#             task_type="RETRIEVAL_DOCUMENT",
#             output_dimensionality= config.EMBED_DIM,
#         ),
#     )
#     return response.embeddings[0].values

def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=config.EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _format_time(second:float)->str:
    """Convert a float number of seconds and display in it MM:SS format"""
    minutes = int(second) // 60
    seconds = int(second) % 60
    return f"{minutes : 02d}:{seconds : 02d}"

def ingest(url:str) -> tuple[OpenAI, chromadb.Collection]:
    """
    Full ingestion pipeline: download, extract, describe, embed, store
    Returns (genai.client, chromadb.Collection) so cli.py can reuse the same
    Gemini client for quering instead of creating a second one
    """

    # client = genai.Client(api_key = config.GEMINI_API_KEY)
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Step 0: Detect already-indexed video (no download needed)
    print(" Checking if video is already indexed...")
    video_id = get_video_id(url)
    collection_name = f"{config.COLLECTION_PREFIX}_{video_id}"

    config.CHROMA_DIR.mkdir(exist_ok=True)
    db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = db.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get()
    if existing["ids"]:
        print(f" ✓ '{collection_name}' already has {len(existing['ids'])} frames. Skipping ingestion.")
        return client, collection

    print(f" New video — collection '{collection_name}' will be created.")

    # Step 1: Download
    print(" Downloading Video...")
    video_path, _ = download_video(url)   # video_id already known
    print(f" Saved to {video_path}")

    # Step 2: Extract Frames
    print(f"\n Extracting Frames (1 per {config.FRAME_INTERVAL_SECONDS}s)...")
    frames = extract_frames(video_path)

    # Step 3,4,5: Describe, Embed, Store (one frame at a time)
    print("\n Describing and Indexing frames...")

    for i, frame in enumerate(frames):
        timestamp_str = _format_time(frame["timestamp"])
        print(f" [{i+1}/{len(frames)}] t={timestamp_str} - describing...", end="", flush=True)

        description = describe_frame(client, frame["path"])
        vector = embed_text(client, description)

        collection.add(
            ids=[f"frame_{frame['frame_index']:06d}"],
            embeddings=[vector],
            documents=[description],
            metadatas=[{
                "timestamp": frame["timestamp"],
                "timestamp_str": timestamp_str,
                "frame_path": str(frame["path"]),
                "frame_index": frame["frame_index"],
            }],
        )

        print(" Done...")

        # Free tier limit : 15 RPM across all Gemini calls
        # Each frame costs 2 calls (describe + embed).
        # Sleeping 2s per frame keeps us well under the limit.

        if i < len(frames) - 1:
            time.sleep(2)

    print(f"\n Indexed {len(frames)} frames into ChromaDB")
    return client, collection    



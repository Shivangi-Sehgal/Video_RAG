import os
import time
from pathlib import Path

import cv2
import yt_dlp
import imagehash
from PIL import Image

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
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "quiet" : True,
        "no_warnings" : True,
        "noplaylist" : True,          
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = config.VIDEO_DIR / f"{info['id']}.{info.get('ext','mp4')}"

    return video_path, info["id"]


def are_frames_similar(img1, img2, threshold: int = None) -> bool:
    """Check if two frames are visually similar using average hash from imagehash library"""
    if threshold is None:
        threshold = config.FRAME_HASH_THRESHOLD
    if img1 is None or img2 is None:
        return False
    try:
        pil_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        pil_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        hash1 = imagehash.average_hash(pil_img1)
        hash2 = imagehash.average_hash(pil_img2)
        return (hash1 - hash2) <= threshold
    except Exception as e:
        print(f"Error comparing hashes: {e}")
        return False


def extract_frames(video_path: Path) -> list[dict]:
    """Extract one jpeg frame every FRAME_INTERVAL_SECONDS from the video, skipping duplicates"""

    config.FRAMES_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames/fps if fps>0 else 0

    print(f"Video : {duration:.0f}s at {fps:.1f} fps")

    frame_interval = max(1, int(fps * config.FRAME_INTERVAL_SECONDS))

    extracted = []
    frame_idx = 0
    last_unique_frame_data = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx/fps
            frame_path = config.FRAMES_DIR/f"frame_{frame_idx:06d}.jpg"

            # Check if this frame is similar to the last unique frame
            if last_unique_frame_data is not None and are_frames_similar(frame, last_unique_frame_data["frame"]):
                extracted.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "path": last_unique_frame_data["path"],
                    "is_duplicate": True,
                    "parent_frame_index": last_unique_frame_data["index"]
                })
            else:
                # Unique frame - save it
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
                )
                last_unique_frame_data = {
                    "frame": frame.copy(),
                    "index": frame_idx,
                    "path": frame_path
                }
                extracted.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "path": frame_path,
                    "is_duplicate": False
                })

        frame_idx += 1

    cap.release()

    duplicates_count = sum(1 for f in extracted if f.get("is_duplicate"))
    print(f"Extracted : {len(extracted)} frames ({duplicates_count} duplicates skipped from API/disk writing)")  
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


def parse_vtt(vtt_path: Path) -> list[dict]:
    """Parse WebVTT file and return list of segments: [{'start': float, 'end': float, 'text': str}]"""
    segments = []
    if not vtt_path.exists():
        return segments

    def time_to_seconds(t_str: str) -> float:
        parts = t_str.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return 0.0

    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_segment = None
    for line in lines:
        line = line.strip()
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start_str = parts[0].split()[0]
                end_str = parts[1].split()[0]
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    "start": time_to_seconds(start_str),
                    "end": time_to_seconds(end_str),
                    "text": ""
                }
        elif line and not line.startswith("WEBVTT") and not line.startswith("NOTE"):
            if current_segment:
                if current_segment["text"]:
                    current_segment["text"] += " " + line
                else:
                    current_segment["text"] = line

    if current_segment:
        segments.append(current_segment)

    cleaned_segments = []
    import re
    for seg in segments:
        text = seg["text"].strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            seg["text"] = text
            cleaned_segments.append(seg)
            
    return cleaned_segments


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file and return list of segments: [{'start': float, 'end': float, 'text': str}]"""
    segments = []
    if not srt_path.exists():
        return segments

    def time_to_seconds(t_str: str) -> float:
        t_str = t_str.replace(",", ".")
        parts = t_str.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    import re
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            time_line = lines[1]
            if "-->" in time_line:
                parts = time_line.split("-->")
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                text = " ".join(l.strip() for l in lines[2:])
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segments.append({
                        "start": time_to_seconds(start_str),
                        "end": time_to_seconds(end_str),
                        "text": text
                    })
    return segments


def chunk_transcript(segments: list[dict], chunk_duration: float = 60.0) -> list[dict]:
    """Group short transcript segments into larger chunks (e.g., 60 seconds) for better embedding context"""
    chunks = []
    if not segments:
        return chunks

    current_chunk = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text_parts": [segments[0]["text"]]
    }

    for seg in segments[1:]:
        if seg["end"] - current_chunk["start"] <= chunk_duration:
            current_chunk["end"] = seg["end"]
            current_chunk["text_parts"].append(seg["text"])
        else:
            chunks.append({
                "start": current_chunk["start"],
                "end": current_chunk["end"],
                "text": " ".join(current_chunk["text_parts"])
            })
            current_chunk = {
                "start": seg["start"],
                "end": seg["end"],
                "text_parts": [seg["text"]]
            }

    if current_chunk["text_parts"]:
        chunks.append({
            "start": current_chunk["start"],
            "end": current_chunk["end"],
            "text": " ".join(current_chunk["text_parts"])
        })

    return chunks

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
        print(f" [Indexed] '{collection_name}' already has {len(existing['ids'])} frames. Skipping ingestion.")
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
    unique_frame_cache = {}

    for i, frame in enumerate(frames):
        timestamp_str = _format_time(frame["timestamp"])
        is_dup = frame.get("is_duplicate", False)

        if is_dup:
            parent_idx = frame["parent_frame_index"]
            description, vector = unique_frame_cache[parent_idx]
            print(f" [{i+1}/{len(frames)}] t={timestamp_str} - duplicate of frame_{parent_idx:06d} (skipping API calls)...", end="", flush=True)
        else:
            print(f" [{i+1}/{len(frames)}] t={timestamp_str} - describing...", end="", flush=True)
            description = describe_frame(client, frame["path"])
            vector = embed_text(client, description)
            unique_frame_cache[frame["frame_index"]] = (description, vector)

        collection.add(
            ids=[f"frame_{frame['frame_index']:06d}"],
            embeddings=[vector],
            documents=[description],
            metadatas=[{
                "timestamp": frame["timestamp"],
                "timestamp_str": timestamp_str,
                "frame_path": str(frame["path"]),
                "frame_index": frame["frame_index"],
                "type": "frame",
                "is_duplicate": is_dup,
            }],
        )

        print(" Done...")

        if not is_dup and i < len(frames) - 1:
            time.sleep(2)

    print(f"\n Indexed {len(frames)} frames into ChromaDB")

    # Step 6: Process Subtitles
    print("\n Processing YouTube Subtitles...")
    sub_path = None
    for p in config.VIDEO_DIR.glob(f"{video_id}.en.*"):
        if p.suffix in (".vtt", ".srt"):
            sub_path = p
            break

    segments = []
    if sub_path:
        print(f" [Subtitles] Found subtitles file: {sub_path.name}")
        try:
            if sub_path.suffix == ".vtt":
                segments = parse_vtt(sub_path)
            else:
                segments = parse_srt(sub_path)
            
            if sub_path.exists():
                sub_path.unlink()
                print(f" Cleaned up subtitles file: {sub_path.name}")
        except Exception as e:
            print(f" Error parsing subtitles file: {e}")
    else:
        print(" English subtitles/captions not found on YouTube. Skipping transcript indexing.")

    if segments:
        print(f" Chunking transcript into {config.FRAME_INTERVAL_SECONDS}s segments...")
        chunks = chunk_transcript(segments, chunk_duration=float(config.FRAME_INTERVAL_SECONDS))
        print(f" Indexing {len(chunks)} transcript chunks into ChromaDB...")
        
        for i, chunk in enumerate(chunks):
            timestamp_str = _format_time(chunk["start"])
            print(f" [Transcript {i+1}/{len(chunks)}] t={timestamp_str} - embedding...", end="", flush=True)
            vector = embed_text(client, chunk["text"])
            
            collection.add(
                ids=[f"transcript_{i:06d}"],
                embeddings=[vector],
                documents=[chunk["text"]],
                metadatas=[{
                    "timestamp": chunk["start"],
                    "timestamp_str": timestamp_str,
                    "end_timestamp": chunk["end"],
                    "type": "transcript",
                }],
            )
            print(" Done...")

    return client, collection    



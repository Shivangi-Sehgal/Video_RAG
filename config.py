import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -API----------------------------------------------
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
# GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
# it is easier to debug the error with the help of os.environ
# as if the keys are not there it will cause an error directly

# -MODELS-------------------------------
# VISION_MODEL: str = "gemini-2.5-flash"
# EMBED_MODEL: str = "gemini-embedding-001"
VISION_MODEL: str = "gpt-4o"
EMBED_MODEL: str = "text-embedding-3-large"
EMBED_DIM: int = 3072

# -FRAME EXTRACTION-----------------------------------
FRAME_INTERVAL_SECONDS: int = 30
JPEG_QUALITY: int = 85
VIDEO_FORMAT: str = "bestvideo[height<=720][ext=mp4]/best[height<=720]"
FRAME_HASH_THRESHOLD: int = 4  # Hamming distance threshold for duplicate frames (0-64)

#-VECTOR STORE------------------------------------------------
# COLLECTION_NAME: str = "video_frames"
COLLECTION_PREFIX: str = "video"        # final name → video_{youtube_id}
TOP_K_RESULTS: int = 3

#-PATH---------------------------------------------------------
FRAMES_DIR: Path = Path("frames")
VIDEO_DIR: Path = Path("videos")
CHROMA_DIR: Path = Path("chroma_db")        # persistent vector store


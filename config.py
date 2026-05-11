import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -API----------------------------------------------
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
# it is easier to debug the error with the help of os.environ
# as if the keys are not there it will cause an error directly

# -MODELS-------------------------------
VISION_MODEL: str = "gemini-2.5-flash"
EMBED_MODEL: str = "gemini-embedding-001"
EMBED_DIM: int = 3072

# -FRAME EXTRACTION-----------------------------------
FRAME_INTERVAL_SECONDS: int = 30
JPEG_QUALITY: int = 85
VIDEO_FORMAT: str = "bestvideo[height<=720][ext=mp4]/best[height<=720]"

#-VECTOR STORE------------------------------------------------
COLLECTION_NAME: str = "video_frames"
TOP_K_RESULTS: int = 3

#-PATH---------------------------------------------------------
FRAMES_DIR: Path = Path("frames")
VIDEO_DIR: Path = Path("videos")

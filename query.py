import os

from openai import OpenAI

# from google import genai
# from google.genai import types

import base64
import config

os.environ['CHROMA_TELEMETRY'] = "false"
import chromadb

# def embed_query(client: genai.Client, question: str) -> list[float]:
#     """Convert a user query into a vector for semantic search"""
#     response = client.models.embed_content(
#         model = config.EMBED_MODEL,
#         contents = question,
#         config = types.EmbedContentConfig(
#             task_type="RETRIEVAL_QUERY",
#             output_dimensionality=config.EMBED_DIM,
#         ),
#     )
#     return response.embeddings[0].values

def embed_query(client: OpenAI, question: str) -> list[float]:
    response = client.embeddings.create(
        model=config.EMBED_MODEL,
        input=question,
    )
    return response.data[0].embedding


def retrieve_context(
       collection : chromadb.Collection,
       query_vector : list[float], 
)-> tuple[list[dict], list[dict]]:
    """Search ChromaDB for the frames and transcript chunks most semantically similar to the question"""

    # Query ChromaDB for twice the top-k results to ensure we retrieve a mix of frames and transcripts
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=config.TOP_K_RESULTS * 2,
        include=["documents", "metadatas", "distances"]
    )

    frames = []
    transcripts = []

    if not results or not results["documents"] or not results["documents"][0]:
        return frames, transcripts

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        item_type = meta.get("type", "frame")
        if item_type == "frame":
            frames.append({
                "description": doc,
                "timestamp_str": meta.get("timestamp_str", "00:00"),
                "frame_path": meta.get("frame_path", ""),
                "similarity": 1 - dist,
                "timestamp": meta.get("timestamp", 0.0),
            })
        else:
            transcripts.append({
                "text": doc,
                "timestamp_str": meta.get("timestamp_str", "00:00"),
                "similarity": 1 - dist,
                "timestamp": meta.get("timestamp", 0.0),
            })
    
    # Sort both chronologically
    frames.sort(key=lambda x: x["timestamp"])
    transcripts.sort(key=lambda x: x["timestamp"])
    
    # Return top K results for each modality
    return frames[:config.TOP_K_RESULTS], transcripts[:config.TOP_K_RESULTS]


def answer_question(client: OpenAI, question: str, frames: list[dict], transcripts: list[dict] = None) -> str:
    if transcripts is None:
        transcripts = []

    context_prompt = (
        f"You are answering a question about a video.\n"
        f"You have been given context retrieved from the video (both visual frames and audio/spoken transcripts) "
        f"that are most relevant to the question.\n"
        f"Use both the visual frame content and the spoken words to provide a complete and accurate answer. "
        f"If the visual frame is static, blank, or empty, rely more heavily on the audio transcripts.\n"
    )

    content = [{"type": "text", "text": context_prompt}]

    # 1. Add Audio Transcript context
    if transcripts:
        transcript_context = "\n### RELEVANT AUDIO TRANSCRIPT CHUNKS:\n"
        for t in transcripts:
            transcript_context += f"[{t['timestamp_str']}]: \"{t['text']}\"\n"
        content.append({"type": "text", "text": transcript_context})

    # 2. Add Visual Frame context
    if frames:
        content.append({"type": "text", "text": "\n### RELEVANT VISUAL FRAMES:\n"})
        for frame in frames:
            content.append({"type": "text", "text": f"\n[Frame at {frame['timestamp_str']} - Visual Description: {frame['description']}]"})
            try:
                with open(frame["frame_path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception as e:
                # Handle cases where the frame image file is missing
                pass

    content.append({"type": "text", "text": f"\nQuestion: {question}"})

    response = client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content.strip()


# This is called Multimodal RAG
# Because:
# retrieval text embeddings pe ho raha hai
# final answering actual images and audio transcripts dekh ke ho raha hai
# So:
# semantic retrieval + visual reasoning + audio context together.

def ask(
    client : OpenAI,
    collection : chromadb.Collection,
    question : str,
)-> str:
    """
    Single entry point for the CLI: embed question, retrieve context, answer.
    Hides the multimodal pipeline behind one clean call.
    """
    query_vector = embed_query(client, question)
    frames, transcripts = retrieve_context(collection, query_vector)

    if not frames and not transcripts:
        return "No relevant frames or audio transcripts found for that question."
    
    return answer_question(client, question, frames, transcripts)

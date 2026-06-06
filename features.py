import json
import chromadb
from openai import OpenAI
import config

def _get_timeline(collection: chromadb.Collection)->str:
    results = collection.get(include=["documents", "metadatas"])

    print("\n ==========RAW RESULTS FROM CHROMDB==============")
    print("Total records stored:", len(results["documents"]))
    
    if not results["documents"]:
        return ""

    print("First Document:", results["documents"][0])
    print("First metadata:", results["metadatas"][0])

    items = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1].get("timestamp", 0.0),
    )

    print("\n =============AFTER SORTING BY TIMESTAMP=================")
    for doc, meta in items:
        item_type = meta.get("type", "frame")
        prefix = "Visual" if item_type == "frame" else "Spoken"
        print(f"[{meta['timestamp_str']}] ({prefix}): {doc[:60]}...")

    timeline_lines = []
    for doc, meta in items:
        item_type = meta.get("type", "frame")
        prefix = "Visual" if item_type == "frame" else "Spoken"
        timeline_lines.append(f"[{meta['timestamp_str']}] ({prefix}): {doc}")

    timeline = "\n".join(timeline_lines)

    print("\n--- FINAL TIMELINE STRING (sent to GPT) ---")
    print(timeline[:300], "...")

    return timeline

def summarize_video(client: OpenAI, collection: chromadb.Collection)->str:
    print("\n---BUILDING TIMELINE FOR SUMMARY---")
    timeline = _get_timeline(collection)

    print("\n---SENDING TO GPT---")
    response = client.chat.completions.create(
        model=config.VISION_MODEL,
        timeout=120,
        messages=[{
            "role":"user",
            "content": (
                "You are analyzing a video. Below are descriptions of frames at regular intervals:\n\n"
                f"{timeline}\n\n"
                "Write a comprehensive 3-5 paragraph summary covering: "
                "what the video is about, main topics discussed, key moments, and the overall takeaway."
            ),
        }],
    )
    result = response.choices[0].message.content.strip()
    print("\n--- GPT RESPONSE ---")
    print(result)

    return result

def get_key_points(client: OpenAI, collection: chromadb.Collection) -> str:
    print("\n--- BUILDING TIMELINE FOR KEY POINTS ---")
    timeline = _get_timeline(collection)

    print("\n--- SENDING TO GPT ---")
    response = client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are analyzing a video. Below are descriptions of frames at regular intervals:\n\n"
                f"{timeline}\n\n"
                "Extract 5-8 key points or takeaways from this video. "
                "Format each as a bullet point starting with a dash."
            ),
        }],
    )

    result = response.choices[0].message.content.strip()

    print("\n--- GPT RESPONSE ---")
    print(result)

    return result

def generate_flashcards(client: OpenAI, collection: chromadb.Collection) -> list[dict]:
    print("\n--- BUILDING TIMELINE FOR FLASHCARDS ---")
    timeline = _get_timeline(collection)

    print("\n--- SENDING TO GPT (JSON MODE) ---")
    response = client.chat.completions.create(
        model=config.VISION_MODEL,
        timeout=120,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": (
                "You are analyzing a video. Below are descriptions of frames at regular intervals:\n\n"
                f"{timeline}\n\n"
                "Generate 5 flashcard Q&A pairs to test understanding of this video.\n"
                'Return JSON in this exact format: {"flashcards": [{"question": "...", "answer": "..."}]}'
            ),
        }],
    )

    raw = response.choices[0].message.content
    print("\n--- RAW JSON FROM GPT ---")
    print(raw)

    data = json.loads(raw)
    print("\n--- PARSED FLASHCARDS ---")
    for i, card in enumerate(data["flashcards"], 1):
        print(f"  Q{i}: {card['question']}")
        print(f"  A{i}: {card['answer']}\n")

    return data["flashcards"]


def run_flashcard_test(flashcards: list[dict]) -> None:
    if not flashcards:
        print("No flashcards generated.")
        return

    print(f"\n{'='*50}")
    print(f"  FLASHCARD TEST  ({len(flashcards)} questions)")
    print(f"{'='*50}")
    print("  Press Enter to reveal each answer, then mark yourself.\n")

    correct = 0

    for i, card in enumerate(flashcards, 1):
        print(f"  Card {i}/{len(flashcards)}")
        print(f"  Q: {card['question']}")
        input("\n  [Press Enter to reveal answer]")
        print(f"  A: {card['answer']}\n")

        while True:
            mark = input("  Got it right? (y/n): ").strip().lower()
            if mark in ("y", "n"):
                break

        if mark == "y":
            correct += 1
        print()

    pct = correct / len(flashcards) * 100
    print(f"{'='*50}")
    print(f"  SCORE: {correct}/{len(flashcards)}  ({pct:.0f}%)")
    if pct == 100:
        print("  Perfect!")
    elif pct >= 70:
        print("  Good job!")
    else:
        print("  Keep reviewing!")
    print(f"{'='*50}\n")


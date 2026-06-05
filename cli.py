import argparse #CLI arguments handle karta hai.mtlb terminal se input lena
import os
import sys

import config
from ingest import ingest
from query import ask

os.environ["CHROMA_TELEMETRY"] = "false"

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Video RAG - ask questions about a YouTube Video"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Youtube Video URl to analyse",
    )
    args = parser.parse_args()

    print("\n Video RAG - AI That Sees YouTube\n")

    # ingest() returns both the OpenAI Client and the ChromaDB Collection.
    # We unpack both so we can reuse the same client for all subsequent questions.

    print("Step 1/2 - Ingesting Video (this runs once per video)\n")

    client, collection = ingest(args.url)

    print("\nStep 2/2 - Ready to Answer Questions\n")
    print("-" * 50)
    print("Type your question and press Enter.")
    print("Type 'quit or press CTRL+C to Exit.")
    print("-" * 50)

    while True:
        try:
            question = input("\nYOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n BYE!...")
            sys.exit(0)
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("BYE!...")
            sys.exit(0)
        
        print("\n AI: ", end="", flush=True)
        answer = ask(client, collection, question)
        print(answer)

if __name__=="__main__":
    main()
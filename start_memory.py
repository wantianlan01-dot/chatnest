"""Memory search service - vectorize on first run, then start API."""
import os, sys, json
from pathlib import Path

os.environ.setdefault("AGENT_APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Ensure memory directory exists
mem_dir = Path(os.environ.get("MEMORY_CHROMA_DIR",
    os.path.join(os.environ["AGENT_APP_ROOT"], "memory", "chroma_db")))
state_file = Path(os.environ.get("MEMORY_STATE_FILE",
    os.path.join(os.environ["AGENT_APP_ROOT"], "memory", "chroma_state.json")))

state_file.parent.mkdir(parents=True, exist_ok=True)

need_vectorize = True
if state_file.exists():
    try:
        with open(state_file) as f:
            data = json.load(f)
        if data.get("version") and data.get("updated_at"):
            need_vectorize = False
            print("Memory index exists, skipping vectorization")
    except:
        pass

if need_vectorize:
    print("Vectorizing memory files (CLAUDE.md)...")
    from memory_search_service.vectorize import main as vectorize
    vectorize()
    print("Vectorization complete!")

# Start the server
from memory_search_service.server import main
main()

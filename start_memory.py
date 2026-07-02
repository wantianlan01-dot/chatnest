"""Memory search service - start server first, vectorize in background."""
import os, sys, json, threading, time, logging
from pathlib import Path

os.environ.setdefault("AGENT_APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Ensure memory directories exist
mem_dir = Path(os.environ.get("MEMORY_CHROMA_DIR",
    os.path.join(os.environ["AGENT_APP_ROOT"], "memory", "chroma_db")))
state_file = Path(os.environ.get("MEMORY_STATE_FILE",
    os.path.join(os.environ["AGENT_APP_ROOT"], "memory", "chroma_state.json")))
state_file.parent.mkdir(parents=True, exist_ok=True)

# Start vectorization in background
def do_vectorize():
    time.sleep(2)  # Give server a moment to start
    need = True
    if state_file.exists():
        try:
            with open(state_file) as f:
                data = json.load(f)
            if data.get("version") and data.get("updated_at"):
                need = False
                print("Memory index exists, skipping vectorization")
        except:
            pass
    if need:
        print("Vectorizing memory files (CLAUDE.md)...")
        from memory_search_service.vectorize import main as vec
        vec()
        print("Vectorization complete!")

t = threading.Thread(target=do_vectorize, daemon=True)
t.start()

# Start server immediately (no waiting for vectorization)
print("Starting memory search server on port 3900...")
from memory_search_service.server import main
main()

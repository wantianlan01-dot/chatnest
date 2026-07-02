"""Memory search service entrypoint - vectorize on first run, then start API."""
import os, sys, json
from pathlib import Path

os.environ.setdefault("AGENT_APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Check if index already exists
state_file = Path(os.environ.get("MEMORY_STATE_FILE", 
    os.path.join(os.environ["AGENT_APP_ROOT"], "memory", "chroma_state.json")))
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
    print("Vectorizing memory files...")
    from memory_search_service.vectorize import run_vectorize
    run_vectorize()
    print("Vectorization complete!")

# Start the server
from memory_search_service.server import main
main()

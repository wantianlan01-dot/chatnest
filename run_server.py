import os, sys

os.environ["AGENT_APP_ROOT"] = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.environ["AGENT_APP_ROOT"])

import uvicorn
import app.main

uvicorn.run(app.main.app, host="0.0.0.0", port=8787, log_level="debug")

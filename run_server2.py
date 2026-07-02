import os, sys

os.environ["AGENT_APP_ROOT"] = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.environ["AGENT_APP_ROOT"])

# Import to check specific endpoints
import dotenv
dotenv.load_dotenv(os.path.join(os.environ["AGENT_APP_ROOT"], ".env"))

from app.splash import current_period, random_line, read_pool
print("Splash pool:", read_pool())
print("Period:", current_period())
print("Line:", random_line(current_period()))

from app.claude import available_models
print("Models:", available_models())

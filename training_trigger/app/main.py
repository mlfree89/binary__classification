import os
import subprocess
import logging
from fastapi import FastAPI, HTTPException
import threading
app = FastAPI()
logger = logging.getLogger(__name__)

PROJECT_DIR = os.environ["COMPOSE_PROJECT_DIR"]




@app.post("/train")
def trigger_training():
    def run():
        try:
            subprocess.run(
                ["docker", "compose", "run", "--rm", "trainer"],
                cwd=PROJECT_DIR,
                check=True,
                timeout=600,
            )
        except (subprocess.SubprocessError, OSError):
            logger.exception("Failed to start training container.")

    threading.Thread(target=run, daemon=True).start()
    return {"status": "training_started"}

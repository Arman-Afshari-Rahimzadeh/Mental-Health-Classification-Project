import logging
import os
from datetime import datetime

# Create timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


# Create 'logs' folder if it doesn't exist
logs_path = os.path.join(os.getcwd(), 'logs')
os.markedirs(logs_path, exist_ok=True)

# Full path to log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure Logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)
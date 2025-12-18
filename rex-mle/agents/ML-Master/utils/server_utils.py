import logging
import os
import random
import time
from pathlib import Path

import requests
logger = logging.getLogger("ml-master")


def _load_server_urls():
    urls = []
    custom_url = os.getenv("ML_MASTER_VALIDATE_URL")
    if custom_url:
        urls.append(custom_url.rstrip("/"))

    workspace_dir = Path(os.getenv("WORKSPACE_DIR", "."))
    grading_port_file = workspace_dir / ".grading_port"
    if grading_port_file.exists():
        try:
            port = grading_port_file.read_text().strip()
            if port:
                urls.append(f"http://127.0.0.1:{port}")
        except OSError:
            logger.warning("Unable to read grading port file at %s", grading_port_file)

    # Fallback to default
    urls.append("http://127.0.0.1:5000")
    return urls


def is_server_online(max_retries=None, timeout=60):
    server_url_list = _load_server_urls()
    if not server_url_list:
        return False, ""
    if max_retries is None:
        max_retries = len(server_url_list)
    retry = 0
    index = random.randrange(len(server_url_list))
    server_url = server_url_list[index]
    while retry < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is online, status code: {response.status_code}")
                return True, server_url
            else:
                logger.warning(f"Server returned non-200 status code: {response.status_code}")
                
        except requests.exceptions.Timeout:
            timeout += 20
            logger.error(f"Connection to {server_url} timed out.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            index += 1
            index = index%(len(server_url_list))
            server_url = server_url_list[index]
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
    logger.error(f"Server is not online after {max_retries} retries.")  
    return False, ""

def call_validate(exp_id, submission_path, timeout=60, max_retries=3):
    online, server_url = is_server_online()
    retry=0
    while retry < max_retries:
        try:
            if online:
                files = {"file": open(submission_path, "rb")}
                response = requests.post(f"{server_url}/validate", files = files,headers = {"exp-id": exp_id}, timeout=timeout)
                print(response)
                response_json = response.json()
                if "error" in response_json:
                    logger.error(f"Server returned error: {response.text}")
                    return False, response_json['details']
                else:
                    return True, response_json
            else:
                return False, f"Server at {server_url} is not online"
        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
        else:
            return False, ""

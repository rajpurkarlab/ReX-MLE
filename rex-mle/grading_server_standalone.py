#!/usr/bin/env python
"""
Standalone grading server that runs without Docker.
This replaces the containerized grading server for local development.
"""
import argparse
import logging
import multiprocessing
import os
import signal
import socket
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request

from rexmle.grade import validate_submission
from rexmle.registry import registry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GradingServer:
    """Standalone grading server that can be run in a separate process."""

    def __init__(self, competition_id: str, private_data_dir: Path, port: int = 5000):
        self.competition_id = competition_id
        self.private_data_dir = private_data_dir
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/validate", methods=["POST"])
        def validate():
            import shutil
            import zipfile

            submission_file = request.files.get("file")
            if not submission_file:
                return jsonify({"error": "No file provided"}), 400

            # Create a unique temporary directory for this validation
            temp_dir = Path(f"/tmp/submission_validate_{os.getpid()}_{int(time.time() * 1000)}")
            temp_dir.mkdir(parents=True, exist_ok=True)

            submission_path = temp_dir / "submission.csv"
            submission_file.save(submission_path)

            # Handle predictions folder if provided (as a zip file)
            predictions_zip = request.files.get("predictions_zip")
            if predictions_zip:
                predictions_zip_path = temp_dir / "predictions.zip"
                predictions_zip.save(predictions_zip_path)

                try:
                    with zipfile.ZipFile(predictions_zip_path, 'r') as zip_ref:
                        # Extract to temp_dir (will create predictions/ folder)
                        zip_ref.extractall(temp_dir)
                    logger.info(f"Extracted predictions folder to {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to extract predictions zip: {e}")
                finally:
                    if predictions_zip_path.exists():
                        predictions_zip_path.unlink()

            try:
                is_valid, result = self.run_validation(submission_path)
            except Exception as e:
                logger.error(f"Validation error: {e}", exc_info=True)
                return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            return jsonify({"is_valid": is_valid, "result": result})

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "running"}), 200

    def run_validation(self, submission: Path) -> tuple[bool, str]:
        """Run validation on a submission."""
        # private_data_dir is: {data_dir}/{challenge_id}/prepared/private
        # We need to go up 3 levels to get to the root data_dir
        root_data_dir = self.private_data_dir.parent.parent.parent
        new_registry = registry.set_data_dir(root_data_dir)
        challenge = new_registry.get_challenge(self.competition_id)
        return validate_submission(submission, challenge)

    def run(self):
        """Start the Flask server."""
        logger.info(f"Starting grading server for competition: {self.competition_id}")
        logger.info(f"Server running on http://localhost:{self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)


def find_free_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


def run_server_process(competition_id: str, private_data_dir: Path, port: int = 5000):
    """Function to run in a separate process."""
    server = GradingServer(competition_id, private_data_dir, port)
    server.run()


class GradingServerManager:
    """Manager for starting/stopping the grading server in a subprocess."""

    def __init__(self, competition_id: str, private_data_dir: Path, port: int = None):
        self.competition_id = competition_id
        self.private_data_dir = private_data_dir
        self.requested_port = port
        self.port = None
        self.process = None

    def start(self):
        """Start the grading server in a separate process."""
        if self.process is not None:
            logger.warning("Grading server is already running")
            return

        attempts = 1 if self.requested_port is not None else 5
        for attempt in range(1, attempts + 1):
            candidate_port = self.requested_port if self.requested_port is not None else find_free_port()
            logger.info(
                f"Starting grading server process for {self.competition_id} on port {candidate_port} "
                f"(attempt {attempt}/{attempts})..."
            )
            process = multiprocessing.Process(
                target=run_server_process,
                args=(self.competition_id, self.private_data_dir, candidate_port)
            )
            process.start()

            time.sleep(2)

            if process.is_alive():
                self.process = process
                self.port = candidate_port
                logger.info(f"Grading server started on port {self.port} (PID: {self.process.pid})")
                return

            logger.warning(f"Grading server failed to start on port {candidate_port}.")
            process.join(timeout=1)

            if self.requested_port is not None:
                break

        raise RuntimeError("Grading server failed to start")

    def stop(self):
        """Stop the grading server process."""
        if self.process is None:
            return

        logger.info(f"Stopping grading server (PID: {self.process.pid})...")

        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)

            if self.process.is_alive():
                logger.warning("Grading server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.join()

        logger.info("Grading server stopped")
        self.process = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone grading server")
    parser.add_argument("--competition-id", required=True, help="Competition ID")
    parser.add_argument("--private-data-dir", required=True, type=Path, help="Path to private data directory")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    args = parser.parse_args()

    server = GradingServer(args.competition_id, args.private_data_dir, args.port)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Shutting down grading server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    server.run()

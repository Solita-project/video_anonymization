# Prepare transcript files for the Streamlit app.
# This script is started by app.py as a background process.
# It extracts audio, creates a transcript and runs automatic transcript anonymization.
# Script output is shown in the terminal where Streamlit is running.

from pathlib import Path
import subprocess
import sys


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define output log file
LOG_FILE = ROOT_DIR / "data" / "output" / "transcript_background.log"

# Define scripts to run
SCRIPTS = [
    "src/audio.py",
    "run/run_transcript.py",
    "run/run_anonymize.py",
]


def write_log(message):
    # Write one message to the log file and terminal
    print(message, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(message + "\n")
        log.flush()


def main():
    # Make sure output folder exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Clear old log file
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("Transcript preparation started.\n")

    try:
        for script in SCRIPTS:
            write_log(f"Running: {script}")

            # Do not hide stdout/stderr, this allows script output to appear in the terminal
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / script),
                ],
                cwd=ROOT_DIR,
                check=True,
            )

            write_log(f"Completed: {script}")

        write_log("Transcript preparation completed.")

    except Exception as error:
        write_log(f"Transcript preparation failed: {error}")
        raise


if __name__ == "__main__":
    main()
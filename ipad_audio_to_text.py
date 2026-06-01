#!/usr/bin/env python3
"""
Transcribe audio files (from iPad or any source) to text using OpenAI Whisper API.

Requirements:
    pip install openai

Usage:
    # Transcribe a single file
    python ipad_audio_to_text.py recording.m4a

    # Transcribe all audio files in a folder
    python ipad_audio_to_text.py /path/to/audio_folder/

    # Save transcripts to a custom output folder
    python ipad_audio_to_text.py /path/to/audio_folder/ --output /path/to/transcripts/

Supported formats: m4a, mp3, mp4, wav, webm, ogg, flac
"""

import argparse
import os
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".wav", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE_MB = 25  # Whisper API limit


def transcribe_file(client, audio_path: Path, language: str = None) -> str:
    """Send one audio file to the Whisper API and return the transcript."""
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"{audio_path.name} is {size_mb:.1f} MB — Whisper API limit is {MAX_FILE_SIZE_MB} MB. "
            "Split the file first (e.g. with ffmpeg: ffmpeg -i input.m4a -f segment -segment_time 600 part_%03d.m4a)"
        )

    with open(audio_path, "rb") as f:
        kwargs = {"model": "whisper-1", "file": f, "response_format": "text"}
        if language:
            kwargs["language"] = language
        transcript = client.audio.transcriptions.create(**kwargs)

    return transcript


def process_path(client, input_path: Path, output_dir: Path, language: str = None):
    """Transcribe a single file or all audio files in a directory."""
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No audio files found in {input_path}")
            return
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in files:
        out_file = output_dir / (audio_file.stem + ".txt")
        if out_file.exists():
            print(f"  Skipping {audio_file.name} (transcript already exists)")
            continue

        print(f"  Transcribing {audio_file.name} ...", end=" ", flush=True)
        try:
            text = transcribe_file(client, audio_file, language)
            out_file.write_text(text, encoding="utf-8")
            word_count = len(text.split())
            print(f"done ({word_count} words) -> {out_file.name}")
        except Exception as exc:
            print(f"FAILED: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe iPad audio files to text using OpenAI Whisper."
    )
    parser.add_argument("input", help="Audio file or folder containing audio files")
    parser.add_argument(
        "--output", "-o",
        help="Output folder for transcripts (default: 'transcripts/' next to input)",
    )
    parser.add_argument(
        "--language", "-l",
        help="Language code to improve accuracy, e.g. 'en', 'es', 'fr' (auto-detected if omitted)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Get your key at https://platform.openai.com/api-keys and run:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run:  pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    input_path = Path(args.input).expanduser().resolve()

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path / "transcripts"
    else:
        output_dir = input_path.parent / "transcripts"

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if args.language:
        print(f"Language: {args.language}")
    print()

    process_path(client, input_path, output_dir, args.language)
    print("\nDone.")


if __name__ == "__main__":
    main()

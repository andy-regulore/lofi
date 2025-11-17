#!/usr/bin/env python3
"""
Basic Workflow Example

Demonstrates complete end-to-end workflow for a single track.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import WorkflowOrchestrator


def main():
    print("=" * 60)
    print("Basic Workflow Example")
    print("=" * 60)
    print()

    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator()

    # Generate single track with all automation
    package = orchestrator.single_track_workflow(
        mood='chill',
        duration=180
    )

    if package:
        print("\n✅ Success!")
        print(f"Audio: {package['audio_path']}")
        print(f"Video: {package['video_path']}")
        print(f"Thumbnail: {package['thumbnail_path']}")
        print(f"Title: {package['metadata']['title']}")
    else:
        print("\n❌ Workflow failed")


if __name__ == '__main__':
    main()

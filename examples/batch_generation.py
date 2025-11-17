#!/usr/bin/env python3
"""
Batch Generation Example

Generate multiple tracks in one go.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import WorkflowOrchestrator


def main():
    print("=" * 60)
    print("Batch Generation Example")
    print("=" * 60)
    print()

    # Initialize
    orchestrator = WorkflowOrchestrator()

    # Generate 5 chill tracks
    print("Generating 5 chill tracks...\n")

    packages = orchestrator.batch_workflow(
        count=5,
        mood='chill'
    )

    print("\n" + "=" * 60)
    print(f"✅ Generated {len(packages)} tracks")
    print("=" * 60)

    for i, pkg in enumerate(packages, 1):
        print(f"\n{i}. {pkg['metadata']['title']}")
        print(f"   Audio: {pkg['audio_path']}")
        print(f"   Video: {pkg['video_path']}")


if __name__ == '__main__':
    main()

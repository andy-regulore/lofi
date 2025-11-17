#!/usr/bin/env python3
"""
API Usage Example

Demonstrates how to interact with the API server programmatically.
"""

import requests
import time
import json


API_URL = "http://localhost:8000"


def check_server():
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False


def generate_track(mood='chill', duration=180):
    """Generate a music track via API."""
    print(f"Generating {mood} track ({duration}s)...")

    response = requests.post(
        f"{API_URL}/api/generate",
        json={
            'mood': mood,
            'duration': duration,
            'count': 1
        }
    )

    data = response.json()
    job_id = data['job_id']

    print(f"Job ID: {job_id}")
    print("Waiting for completion...")

    # Poll for completion
    while True:
        job_response = requests.get(f"{API_URL}/api/jobs/{job_id}")
        job = job_response.json()

        status = job['status']
        progress = job['progress']

        print(f"Status: {status} ({progress}%)")

        if status == 'completed':
            print("✅ Generation complete!")
            return job['result']
        elif status == 'failed':
            print(f"❌ Generation failed: {job.get('error')}")
            return None

        time.sleep(2)


def get_analytics():
    """Get system analytics."""
    response = requests.get(f"{API_URL}/api/analytics")
    return response.json()


def list_tracks():
    """List all generated tracks."""
    response = requests.get(f"{API_URL}/api/tracks")
    return response.json()


def main():
    print("=" * 60)
    print("API Usage Example")
    print("=" * 60)
    print()

    # Check server
    if not check_server():
        print("❌ API server not running")
        print("Start it with: python api_server.py")
        return

    print("✅ API server is running\n")

    # Generate track
    result = generate_track(mood='focus', duration=180)

    if result:
        print("\nGenerated tracks:")
        for track in result:
            print(f"  - {track['title']}")

    # Get analytics
    print("\n" + "-" * 60)
    print("System Analytics:")
    analytics = get_analytics()
    print(f"Total tracks: {analytics['summary']['total_tracks']}")
    print(f"Total videos: {analytics['summary']['total_videos']}")

    # List tracks
    print("\n" + "-" * 60)
    print("Recent Tracks:")
    tracks_data = list_tracks()
    for track in tracks_data['tracks'][:5]:
        print(f"  - {track['title']} ({track['mood']})")


if __name__ == '__main__':
    main()

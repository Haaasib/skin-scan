"""CLI tool for batch skin scanning."""
import argparse
import json
import sys
from pathlib import Path
import cv2

from ..pipeline.compose import run_scan
from ..app.utils_io import resize_max


def main():
    parser = argparse.ArgumentParser(description="Scan facial image for skin analysis")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--out", "-o", default=".", help="Output directory")
    parser.add_argument("--save-overlays", action="store_true", help="Save overlay images")
    args = parser.parse_args()

    # Load image
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading image: {input_path}")
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Error: Could not read image: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Resize if needed
    img = resize_max(img, 2048)

    print("Running skin scan analysis...")
    try:
        result = run_scan(img)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save scores to JSON
    scores_file = out_dir / f"{input_path.stem}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(
            {
                "scores": result["scores"],
                "regions": result["regions"],
                "issues": result.get("issues", []),
                "concern_tags": result.get("concern_tags", []),
                "detections": result.get("detections", []),
                "ml": result.get("ml", {}),
            },
            f,
            indent=2,
        )
    print(f"Saved scores to: {scores_file}")

    print("\nIssues:")
    for issue in result.get("issues", []):
        print(f"  [{issue['severity']}] {issue['label']}: {issue['summary']}")

    print("\nConcern tags:", ", ".join(result.get("concern_tags", [])) or "(none)")

    print("\nScores:")
    for category, score in result["scores"].items():
        print(f"  {category:16s}: {score:.2f}")

    # Optionally save overlay images
    if args.save_overlays:
        print("\nSaving overlay images...")
        for category, overlay_data in result["overlays"].items():
            # Extract base64 data
            if overlay_data.startswith("data:image/png;base64,"):
                import base64
                b64_data = overlay_data.split(",")[1]
                img_bytes = base64.b64decode(b64_data)

                # Save to file
                overlay_file = out_dir / f"{input_path.stem}_{category}_overlay.png"
                with open(overlay_file, "wb") as f:
                    f.write(img_bytes)
                print(f"  Saved {category} overlay to: {overlay_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()

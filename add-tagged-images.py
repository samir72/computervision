from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os, json, time, math

BATCH_LIMIT = 64
MAX_RETRIES = 5

def main():
    global training_client, custom_vision_project
    try:
        # Clear console (best-effort)
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            pass

        # ---- Load & validate env ----
        load_dotenv()
        training_endpoint = os.getenv('ObjectTrainEndpoint')
        training_key = os.getenv('ObjectTraincredential')
        project_id = os.getenv('Objectprojectid')

        missing_env = [k for k, v in {
            'ObjectTrainEndpoint': training_endpoint,
            'ObjectTraincredential': training_key,
            'Objectprojectid': project_id
        }.items() if not v]
        if missing_env:
            raise RuntimeError(f"Missing required env var(s): {', '.join(missing_env)}")

        # ---- Auth + project ----
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)
        custom_vision_project = training_client.get_project(project_id)

        # ---- Upload ----
        json_path='tagged-images.json'
        folder = 'obj-training-images-code/'
        create_dir(folder)
        Upload_Images(folder, json_path)
    except Exception as e:
        print(f"Error: {e}")


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print(f"An error occurred: {e}")

def _within_01(x, eps=1e-7):
    return -eps <= float(x) <= 1.0 + eps

def _clamp01(x):
    return max(0.0, min(1.0, float(x)))

def _retryable_upload(batch):
    """Upload one batch with retries on throttling (HTTP 429) or transient failures."""
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return training_client.create_images_from_files(
                custom_vision_project.id,
                ImageFileCreateBatch(images=batch)
            )
        except Exception as e:
            msg = str(e).lower()
            is_throttle = "429" in msg or "too many requests" in msg or "throttle" in msg
            is_transient = any(s in msg for s in ["timeout", "temporar", "connection", "reset"])
            if attempt < MAX_RETRIES and (is_throttle or is_transient):
                print(f"  Transient error (attempt {attempt}/{MAX_RETRIES}). Sleeping {delay:.1f}s…")
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            raise  # rethrow if not retryable or out of attempts


def Upload_Images(folder, json_path):
    try:
        print("Uploading images...")

        # ---- Tags in project -> map ----
        proj_tags = training_client.get_tags(custom_vision_project.id)
        tag_map = {t.name: t.id for t in proj_tags}
        if not tag_map:
            raise RuntimeError("No tags found in the project. Create tags first (e.g., apple, banana, orange).")

        # ---- Read JSON ----
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")

        with open(json_path, 'r') as jf:
            meta = json.load(jf)

        files = meta.get('files') or []
        if not files:
            raise ValueError(f"No 'files' array in {json_path}")

        # ---- Build entries with validation ----
        entries = []
        missing_files = []
        missing_tags = {}
        bad_boxes = []
        total_regions = 0

        for item in files:
            fname = item.get('filename')
            if not fname:
                continue

            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                missing_files.append(fname)
                continue

            regions = []
            for r in item.get('tags', []):
                total_regions += 1
                tname = r.get('tag')
                if tname not in tag_map:
                    missing_tags[tname] = missing_tags.get(tname, 0) + 1
                    continue

                left = r.get('left'); top = r.get('top'); width = r.get('width'); height = r.get('height')
                # Expect normalized; validate and clamp
                if not all(_within_01(v) for v in (left, top, width, height)):
                    bad_boxes.append((fname, r))
                    continue

                left = _clamp01(left); top = _clamp01(top)
                width = _clamp01(width); height = _clamp01(height)

                # Keep inside [0,1] bounds
                width = min(width, 1.0 - left)
                height = min(height, 1.0 - top)

                if width <= 0 or height <= 0:
                    bad_boxes.append((fname, r))
                    continue

                regions.append(Region(tag_id=tag_map[tname], left=left, top=top, width=width, height=height))

            if not regions:
                # No valid regions for this image; skip
                continue

            with open(fpath, "rb") as f:
                entries.append(ImageFileCreateEntry(name=fname, contents=f.read(), regions=regions))

        if not entries:
            print("Nothing to upload after validation.")
            if missing_files:
                print(f"- Missing files ({len(missing_files)}): " + ", ".join(missing_files[:10]) + (" ..." if len(missing_files) > 10 else ""))
            if missing_tags:
                print("- Missing tag names (create in project or fix JSON):")
                for t, c in missing_tags.items():
                    print(f"  • {t}: {c} occurrence(s)")
            if bad_boxes:
                print(f"- Invalid boxes ({len(bad_boxes)}), e.g.: {bad_boxes[0]}")
            return

        # ---- Batch upload ----
        total = len(entries)
        print(f"Prepared {total} image(s), {total_regions} region annotation(s).")
        print(f"Uploading in batches of {BATCH_LIMIT}…")

        failures = 0
        uploaded = 0

        for i in range(0, total, BATCH_LIMIT):
            batch = entries[i:i + BATCH_LIMIT]
            try:
                result = _retryable_upload(batch)
            except Exception as e:
                print(f"Batch {i//BATCH_LIMIT + 1} failed hard: {e}")
                failures += len(batch)
                continue

            if not result.is_batch_successful:
                print(f"Batch {i//BATCH_LIMIT + 1} reported failures:")
                for img in result.images:
                    if getattr(img, "status", "").lower() != "ok":
                        failures += 1
                        print(f"  - {img.image or img.source_url or 'unknown'}: {img.status}")
            else:
                print(f"Batch {i//BATCH_LIMIT + 1} OK ({len(batch)} images).")
                uploaded += len(batch)

        # ---- Summary ----
        print("\nSummary")
        print("-------")
        print(f"Uploaded: {uploaded}/{total} images")
        if failures:
            print(f"Failures: {failures}")
        if missing_files:
            print(f"Missing files ({len(missing_files)}): " + ", ".join(missing_files[:10]) + (" ..." if len(missing_files) > 10 else ""))
        if missing_tags:
            print("Missing tag names encountered (not uploaded for those regions):")
            for t, c in missing_tags.items():
                print(f"  • {t}: {c} occurrence(s)")
        if bad_boxes:
            print(f"Invalid/Out-of-bounds boxes skipped: {len(bad_boxes)}")
    except Exception as e:
        print(f"Error during upload: {e}")
        return

if __name__ == '__main__':
    main()

"""
Phase 2 + 3: Label each trajectory frame with its current subgoal, then
write a new LeRobot dataset where the `task` string per frame is the subgoal
instruction instead of the full task.

Usage:
    python scripts/subgoal_label_dataset.py \
        --dataset_path /path/to/lerobot/dataset \
        --subgoals_json subgoals.json \
        --output_repo your_hf_username/libero-subgoal \
        --frames_per_episode 100   # how many frames to send GPT per episode (max ~150)

How it works:
  1. For each episode, uniformly sample `frames_per_episode` frames.
  2. Send all sampled frames + the pre-defined subgoal list to GPT-4.5.
  3. GPT returns the sampled-frame index boundary where each subgoal ends.
  4. Interpolate boundaries back to all T frames → per-frame subgoal index.
  5. Write a new LeRobot dataset with per-frame task strings = subgoal instructions.

The new dataset is drop-in compatible with the existing training pipeline because
`PromptFromLeRobotTask` reads `task_index` per frame, which will now point to
the current subgoal string.
"""

import argparse
import base64
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from openai import OpenAI
from tqdm import tqdm

MODEL = "gpt-5.4"
MAX_RETRIES = 3
MAX_FRAMES_PER_CALL = 150  # conservative safety ceiling; GPT-4.5 can handle ~500


def encode_image(img: np.ndarray) -> str:
    """Encode HWC uint8 RGB array to base64 JPEG string."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def sample_frames(frames: np.ndarray, n: int) -> tuple[list[int], np.ndarray]:
    """Uniformly sample n frames from (T, H, W, 3) array."""
    T = len(frames)
    n = min(n, T)
    indices = np.linspace(0, T - 1, n, dtype=int).tolist()
    return indices, frames[indices]


def label_episode(
    client: OpenAI,
    task_str: str,
    subgoals: list[str],
    frames: np.ndarray,
    frame_indices: list[int],
) -> list[int]:
    """
    Ask GPT to find where each subgoal transitions in the episode.

    Returns: per-frame subgoal index for ALL T frames (not just sampled ones).
    frame_indices maps sampled positions back to original frame indices.
    """
    n_sampled = len(frames)
    n_subgoals = len(subgoals)
    T = frame_indices[-1] + 1  # upper bound on total frames

    subgoals_str = "\n".join(f"  {i}: \"{sg}\"" for i, sg in enumerate(subgoals))

    content = [
        {
            "type": "text",
            "text": (
                f"You are labeling a robot manipulation trajectory.\n"
                f"Full task: \"{task_str}\"\n\n"
                f"The task has {n_subgoals} subgoals in order:\n{subgoals_str}\n\n"
                f"Below are {n_sampled} frames sampled from the episode "
                f"(frame indices shown). For each frame, the robot is performing "
                f"exactly one of the subgoals above.\n\n"
                f"Find the LAST sampled frame index (0-indexed into this list of "
                f"{n_sampled} frames, not the original frame number) where each "
                f"subgoal is still active before transitioning to the next.\n\n"
                f"Example for 3 subgoals with {n_sampled} frames:\n"
                f'  {{"boundaries": [12, 28, {n_sampled - 1}]}}\n'
                f"  means subgoal 0 covers sampled frames 0-12, subgoal 1 covers "
                f"13-28, subgoal 2 covers 29-{n_sampled - 1}.\n\n"
                f"Rules:\n"
                f"- boundaries must be strictly increasing\n"
                f"- last boundary must always be {n_sampled - 1}\n"
                f"- list length must equal {n_subgoals}\n\n"
                f'Respond only as JSON: {{"boundaries": [int, ...]}}'
            ),
        }
    ]

    for i, (orig_idx, frame) in enumerate(zip(frame_indices, frames)):
        content.append({"type": "text", "text": f"Sampled frame {i} (original idx {orig_idx}):"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(frame)}",
                "detail": "low",
            },
        })

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=256,
            )
            result = json.loads(response.choices[0].message.content)
            boundaries = result["boundaries"]

            # Validate
            assert len(boundaries) == n_subgoals, (
                f"Expected {n_subgoals} boundaries, got {len(boundaries)}"
            )
            assert boundaries[-1] == n_sampled - 1, (
                f"Last boundary must be {n_sampled - 1}, got {boundaries[-1]}"
            )
            assert all(
                boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1)
            ), "Boundaries must be strictly increasing"

            break
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            elif attempt == MAX_RETRIES - 1:
                # Fallback: divide equally
                print("    Using equal-split fallback")
                step = n_sampled // n_subgoals
                boundaries = [step * (i + 1) - 1 for i in range(n_subgoals)]
                boundaries[-1] = n_sampled - 1

    # Convert sampled-frame boundaries → original frame indices
    # boundaries[i] = last sampled index where subgoal i is active
    # frame_indices[boundaries[i]] = last original frame index of subgoal i
    orig_boundaries = [frame_indices[b] for b in boundaries]

    # Build per-frame label array for ALL T original frames
    # (T inferred from max frame index seen)
    total_frames = frame_indices[-1] + 1
    # We need to know actual T; use the passed frame count
    per_frame_labels = np.zeros(total_frames, dtype=int)
    prev = 0
    for sg_id, end_orig in enumerate(orig_boundaries):
        per_frame_labels[prev : end_orig + 1] = sg_id
        prev = end_orig + 1
    # Any remaining frames get the last subgoal
    per_frame_labels[prev:] = n_subgoals - 1

    return per_frame_labels.tolist()


def get_episode_frames(dataset: LeRobotDataset, ep_idx: int) -> tuple[np.ndarray, str]:
    """
    Load all base-camera frames for an episode.
    Returns (frames array T×H×W×3, task_string).
    """
    from_idx = dataset.episode_data_index["from"][ep_idx].item()
    to_idx = dataset.episode_data_index["to"][ep_idx].item()

    frames = []
    task_str = None
    for i in range(from_idx, to_idx):
        item = dataset.hf_dataset[i]
        img = item["observation.image"]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        frames.append(img)
        if task_str is None:
            task_str = dataset.meta.tasks[item["task_index"]]

    return np.stack(frames, axis=0), task_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Source LeRobot dataset path or HF repo id")
    parser.add_argument("--subgoals_json", required=True,
                        help="Path to subgoals.json from subgoal_decompose.py")
    parser.add_argument("--output_repo", required=True,
                        help="Output repo name, e.g. 'myuser/libero-subgoal'")
    parser.add_argument("--frames_per_episode", type=int, default=100,
                        help=f"Frames to sample per episode (max {MAX_FRAMES_PER_CALL})")
    parser.add_argument("--labels_cache", default="subgoal_labels_cache.json",
                        help="Cache file for per-episode labels (allows resuming)")
    args = parser.parse_args()

    args.frames_per_episode = min(args.frames_per_episode, MAX_FRAMES_PER_CALL)

    client = OpenAI()

    with open(args.subgoals_json) as f:
        subgoals_map: dict[str, list[str]] = json.load(f)
    print(f"Loaded subgoals for {len(subgoals_map)} tasks")

    print(f"Loading source dataset from {args.dataset_path} ...")
    src = LeRobotDataset(args.dataset_path)
    print(f"  {src.num_episodes} episodes")

    # ── Phase 2: Label all episodes ─────────────────────────────────────────

    # Load cache (allows resuming if interrupted)
    cache_path = Path(args.labels_cache)
    if cache_path.exists():
        with open(cache_path) as f:
            labels_cache: dict[str, list[int]] = json.load(f)
        print(f"  Resuming labeling: {len(labels_cache)} episodes cached")
    else:
        labels_cache = {}

    for ep_idx in tqdm(range(src.num_episodes), desc="Labeling episodes"):
        ep_key = str(ep_idx)
        if ep_key in labels_cache:
            continue

        frames, task_str = get_episode_frames(src, ep_idx)
        T = len(frames)

        if task_str not in subgoals_map:
            # No subgoals defined for this task — keep original task string
            # Signal with None; handled during dataset creation below
            labels_cache[ep_key] = None
        else:
            subgoals = subgoals_map[task_str]
            sample_indices, sampled_frames = sample_frames(
                frames, args.frames_per_episode
            )
            per_frame_labels = label_episode(
                client, task_str, subgoals, sampled_frames, sample_indices
            )
            # per_frame_labels may be shorter than T if last frame index < T-1
            # Pad if needed
            if len(per_frame_labels) < T:
                last = per_frame_labels[-1]
                per_frame_labels += [last] * (T - len(per_frame_labels))
            labels_cache[ep_key] = per_frame_labels[:T]

        # Save cache after each episode
        with open(cache_path, "w") as f:
            json.dump(labels_cache, f)

    print("Labeling complete.")

    # ── Phase 3: Write new dataset ───────────────────────────────────────────

    output_path = HF_LEROBOT_HOME / args.output_repo
    if output_path.exists():
        print(f"Removing existing output at {output_path}")
        shutil.rmtree(output_path)

    print(f"Creating new dataset at {output_path} ...")
    dst = LeRobotDataset.create(
        repo_id=args.output_repo,
        robot_type=src.meta.robot_type,
        fps=src.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for ep_idx in tqdm(range(src.num_episodes), desc="Writing episodes"):
        from_idx = src.episode_data_index["from"][ep_idx].item()
        to_idx = src.episode_data_index["to"][ep_idx].item()
        T = to_idx - from_idx

        ep_key = str(ep_idx)
        per_frame_labels = labels_cache.get(ep_key)

        frames_in_ep = range(from_idx, to_idx)
        for local_t, global_i in enumerate(frames_in_ep):
            item = src.hf_dataset[global_i]
            orig_task_str = src.meta.tasks[item["task_index"]]

            # Determine subgoal string for this frame
            if per_frame_labels is None or orig_task_str not in subgoals_map:
                task_str = orig_task_str  # fallback: original task
            else:
                sg_idx = per_frame_labels[local_t]
                subgoals = subgoals_map[orig_task_str]
                sg_idx = min(sg_idx, len(subgoals) - 1)
                task_str = subgoals[sg_idx]

            img = item["observation.image"]
            wrist = item["observation.wrist_image"]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if not isinstance(wrist, np.ndarray):
                wrist = np.array(wrist)

            dst.add_frame({
                "image": img,
                "wrist_image": wrist,
                "state": np.array(item["observation.state"], dtype=np.float32),
                "actions": np.array(item["action"], dtype=np.float32),
                "task": task_str,  # ← subgoal instruction for this frame
            })

        dst.save_episode()

    print(f"\nDone. New dataset written to {output_path}")
    print(f"Total unique subgoal task strings: {len(dst.meta.tasks)}")
    for idx, t in sorted(dst.meta.tasks.items()):
        print(f"  [{idx}] {t}")


if __name__ == "__main__":
    main()

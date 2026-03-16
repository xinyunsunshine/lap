"""
Phase 1: Vision-grounded subgoal decomposition for LIBERO-Goal.

For each unique task in the dataset, samples key frames from a representative
episode and sends them to GPT-4.5 vision to watch the trajectory and decompose
the task into ordered subgoals.

Usage:
    python scripts/subgoal_decompose.py \
        --dataset_path /path/to/lerobot/dataset \
        --output subgoals.json \
        --n_key_frames 16 \
        --task_filter libero_goal   # optional: only process tasks containing this substring

Output (subgoals.json):
    {
      "open the top drawer of the cabinet": [
        "move the gripper toward the top drawer handle",
        "grasp the top drawer handle",
        "pull the top drawer open"
      ],
      ...
    }
"""

import argparse
import base64
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openai import OpenAI
from tqdm import tqdm

MODEL = "gpt-5.4"
MAX_RETRIES = 3


def encode_image(img: np.ndarray) -> str:
    """Encode HWC uint8 RGB array to base64 JPEG string."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def sample_key_frames(frames: np.ndarray, n: int) -> tuple[list[int], np.ndarray]:
    """Uniformly sample n key frames from episode frames (T, H, W, 3)."""
    T = len(frames)
    indices = np.linspace(0, T - 1, n, dtype=int).tolist()
    return indices, frames[indices]


def decompose_task(client: OpenAI, task_str: str, frames: np.ndarray,
                   frame_indices: list[int]) -> list[str]:
    """
    Send key frames to GPT vision and ask it to decompose the task into subgoals.
    Returns a list of subgoal instruction strings in order.
    """
    n = len(frames)

    # Build the message: text prompt first, then interleaved frame labels + images
    content = [
        {
            "type": "text",
            "text": (
                f"You are watching a robot manipulation episode. "
                f"The robot's task is: \"{task_str}\"\n\n"
                f"Below are {n} key frames uniformly sampled from the full trajectory "
                f"(frame indices shown). Watch what the robot does step by step and "
                f"decompose the task into 2–4 ordered subgoals.\n\n"
                "Rules:\n"
                "- Each subgoal must be a clear, concise language instruction\n"
                "- Subgoals must be sequential and cover the entire task\n"
                "- Subgoals must be grounded in what you actually see the robot doing\n"
                "- 2 subgoals for simple tasks, 3–4 for multi-step tasks\n\n"
                'Respond only as JSON: {"subgoals": ["instruction 0", "instruction 1", ...]}'
            ),
        }
    ]

    for idx, frame in zip(frame_indices, frames):
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(frame)}",
                "detail": "low",  # save tokens; 256x256 is already small
            },
        })

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=512,
            )
            result = json.loads(response.choices[0].message.content)
            subgoals = result["subgoals"]
            assert isinstance(subgoals, list) and len(subgoals) >= 2
            return [s.strip() for s in subgoals]
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to decompose task: {task_str}")


def get_episode_frames(dataset: LeRobotDataset, ep_idx: int) -> np.ndarray:
    """
    Load all base-camera frames for an episode as (T, H, W, 3) uint8 array.
    Returns the 'image' (third-person) camera.
    """
    ep_data = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == ep_idx, keep_in_memory=True
    )
    # images come out as PIL or numpy depending on lerobot version
    frames = []
    for item in ep_data:
        img = item["observation.image"]  # PIL Image or np array
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        frames.append(img)
    return np.stack(frames, axis=0)


def build_task_to_episode_map(dataset: LeRobotDataset) -> dict[str, int]:
    """Map each unique task string to the index of its first episode."""
    task_to_ep: dict[str, int] = {}
    for ep_idx in range(dataset.num_episodes):
        # episode_data dict contains task string(s) for this episode
        task_idx = dataset.episode_data_index["from"][ep_idx].item()
        task_str = dataset.meta.tasks[
            dataset.hf_dataset[task_idx]["task_index"]
        ]
        if task_str not in task_to_ep:
            task_to_ep[task_str] = ep_idx
    return task_to_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to LeRobot dataset directory or HF repo id")
    parser.add_argument("--output", default="subgoals.json",
                        help="Output JSON file path")
    parser.add_argument("--n_key_frames", type=int, default=64,
                        help="Number of key frames to sample per representative episode")
    parser.add_argument("--task_filter", default=None,
                        help="Only process tasks containing this substring (e.g. 'libero_goal')")
    args = parser.parse_args()

    client = OpenAI()

    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = LeRobotDataset(args.dataset_path)
    print(f"  {dataset.num_episodes} episodes, {len(dataset.meta.tasks)} unique tasks")

    # Filter tasks if requested
    tasks_to_process = {
        idx: task_str
        for idx, task_str in dataset.meta.tasks.items()
        if args.task_filter is None or args.task_filter.lower() in task_str.lower()
    }
    print(f"  Processing {len(tasks_to_process)} tasks")

    # Map each task string to its first episode
    task_to_ep = build_task_to_episode_map(dataset)

    # Load existing output to allow resuming
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            subgoals_map = json.load(f)
        print(f"  Resuming: {len(subgoals_map)} tasks already processed")
    else:
        subgoals_map = {}

    for task_str in tqdm(tasks_to_process.values(), desc="Decomposing tasks"):
        if task_str in subgoals_map:
            continue  # already done

        ep_idx = task_to_ep[task_str]
        print(f"\nTask: \"{task_str}\"  (episode {ep_idx})")

        frames = get_episode_frames(dataset, ep_idx)
        frame_indices, key_frames = sample_key_frames(frames, args.n_key_frames)

        subgoals = decompose_task(client, task_str, key_frames, frame_indices)
        print(f"  → {len(subgoals)} subgoals: {subgoals}")

        subgoals_map[task_str] = subgoals

        # Save after each task so progress isn't lost
        with open(output_path, "w") as f:
            json.dump(subgoals_map, f, indent=2)

    print(f"\nDone. Subgoals saved to {output_path}")
    print(json.dumps(subgoals_map, indent=2))


if __name__ == "__main__":
    main()

"""
Test the subgoal decomposition + labeling pipeline on a single LIBERO episode.

Runs Phase 1 (decompose) and Phase 2 (label) end-to-end on one episode,
then saves a visualization showing:
  - The subgoal strings
  - The full trajectory contact sheet with subgoal color-coding
  - Close-up of each subgoal boundary (last frame of subgoal N, first frame of N+1)

Usage:
    python scripts/test_subgoal_pipeline.py \
        --dataset_path /proj/inf-scaling/zwhong/idan/mem-vla/data/lerobot/combined/physical-intelligence/libero \
        --task_index 12 \
        --output_dir /tmp/subgoal_test \
        --frames_per_label 100 \
        --n_key_frames 64
"""

import argparse
import base64
import glob
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

MODEL = "gpt-5.4"
MAX_RETRIES = 3
MIN_SUBGOAL_FRAMES = 4

# Switch between "v1" (original, longer subgoals) and "v2" (short, fine-grained)
DECOMPOSE_PROMPT_VERSION = "v2"

# Subgoal colors (BGR for cv2, RGB for PIL)
PALETTE = [
    (220, 50,  50),   # red
    (50,  150, 220),  # blue
    (50,  180, 80),   # green
    (200, 130, 30),   # orange
    (160, 60,  200),  # purple
]


# ── Image utilities ────────────────────────────────────────────────────────────

def load_episode_from_rlds(rlds_dataset: str, rlds_data_dir: str, ep_idx: int):
    """Load base frames, wrist frames, and task string from an RLDS episode by index."""
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    ds = tfds.load(rlds_dataset, data_dir=rlds_data_dir, split=f"train[{ep_idx}:{ep_idx+1}]")
    episode = next(iter(ds))
    steps = list(episode["steps"])
    base_frames = np.stack([step["observation"]["image"].numpy() for step in steps])
    wrist_frames = np.stack([step["observation"]["wrist_image"].numpy() for step in steps])
    actions = np.stack([step["action"].numpy() for step in steps])
    obs_states = np.stack([step["observation"]["state"].numpy() for step in steps])
    task_str = steps[0]["language_instruction"].numpy().decode("utf-8")
    return base_frames, wrist_frames, actions, obs_states, task_str


def load_frames_from_parquet(parquet_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load base and wrist frames from a single episode parquet file.
    Returns (base_frames T×H×W×3, wrist_frames T×H×W×3, actions T×7, obs_states T×8).
    """
    df = pd.read_parquet(parquet_path)
    base_frames, wrist_frames = [], []
    for _, row in df.iterrows():
        base_frames.append(np.array(Image.open(io.BytesIO(row["image"]["bytes"]))))
        wrist_frames.append(np.array(Image.open(io.BytesIO(row["wrist_image"]["bytes"]))))
    actions = np.stack(df["actions"].values)
    obs_states = np.stack(df["state"].values)
    return np.stack(base_frames), np.stack(wrist_frames), actions, obs_states


def encode_image(img: np.ndarray) -> str:
    """Encode HWC uint8 RGB array to base64 JPEG."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def sample_frames(frames: np.ndarray, n: int) -> tuple[list[int], np.ndarray]:
    T = len(frames)
    n = min(n, T)
    indices = np.linspace(0, T - 1, n, dtype=int).tolist()
    return indices, frames[indices]


GRIPPER_CLOSED_THRESHOLD = 0.02  # width (m) below which gripper is considered closed
                                 # open ≈ 0.08m, closed ≈ 0.003m


def gripper_width_from_state(obs_states: np.ndarray) -> np.ndarray:
    """
    Compute per-frame physical gripper width from observation state (T×8).
    State indices 6 and 7 are the two finger positions (symmetric, opposite signs).
    width ≈ 0.08 when open, ≈ 0 when closed.
    """
    return np.abs(obs_states[:, 6]) + np.abs(obs_states[:, 7])


# ── Phase 1: Decompose ─────────────────────────────────────────────────────────

def _decompose_prompt_v1(task_str: str, n: int) -> str:
    """Original prompt — longer subgoal strings, 2–10 subgoals."""
    return (
        f'You are watching a robot manipulation episode.\n'
        f'The robot\'s task is: "{task_str}"\n\n'
        f'Below are {n} key frames uniformly sampled from the full '
        f'trajectory (frame indices shown).\n'
        f'Watch what the robot does and decompose the task into 2–10 '
        f'ordered subgoals.\n\n'
        f'Rules:\n'
        f'- Each subgoal must be a clear, concise language instruction\n'
        f'- Subgoals must be sequential and cover the entire task\n'
        f'- Ground subgoals in what you actually see the robot doing\n'
        f'- Prefer more subgoals over fewer — err on the side of finer granularity\n'
        f'- Each subgoal must span at least {MIN_SUBGOAL_FRAMES} frames of the trajectory; '
        f'avoid decompositions where a subgoal covers only 1–3 frames\n\n'
        f'Examples of good subgoal decompositions:\n'
        f'  Task: "put the mushroom in the pot"\n'
        f'  Subgoals: ["reach for the mushroom", "grasp the mushroom", '
        f'"lift the mushroom", "move the mushroom above the pot", '
        f'"release the mushroom into the pot"]\n\n'
        f'  Task: "wipe the table with the cloth"\n'
        f'  Subgoals: ["move to the cloth", "slide the cloth to the left along the table", '
        f'"slide the cloth to the right along the table", '
        f'"slide the cloth to the left along the table", "release the cloth"]\n\n'
        f'  Task: "open the cabinet door"\n'
        f'  Subgoals: ["move to the cabinet door handle", "grasp the handle", '
        f'"pull the door open", "swing the arm to the left to open the door", '
        f'"reposition arm to right of door", "push cabinet door fully open to the left"]\n\n'
        f'  Task: "pour the cup into the bowl"\n'
        f'  Subgoals: ["move to the cup", "grasp the cup", '
        f'"move above the bowl", "rotate to pour the cup"]\n\n'
        f'Respond only as JSON: {{"subgoals": ["instruction 0", ...]}}'
    )


def _decompose_prompt_v2(task_str: str, n: int) -> str:
    """V2 prompt — concise subgoals, finer granularity (4–12 subgoals)."""
    return (
        f'You are watching a robot manipulation episode.\n'
        f'Task: "{task_str}"\n\n'
        f'Below are {n} key frames (indices shown). '
        f'Decompose the task into 6–15 ordered subgoals.\n\n'
        f'STRICT RULES (violations are wrong answers):\n'
        f'- NEVER use the word "and" — if you need "and", split into two subgoals\n'
        f'- Each subgoal = ONE atomic action only\n'
        f'- Each subgoal must be at most 7 words\n'
        f'- Always prefer MORE subgoals — every distinct motion phase is its own subgoal\n'
        f'- Separate approach, grasp, lift, move, lower, release into individual subgoals\n'
        f'- Sequential, covering the full task; ground in what you see\n\n'
        f'Examples of good subgoal decompositions:\n'
        f'  Task: "put the mushroom in the pot"\n'
        f'  Subgoals: ["move above the mushroom", "lower to the mushroom", '
        f'"grasp the mushroom", "lift the mushroom up", '
        f'"move above the pot", "lower the mushroom into the pot", '
        f'"release the mushroom"]\n\n'
        f'  Task: "wipe the table with the cloth"\n'
        f'  Subgoals: ["move above the cloth", "lower to the cloth", "grasp the cloth", '
        f'"lift the cloth", "slide the cloth left", "slide the cloth right", '
        f'"slide the cloth left", "release the cloth"]\n\n'
        f'  Task: "open the cabinet door"\n'
        f'  Subgoals: ["move to the cabinet handle", "lower to the handle", '
        f'"grasp the handle", "pull the door open", "swing arm left", '
        f'"reposition arm right of door", "push door fully open"]\n\n'
        f'  Task: "pour the cup into the bowl"\n'
        f'  Subgoals: ["move above the cup", "lower to the cup", "grasp the cup", '
        f'"lift the cup", "move above the bowl", "tilt to pour", "place cup down"]\n\n'
        f'Respond only as JSON: {{"subgoals": ["instruction 0", ...]}}'
    )


def decompose_task(client: OpenAI, task_str: str, frames: np.ndarray,
                   frame_indices: list[int],
                   prompt_version: str = DECOMPOSE_PROMPT_VERSION) -> list[str]:
    """Send key frames to GPT to decompose task into subgoals."""
    n = len(frames)
    if prompt_version == "v2":
        prompt_text = _decompose_prompt_v2(task_str, n)
    else:
        prompt_text = _decompose_prompt_v1(task_str, n)

    content = [
        {
            "type": "text",
            "text": prompt_text,
        }
    ]
    for idx, frame in zip(frame_indices, frames):
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(frame)}",
                "detail": "low",
            },
        })

    messages = [{"role": "user", "content": content}]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=1024,
            )
            result = json.loads(resp.choices[0].message.content)
            subgoals = [s.strip() for s in result["subgoals"]]
            assert len(subgoals) >= 2

            # Reject if any subgoal contains "and" — ask GPT to fix
            bad = [s for s in subgoals if " and " in s.lower()]
            if bad:
                print(f"  Decompose attempt {attempt + 1}: subgoals contain 'and': {bad} — retrying")
                messages = messages + [
                    {"role": "assistant", "content": resp.choices[0].message.content},
                    {"role": "user", "content": (
                        f'Some subgoals contain "and": {bad}\n'
                        f'This is not allowed. Split each of them into separate subgoals '
                        f'(one action per subgoal, no "and"). '
                        f'Return the full corrected list as JSON: {{"subgoals": ["...", ...]}}'
                    )},
                ]
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue

            return subgoals
        except Exception as e:
            print(f"  Decompose attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError("Decompose failed after retries")


# ── Subgoal refinement ─────────────────────────────────────────────────────────

def refine_subgoals_for_min_frames(
    client: OpenAI,
    task_str: str,
    subgoals: list[str],
    orig_boundaries: list[int],
    T: int,
    min_frames: int = MIN_SUBGOAL_FRAMES,
) -> list[str]:
    """
    If any subgoal has fewer than min_frames frames after Phase 2 labeling,
    ask GPT to revise the decomposition so every subgoal meets the minimum.
    Returns a new subgoals list (same length or shorter after merges).
    """
    starts = [0] + [b + 1 for b in orig_boundaries[:-1]]
    lengths = [orig_boundaries[i] - starts[i] + 1 for i in range(len(subgoals))]

    sg_summary = "\n".join(
        f"  {i}: \"{sg}\"  [{lengths[i]} frames{'  ← TOO SHORT' if lengths[i] < min_frames else ''}]"
        for i, sg in enumerate(subgoals)
    )
    short_count = sum(1 for l in lengths if l < min_frames)

    prompt = (
        f'You previously decomposed the robot task "{task_str}" into these subgoals '
        f'(frame counts shown after Phase 2 labeling):\n\n'
        f'{sg_summary}\n\n'
        f'{short_count} subgoal(s) are too short (< {min_frames} frames). '
        f'Merge each short subgoal into its nearest neighbor so that every subgoal '
        f'spans at least {min_frames} frames.\n\n'
        f'Rules:\n'
        f'- Keep subgoals in the same order\n'
        f'- Only merge; do not split existing subgoals\n'
        f'- When merging, write a new concise instruction of at most 7 words '
        f'(e.g. "reach and grasp the handle" → "grasp the handle")\n'
        f'- NEVER use the word "and" — rewrite the merged subgoal as a single action\n'
        f'- Every subgoal in the output must have been reachable with >= {min_frames} frames\n\n'
        f'Respond only as JSON: {{"subgoals": ["instruction 0", ...]}}'
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=1024,
            )
            result = json.loads(resp.choices[0].message.content)
            refined = [s.strip() for s in result["subgoals"]]
            assert len(refined) >= 1
            return refined
        except Exception as e:
            print(f"  Refine attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    print("  Refine failed — keeping original subgoals")
    return subgoals


# ── Phase 2: Label ─────────────────────────────────────────────────────────────

def label_episode(client: OpenAI, task_str: str, subgoals: list[str],
                  frames: np.ndarray, frame_indices: list[int],
                  gripper_widths: np.ndarray | None = None) -> list[int]:
    """
    Send sampled frames + subgoals to GPT. Returns per-frame subgoal index
    for all T original frames.
    """
    n_sampled = len(frames)
    n_sg = len(subgoals)
    subgoals_str = "\n".join(f"  {i}: \"{sg}\"" for i, sg in enumerate(subgoals))

    content = [
        {
            "type": "text",
            "text": (
                f'You are labeling a robot manipulation trajectory.\n'
                f'Full task: "{task_str}"\n\n'
                f'Subgoals in order:\n{subgoals_str}\n\n'
                f'Below are {n_sampled} frames sampled from the episode '
                f'(frame indices shown). Find the LAST sampled-list index '
                f'(0 to {n_sampled - 1}) for each subgoal.\n\n'
                f'IMPORTANT boundary rule: assign each frame to the subgoal '
                f'whose action it INITIATES. If a frame shows the robot '
                f'beginning a new action (e.g. starting to lower, starting to '
                f'open the gripper), that frame belongs to the NEW subgoal — '
                f'NOT the previous one. For example: if subgoal 0 is '
                f'"carry the bowl over the drawer" and subgoal 1 is '
                f'"lower the bowl into the drawer and release it", then the '
                f'first frame where the robot starts moving downward belongs '
                f'to subgoal 1.\n\n'
                f'Example for {n_sg} subgoals:\n'
                f'  {{"boundaries": [12, 28, {n_sampled - 1}]}}\n'
                f'  → subgoal 0: sampled frames 0–12, subgoal 1: 13–28, '
                f'subgoal {n_sg-1}: 29–{n_sampled - 1}\n\n'
                f'Rules:\n'
                f'- strictly increasing\n'
                f'- last value must be {n_sampled - 1}\n'
                f'- exactly {n_sg} values\n'
                f'- for subgoals that involve grasping (e.g. "grasp the bowl"), '
                f'the boundary must be a frame where gripper state is closing\n'
                f'- for subgoals that involve releasing (e.g. "release the bowl"), '
                f'the boundary must be a frame where gripper state is open\n\n'
                f'Respond only as JSON: {{"boundaries": [int, ...]}}'
            ),
        }
    ]
    for i, (orig_idx, frame) in enumerate(zip(frame_indices, frames)):
        if gripper_widths is not None and orig_idx < len(gripper_widths):
            g_state = "closed" if gripper_widths[orig_idx] < GRIPPER_CLOSED_THRESHOLD else "open"
            frame_label = f"Sampled [{i}] (orig frame {orig_idx}, gripper: {g_state}):"
        else:
            frame_label = f"Sampled [{i}] (orig frame {orig_idx}):"
        content.append({"type": "text", "text": frame_label})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(frame)}",
                "detail": "low",
            },
        })

    boundaries = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=256,
            )
            result = json.loads(resp.choices[0].message.content)
            boundaries = result["boundaries"]
            assert len(boundaries) == n_sg
            assert boundaries[-1] == n_sampled - 1
            assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1))
            break
        except Exception as e:
            print(f"  Label attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    if boundaries is None:
        print("  Using equal-split fallback")
        step = n_sampled // n_sg
        boundaries = [step * (i + 1) - 1 for i in range(n_sg)]
        boundaries[-1] = n_sampled - 1

    # Map sampled-list boundaries → original frame indices
    orig_boundaries = [frame_indices[b] for b in boundaries]

    # Build per-frame label for all T frames
    T = frame_indices[-1] + 1
    per_frame = np.zeros(T, dtype=int)
    prev = 0
    for sg_id, end in enumerate(orig_boundaries):
        per_frame[prev : end + 1] = sg_id
        prev = end + 1
    per_frame[prev:] = n_sg - 1

    return per_frame.tolist(), orig_boundaries


# ── Visualization ──────────────────────────────────────────────────────────────

def add_label(img: np.ndarray, text: str, color: tuple, frame_idx: int) -> np.ndarray:
    """Overlay subgoal label and frame index on an image."""
    out = img.copy()
    H, W = out.shape[:2]
    # color bar at top
    out[:18, :] = color
    # frame index
    cv2.putText(out, f"f{frame_idx}", (4, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(frames: np.ndarray, per_frame_labels: list[int],
                       subgoals: list[str], thumb_w: int = 96) -> np.ndarray:
    """
    Full trajectory contact sheet. Each frame is a thumbnail with a color
    border indicating its subgoal.
    """
    T = len(frames)
    H, W = frames.shape[1:3]
    scale = thumb_w / W
    thumb_h = int(H * scale)
    cols = 16
    rows = (T + cols - 1) // cols

    canvas = np.ones((rows * (thumb_h + 2), cols * (thumb_w + 2), 3), dtype=np.uint8) * 40

    for i, (frame, sg) in enumerate(zip(frames, per_frame_labels)):
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        color = PALETTE[sg % len(PALETTE)]
        # color border (2px)
        x0 = c * (thumb_w + 2)
        y0 = r * (thumb_h + 2)
        canvas[y0 : y0 + thumb_h + 2, x0 : x0 + thumb_w + 2] = color
        canvas[y0 + 1 : y0 + thumb_h + 1, x0 + 1 : x0 + thumb_w + 1] = thumb

    return canvas


def make_boundary_strip(frames: np.ndarray, orig_boundaries: list[int],
                        subgoals: list[str], context: int = 4) -> np.ndarray:
    """
    For each subgoal transition, show the last `context` frames of subgoal N
    and first `context` frames of subgoal N+1, separated by a divider.
    Returns a vertical stack of transition strips.
    """
    T = len(frames)
    H, W = frames.shape[1:3]
    strips = []

    for transition_idx in range(len(subgoals) - 1):
        boundary = orig_boundaries[transition_idx]  # last frame of subgoal N
        sg_a = transition_idx
        sg_b = transition_idx + 1
        color_a = PALETTE[sg_a % len(PALETTE)]
        color_b = PALETTE[sg_b % len(PALETTE)]

        # Collect context frames on each side
        left_indices = list(range(max(0, boundary - context + 1), boundary + 1))
        right_indices = list(range(boundary + 1, min(T, boundary + 1 + context)))

        row_imgs = []
        for idx in left_indices:
            img = add_label(frames[idx], f"SG{sg_a}", color_a, idx)
            row_imgs.append(img)

        # Divider
        divider = np.ones((H, 6, 3), dtype=np.uint8) * 255
        row_imgs.append(divider)

        for idx in right_indices:
            img = add_label(frames[idx], f"SG{sg_b}", color_b, idx)
            row_imgs.append(img)

        strip = np.concatenate(row_imgs, axis=1)

        # Add text header above the strip
        label_bar = np.zeros((36, strip.shape[1], 3), dtype=np.uint8)
        text_a = f'SG{sg_a}: "{subgoals[sg_a]}"'
        text_b = f'SG{sg_b}: "{subgoals[sg_b]}"'
        cv2.putText(label_bar, f"Transition {sg_a}→{sg_b}  |  boundary after frame {boundary}",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1)
        cv2.putText(label_bar, text_a, (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tuple(color_a), 1)
        cv2.putText(label_bar, f"  →  {text_b}", (6 + len(text_a) * 8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tuple(color_b), 1)

        strips.append(np.concatenate([label_bar, strip], axis=0))
        # spacer
        strips.append(np.zeros((8, strip.shape[1], 3), dtype=np.uint8))

    # Pad all strips to same width
    max_w = max(s.shape[1] for s in strips)
    padded = []
    for s in strips:
        pad = max_w - s.shape[1]
        if pad > 0:
            s = np.concatenate([s, np.zeros((s.shape[0], pad, 3), dtype=np.uint8)], axis=1)
        padded.append(s)

    return np.concatenate(padded, axis=0)


def make_legend(subgoals: list[str], width: int = 800) -> np.ndarray:
    bar_h = 40
    canvas = np.zeros((bar_h * len(subgoals), width, 3), dtype=np.uint8)
    for i, sg in enumerate(subgoals):
        color = PALETTE[i % len(PALETTE)]
        canvas[i * bar_h : (i + 1) * bar_h, :60] = color
        cv2.putText(canvas, f"SG{i}: {sg}",
                    (68, i * bar_h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def find_parquet_for_task(dataset_path: str, task_index: int) -> str:
    """Find the parquet file of the first episode with the given task_index."""
    parquets = sorted(glob.glob(f"{dataset_path}/data/chunk-*/*.parquet"))
    for p in parquets:
        df = pd.read_parquet(p, columns=["task_index"])
        if int(df["task_index"].iloc[0]) == task_index:
            return p
    raise ValueError(f"No episode found with task_index={task_index}")


def load_tasks_json(dataset_path: str) -> dict[int, str]:
    tasks = {}
    with open(f"{dataset_path}/meta/tasks.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            tasks[entry["task_index"]] = entry["task"]
    return tasks


def main():
    parser = argparse.ArgumentParser()
    # RLDS mode (original LIBERO RLDS datasets)
    parser.add_argument("--rlds_dataset", type=str, default=None,
                        help="TFDS dataset name, e.g. libero_goal_no_noops")
    parser.add_argument("--rlds_data_dir", type=str, default=None,
                        help="Directory containing the TFDS dataset")
    parser.add_argument("--episode_index", type=int, default=None,
                        help="Episode index within the RLDS train split")
    # Parquet mode (LeRobot format)
    parser.add_argument("--dataset_path",
                        default="/proj/inf-scaling/zwhong/idan/mem-vla/data/lerobot/combined/physical-intelligence/libero")
    parser.add_argument("--task_index", type=int, default=None,
                        help="Task index — finds the first episode with this task_index")
    parser.add_argument("--episode_file", type=str, default=None,
                        help="Path to a specific episode parquet file (overrides --task_index)")
    parser.add_argument("--output_dir", default="/tmp/subgoal_test")
    parser.add_argument("--n_key_frames", type=int, default=64,
                        help="Key frames for Phase 1 decomposition")
    parser.add_argument("--frames_per_label", type=int, default=100,
                        help="Frames for Phase 2 labeling")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    # ── Load episode frames and task string ──────────────────────────────────
    if args.rlds_dataset:
        if args.episode_index is None:
            parser.error("--episode_index is required with --rlds_dataset")
        ep_idx = args.episode_index
        print(f"\nLoading RLDS episode {ep_idx} from {args.rlds_dataset}")
        base_frames, wrist_frames, actions, obs_states, task_str = load_episode_from_rlds(
            args.rlds_dataset, args.rlds_data_dir, ep_idx
        )
        episode_meta = {"episode_index": ep_idx, "task": task_str}
        print(f"Task: \"{task_str}\"")
    else:
        if args.episode_file is None and args.task_index is None:
            parser.error("One of --rlds_dataset, --task_index, or --episode_file is required")
        tasks = load_tasks_json(args.dataset_path)
        if args.episode_file:
            parquet_path = args.episode_file
            df_meta = pd.read_parquet(parquet_path, columns=["task_index"])
            task_idx = int(df_meta["task_index"].iloc[0])
        else:
            task_idx = args.task_index
            parquet_path = find_parquet_for_task(args.dataset_path, task_idx)
        task_str = tasks[task_idx]
        print(f"\nTask [{task_idx}]: \"{task_str}\"")
        print(f"Episode file: {parquet_path}")
        base_frames, wrist_frames, actions, obs_states = load_frames_from_parquet(parquet_path)
        episode_meta = {"task_index": task_idx, "task": task_str}
    T = len(base_frames)
    print(f"Loaded {T} frames  ({base_frames.shape[1]}×{base_frames.shape[2]})")
    gripper_widths = gripper_width_from_state(obs_states)

    # ── Phase 1: Decompose ──────────────────────────────────────────────────
    print(f"\n── Phase 1: Decompose ({args.n_key_frames} key frames) ──")
    key_indices, key_frames = sample_frames(base_frames, args.n_key_frames)
    subgoals = decompose_task(client, task_str, key_frames, key_indices)

    print(f"\nSubgoals ({len(subgoals)}):")
    for i, sg in enumerate(subgoals):
        print(f"  {i}: {sg}")

    # ── Phase 2: Label ──────────────────────────────────────────────────────
    print(f"\n── Phase 2: Label ({args.frames_per_label} frames) ──")
    label_indices, label_frames = sample_frames(base_frames, args.frames_per_label)
    per_frame_labels, orig_boundaries = label_episode(
        client, task_str, subgoals, label_frames, label_indices, gripper_widths
    )

    # Iteratively refine subgoals that are too short (up to 3 rounds)
    for refine_round in range(3):
        starts = [0] + [b + 1 for b in orig_boundaries[:-1]]
        short_ids = [
            i for i in range(len(subgoals))
            if orig_boundaries[i] - starts[i] + 1 < MIN_SUBGOAL_FRAMES
        ]
        if not short_ids:
            break
        print(f"\n  Round {refine_round + 1}: {len(short_ids)} subgoal(s) have "
              f"<{MIN_SUBGOAL_FRAMES} frames (SG{short_ids}), refining...")
        subgoals = refine_subgoals_for_min_frames(
            client, task_str, subgoals, orig_boundaries, T
        )
        print(f"  Refined to {len(subgoals)} subgoals. Re-running Phase 2...")
        for i, sg in enumerate(subgoals):
            print(f"    {i}: {sg}")
        per_frame_labels, orig_boundaries = label_episode(
            client, task_str, subgoals, label_frames, label_indices, gripper_widths
        )

    print(f"\nBoundary frames (last frame of each subgoal):")
    prev = 0
    for i, (sg, end) in enumerate(zip(subgoals, orig_boundaries)):
        print(f"  SG{i} frames {prev:3d}–{end:3d}  ({end - prev + 1} frames): \"{sg}\"")
        prev = end + 1

    # Enforce monotonicity: if any boundary is non-increasing, fix by clamping
    for i in range(1, len(orig_boundaries)):
        if orig_boundaries[i] <= orig_boundaries[i - 1]:
            orig_boundaries[i] = orig_boundaries[i - 1] + 1
            print(f"  Warning: fixed non-monotonic boundary at SG{i} → {orig_boundaries[i]}")
    orig_boundaries[-1] = min(orig_boundaries[-1], T - 1)

    # ── Save results JSON ───────────────────────────────────────────────────
    results = {
        **episode_meta,
        "total_frames": T,
        "subgoals": subgoals,
        "boundaries": orig_boundaries,
        "per_frame_labels": per_frame_labels,
    }
    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Visualizations ──────────────────────────────────────────────────────
    print("\n── Generating visualizations ──")

    # 1. Legend
    legend = make_legend(subgoals, width=900)
    cv2.imwrite(str(out / "legend.png"), cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))

    # 2. Full trajectory contact sheet
    contact = make_contact_sheet(base_frames, per_frame_labels[:T], subgoals)
    cv2.imwrite(str(out / "contact_sheet.png"), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))

    # 3. Boundary close-ups
    boundary_strip = make_boundary_strip(base_frames, orig_boundaries, subgoals, context=5)
    cv2.imwrite(str(out / "boundaries.png"), cv2.cvtColor(boundary_strip, cv2.COLOR_RGB2BGR))

    # 4. Save individual boundary frames (last of SG_n, first of SG_{n+1})
    for i, boundary in enumerate(orig_boundaries[:-1]):
        last_frame = base_frames[boundary]
        first_frame = base_frames[boundary + 1]
        pair = np.concatenate([last_frame, first_frame], axis=1)
        # Add divider line
        pair[:, last_frame.shape[1] - 1 : last_frame.shape[1] + 1] = 255
        cv2.imwrite(
            str(out / f"boundary_{i}_to_{i+1}_frame{boundary}.png"),
            cv2.cvtColor(pair, cv2.COLOR_RGB2BGR),
        )

    print(f"\nSaved to {out}/")
    print("  legend.png         — subgoal color legend")
    print("  contact_sheet.png  — full trajectory with color-coded subgoals")
    print("  boundaries.png     — close-up of each subgoal transition")
    print("  boundary_*.png     — side-by-side last/first frames at each cutoff")


if __name__ == "__main__":
    main()

"""
generate_steering_commands.py

Phase 3 of the subgoal pipeline. For each subgoal in an episode, generates a
structured "steering command" describing the motion from multiple levels:

    {
      "task":                        "put the mushroom in the pot",
      "subgoal":                     "move the mushroom above the pot",
      "spatial_command":             "position the mushroom directly above the pot opening",
      "low_level_motion":            "move up and forward",
      "low_level_motion_with_numbers": "move up 8m, move forward 5m, close gripper"
    }

The numbers in low_level_motion_with_numbers are computed exactly from the summed
action deltas over each subgoal's frame range. GPT receives these numbers and
generates the other fields from the visual context (first/last frame of subgoal).

Action convention (7D EEF delta):
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    dx > 0  → forward    dx < 0  → back
    dy > 0  → left       dy < 0  → right
    dz > 0  → up         dz < 0  → down
    gripper +1 → close   gripper -1 → open

Usage:
    python scripts/generate_steering_commands.py \\
        --dataset_path /proj/.../libero \\
        --results_json  outputs/subgoal_test_task12/results.json \\
        --output_json   outputs/subgoal_test_task12/steering_commands.json

Batch (directory of results.json):
    python scripts/generate_steering_commands.py \\
        --dataset_path /proj/.../libero \\
        --results_dir  outputs/subgoal_results/
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
from PIL import Image

MODEL = "gpt-4o"
MAX_RETRIES = 3
N_FRAMES = 16  # frames sampled per subgoal

# Action indices
IDX_X, IDX_Y, IDX_Z, IDX_GRIPPER = 0, 1, 2, 6
MIN_DELTA_M = 0.5  # drop components that round to 0m


# ── Action utilities ───────────────────────────────────────────────────────────

def load_actions_from_parquet(parquet_path: str) -> np.ndarray:
    """Returns (T, 7) float32 action array from parquet."""
    df = pd.read_parquet(parquet_path)
    return np.stack(df["actions"].values)  # (T, 7)


def load_episode_from_rlds(rlds_dataset: str, rlds_data_dir: str, ep_idx: int):
    """Load frames and actions from an RLDS episode by index."""
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    ds = tfds.load(rlds_dataset, data_dir=rlds_data_dir, split=f"train[{ep_idx}:{ep_idx+1}]")
    episode = next(iter(ds))
    steps = list(episode["steps"])
    frames = np.stack([step["observation"]["image"].numpy() for step in steps])
    actions = np.stack([step["action"].numpy() for step in steps])
    return frames, actions


def detect_gripper_transition(actions: np.ndarray, start: int, end: int) -> str | None:
    """
    Returns 'open gripper', 'close gripper', or None.
    Uses the mean gripper action at the start vs end of the subgoal to detect
    an actual open↔close transition, ignoring mid-subgoal noise.
    """
    window = max(1, (end - start + 1) // 4)  # use first/last 25% of frames
    start_gripper = actions[start : start + window, IDX_GRIPPER].mean()
    end_gripper   = actions[end - window + 1 : end + 1, IDX_GRIPPER].mean()

    # Convention: action[6] > 0 = close command, action[6] < 0 = open command
    was_opening = start_gripper < 0
    is_closing  = end_gripper > 0
    was_closing = start_gripper > 0
    is_opening  = end_gripper < 0

    if was_opening and is_closing:
        return "close gripper"
    if was_closing and is_opening:
        return "open gripper"
    return None


def detect_gripper_start_state(actions: np.ndarray, start: int) -> str:
    """
    Returns 'open' or 'closed' based on gripper commands at the start of the subgoal.
    gripper action > 0 = open command, < 0 = close command.
    """
    window = max(1, min(5, len(actions) - start))
    mean = actions[start : start + window, IDX_GRIPPER].mean()
    return "closed" if mean < 0 else "open"


# Keywords for text-based gripper inference (more reliable than action detection)
# Uses regex word boundaries to avoid "grip" matching "gripper", etc.
import re as _re

_GRASP_PATTERNS = [
    r'\bgrasp\b', r'\bgrab\b', r'\bpick up\b', r'\bgrasping\b', r'\bclose gripper\b',
    r'\bgrip\b',   # whole word only — does NOT match "gripper"
]
_RELEASE_PATTERNS = [
    r'\brelease\b', r'\bdrop\b', r'\bput down\b', r'\breleasing\b',
    r'\blet go\b', r'\bopen gripper\b', r'\bplace\b',
]


def infer_gripper_transition_from_text(subgoal: str) -> str | None:
    """
    Infers gripper transition from subgoal language using whole-word matching.
    Returns 'close gripper', 'open gripper', or None if ambiguous.
    Text-based inference is more reliable than action signal detection.
    """
    s = subgoal.lower()
    if any(_re.search(p, s) for p in _GRASP_PATTERNS):
        return "close gripper"
    if any(_re.search(p, s) for p in _RELEASE_PATTERNS):
        return "open gripper"
    return None


def sum_actions_for_subgoal(actions: np.ndarray, start: int, end: int) -> dict:
    """
    Sum action deltas over frames [start, end] inclusive.
    Returns dict with human-readable signed components in meters (rounded to int).
    Gripper command only included if there is an actual open↔close transition.
    """
    chunk = actions[start : end + 1]  # (N, 7)
    total = chunk.sum(axis=0)         # (7,)

    dx = float(total[IDX_X])
    dy = float(total[IDX_Y])
    dz = float(total[IDX_Z])
    gripper_delta = float(total[IDX_GRIPPER])

    components = []
    if abs(dz) >= MIN_DELTA_M:
        direction = "up" if dz > 0 else "down"
        components.append(f"move {direction} {round(abs(dz))}m")
    if abs(dx) >= MIN_DELTA_M:
        direction = "forward" if dx > 0 else "back"
        components.append(f"move {direction} {round(abs(dx))}m")
    if abs(dy) >= MIN_DELTA_M:
        direction = "left" if dy > 0 else "right"
        components.append(f"move {direction} {round(abs(dy))}m")

    gripper_cmd = detect_gripper_transition(actions, start, end)

    return {
        "dx_m": round(dx, 3),
        "dy_m": round(dy, 3),
        "dz_m": round(dz, 3),
        "gripper_delta": round(gripper_delta, 3),
        "gripper_transition": gripper_cmd,
        "spatial_components": ", ".join(components) if components else "hold position",
    }


# ── Image utilities ────────────────────────────────────────────────────────────

def load_frames_from_parquet(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    frames = []
    for _, row in df.iterrows():
        frames.append(np.array(Image.open(io.BytesIO(row["image"]["bytes"]))))
    return np.stack(frames)


def encode_image(img: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ── GPT prompt ─────────────────────────────────────────────────────────────────

SPATIAL_EXAMPLES = """
Examples of spatial commands (include gripper action at end if one occurs):
  - "move the gripper down and to the right to align with the drawer handle, then grasp it"
  - "move the cloth to the left side of the table surface"
  - "lift the bowl up, move it backward and to the right over the open drawer"
  - "lower the bowl into the cabinet opening, then release it"
  - "bring the cup above the bowl rim, then tilt it forward"
  - "slide the cloth across the table to the left"
  - "swing the door to the left until flush with the wall"
  - "position the mushroom directly above the pot opening, then release it"
""".strip()

MOTION_EXAMPLES = """
Examples of low-level motion instructions (no numbers, include gripper if transition):
  - "move down and right, then grasp"
  - "move the cloth left"
  - "pull the drawer handle backward"
  - "lift up and move forward"
  - "move up, backward, and right"
  - "lower down and release"
""".strip()

OUTCOME_EXAMPLES = """
Examples of outcome (final scene state at end of THIS subgoal only):
  - "mushroom is held in the gripper above the pot opening"
  - "cloth is flat on the left half of the table"
  - "gripper is closed around the drawer handle"
  - "top drawer is fully open"
  - "bowl is held in the gripper above the open drawer"
  - "bowl is resting inside the open drawer, gripper released"
  - "door is pushed fully to the left against the cabinet frame"
""".strip()

AFFORDANCE_EXAMPLES = """
Examples of object affordance (how gripper interacts with the object named in the SUBGOAL):
  Grasping (gripper closes during subgoal):
    - "grasp the mushroom cap from above with a top-down pinch grip"
    - "hook gripper around the drawer handle from the front"
    - "wrap gripper around the cup handle on the right side"
  No transition (describe grip + motion on the subgoal object):
    - "grasp the bowl from above with a secure grip and lower it into the drawer"
    - "hook gripper around the drawer handle and pull it backward"
    - "grip the drawer handle centrally and push it forward"
    - "hold the mushroom in a pinch grip and carry it over the pot"
  Releasing (gripper opens during subgoal):
    - "release the bowl into the drawer from directly above"
    - "open gripper to drop the mushroom into the pot"
    - "release the drawer handle after pulling it open"
""".strip()


def sample_subgoal_frames(
    frames: np.ndarray, start: int, end: int, n: int = N_FRAMES
) -> list[tuple[int, np.ndarray]]:
    """Uniformly sample up to n frames from [start, end] inclusive."""
    length = end - start + 1
    n = min(n, length)
    indices = np.linspace(start, end, n, dtype=int).tolist()
    return [(idx, frames[idx]) for idx in indices]


def _gripper_context_str(start_state: str, transition: str | None) -> str:
    """
    Returns a human-readable gripper context string combining start state and transition.
    Intentionally avoids interpretive labels like "holding object" — GPT should infer
    what is being held (if anything) from the frames and subgoal text.
    """
    if transition == "close gripper":
        return f"starts {start_state}, closes during subgoal"
    elif transition == "open gripper":
        return f"starts {start_state}, opens during subgoal"
    else:
        return f"{start_state} throughout (no open/close transition)"


def generate_steering_command(
    client: OpenAI,
    task_str: str,
    subgoal: str,
    motion_with_numbers: str,
    gripper_transition: str | None,
    gripper_start_state: str,
    subgoal_frames: list[tuple[int, np.ndarray]],
) -> dict:
    """
    Given the subgoal, pre-computed motion numbers, and sampled frames across
    the subgoal, ask GPT to generate spatial_command, low_level_motion,
    outcome, and object_affordance.
    """
    n = len(subgoal_frames)
    gripper_ctx = _gripper_context_str(gripper_start_state, gripper_transition)
    prompt = (
        f'You are describing a robot manipulation subgoal.\n\n'
        f'Full task: "{task_str}"\n'
        f'Subgoal: "{subgoal}"\n'
        f'Exact motion (computed from actions): "{motion_with_numbers}"\n'
        f'Gripper: {gripper_ctx}\n\n'
        f'IMPORTANT rules:\n'
        f'- For spatial_command and object_affordance: only refer to objects explicitly '
        f'named in the SUBGOAL above — do NOT use objects that only appear in the full task\n'
        f'- spatial_command must cover the FULL displacement (all axes in motion_with_numbers) '
        f'and end with the gripper action (grasp/release) if one occurs\n'
        f'- Do NOT use forward-looking phrases like "before releasing", "in preparation for", '
        f'"to position for" — only describe what happens in THIS subgoal\n'
        f'- outcome describes ONLY the state at the END of this subgoal; '
        f'do NOT describe what happens in future subgoals\n'
        f'- object_affordance: look at the frames to determine what the gripper is actually '
        f'interacting with in this subgoal; '
        f'if gripper closes → describe how to grasp the object; '
        f'if gripper opens → describe how/where to release the object; '
        f'if no transition → describe the grip AND the motion being performed '
        f'(e.g. "grasp the bowl from above and lower it into the drawer", '
        f'"hook gripper around the handle and pull it backward"); '
        f'only reference objects named in the SUBGOAL (NOT objects only in the full task)\n\n'
        f'{SPATIAL_EXAMPLES}\n\n'
        f'{MOTION_EXAMPLES}\n\n'
        f'{OUTCOME_EXAMPLES}\n\n'
        f'{AFFORDANCE_EXAMPLES}\n\n'
        f'You are shown {n} frames uniformly sampled across this subgoal '
        f'(frame indices shown, first → last). Study them carefully.\n\n'
        f'Generate all four fields. After drafting each one, verify it is consistent '
        f'with the subgoal text, the gripper context, and what you see in the frames.\n\n'
        f'1. spatial_command: spatial/relational description covering ALL directions moved '
        f'and ending with gripper action (grasp/release) if applicable. One sentence.\n'
        f'2. low_level_motion: concise motion words only, no numbers, include gripper '
        f'action (e.g. "grasp", "release") if one occurs. One short phrase.\n'
        f'3. outcome: final scene state at the END of THIS subgoal only. One sentence.\n'
        f'4. object_affordance: how the gripper interacts with the specific object — '
        f'grasp description, carry grip, or release point depending on gripper context. '
        f'One short phrase.\n\n'
        f'Respond only as JSON:\n'
        f'{{"spatial_command": "...", "low_level_motion": "...", '
        f'"outcome": "...", "object_affordance": "..."}}'
    )

    content = [{"type": "text", "text": prompt}]
    for idx, frame in subgoal_frames:
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(frame)}",
            "detail": "low",
        }})

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=512,
            )
            result = json.loads(resp.choices[0].message.content)
            assert all(k in result for k in (
                "spatial_command", "low_level_motion", "outcome", "object_affordance"
            ))
            return result
        except Exception as e:
            print(f"    GPT attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {"spatial_command": "", "low_level_motion": "", "outcome": "", "object_affordance": ""}


# ── Main logic ─────────────────────────────────────────────────────────────────

def find_parquet_for_task(dataset_path: str, task_index: int) -> str:
    parquets = sorted(glob.glob(f"{dataset_path}/data/chunk-*/*.parquet"))
    for p in parquets:
        df = pd.read_parquet(p, columns=["task_index"])
        if int(df["task_index"].iloc[0]) == task_index:
            return p
    raise ValueError(f"No episode found with task_index={task_index}")


def process_episode(
    client: OpenAI,
    results_json: Path,
    output_json: Path,
    dataset_path: str | None = None,
    rlds_dataset: str | None = None,
    rlds_data_dir: str | None = None,
) -> None:
    with open(results_json) as f:
        results = json.load(f)

    task_str = results["task"]
    subgoals = results["subgoals"]
    boundaries = results["boundaries"]

    # Determine mode: RLDS or parquet
    use_rlds = "episode_index" in results
    if use_rlds:
        ep_idx = results["episode_index"]
        print(f'  Episode [{ep_idx}]: "{task_str}"')
        frames, actions = load_episode_from_rlds(rlds_dataset, rlds_data_dir, ep_idx)
        id_key, id_val = "episode_index", ep_idx
    else:
        task_index = results["task_index"]
        print(f'  Task [{task_index}]: "{task_str}"')
        parquet_path = find_parquet_for_task(dataset_path, task_index)
        frames = load_frames_from_parquet(parquet_path)
        actions = load_actions_from_parquet(parquet_path)
        id_key, id_val = "task_index", task_index

    T = len(frames)

    steering_commands = []
    prev = 0
    gripper_closed = False  # cumulative gripper state; assume open at episode start
    for i, (subgoal, end) in enumerate(zip(subgoals, boundaries)):
        start = prev
        end_clamp = min(end, T - 1)

        # Compute exact motion deltas from actions
        motion = sum_actions_for_subgoal(actions, start, end_clamp)

        # Gripper start state from cumulative tracking (more reliable than per-subgoal detection)
        gripper_start = "closed" if gripper_closed else "open"

        # Gripper transition: prefer text-based inference, fall back to action-based
        gripper_transition = infer_gripper_transition_from_text(subgoal)
        if gripper_transition is None:
            gripper_transition = motion["gripper_transition"]

        # Update cumulative state for next subgoal
        if gripper_transition == "close gripper":
            gripper_closed = True
        elif gripper_transition == "open gripper":
            gripper_closed = False

        # Build motion_with_numbers now that gripper_transition is finalised
        parts = [motion["spatial_components"]] if motion["spatial_components"] != "hold position" else []
        if gripper_transition:
            parts.append(gripper_transition)
        motion_with_numbers = ", ".join(parts) if parts else "hold position"

        print(f"  SG{i} frames {start}–{end_clamp}: \"{subgoal}\"")

        # Skip subgoals with inverted boundaries (non-monotonic results.json)
        if start > end_clamp:
            print(f"    [skip] inverted frame range ({start} > {end_clamp})")
            prev = end_clamp + 1
            continue

        print(f"    motion_with_numbers: {motion_with_numbers}")
        print(f"    gripper: {_gripper_context_str(gripper_start, gripper_transition)}")

        # Ask GPT to generate spatial + motion descriptions
        subgoal_frames = sample_subgoal_frames(frames, start, end_clamp, n=N_FRAMES)
        gpt_result = generate_steering_command(
            client, task_str, subgoal,
            motion_with_numbers,
            gripper_transition,
            gripper_start,
            subgoal_frames,
        )

        steering_commands.append({
            "subgoal_index": i,
            "task": task_str,
            "subgoal": subgoal,
            "spatial_command": gpt_result["spatial_command"],
            "low_level_motion": gpt_result["low_level_motion"],
            "low_level_motion_with_numbers": motion_with_numbers,
            "outcome": gpt_result["outcome"],
            "object_affordance": gpt_result["object_affordance"],
            "action_deltas": {
                "dx_m": motion["dx_m"],
                "dy_m": motion["dy_m"],
                "dz_m": motion["dz_m"],
                "gripper_delta": motion["gripper_delta"],
            },
            "gripper_start_state": gripper_start,
            "gripper_transition": gripper_transition,
            "frame_range": [start, end_clamp],
        })

        print(f"    spatial_command:    {gpt_result['spatial_command']}")
        print(f"    low_level_motion:   {gpt_result['low_level_motion']}")
        print(f"    outcome:            {gpt_result['outcome']}")
        print(f"    object_affordance:  {gpt_result['object_affordance']}")
        prev = end + 1

    output = {
        id_key: id_val,
        "task": task_str,
        "subgoals": subgoals,
        "steering_commands": steering_commands,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=None,
                        help="Parquet dataset path (parquet mode only)")
    parser.add_argument("--rlds_dataset", default=None,
                        help="RLDS dataset name, e.g. libero_goal_no_noops")
    parser.add_argument("--rlds_data_dir", default=None,
                        help="RLDS data directory, e.g. /proj/inf-scaling/SWE_sim/modified_libero_rlds_new")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results_json", type=str,
                       help="Path to a single results.json")
    group.add_argument("--results_dir", type=str,
                       help="Directory containing results.json files (searched recursively)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output path (single-file mode only). "
                             "Defaults to steering_commands.json next to results.json.")
    args = parser.parse_args()

    client = OpenAI()

    kwargs = dict(
        dataset_path=args.dataset_path,
        rlds_dataset=args.rlds_dataset,
        rlds_data_dir=args.rlds_data_dir,
    )

    if args.results_json:
        results_path = Path(args.results_json)
        out_path = Path(args.output_json) if args.output_json else \
            results_path.parent / "steering_commands.json"
        print(f"\n── Processing {results_path} ──")
        process_episode(client, results_path, out_path, **kwargs)
    else:
        paths = sorted(Path(args.results_dir).rglob("results.json"))
        print(f"Found {len(paths)} results.json files")
        for i, results_path in enumerate(paths):
            out_path = results_path.parent / "steering_commands.json"
            print(f"\n[{i+1}/{len(paths)}] {results_path}")
            process_episode(client, results_path, out_path, **kwargs)


if __name__ == "__main__":
    main()

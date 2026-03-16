"""
generate_subgoal_variations.py

Given a results.json (from test_subgoal_pipeline.py), asks GPT to generate N
alternative phrasings for each subgoal. Writes an updated results.json with a
new field:

    "subgoal_variations": [
        ["canonical sg0", "alt phrasing 1", "alt phrasing 2", ...],
        ["canonical sg1", "alt phrasing 1", ...],
        ...
    ]

The canonical subgoal is always index 0 in each variation list.

Usage (single episode):
    python scripts/generate_subgoal_variations.py \
        --results_json outputs/subgoal_test_task12/results.json \
        --n_variations 4

Usage (whole directory of results.json files):
    python scripts/generate_subgoal_variations.py \
        --results_dir outputs/subgoal_results/ \
        --n_variations 4
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

MODEL = "gpt-4o"
MAX_RETRIES = 3


def generate_variations(
    client: OpenAI,
    task_str: str,
    subgoals: list[str],
    n_variations: int,
) -> list[list[str]]:
    """
    For each subgoal, generate n_variations alternative phrasings.
    Returns list[list[str]] where result[i][0] is the canonical subgoal.
    """
    subgoals_str = "\n".join(f"  {i}: \"{sg}\"" for i, sg in enumerate(subgoals))

    prompt = (
        f'You are helping train a robot to follow step-by-step instructions.\n'
        f'The full task is: "{task_str}"\n\n'
        f'The task has been decomposed into these subgoals:\n{subgoals_str}\n\n'
        f'For each subgoal, generate {n_variations} alternative phrasings that mean the same '
        f'thing but use different wording. Variations should:\n'
        f'- Describe the same robot motion/action\n'
        f'- Use natural language a human might say\n'
        f'- Vary in phrasing, verb choice, and specificity\n'
        f'- Be concise (under 12 words each)\n\n'
        f'Respond only as JSON:\n'
        f'{{"variations": [["alt1 for sg0", "alt2 for sg0", ...], ["alt1 for sg1", ...], ...]}}\n'
        f'Each inner list must have exactly {n_variations} entries.'
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
            variations = result["variations"]
            assert len(variations) == len(subgoals), (
                f"Expected {len(subgoals)} variation lists, got {len(variations)}"
            )
            for i, vlist in enumerate(variations):
                assert len(vlist) == n_variations, (
                    f"Subgoal {i}: expected {n_variations} variations, got {len(vlist)}"
                )
            # Prepend canonical as index 0
            return [[subgoals[i]] + variations[i] for i in range(len(subgoals))]
        except Exception as e:
            print(f"  Variation attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    print("  Falling back to canonical-only variations")
    return [[sg] for sg in subgoals]


def process_results_json(client: OpenAI, path: Path, n_variations: int) -> None:
    with open(path) as f:
        results = json.load(f)

    if "subgoal_variations" in results:
        print(f"  Already has variations, skipping: {path}")
        return

    task_str = results["task"]
    subgoals = results["subgoals"]
    print(f"  Task: \"{task_str}\"")
    print(f"  Subgoals: {subgoals}")

    variations = generate_variations(client, task_str, subgoals, n_variations)
    results["subgoal_variations"] = variations

    print(f"  Generated variations:")
    for i, vlist in enumerate(variations):
        print(f"    SG{i}: {vlist}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results_json", type=str,
                       help="Path to a single results.json file")
    group.add_argument("--results_dir", type=str,
                       help="Directory containing results.json files (searched recursively)")
    parser.add_argument("--n_variations", type=int, default=4,
                        help="Number of alternative phrasings per subgoal (default: 4)")
    args = parser.parse_args()

    client = OpenAI()

    if args.results_json:
        paths = [Path(args.results_json)]
    else:
        paths = sorted(Path(args.results_dir).rglob("results.json"))
        print(f"Found {len(paths)} results.json files")

    for i, path in enumerate(paths):
        print(f"\n[{i+1}/{len(paths)}] {path}")
        process_results_json(client, path, args.n_variations)


if __name__ == "__main__":
    main()

"""Convert handover_block RoboTwin HDF5 endpose demos to BiRelPoseDP .npz.

Dry-run mode inspects HDF5 structure and refuses to fake object poses if no
explicit per-frame object pose key is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

POSEINSERT_ROOT = Path(__file__).resolve().parents[1]
if str(POSEINSERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEINSERT_ROOT))

from robotwin_birelpose.conversion import (
    convert_episode_to_birelpose,
    find_hdf5_files,
    find_object_pose_candidates,
    list_hdf5_datasets,
    load_object_pose_sequence,
    save_converted_episode,
)
from robotwin_birelpose.hdf5_endpose_reader import ENDPOSE_DATASETS, load_endpose_episode


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", default="handover_block")
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--object_pose_key", default=None)
    parser.add_argument("--target_pose_key", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _print_episode_inspection(hdf5_path: Path):
    print(f"episode: {hdf5_path}")
    datasets = list_hdf5_datasets(hdf5_path)
    root_keys = sorted({info.key.strip("/").split("/")[0] for info in datasets})
    print(f"root keys: {root_keys}")
    episode = load_endpose_episode(hdf5_path)
    print("endpose fields:")
    print(f"  {ENDPOSE_DATASETS['left_ee_pose']}: {episode.left_ee_pose.shape} {episode.left_ee_pose.dtype}")
    print(f"  {ENDPOSE_DATASETS['right_ee_pose']}: {episode.right_ee_pose.shape} {episode.right_ee_pose.dtype}")
    print(f"  {ENDPOSE_DATASETS['left_gripper']}: {episode.left_gripper.shape} {episode.left_gripper.dtype}")
    print(f"  {ENDPOSE_DATASETS['right_gripper']}: {episode.right_gripper.shape} {episode.right_gripper.dtype}")
    print("first frame:")
    print(f"  left_endpose={episode.left_ee_pose[0]}")
    print(f"  right_endpose={episode.right_ee_pose[0]}")
    print(f"  left_gripper={episode.left_gripper[0]}")
    print(f"  right_gripper={episode.right_gripper[0]}")

    candidates = find_object_pose_candidates(hdf5_path, expected_frames=episode.num_frames)
    print(f"object pose candidates with shape (T, 7): {len(candidates)}")
    for candidate in candidates:
        print(f"  {candidate.key}: shape={candidate.shape}, dtype={candidate.dtype}")
    return episode, candidates


def main():
    args = _parse_args()
    hdf5_files = find_hdf5_files(args.input_dir, max_episodes=args.max_episodes)
    print(f"input_dir: {args.input_dir}")
    print(f"found_hdf5: {len(hdf5_files)}")
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.input_dir}")

    wrote = 0
    for hdf5_path in hdf5_files:
        episode, candidates = _print_episode_inspection(hdf5_path)

        if args.object_pose_key is None:
            print(
                "HDF5 has no confirmed object_pose key. Need replay extractor or "
                "recollect demos with object pose. Use --object_pose_key only after "
                "confirming a real per-frame object pose dataset."
            )
            if not args.dry_run:
                raise RuntimeError("Refusing to convert without --object_pose_key")
            continue

        object_pose = load_object_pose_sequence(hdf5_path, args.object_pose_key)
        if object_pose.shape[0] != episode.num_frames:
            raise ValueError(
                f"object pose frame count {object_pose.shape[0]} does not match endpose frame count {episode.num_frames}"
            )
        target_pose = None
        if args.target_pose_key is not None:
            target_pose = load_object_pose_sequence(hdf5_path, args.target_pose_key)
            if target_pose.shape[0] != episode.num_frames:
                raise ValueError(
                    f"target pose frame count {target_pose.shape[0]} does not match endpose frame count {episode.num_frames}"
                )
        if args.dry_run:
            print(f"dry_run: would convert using object_pose_key={args.object_pose_key}")
            if args.target_pose_key is not None:
                print(f"dry_run: would load target_pose_key={args.target_pose_key}")
            continue

        converted = convert_episode_to_birelpose(
            hdf5_path,
            object_pose=object_pose,
            task_name=args.task_name,
            target_pose=target_pose,
        )
        output_path = Path(args.output_dir) / f"{hdf5_path.stem}.npz"
        save_converted_episode(output_path, converted)
        print(f"wrote: {output_path}")
        print(f"  rel_obs: {converted['rel_obs'].shape}")
        print(f"  action_ee: {converted['action_ee'].shape}")
        print(f"  object_pose: {converted['object_pose'].shape}")
        wrote += 1

    print(f"converted_episodes: {wrote}")


if __name__ == "__main__":
    main()

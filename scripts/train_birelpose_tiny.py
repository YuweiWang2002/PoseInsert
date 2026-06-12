import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotwin_birelpose.dataset import BiRelPoseNpzDataset, compute_normalizer, normalize_batch
from robotwin_birelpose.model import BiRelPoseDiffusionPolicy, BiRelPoseMlpPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(items):
    return {
        "obs": np.stack([item["obs"] for item in items], axis=0),
        "action": np.stack([item["action"] for item in items], axis=0),
        "episode_id": np.asarray([item["episode_id"] for item in items], dtype=np.int64),
        "t": np.asarray([item["t"] for item in items], dtype=np.int64),
    }


def make_model(args):
    if args.model_type == "diffusion":
        return BiRelPoseDiffusionPolicy(
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            obs_feature_dim=args.obs_feature_dim,
            hidden_dim=args.hidden_dim,
            down_dims=tuple(args.down_dims),
            num_inference_steps=args.num_inference_steps,
            prediction=args.prediction,
        )
    if args.model_type == "mlp":
        return BiRelPoseMlpPolicy(
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            hidden_dim=args.hidden_dim,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--obs_key", default="rel_obs")
    parser.add_argument("--action_key", default="action_ee")
    parser.add_argument("--obs_dim", type=int, default=29)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--model_type", choices=["diffusion", "mlp"], default="diffusion")
    parser.add_argument("--obs_feature_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--down_dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--prediction", choices=["sample", "epsilon"], default="sample")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)

    dataset = BiRelPoseNpzDataset(
        args.data_dir,
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        obs_key=args.obs_key,
        action_key=args.action_key,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
    normalizer = compute_normalizer(dataset)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.num_steps * args.batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
    )

    print(f"data_dir: {args.data_dir}")
    print(f"num_episodes: {len(dataset.episodes)}")
    print(f"num_samples: {len(dataset)}")
    print(f"obs_dim: {args.obs_dim}")
    print(f"action_dim: {args.action_dim}")
    print(f"obs_key: {args.obs_key}")
    print(f"action_key: {args.action_key}")
    print(f"horizon: {args.horizon}")
    print(f"n_obs_steps: {args.n_obs_steps}")
    print(f"model_type: {args.model_type}")
    print(f"device: {device}")

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    losses = []
    model.train()
    for step, batch in enumerate(dataloader, start=1):
        batch = normalize_batch(batch, normalizer, to_torch=True)
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)

        loss = model.forward_loss(obs, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.num_steps:
            print(f"step {step:04d} loss {loss_value:.6f}")
        if step >= args.num_steps:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "latest.pt"
    config = vars(args).copy()
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": normalizer,
            "config": config,
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "horizon": args.horizon,
            "n_obs_steps": args.n_obs_steps,
            "losses": losses,
        },
        ckpt_path,
    )
    print(f"initial_loss: {losses[0]:.6f}")
    print(f"final_loss: {losses[-1]:.6f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    main()

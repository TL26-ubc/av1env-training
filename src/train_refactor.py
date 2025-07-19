import argparse
from pathlib import Path
import math

from av1gym.environment import Av1GymEnv, Av1GymObsNormWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from wandb_callback import WandbCallback
import wandb
from av1gym.environment.actorcritic import SBGlobalActorCriticPolicy


def exponential_lr_schedule(initial_lr: float, decay_rate: float = 0.96):
    """
    Exponential learning rate scheduler function for Stable Baselines3.
    
    Args:
        initial_lr: Starting learning rate
        decay_rate: Exponential decay rate (< 1.0 for decay)
    
    Returns:
        Callable that takes progress and returns current learning rate
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        We want lr to decrease as progress_remaining decreases.
        """
        progress_used = 1.0 - progress_remaining  # 0 to 1 as training progresses
        current_lr = initial_lr * (decay_rate ** progress_used)
        return current_lr
    return schedule
    
def prase_arg():
    parser = argparse.ArgumentParser(description="Train RL agent for video encoding")

    # Video and output
    parser.add_argument(
        "--file", help="Input video file", default="Data/akiyo_qcif.y4m"
    )
    parser.add_argument(
        "--output_dir", default="logs/", help="Output directory for models and logs"
    )

    # RL parameters
    parser.add_argument(
        "--algorithm", choices=["ppo", "dqn"], default="ppo", help="RL algorithm to use"
    )
    parser.add_argument(
        "--total_iteration", type=int, default=50, help="Total training loop iterations, number of times the environment is reset"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.96, help="Exponential learning rate decay rate (< 1.0 for decay)"
    )
    parser.add_argument("--batch_size", type=int, default=65, help="Batch size")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="Number of steps per update (PPO only), should match number of frames in the video, -1 would search for the video length",
    )

    # Environment parameters
    parser.add_argument(
        "--lambda_rd", type=float, default=0.1, help="Rate-distortion lambda"
    )
    parser.add_argument(
        "--max_frames", type=int, default=100, help="Maximum frames per episode"
    )

    # Training parameters
    parser.add_argument(
        "--eval_freq", type=int, default=5000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10000, help="Model save frequency"
    )

    parser.add_argument(
        "--disable_observation_normalization", 
        action="store_true", 
        help="Disable observation state normalization"
    )
    
    parser.add_argument(
        "--wandb",
        help="enable wandb logging, put any value here to enable",
        default=None,
    )
    
    parser.add_argument(
        "--tbr", type=int, default=1000, help="Target bitrate for the original encoder"
    )
    
    parser.add_argument(
        "--video_name", type=str, default="default_video_name", help="Name of the video being encoded"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = prase_arg()
    # Initialize wandb
    if args.wandb is not None:
        wandb.init(
            project="av1-video-encoding",
            config={
                "algorithm": args.algorithm,
                "learning_rate": args.learning_rate,
                "lr_decay_rate": args.lr_decay_rate,
                "batch_size": args.batch_size,
                "n_steps": args.n_steps,
                "lambda_rd": args.lambda_rd,
                "total_iteration": args.total_iteration,
                "video_file": args.file,
                "video_name": args.video_name,
            },
            name=f"{args.video_name}_tbr_{args.tbr}",
        )
        wandb_callback = WandbCallback()
    else:
        wandb_callback = None

    # create envirnment
    base_output_path = Path(args.output_dir)
    gym_env = Av1GymEnv(
        video_path=args.file,
        output_dir=str(base_output_path),
        lambda_rd=args.lambda_rd,
        tbr=args.tbr,
    )
    wrapped_gym_env = Av1GymObsNormWrapper(gym_env)
    
    env = Monitor(wrapped_gym_env, str(base_output_path / "monitor"))
    if args.n_steps == -1:
        # Automatically determine n_steps based on video length
        video_length = wrapped_gym_env.env.num_frames
        args.n_steps = video_length if video_length > 0 else 1000  # Fallback to 1000 if length is unknown

    model = None
    if args.algorithm.lower() != "ppo":
         raise ValueError("Only PPO is supported with the per-SB actor head")

    # Standard SB3 PPO; we pass the *module instance* as the policy
    model = PPO(
        policy=SBGlobalActorCriticPolicy,
        env=env,
        learning_rate=exponential_lr_schedule(args.learning_rate, args.lr_decay_rate),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        verbose=1,
        device="auto",
        policy_kwargs={
            # Model architecture configuration - edit these directly for different model sizes
            "sb_channels": 128,
            "glob_layers": [128, 128, 64],       # 2 layers: input → 128 → 64
            "value_layers": [256, 128, 1],  # 2 hidden layers: input → 128 → 1
            
            # Examples for different model sizes:
            # Small model:  sb_channels=32,  glob_layers=[32, 32],    value_layers=[64, 1]
            # Large model:  sb_channels=128, glob_layers=[128, 128], value_layers=[256, 256, 1]
            # Deep model:   sb_channels=64,  glob_layers=[64, 64, 64, 64], value_layers=[128, 128, 128, 1]
        },
    )
        
    total_timesteps = args.total_iteration * wrapped_gym_env.env.num_frames
    
    # training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=wandb_callback,
            tb_log_name=f"{args.algorithm}_run",
        )

        # Save final model
        final_model_path = base_output_path / f"final_{args.algorithm}_model"
        model.save(str(final_model_path))
        wrapped_gym_env.env.save_bitstream_to_file(
            str(base_output_path / "final_encoder_video.ivf")
        )
        print(f"Training completed! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current model
        interrupted_model_path = (
            base_output_path / f"interrupted_{args.algorithm}_model"
        )
        model.save(str(interrupted_model_path))
        wrapped_gym_env.env.save_bitstream_to_file(
            str(base_output_path / "interrupted_encoder_video.ivf"),
        )
        print(f"Model saved to: {interrupted_model_path}")
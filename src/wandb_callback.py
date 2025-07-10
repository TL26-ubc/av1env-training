from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_psnr_y = []
        self.episode_psnr_cb = []
        self.episode_psnr_cr = []
        self.episode_bitrates = []

    def _on_step(self) -> bool:
        # Log step-level metrics
        step_metrics = {}
        
        # Add model training metrics if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float, np.number)):
                        step_metrics[f"train/{key}"] = value
        
        # Add learning rate
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                step_metrics["train/learning_rate"] = self.model.learning_rate(self.model._current_progress_remaining)
            else:
                step_metrics["train/learning_rate"] = self.model.learning_rate
        
        # Log step metrics if any
        if step_metrics:
            wandb.log(step_metrics)
        
        self.current_episode_reward += self.locals.get('rewards', [0])[-1]
        self.current_episode_length += 1
        
        # Get PSNR values and bitrate from environment info
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'y_psnr' in info:
                    self.episode_psnr_y.append(info['y_psnr'])
                if 'cb_psnr' in info:
                    self.episode_psnr_cb.append(info['cb_psnr'])
                if 'cr_psnr' in info:
                    self.episode_psnr_cr.append(info['cr_psnr'])
                if 'bitstream_size' in info:
                    self.episode_bitrates.append(info['bitstream_size'])
        
        # Check if episode is done
        if self.locals.get('dones', [False])[-1]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode metrics
            episode_metrics = {
                "episode/reward": self.current_episode_reward,
                "episode/length": self.current_episode_length,
                "episode/number": len(self.episode_rewards),
                "episode/mean_y_psnr": np.mean(self.episode_psnr_y) if self.episode_psnr_y else 0,
                "episode/mean_cb_psnr": np.mean(self.episode_psnr_cb) if self.episode_psnr_cb else 0,
                "episode/mean_cr_psnr": np.mean(self.episode_psnr_cr) if self.episode_psnr_cr else 0,
                "episode/mean_bitrate": np.mean(self.episode_bitrates) if self.episode_bitrates else 0,
                "episode/total_bitrate": np.sum(self.episode_bitrates) if self.episode_bitrates else 0,
            }
            
            wandb.log(episode_metrics)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_psnr_y = []
            self.episode_psnr_cb = []
            self.episode_psnr_cr = []
            self.episode_bitrates = []
            
        return True

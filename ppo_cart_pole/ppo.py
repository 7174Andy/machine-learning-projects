import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import hydra
from omegaconf import DictConfig
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import layer_init
from config import Args
from tqdm import trange



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def plot_metrics(metrics, run_name):
    """Plot training metrics and save to file.

    Args:
        metrics: dict of metric_name -> list of values
        run_name: string used for the output filename

    Available metrics in the dict:
        - global_step: x-axis values (timesteps)
        - learning_rate, value_loss, policy_loss, entropy
        - approx_kl, clipfrac, explained_variance, SPS
    """
    steps = metrics["global_step"]
    plot_groups = [
        ("Value Loss", ["value_loss"]),
        ("Policy Loss", ["policy_loss"]),
        ("Entropy", ["entropy"]),
        ("Approx KL", ["approx_kl"]),
        ("Clip Fraction", ["clipfrac"]),
        ("Explained Variance", ["explained_variance"]),
        ("Learning Rate", ["learning_rate"]),
        ("SPS", ["SPS"]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (title, keys) in zip(axes.flat, plot_groups):
        for key in keys:
            ax.plot(steps, metrics[key], label=key)
        ax.set_title(title)
        ax.set_xlabel("global_step")
        if len(keys) > 1:
            ax.legend()

    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(f"{run_name}_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved metrics plot to {run_name}_metrics.png")


def evaluate(agent: Agent, run_name, device, num_episodes=5):
    """Record evaluation episodes with the trained agent.

    Args:
        agent: trained Agent network
        run_name: used for video output directory
        device: torch device
        num_episodes: number of episodes to record
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}_eval", episode_trigger=lambda x: True)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    for ep in trange(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=1).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        ep_return = info["episode"]["r"]
        ep_length = info["episode"]["l"]
        print(f"eval episode {ep}: return={ep_return}, length={ep_length}")

    env.close()


@hydra.main(config_name="config", version_base=None)
def main(cfg: DictConfig):
    args: Args = Args(**cfg)

    # Compute runtime values
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}__{args.seed}"

    def make_env(idx):
        def thunk():
            env = gym.make("CartPole-v1", render_mode="rgb_array" if args.capture_video else None)
            if args.capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.NormalizeReward(env)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent: Agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Metrics tracking
    metrics = defaultdict(list)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iter in trange(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iter - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = log_prob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record metrics for plotting
        sps = int(global_step / (time.time() - start_time))
        metrics["global_step"].append(global_step)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
        metrics["value_loss"].append(v_loss.item())
        metrics["policy_loss"].append(pg_loss.item())
        metrics["entropy"].append(entropy_loss.item())
        metrics["approx_kl"].append(approx_kl.item())
        metrics["clipfrac"].append(np.mean(clipfracs))
        metrics["explained_variance"].append(explained_var)
        metrics["SPS"].append(sps)

    envs.close()
    plot_metrics(metrics, run_name)
    evaluate(agent, run_name, device)


if __name__ == "__main__":
    main()

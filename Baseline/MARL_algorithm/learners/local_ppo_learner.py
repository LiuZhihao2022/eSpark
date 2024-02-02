import torch
from torch.optim import Adam

from components.action_selectors import categorical_entropy
from components.episode_buffer import EpisodeBatch
from modules.critics import REGISTRY as critic_resigtry
from utils.rl_utils import build_gae_targets, build_gae_targets_with_T
from utils.value_norm import ValueNorm
import pickle
import wandb
import os

class LocalPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.opt_load_path = None
        # a trick to reuse mac
        # dummy_args = copy.deepcopy(args)
        # dummy_args.n_actions = 1
        # self.critic = NMAC(scheme, None, dummy_args)
        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.params = list(self.mac.parameters()) + list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr

        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.args.device)
        self.training_curve_log = []

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, current_iter: str = ""):
        # Get the relevant quantities
        if self.args.use_individual_rewards:
            rewards = batch["individual_rewards"][:, :-1].to(batch.device)
        else:
            rewards = (
                batch["reward"][:, :-1]
                .to(batch.device)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )
            if self.args.use_mean_team_reward:
                rewards /= self.n_agents
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = torch.log(torch.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # targets and advantages
        with torch.no_grad():
            if "rnn" in self.args.critic_type:
                old_values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = self.critic.forward(batch, t=t)
                    old_values.append(agent_outs)
                old_values = torch.stack(old_values, dim=1)
            else:
                old_values = self.critic(batch)

            if self.use_value_norm:
                value_shape = old_values.shape
                values = self.value_norm.denormalize(old_values.view(-1)).view(
                    value_shape
                )
            else:
                values = old_values
            
            advantages, targets = build_gae_targets(
                rewards * 100, #.unsqueeze(2).repeat(1, 1, self.n_agents, 1),
                mask_agent,
                values,
                self.args.gamma,
                self.args.gae_lambda,
            )
        advantage_normalization = self.args.env_args.get("advantage_normalization", True)
        if advantage_normalization is False:
            normed_advantages = advantages
            # normed_advantages = rewards
        else:
            normed_advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-6)

        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Critic
            if "rnn" in self.args.critic_type:
                values = []
                self.critic.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    agent_outs = self.critic.forward(batch, t=t)
                    values.append(agent_outs)
                values = torch.stack(values, dim=1)
            else:
                values = self.critic(batch)[:, :-1]

            # # value clip
            # values_clipped = old_values[:, :-1] + (values - old_values[:, :-1]).clamp(
            #     -self.args.eps_clip, self.args.eps_clip
            # )

            # # 0-out the targets that came from padded data
            # td_error = torch.max(
            #     (values - targets.detach()) ** 2,
            #     (values_clipped - targets.detach()) ** 2,
            # )
            
            td_error = (values - targets.detach()) ** 2
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t, t_env)
                pi.append(agent_outs)
            pi = torch.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = torch.gather(pi, dim=3, index=actions)
            log_pi_taken = torch.log(pi_taken)

            ratios = torch.exp(log_pi_taken - old_logprob) 
            surr1 = ratios * normed_advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * normed_advantages
            )
            actor_loss = (
                -(torch.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            )

            # entropy
            entropy_loss = categorical_entropy(pi).mean(
                -1, keepdim=True
            )  # mean over agents
            entropy_loss[mask == 0] = 0  # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()

            
            loss = (
                actor_loss
                + self.args.critic_coef * critic_loss
                - self.args.entropy_coef * entropy_loss
            )

            # Optimise agents
            # TODO:optimizer的梯度有问题，有cpu和gpt两个设备
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params, self.args.grad_norm_clip
            )
            self.optimiser.step()

            if _ == self.args.mini_epochs - 1:
                if "gpt" not in self.args.runner:
                    if self.args.use_wandb:
                        wandb.log({'loss': loss.item(),
                                'actor_loss': actor_loss.item(),
                                'critic_loss': critic_loss.item(),
                                'values': values.mean().item(),
                                'targets': targets.mean().item(),
                                'reward': rewards.mean().detach().cpu().numpy(),
                                'advantage': normed_advantages.mean().item()
                                })
                else:
                    self.training_curve_log.append({'loss': loss.item(),
                            'actor_loss': actor_loss.item(),
                            'critic_loss': critic_loss.item(),
                            'values': values.mean().item(),
                            'targets': targets.mean().item(),
                            'reward': rewards.mean().detach().cpu().numpy(),
                            'advantage': normed_advantages.mean().item()
                            })

    def cuda(self, gpu_id = 0):
        self.mac.cuda(gpu_id)
        self.critic.cuda(gpu_id)
        self.params = list(self.mac.parameters()) + list(self.critic.parameters())
        self.optimiser = Adam(params=self.params, lr=self.args.lr) #TODO:这里一定要重新加载！不然会在两个设备上发现参数！
        if self.opt_load_path is not None:
            self.optimiser.load_state_dict(
                        torch.load(
                            self.opt_load_path,
                            map_location=lambda storage, loc: storage,
                        )
                    )


    def save_models(self, path, postfix = ""):
        # TODO:learner的一切都要保存，actor，critic，optimizer，还有各个属性。
        self.mac.save_models(path, postfix)
        self.critic.save_models(path, postfix)
        torch.save(self.optimiser.state_dict(), "{}/agent_opt".format(path)+postfix+".th")
        pkl_path = os.path.join(path, "property.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump((self.last_target_update_step, self.critic_training_steps, self.last_lr), f)



    def load_models(self, path, postfix = ""):
        self.mac.load_models(path, postfix)
        # Not quite right but I don't want to save target networks
        # 优化器需要在模型参数加载时候，由模型参数重新创建优化器，随后再加载优化器参数
        self.critic.load_models(path, postfix)
        self.params = list(self.mac.parameters()) + list(self.critic.parameters())
        self.opt_load_path = "{}/agent_opt".format(path)+postfix+".th"
        self.optimiser.load_state_dict(
            torch.load(
                "{}/agent_opt".format(path)+postfix+".th",
                map_location=lambda storage, loc: storage,
            )
        )
        if "property.pkl" in os.listdir(path):
            pkl_path = os.path.join(path, "property.pkl")
            with open(pkl_path, "rb") as f:
                self.last_target_update_step, self.critic_training_steps, self.last_lr = pickle.load(f)
                
        
    def load_other_learner(self, other_learner):
        self.args = other_learner.args
        self.n_agents = other_learner.args.n_agents
        self.n_actions = other_learner.args.n_actions
        self.mac.load_state(other_learner.mac)
        self.logger = other_learner.logger

        self.last_target_update_step = other_learner.last_target_update_step
        self.critic_training_steps = other_learner.critic_training_steps

        self.log_stats_t = other_learner.log_stats_t

        # a trick to reuse mac
        # dummy_args = copy.deepcopy(args)
        # dummy_args.n_actions = 1
        # self.critic = NMAC(scheme, None, dummy_args)
        self.critic.load_state_dict(other_learner.critic.state_dict())
        self.params = list(self.mac.parameters()) + list(self.critic.parameters())

        self.optimiser.load_state_dict(other_learner.optimiser.state_dict())
        self.last_lr = other_learner.last_lr
        
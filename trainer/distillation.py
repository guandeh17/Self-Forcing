import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Lookahead Forcing: head/fusion params live in their own optimizer so
        # they can also be stepped on critic steps (free drafter data,
        # mtp.md section 4); the generator optimizer excludes them.
        lookahead_cfg = getattr(config, "lookahead", None)
        self.lookahead_enabled = bool(lookahead_cfg and lookahead_cfg.get("enabled", False))
        self.generator_optimizer = torch.optim.AdamW(
            [param for name, param in self.model.generator.named_parameters()
             if param.requires_grad and "lookahead" not in name],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.lookahead_optimizer = None
        if self.lookahead_enabled:
            lookahead_params = [
                param for name, param in self.model.generator.named_parameters()
                if param.requires_grad and "lookahead" in name]
            assert len(lookahead_params) > 0, "lookahead enabled but no lookahead params found"
            self.lookahead_optimizer = torch.optim.AdamW(
                lookahead_params,
                lr=lookahead_cfg.get("head_lr", None) or config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
            # lambda state for the grad-norm-ratio controller (+ warmup)
            self.lookahead_lambda = float(lookahead_cfg.get("lambda_init", 0.25))
            self.lookahead_warmup_steps = int(lookahead_cfg.get("lambda_warmup_steps", 1500))
            ratio_lo, ratio_hi = lookahead_cfg.get("grad_ratio_target", [0.10, 0.25])
            self.lookahead_ratio_lo, self.lookahead_ratio_hi = float(ratio_lo), float(ratio_hi)
            self.lookahead_ratio_interval = int(lookahead_cfg.get("grad_ratio_interval", 100))
            self.lookahead_gen_updates = 0

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            # strict=False so a lookahead-augmented generator can load the
            # released (head-less) checkpoint; assert nothing else is off
            missing_keys, unexpected_keys = self.model.generator.load_state_dict(
                state_dict, strict=False
            )
            assert len(unexpected_keys) == 0, \
                f"unexpected keys in generator checkpoint: {unexpected_keys}"
            non_lookahead_missing = [k for k in missing_keys if "lookahead" not in k]
            assert len(non_lookahead_missing) == 0, \
                f"missing non-lookahead keys in generator checkpoint: {non_lookahead_missing}"

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _generator_backbone_grad_norm(self):
        """Global L2 norm of non-lookahead generator grads (FSDP-sharded)."""
        sq = torch.zeros(1, device=self.device, dtype=torch.float32)
        for name, param in self.model.generator.named_parameters():
            if param.grad is not None and "lookahead" not in name:
                sq += param.grad.detach().float().pow(2).sum()
        dist.all_reduce(sq)
        return sq.sqrt().item()

    def _compose_lookahead_loss(self, lookahead_losses, lam):
        """Split LSC terms by tap source: exit terms are lambda-weighted (they
        carry backbone gradient via h_fuse); context terms train heads only."""
        zero = torch.zeros((), device=self.device)
        la_exit = sum((v for k, v in lookahead_losses.items() if k.endswith("_exit")), zero)
        la_context = sum((v for k, v in lookahead_losses.items() if k.endswith("_context")), zero)
        return la_exit, la_context, lam * la_exit + la_context

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            lookahead_losses = generator_log_dict.pop("lookahead_losses", None)
            if lookahead_losses:
                self.lookahead_gen_updates += 1
                warmup = min(1.0, self.lookahead_gen_updates / max(self.lookahead_warmup_steps, 1))
                lam = self.lookahead_lambda * warmup
                la_exit, la_context, lookahead_total = self._compose_lookahead_loss(
                    lookahead_losses, lam)

                measure = (
                    torch.is_tensor(la_exit) and la_exit.requires_grad
                    and self.lookahead_gen_updates % self.lookahead_ratio_interval == 0
                )
                if measure:
                    # grad-norm-ratio controller (mtp.md section 4): probe the
                    # backbone grad norms of the two terms separately, adjust
                    # lambda multiplicatively, then accumulate the true total.
                    (lam * la_exit).backward(retain_graph=True)
                    norm_lsc = self._generator_backbone_grad_norm()
                    self.generator_optimizer.zero_grad(set_to_none=True)
                    self.lookahead_optimizer.zero_grad(set_to_none=True)
                    generator_loss.backward(retain_graph=True)
                    norm_dmd = self._generator_backbone_grad_norm()
                    lookahead_total.backward()
                    ratio = norm_lsc / (norm_dmd + 1e-8)
                    if ratio > self.lookahead_ratio_hi:
                        self.lookahead_lambda /= 1.5
                    elif ratio < self.lookahead_ratio_lo:
                        self.lookahead_lambda *= 1.5
                    generator_log_dict["lookahead_grad_ratio"] = torch.tensor(ratio)
                else:
                    (generator_loss + lookahead_total).backward()

                generator_log_dict.update({
                    "lookahead_loss_exit": la_exit.detach() if torch.is_tensor(la_exit) else la_exit,
                    "lookahead_loss_context": la_context.detach() if torch.is_tensor(la_context) else la_context,
                    "lookahead_lambda": torch.tensor(lam),
                })
            else:
                generator_loss.backward()

            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        # Lookahead: head-only update from the critic-step rollout (features
        # are detached, so no backbone/critic interference)
        lookahead_losses = critic_log_dict.pop("lookahead_losses", None)
        if lookahead_losses:
            _, _, lookahead_total = self._compose_lookahead_loss(
                lookahead_losses, self.lookahead_lambda)
            if torch.is_tensor(lookahead_total) and lookahead_total.requires_grad:
                lookahead_total.backward()
                critic_log_dict["lookahead_loss_critic_step"] = lookahead_total.detach()

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                if self.lookahead_optimizer is not None:
                    self.lookahead_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.lookahead_optimizer is not None:
                    self.lookahead_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            if self.lookahead_optimizer is not None:
                self.lookahead_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()
            if self.lookahead_optimizer is not None:
                self.lookahead_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    for key in ("lookahead_loss_exit", "lookahead_loss_context",
                                "lookahead_lambda", "lookahead_grad_ratio"):
                        if key in generator_log_dict:
                            wandb_loss_dict[key] = generator_log_dict[key].mean().item()
                if "lookahead_loss_critic_step" in critic_log_dict:
                    wandb_loss_dict["lookahead_loss_critic_step"] = \
                        critic_log_dict["lookahead_loss_critic_step"].mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

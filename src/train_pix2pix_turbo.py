import os
import gc
import shutil
import glob
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from huggingface_hub import HfApi
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset


def main(args):
    config = ProjectConfiguration(project_dir=args.output_dir, total_limit=1)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=config,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir,
                    "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type=args.gan_loss_type, device=str(accelerator.device))
    else:
        raise NotImplementedError(
            f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.to(accelerator.device)
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
    net_clip, _ = clip.load("ViT-B/32", device=accelerator.device)
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(
                                      args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
                                       betas=(
                                           args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder,
                                  image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_folder=args.dataset_folder,
                                image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0)

    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)

    t_clip_renorm = transforms.Normalize(mean=(
        0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(dtype=weight_dtype)
    net_disc.to(dtype=weight_dtype)
    net_lpips.to(dtype=weight_dtype)
    net_clip.to(dtype=weight_dtype)

    for p in layers_to_opt:
        p.data = p.data.to(torch.float32)

    for p in net_disc.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name, config=tracker_config)

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", str(
            accelerator.device), use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)
        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                                        shuffle=False, seed=0, batch_size=8, device=accelerator.device,
                                        mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    # ==========================================
    # CHECKPOINT RESUMING LOGIC
    # ==========================================
    global_step = 0
    initial_epoch = 0
    resume_step = 0

    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)

    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(checkpoint_dir, path)

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            accelerator.print(f"Resuming from checkpoint: {path}")
            accelerator.load_state(path)
            global_step = int(os.path.basename(path).split("-")[1])
            steps_per_epoch = len(dl_train)
            initial_epoch = global_step // steps_per_epoch
            resume_step = global_step % steps_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process,)

    # ==========================================
    # START THE TRAINING LOOP
    # ==========================================
    for epoch in range(initial_epoch, args.num_training_epochs):
        active_dataloader = dl_train
        if resume_from_checkpoint and epoch == initial_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(
                dl_train, resume_step)

        for step, batch in enumerate(active_dataloader):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape

                x_tgt_pred = net_pix2pix(
                    x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                loss_l2 = F.mse_loss(
                    x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(
                    x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(
                        batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                x_tgt_pred = net_pix2pix(
                    x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                lossG = net_disc(
                    x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                lossD_real = net_disc(
                    x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                lossD_fake = net_disc(
                    x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step >= args.max_train_steps:
                    accelerator.print(
                        f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
                    break

                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # ==========================================
                    # NEW: AGGRESSIVE LOCAL & REMOTE CHECKPOINTING
                    # ==========================================
                    is_main_save = (global_step %
                                    args.checkpointing_steps == 1)
                    is_intermediate_save = (global_step % args.checkpointing_steps == (
                        args.checkpointing_steps - 100))

                    if is_main_save or is_intermediate_save:
                        outf = os.path.join(
                            args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                        save_path = os.path.join(
                            args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(
                            f"\n--- Saved full training state to {save_path} ---")

                        try:
                            import time
                            print(
                                f"Starting Hugging Face backup for step {global_step}...")

                            # --- FIX: Safe Environment Variable Pull ---
                            hf_token = os.environ.get("HF_TOKEN")
                            if not hf_token:
                                print(
                                    "WARNING: HF_TOKEN environment variable not found. Skipping HF push.")
                            else:
                                api = HfApi(token=hf_token)
                                repo_id = os.environ.get(
                                    "HF_REPO_ID", "PurveshG/img2img-turbo-checkpoints")
                                api.create_repo(
                                    repo_id=repo_id, private=True, exist_ok=True)

                                for attempt in range(3):
                                    try:
                                        api.upload_folder(
                                            folder_path=save_path,
                                            path_in_repo=f"checkpoint-{global_step}",
                                            repo_id=repo_id,
                                            repo_type="model",
                                            commit_message=f"Upload checkpoint {global_step} - attempt {attempt+1}"
                                        )
                                        print(
                                            f"✅ Successfully uploaded checkpoint-{global_step} to HF!")
                                        break
                                    except Exception as e:
                                        print(
                                            f"⚠️ Folder upload attempt {attempt+1} failed: {e}")
                                        if attempt == 2:
                                            raise e
                                        time.sleep(15)

                                for attempt in range(3):
                                    try:
                                        api.upload_file(
                                            path_or_fileobj=outf,
                                            path_in_repo=f"model_{global_step}.pkl",
                                            repo_id=repo_id,
                                            repo_type="model",
                                            commit_message=f"Upload pkl {global_step} - attempt {attempt+1}"
                                        )
                                        print(
                                            f"✅ Successfully uploaded model_{global_step}.pkl to HF!")
                                        break
                                    except Exception as e:
                                        print(
                                            f"⚠️ File upload attempt {attempt+1} failed: {e}")
                                        if attempt == 2:
                                            raise e
                                        time.sleep(15)

                        except Exception as e:
                            print(
                                f"WARNING: Hugging Face backup completely failed! Error: {e}")

                        # ----------------------------------------------------
                        # 4. EXPLICIT LOCAL CLEANUP (Fixes the Kaggle Out of Space Crash)
                        # ----------------------------------------------------
                        print("Running aggressive local cleanup...")
                        checkpoint_dir = os.path.join(
                            args.output_dir, "checkpoints")

                        for pkl in glob.glob(os.path.join(checkpoint_dir, "model_*.pkl")):
                            if f"model_{global_step}.pkl" not in pkl:
                                try:
                                    os.remove(pkl)
                                    print(f"Local Cleanup: Deleted {pkl}")
                                except Exception as e:
                                    pass

                        for ckpt in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
                            if f"checkpoint-{global_step}" not in ckpt:
                                try:
                                    shutil.rmtree(ckpt)
                                    print(f"Local Cleanup: Deleted {ckpt}")
                                except Exception as e:
                                    pass

                        # ----------------------------------------------------
                        # 5. EXPLICIT REMOTE CLEANUP (Hugging Face)
                        # ----------------------------------------------------
                        print("Running remote HF cleanup...")

                        # Only run remote cleanup if token exists
                        if os.environ.get("HF_TOKEN"):
                            steps_to_delete = []
                            if is_main_save:
                                steps_to_delete.extend(
                                    [global_step - 101, global_step - args.checkpointing_steps])
                            elif is_intermediate_save:
                                steps_to_delete.append(
                                    global_step - (args.checkpointing_steps - 100) + 1)

                            for s in steps_to_delete:
                                if s > 0:
                                    try:
                                        api.delete_folder(
                                            path_in_repo=f"checkpoint-{s}", repo_id=repo_id, repo_type="model")
                                        print(
                                            f"Remote Cleanup: Deleted checkpoint-{s} from HF.")
                                    except Exception:
                                        pass

                                    try:
                                        api.delete_file(
                                            path_in_repo=f"model_{s}.pkl", repo_id=repo_id, repo_type="model")
                                        print(
                                            f"Remote Cleanup: Deleted model_{s}.pkl from HF.")
                                    except Exception:
                                        pass

                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(
                                args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break

                            # --- FIX: Dynamic Device Loading for Validation Tensors ---
                            x_src = batch_val["conditioning_pixel_values"].to(
                                accelerator.device)
                            x_tgt = batch_val["output_pixel_values"].to(
                                accelerator.device)
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(
                                    x_src, prompt_tokens=batch_val["input_ids"].to(accelerator.device), deterministic=True)
                                loss_l2 = F.mse_loss(
                                    x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(
                                    x_tgt_pred.float(), x_tgt.float()).mean()
                                x_tgt_pred_renorm = t_clip_renorm(
                                    x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(
                                    x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(
                                    batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(
                                    x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(
                                    x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(
                                    args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                                                             shuffle=False, seed=0, batch_size=8, device=accelerator.device,
                                                             mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)

"""
Training loop for fine-tuning
"""
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, classification_report
import numpy as np


def build_scheduler(optimizer, warmup_epochs, decay_epochs, steps_per_epoch):
    """
    Linear warmup followed by cosine decay.
    - Classification: warmup_epochs=20, decay_epochs=30  (50 total)
    - Segmentation:   warmup_epochs=20, decay_epochs=80  (100 total)
    """
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,          # start near zero
        end_factor=1.0,
        total_iters=warmup_epochs * steps_per_epoch,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_epochs * steps_per_epoch,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch],
    )
    return scheduler


def classification_loop(model,
                        model_name,
                        epochs,
                        train_dl,
                        val_dl,
                        save_dir,
                        wandb_exp,
                        device,
                        optimizer,
                        criterion,
                        scheduler=None,
                        waves=None,
                        multi_label=False,
                        checkpoint_every=0,
                        start_epoch=0,
                        global_step_start=0,
                        seed=None,
                        dataset_pair=None):
    """
    Training loop for fine-tuning classification
    Supports both single-label and multi-label with proper metrics
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    import numpy as np
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_step = global_step_start

    for epoch in range(start_epoch, epochs):
        print('Training epoch {}'.format(epoch + 1))
        train_correct = 0
        train_total = 0
        model.train()
        total_loss = 0

        # For multi-label: collect predictions for F1 calculation
        if multi_label:
            train_all_preds = []
            train_all_labels = []

        for batch_idx, (imgs, labels) in enumerate(train_dl):
            if batch_idx == 0:
                print("First batch loaded")
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            if model_name.lower() == 'dofa':
                logits = model(imgs, waves)
            else:
                logits = model(imgs)

            # Different handling for multi-label vs single-label
            if multi_label:
                # Multi-label: BCEWithLogitsLoss expects float targets
                labels_float = labels.float()
                loss = criterion(logits, labels_float)

                # Predictions: apply sigmoid and threshold at 0.5
                preds = (torch.sigmoid(logits) > 0.5).float()

                # Collect for F1 calculation
                train_all_preds.append(preds.cpu().numpy())
                train_all_labels.append(labels.cpu().numpy())

                # Element-wise accuracy (inflated, for reference)
                train_correct += (preds == labels_float).sum().item()
                train_total += labels_float.numel()
            else:
                # Single-label: standard classification
                preds = logits.argmax(dim=1)
                loss = criterion(logits, labels)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            total_loss += loss.item()

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dl)
        current_lr = optimizer.param_groups[0]['lr']

        if multi_label:
            # Calculate F1 metrics
            train_all_preds = np.vstack(train_all_preds)
            train_all_labels = np.vstack(train_all_labels)

            train_acc = train_correct / train_total  # Inflated accuracy
            train_f1_micro = f1_score(train_all_labels, train_all_preds, average='micro', zero_division=0)
            train_f1_macro = f1_score(train_all_labels, train_all_preds, average='macro', zero_division=0)
            train_precision = precision_score(train_all_labels, train_all_preds, average='micro', zero_division=0)
            train_recall = recall_score(train_all_labels, train_all_preds, average='micro', zero_division=0)

            print(f"Epoch {epoch + 1}: train loss={avg_loss:.4f}, train acc={train_acc:.4f} (inflated), "
                  f"F1-micro={train_f1_micro:.4f}, F1-macro={train_f1_macro:.4f}, lr={current_lr:.6f}")

            wandb_exp.log({
                'train loss': avg_loss,
                'train acc': train_acc,
                'train F1-micro': train_f1_micro,
                'train F1-macro': train_f1_macro,
                'train precision': train_precision,
                'train recall': train_recall,
                'learning_rate': current_lr,
                'epoch': epoch + 1,
                'step': global_step
            })
        else:
            train_acc = train_correct / train_total
            print(f"Epoch {epoch + 1}: train loss={avg_loss:.4f}, train acc={train_acc:.4f}, lr={current_lr:.6f}")

            wandb_exp.log({
                'train loss': avg_loss,
                'train acc': train_acc,
                'learning_rate': current_lr,
                'epoch': epoch + 1,
                'step': global_step
            })

        # Validation
        if val_dl is not None and len(val_dl) > 0:
            model.eval()
            correct = 0
            total = 0
            val_total_loss = 0

            # For multi-label: collect predictions
            if multi_label:
                val_all_preds = []
                val_all_labels = []

            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)

                    if model_name.lower() == 'dofa':
                        logits = model(imgs, waves)
                    else:
                        logits = model(imgs)

                    # Different handling for multi-label vs single-label
                    if multi_label:
                        # Multi-label validation
                        labels_float = labels.float()
                        vloss = criterion(logits, labels_float)

                        preds = (torch.sigmoid(logits) > 0.5).float()

                        # Collect for F1 calculation
                        val_all_preds.append(preds.cpu().numpy())
                        val_all_labels.append(labels.cpu().numpy())

                        correct += (preds == labels_float).sum().item()
                        total += labels_float.numel()
                    else:
                        # Single-label validation
                        preds = logits.argmax(dim=1)
                        vloss = criterion(logits, labels)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    val_total_loss += vloss.item()

            avg_vloss = val_total_loss / len(val_dl)

            if multi_label:
                # Calculate F1 metrics
                val_all_preds = np.vstack(val_all_preds)
                val_all_labels = np.vstack(val_all_labels)

                val_acc = correct / total
                val_f1_micro = f1_score(val_all_labels, val_all_preds, average='micro', zero_division=0)
                val_f1_macro = f1_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
                val_precision = precision_score(val_all_labels, val_all_preds, average='micro', zero_division=0)
                val_recall = recall_score(val_all_labels, val_all_preds, average='micro', zero_division=0)

                print(f"  val loss={avg_vloss:.4f}, val acc={val_acc:.4f} (inflated), "
                      f"F1-micro={val_f1_micro:.4f}, F1-macro={val_f1_macro:.4f}")

                wandb_exp.log({
                    'val loss': avg_vloss,
                    'val acc': val_acc,
                    'val F1-micro': val_f1_micro,
                    'val F1-macro': val_f1_macro,
                    'val precision': val_precision,
                    'val recall': val_recall,
                    'epoch': epoch + 1,
                    'step': global_step
                })
            else:
                val_acc = correct / total
                print(f"  val loss={avg_vloss:.4f}, val acc={val_acc:.4f}")

                wandb_exp.log({
                    'val loss': avg_vloss,
                    'val acc': val_acc,
                    'epoch': epoch + 1,
                    'step': global_step
                })

        # Periodic checkpoint
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            seed_str = f'_seed{seed}' if seed is not None else ''
            pair_str = f'_{dataset_pair}' if dataset_pair is not None else ''
            ckpt_name = f'checkpoint_{model_name}{pair_str}{seed_str}_epoch{epoch + 1}.pt'
            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

def segmentation_loop(
    model,
    model_name,
    epochs,
    train_dl,
    val_dl,
    device,
    save_dir,
    wandb_exp,
    optimizer,
    criterion,
    scheduler=None,
    waves=None,
    num_classes=None,
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_miou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
    val_miou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)

    global_step = 0
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_miou.reset()
        total_loss = 0.0

        for imgs, masks in train_dl:
            imgs = imgs.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()

            if model_name.lower() == 'dofa':
                logits = model(imgs, waves)
            else:
                logits = model(imgs)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            global_step += 1

            preds = logits.argmax(dim=1)
            train_miou.update(preds, masks)

        avg_loss = total_loss / len(train_dl)
        mean_iou_score = train_miou.compute().item()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}: train loss={avg_loss:.4f}, train mIoU={mean_iou_score:.4f}, lr={current_lr:.6f}")

        wandb_exp.log({
            "train loss": avg_loss,
            "train mIoU": mean_iou_score,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
            "step": global_step,
        })

        # ---- Validation ----
        if val_dl is not None and len(val_dl) > 0:
            model.eval()
            val_miou.reset()
            val_total_loss = 0.0

            with torch.no_grad():
                for imgs, masks in val_dl:
                    imgs = imgs.to(device)
                    masks = masks.to(device).long()

                    if model_name.lower() == 'dofa':
                        logits = model(imgs, waves)
                    else:
                        logits = model(imgs)
                    loss = criterion(logits, masks)

                    val_total_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_miou.update(preds, masks)

            val_avg_loss = val_total_loss / len(val_dl)
            val_mean_iou_score = val_miou.compute().item()

            print(f"  val loss={val_avg_loss:.4f}, val mIoU={val_mean_iou_score:.4f}")

            wandb_exp.log({
                "val loss": val_avg_loss,
                "val mIoU": val_mean_iou_score,
                "epoch": epoch + 1,
                "step": global_step,
            })


def finetune(model,
             model_name,
             train_dataset,
             val_dataset,
             task,
             finetune,
             save_dir,
             wandb_exp,
             epochs: int,
             batch_size: int,
             learning_rate: float,
             weight_decay: float = 0,
             waves=None,
             num_classes=None,
             dataset_pair=None,
             checkpoint_every: int = 0,
             resume_checkpoint: str = None,
             seed: int = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if val_dataset is not None:
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_dl = None

    # Criteria
    multi_label = False
    if task == 'class':
        if dataset_pair == 'BenV2-S2-S1':
            class_criterion = torch.nn.BCEWithLogitsLoss()
            multi_label = True
        else:
            class_criterion = torch.nn.CrossEntropyLoss()
    else:
        semseg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # --- Freeze backbone if just want to train head of model
    if finetune == 'head':
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        elif hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "decode_head"):
            for p in model.decode_head.parameters():
                p.requires_grad = True
        elif hasattr(model, "decoder"):
            for p in model.decoder.parameters():
                p.requires_grad = True
        elif hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
            for p in model.backbone.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "backbone") and hasattr(model.backbone, "heads"):
            for p in model.backbone.heads.parameters():
                p.requires_grad = True
        elif hasattr(model, "backbone") and hasattr(model.backbone, "head"):
            for p in model.backbone.head.parameters():
                p.requires_grad = True
        try:
            if hasattr(model.heads, "head"):
                for p in model.heads.parameters():
                    p.requires_grad = True
        except AttributeError as e:
            print('continuing')

    elif finetune == "full":
        for p in model.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --- Build LR scheduler
    steps_per_epoch = len(train_dl)
    if task == 'class':
        warmup_epochs = 20
        decay_epochs = 30
    else:
        warmup_epochs = 20
        decay_epochs = 80

    # Clamp warmup to not exceed total epochs
    warmup_epochs = min(warmup_epochs, epochs)
    decay_epochs = max(epochs - warmup_epochs, 1)

    scheduler = build_scheduler(optimizer, warmup_epochs, decay_epochs, steps_per_epoch)

    # Put everything on device
    model.to(device)
    if waves is not None and isinstance(waves, torch.Tensor):
        waves = waves.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step_start = 0
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step_start = ckpt.get('global_step', 0)
        print(f"  Resuming from epoch {start_epoch} (global step {global_step_start})")

    # Fine-tune
    if task == 'class':
        classification_loop(model, model_name, epochs, train_dl, val_dl, save_dir, wandb_exp, device,
                            optimizer, class_criterion, scheduler=scheduler, waves=waves, multi_label=multi_label,
                            checkpoint_every=checkpoint_every, start_epoch=start_epoch,
                            global_step_start=global_step_start, seed=seed,
                            dataset_pair=dataset_pair)
    if task == 'semseg':
        segmentation_loop(model, model_name, epochs, train_dl, val_dl, device, save_dir, wandb_exp,
                          optimizer, semseg_criterion, scheduler=scheduler, waves=waves,
                          num_classes=num_classes)
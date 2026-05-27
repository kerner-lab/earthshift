"""
Testing loop for fine-tuning
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from model_manager import ModelManager


def classification_inference(model,
                             model_name,
                             test_dl,
                             save_dir,
                             wandb_exp,
                             device,
                             criterion,
                             waves,
                             multi_label=False):
    """
    Running inference for classification metrics
    Returns metrics dict for multi-label, single accuracy for single-label
    """

    correct = 0
    total = 0
    total_loss = 0

    # For multi-label: collect all predictions
    if multi_label:
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(device), labels.to(device)

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

                # Collect for sklearn metrics
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                correct += (preds == labels_float).sum().item()
                total += labels_float.numel()
            else:
                # Single-label: standard classification
                preds = logits.argmax(dim=1)
                loss = criterion(logits, labels)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_dl)

    if multi_label:
        # Calculate proper multi-label metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        test_acc = correct / total  # Inflated accuracy
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

        # Return dict of metrics
        metrics = {
            'loss': avg_test_loss,
            'acc': test_acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall
        }
        return metrics
    else:
        # Single-label: return simple accuracy
        test_acc = correct / total
        return avg_test_loss, test_acc


def segmentation_inference(model,
                           model_name,
                           test_dl,
                           wandb_exp,
                           device,
                           criterion,
                           waves,
                           num_classes):
    """
    Running inference for segmentation metrics
    :param model:
    :param test_dl:
    :param device:
    :param num_classes:
    :param waves:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting up miou
    miou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)

    total_loss = 0
    with torch.no_grad():
        for imgs, masks in test_dl:
            imgs, masks = imgs.to(device), masks.to(device).long()

            if model_name.lower() == 'dofa':
                logits = model(imgs, waves)
            else:
                logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds_indices = torch.argmax(logits, dim=1)
            miou_metric.update(preds_indices, masks)
            mean_iou_score = miou_metric.compute()

    avg_test_loss = total_loss / len(test_dl)

    return avg_test_loss, mean_iou_score


def predict(model,
            model_name,
            test_dataset,
            finetune_test_dataset,
            task,
            save_dir,
            wandb_exp,
            batch_size: int,
            waves=None,
            num_classes=None,
            dataset_pair=None
            ):

    # Load the test dataset from the out of distribution set
    if test_dataset is not None:
        out_test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        out_test_dl = None

    # Load the test dataset from the in distribution set
    if finetune_test_dataset is not None:
        in_test_dl = DataLoader(finetune_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        in_test_dl = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Maintain criterion loop for classification and segmentation
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

    # Running model inference
    model.eval()
    if task == 'class':
        # OOD test
        if out_test_dl is not None:
            if multi_label:
                ood_metrics = classification_inference(model, model_name, out_test_dl, save_dir, wandb_exp, device,
                                                       class_criterion, waves, multi_label=True)
                print(f"OOD Test set - Acc: {ood_metrics['acc']:.4f} (inflated), "
                      f"F1-micro: {ood_metrics['f1_micro']:.4f}, F1-macro: {ood_metrics['f1_macro']:.4f}")

                wandb_exp.log({
                    'test loss OOD': ood_metrics['loss'],
                    'test acc OOD': ood_metrics['acc'],
                    'test F1-micro OOD': ood_metrics['f1_micro'],
                    'test F1-macro OOD': ood_metrics['f1_macro'],
                    'test precision OOD': ood_metrics['precision'],
                    'test recall OOD': ood_metrics['recall']
                })
            else:
                ood_loss, ood_acc = classification_inference(model, model_name, out_test_dl, save_dir, wandb_exp,
                                                             device,
                                                             class_criterion, waves, multi_label=False)
                print(f'OOD Test set accuracy: {ood_acc:.4f}')
                wandb_exp.log({
                    'test loss OOD': ood_loss,
                    'test acc OOD': ood_acc
                })

        # ID test
        if in_test_dl is not None:
            if multi_label:
                in_metrics = classification_inference(model, model_name, in_test_dl, save_dir, wandb_exp, device,
                                                      class_criterion, waves, multi_label=True)
                print(f"ID Test set - Acc: {in_metrics['acc']:.4f} (inflated), "
                      f"F1-micro: {in_metrics['f1_micro']:.4f}, F1-macro: {in_metrics['f1_macro']:.4f}")

                wandb_exp.log({
                    'test loss ID': in_metrics['loss'],
                    'test acc ID': in_metrics['acc'],
                    'test F1-micro ID': in_metrics['f1_micro'],
                    'test F1-macro ID': in_metrics['f1_macro'],
                    'test precision ID': in_metrics['precision'],
                    'test recall ID': in_metrics['recall']
                })
            else:
                in_loss, in_acc = classification_inference(model, model_name, in_test_dl, save_dir, wandb_exp, device,
                                                           class_criterion, waves, multi_label=False)
                print(f'ID Test set accuracy: {in_acc:.4f}')
                wandb_exp.log({
                    'test loss ID': in_loss,
                    'test acc ID': in_acc
                })

    if task == 'semseg':
        ood_loss, ood_miou = (None, None)
        in_loss, in_miou = (None, None)

        if out_test_dl is not None:
            ood_loss, ood_miou = segmentation_inference(model, model_name, out_test_dl, wandb_exp, device,
                                                        semseg_criterion, waves, num_classes)
            print('OOD Test set miou: {}'.format(ood_miou))

        if in_test_dl is not None:
            in_loss, in_miou = segmentation_inference(model, model_name, in_test_dl, wandb_exp, device,
                                                      semseg_criterion, waves, num_classes)
            print('ID Test set miou: {}'.format(in_miou))

        wandb_exp.log({
            'test loss OOD': ood_loss,
            'test miou OOD': ood_miou,
            'test loss ID': in_loss,
            'test miou ID': in_miou,
        })


def run_s2_to_s1_sensor_shift(model, model_type, test_dataset, finetune_dataset_test, task, save_dir, experiment,
                               batch_size, config_task, train_waves, device, data_pair):
    """
    Handles inference for S2->S1 sensor shift experiments (BenV2-S2-S1, Sen1Floods11-S2-S1).
    First evaluates the trained S2 model on the in-distribution test set, then transfers
    compatible weights to a 2-channel S1 model and runs inference on the OOD S1 test set.
    """
    print('Run inference first on the in-distribution dataset...')
    predict(model, model_type, None, finetune_dataset_test, task, save_dir, experiment,
            batch_size=batch_size, waves=train_waves, num_classes=len(config_task['test_classes']),
            dataset_pair=data_pair)

    print('Running test set on Sentinel-1 using trained Sentinel-2 model...')
    rgb_weights = model.state_dict()

    modelmanager_s1 = ModelManager(model_name=model_type, task=task, num_classes=len(config_task['test_classes']),
                                   img_channels=2, wavelength=config_task['test_wavelength'], device=device)
    model_s1 = modelmanager_s1.get_model()
    s1_state = model_s1.state_dict()
    for key in s1_state.keys():
        if 'channel_adapter' not in key and key in rgb_weights:
            if s1_state[key].shape == rgb_weights[key].shape:
                s1_state[key] = rgb_weights[key]
    model_s1.load_state_dict(s1_state, strict=False)
    print('Transferred S2-trained weights to S1 model (channel adapter untrained)')

    model_s1 = model_s1.to(device)
    test_s1_waves = modelmanager_s1.get_waves()
    if test_s1_waves is not None and isinstance(test_s1_waves, torch.Tensor):
        test_s1_waves = test_s1_waves.to(device)

    print('Running S1 test set...')
    predict(model_s1, model_type, test_dataset, None, task, save_dir, experiment,
            batch_size=batch_size, waves=test_s1_waves, num_classes=len(config_task['test_classes']),
            dataset_pair=data_pair)







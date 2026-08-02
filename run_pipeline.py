"""
Script for scratch testing different models and trying things out
"""

import argparse
from model_manager import *
from data_manager import *
import json
from testing import predict, run_s2_to_s1_sensor_shift
from training import *
import wandb
import sys
from torch.utils.data import random_split
import math
from random import randint



def get_args():
    parser = argparse.ArgumentParser(description='Demo of earthshift setup')
    parser.add_argument('--root_dir', help='Root directory for saving things')
    parser.add_argument('--data_dir', help='Data directory of dataset')
    parser.add_argument('--task', help='Model task. Must be one of class, semseg, od')
    parser.add_argument('--model', help='Model to test')
    parser.add_argument('--shift', help='Shift type to run experiment. Must be one of data, sensor, '
                                        'location, temporal.')
    parser.add_argument('--dataset_pair', help='Dataset pair for experiment. Must be one of:'
                                               'RESISC45-UCMerced')
    parser.add_argument('--finetune_type', help='Finetune task, either the final head or the entire model,'
                                                'must be one of full, head')
    parser.add_argument('--save_dir', help='Save directory')
    parser.add_argument('--wandb_project', help='Wandb project to save runs to')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--seed', default=None, help='Seed')
    parser.add_argument('--tag', default='EarthShift', help='Wandb tag')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save a checkpoint every N epochs (0 = disabled)')
    parser.add_argument('--resume_checkpoint', default=None,
                        help='Path to checkpoint .pt file to resume training from')
    #parser.add_argument('--download', action='store_true', help='Specify whether to download dataset or not')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_dir = args.data_dir
    root_dir = args.root_dir
    task = args.task
    model_type = args.model
    finetune_type = args.finetune_type
    save_dir = args.save_dir
    shift = args.shift
    data_pair = args.dataset_pair
    wandb_project = args.wandb_project
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    tag = args.tag
    seed = args.seed
    checkpoint_every = args.checkpoint_every
    resume_checkpoint = args.resume_checkpoint

    # Set seed
    if seed is not None:
        seed = int(seed)
    else:
        seed = randint(0, 9999)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


    # Initialize wandb experiment for tracking
    experiment = wandb.init(project=args.wandb_project,
                            dir='{}/tmp/wandb'.format(root_dir),
                            resume='allow',
                            anonymous='must',
                            tags=[tag])


    # Load json data that stores information to query throughout the pipeline
    with open('configs/{}-shift-exp.json'.format(shift), 'r') as file:
        config = json.load(file)

    # Get config dictionary that describes the task we are interested in running
    config_task = config['task'][task]['data-pairs'][data_pair]
    config_task['finetune_type'] = finetune_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate ModelManager
    modelmanager = ModelManager(model_name=model_type, task=task, num_classes=len(config_task['finetune_classes']),
                                img_channels=config_task['img_channels'], wavelength=config_task['finetune_wavelength'],
                                device=device)

    # Load model
    model = modelmanager.get_model()

    # Get waves if needed for model - may be different if doing sensor shift experiment
    train_waves = modelmanager.get_waves(config_task['finetune_wavelength'])

    test_waves = modelmanager.get_waves(config_task['test_wavelength'])

    # Instantiate DataManager
    datamanager = DataManager(data_dir, model_type, task, data_pair, config_task, data_pair)

    # ------ Dataset checking  ------
    # Load datasets for finetuning and testing and apply normalization
    # Splitting data grabs into steps to account for missing data in certain datasets
    finetune_dataset_train, finetune_dataset_val, finetune_dataset_test = None, None, None
    try:
        finetune_dataset_train = datamanager.get_filtered_dataset(finetune_state='in', split='train',
                                                                     keep_classes=config_task['finetune_classes'],
                                                                  filter=config_task['filter'])
    except FileNotFoundError as e:
        print('Missing data for finetuning training! Checking...')
    try:
        finetune_dataset_val = datamanager.get_filtered_dataset(finetune_state='in',split='val',
                                                                    keep_classes=config_task['finetune_classes'],
                                                                filter=config_task['filter'])
    except FileNotFoundError as e:
        print('Missing data for finetuning validation! Checking...')
    try:
        finetune_dataset_test = datamanager.get_filtered_dataset(finetune_state='in',split='test',
                                                                keep_classes=config_task['finetune_classes'],
                                                                 filter=config_task['filter'])
    except FileNotFoundError as e:
        print('Missing data for finetuning testing! Checking...')

    if finetune_dataset_train is None or len(finetune_dataset_train) == 0:
        print('No training data, exiting...')
        sys.exit(0)
    else:
        print('Finetuning training data loaded!')
    if finetune_dataset_val is None or len(finetune_dataset_val) == 0:
        print('Validation data for finetuning is empty, setting to None')
        finetune_dataset_val = None     # reset to None
    else:
        print('Finetuning validation data loaded!')
    if finetune_dataset_test is None or len(finetune_dataset_test) == 0:
        print('Finetune test data is empty, splitting finetuning training set into train and test and skipping val...')
        train_size = math.ceil(len(finetune_dataset_train) * 0.8)
        test_size = len(finetune_dataset_train) - train_size
        finetune_dataset_train, finetune_dataset_test = random_split(
            finetune_dataset_train, [train_size, test_size]
        )
    else:
        print('Finetuning testing data loaded!')

    # Loading OOD data
    try:
        if data_pair in ('RESISC45-UCMerced', 'RESISC45-UCMerced-sub'):
            split = 'train'
        else:
            split = 'test'
        test_dataset = datamanager.get_filtered_dataset(finetune_state='out', split=split,
                                                        keep_classes=config_task['test_classes'],
                                                        filter=config_task['filter'])
        print('OOD testing data loaded!')
    except (AssertionError, FileNotFoundError) as e:
        print('Test set data from OOD dataset {} not available. Using the other data from {} for OOD inference... '.
              format(config_task['test'], config_task['test']))
        split = 'train'
        test_dataset = datamanager.get_filtered_dataset(finetune_state='out', split=split,
                                                        keep_classes=config_task['test_classes'],
                                                        filter=config_task['filter'])
        print('Loaded {} data to be used to test ood performance after finetuning'.format(split))

        if len(test_dataset) == 0:
            print('ERROR: Training data is also empty! Cannot run OOD testing.')
            sys.exit(0)
        else:
            print('Successfully loaded {} samples from the {} split'.format(len(test_dataset), split))


    # For lr-sweep on large sensor-shift datasets, use 20% subset to speed up experiment
    if tag == 'lr-sweep' and data_pair in ('Sen1Floods11-S2-S1', 'BenV2-S2-S1'):
        subset_generator = torch.Generator().manual_seed(seed)
        for name, ds in [('train', finetune_dataset_train), ('val', finetune_dataset_val),
                         ('test', finetune_dataset_test), ('ood_test', test_dataset)]:
            if ds is None:
                continue
            subset_frac = 0.2 if data_pair == 'Sen1Floods11-S2-S1' else 0.1
            keep = max(1, math.ceil(len(ds) * subset_frac))
            drop = len(ds) - keep
            kept, _ = random_split(ds, [keep, drop], generator=subset_generator)
            if name == 'train':
                finetune_dataset_train = kept
            elif name == 'val':
                finetune_dataset_val = kept
            elif name == 'test':
                finetune_dataset_test = kept
            elif name == 'ood_test':
                test_dataset = kept
            print(f'lr-sweep subset: {name} reduced to {keep} samples')

    if data_pair == 'ftw-germany-year':
        split_generator = torch.Generator().manual_seed(42)
        # Random split with isolated generator
        finetune_dataset_train, finetune_dataset_test = random_split(
            finetune_dataset_train,
            [412, 200],
            generator=split_generator  # Uses separate generator
        )

    # Update wandb experiment before running training, inference
    experiment.config.update(
        dict(model=model_type,
             shift=shift,
             finetune_type=finetune_type,
             dataset_pair=data_pair,
             task=task,
             epochs=epochs,
             batch_size=batch_size,
             learning_rate=lr,
             finetune_wavelength=config_task['finetune_wavelength'],
             test_wavelength=config_task['test_wavelength'],
             seed=seed
             )
    )

    # Train the model to finetune
    print('Fine-tuning model...')
    finetune(model, model_type, finetune_dataset_train, finetune_dataset_val, task, finetune_type, save_dir, experiment,
                               epochs=epochs, batch_size=batch_size, learning_rate=lr, weight_decay=0,
             waves=train_waves, num_classes=len(config_task['test_classes']), dataset_pair=data_pair,
             checkpoint_every=checkpoint_every, resume_checkpoint=resume_checkpoint, seed=seed)

    if 'BenV2-S2-S1' in data_pair or 'Sen1Floods11-S2-S1' in data_pair:
        run_s2_to_s1_sensor_shift(model, model_type, test_dataset, finetune_dataset_test, task, save_dir, experiment,
                                   batch_size, config_task, train_waves, device, data_pair)
    else:
        # Now run on test set
        print('Running Test set...')
        predict(model, model_type, test_dataset, finetune_dataset_test, task, save_dir, experiment,
                batch_size=batch_size, waves=test_waves, num_classes=len(config_task['test_classes']),
                dataset_pair=data_pair)
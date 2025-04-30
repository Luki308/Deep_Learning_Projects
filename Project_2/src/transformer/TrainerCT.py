import logging
import os
import pickle
from datetime import datetime

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Project_2.src.transformer.CnnTransformer import CNNTransformer
from Project_2.src.utils.DataHelperForTransformers import encode_labels, build_label_mapping, get_data_loaders_2

torch.manual_seed(0)

logging.basicConfig(filename=f'runs/CT/{datetime.now().strftime("%d.%m-%H.%M")}.log', filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_run_name(config):
    """Create a unique name for each experiment run"""
    timestamp = datetime.now().strftime("%d.%m-%H.%M")
    return f"{timestamp}---{config['model']}-h{config['num_heads']}l{config['num_layers']}-cl{config['num_classes']}-subset{config['subset_fraction']}_batch{config['batch_size']}"


def train_one_epoch(model, device, loader, optimizer, criterion, label_to_idx, dtype):
    model.train()
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    running_loss = 0.0
    correct = 0
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name} is frozen")

    for specs, labels in tqdm(loader):
        specs = specs.to(device)
        # print(specs.shape)
        # encode to integers
        targets = encode_labels(labels, label_to_idx).to(device)

        optimizer.zero_grad()
        outputs = model(specs)  # [B, num_classes]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += specs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, device, loader, criterion, label_to_idx, dtype, conf=False):
    model.eval()
    model = ipex.optimize(model)
    running_loss = 0.0
    correct = 0
    total = 0
    conf_matrix = None
    if conf:
        conf_matrix = np.zeros((config['num_classes'], config['num_classes']))

    with torch.no_grad(), torch.amp.autocast('xpu', enabled=True, dtype=dtype,
                                             cache_enabled=False):
        for specs, labels in tqdm(loader):
            specs = specs.to(device)
            targets = encode_labels(labels, label_to_idx).to(device)

            outputs = model(specs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * specs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += specs.size(0)

            if conf:
                for label, prediction in zip(targets, preds):
                    conf_matrix[label.item(), prediction.item()] += 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, conf_matrix


config = {
    'model': 'CT',
    'batch_size': 32,
    'subset_fraction': None,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'num_epochs': 50,
    'early_stopping_patience': 5,
    'dim': 64,
    'num_heads': 8,
    'num_layers': 2,
    'num_classes': 12,
    'target_subset': ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"],
    'save_name': f'{datetime.now().strftime("%d.%m-%H.%M")}-CNNT.pth'

}


def run():
    # tensorbord
    if not config['model'] in os.listdir('../tensorboard_runs'):
        os.makedirs(f'tensorboard_runs/{config['model']}')
    writer = SummaryWriter(f'../tensorboard_runs/{config['model']}/{get_run_name(config)}')

    num_epochs = config['num_epochs']

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    dtype = torch.float16 if torch.xpu.is_available() else torch.bfloat16

    train_loader, val_loader, test_loader = get_data_loaders_2(batch_size=config['batch_size'],
                                                               target_classes=config['target_subset'],
                                                               subset_fraction=config['subset_fraction'])

    label_to_idx = build_label_mapping(train_loader)
    logging.info(f"Label mapping{label_to_idx}")

    model = CNNTransformer(time_steps=500, freq_bins=64, embed_dim=config['dim'], num_heads=config['num_heads'],
                           num_layers=config['num_layers'],
                           num_classes=config['num_classes'])  # frec_bins must match n_mels from ToLogMelSpec

    model = model.to(device)

    # writer.add_graph(model, torch.randn([32, 500, 64], device=device))

    # optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    patience = config['early_stopping_patience']
    k = 0
    best_val_loss = np.inf
    best_val_acc = 0
    best_model = None

    # Training loop
    logging.info("Started training")
    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, label_to_idx, dtype)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        logging.info("Started validation")
        val_loss, val_acc, _ = validate_one_epoch(model, device, val_loader, criterion, label_to_idx, dtype)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            k = 0
        elif k >= patience - 1:
            logging.info("Stopping early")
            break
        else:
            k += 1
            logging.info(f"No improvement for {k}/{patience} epochs")
        logging.info(
            f"Epoch [{epoch}/{num_epochs}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    logging.info(f"Best val loss: {best_val_loss:.4f}, best val acc: {best_val_acc:.4f}")
    torch.save(best_model, config['save_name'])
    model.load_state_dict(torch.load(config['save_name']))
    test_loss, test_acc, conf_matrix = validate_one_epoch(model, device, test_loader, criterion, label_to_idx, dtype,
                                                          conf=True)

    writer.add_hparams(
        {
            key: ', '.join(value) if isinstance(value, list) else value
            for key, value in config.items()
            if key not in ['save_name', 'model']
        },
        {'hparam/accuracy': test_acc, 'hparam/loss': test_loss}
    )
    writer.flush()
    writer.close()

    logging.info(
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    logging.info(config)
    with open(f'CT{datetime.now().strftime("%d.%m-%H.%M")}matrix.pickle', 'wb') as f:
        pickle.dump((conf_matrix, label_to_idx), f)


def main():
    heads=[16]
    layers=[1]
    for layer in layers:
        config['num_layers'] = layer
        for head in heads:
            config['num_heads'] = head
            run()
            run()
            run()


if __name__ == "__main__":
    main()

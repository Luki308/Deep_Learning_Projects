import torch
import intel_extension_for_pytorch as ipex

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import os
from datetime import datetime
torch.manual_seed(0)


def get_run_name(config):
    """Create a unique name for each experiment run"""
    timestamp = datetime.now().strftime("%d.%m-%H.%M")
    return f"{timestamp}---{config['model']}-subset{config['subset_percentage']}_batch{config['batch_size']}"


class FineTuneHelper:
    def __init__(self, model_name: str, run_name: str, num_classes=10, weights="IMAGENET1K_V1"):
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = torch.float16 if torch.xpu.is_available() else torch.bfloat16
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.model = self._load_model(model_name, weights)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        if not self.model_name in os.listdir('tensorboard_runs'):
            os.makedirs(f'tensorboard_runs/{self.model_name}')
        self.writer = SummaryWriter(f'tensorboard_runs/{self.model_name}/{run_name}')

    def _load_model(self, model_name, weights):
        """Load pre-trained model and set classifier to right amount of classes"""
        model = torch.hub.load("pytorch/vision", model_name, weights=weights)

        if "efficientnet" in model_name:
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        elif "vit" in model_name:
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported for fine-tuning.")

        logging.info(f"Model loaded: {model_name}")

        return model.to(self.device)

    def freeze_main_model(self):
        """Freeze all layers to train only the classifier"""
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze only classifier
        if "efficientnet" in self.model_name:
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True
        elif "vit" in self.model_name:
            for param in self.model.heads.head.parameters():
                param.requires_grad = True

        logging.info(f"Frozen main model")

    def unfreeze_last_layers(self, num_layers=3, lr=1e-5, weight_decay=1e-4):
        """Unfreeze last few layers for fine-tuning"""
        if "efficientnet" in self.model_name:
            for param in list(self.model.features.parameters())[-num_layers:]:
                param.requires_grad = True
        elif "vit" in self.model_name:
            for param in list(self.model.encoder.layers.parameters())[-num_layers:]:
                param.requires_grad = True

        # update optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        logging.info(f"Unfrozen last layers and changed optimizer")

    def set_train_options(self, lr=0.001, weight_decay=1e-4):
        """Set optimizer, loss function, and learning rate scheduler."""
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        logging.info("Set train options")

    def train(self, train_loader, val_loader, num_epochs=10, save_name="best_model.pth", patience=5,
              best_val_loss=float("inf")):
        """
        Train with early stopping based on validation loss. Save best model so far.
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param num_epochs: Number of epochs to train for
        :param save_name: Name to save model under
        :param patience: Number of epochs to wait before early stopping
        :param best_val_loss: best validation loss so far (if there was some training before).
        :return: best validation loss
        """
        logging.info("Starting training.")
        counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            self.model.to(self.device)
            self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer)

            running_loss = 0.0
            correct, total = 0, 0

            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Trained.")

            train_acc = 100 * correct / total
            avg_train_loss = running_loss / len(train_loader)

            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)

            # validate
            val_loss, val_acc = self._evaluate(val_loader)

            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/validation', val_acc, epoch)

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

            # save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_name)
                counter = 0
                logging.info("Model improved! Saving best model.")
            else:
                counter += 1
                logging.info(f"No improvement for {counter}/{patience} epochs.")

            # early stopping
            if counter >= patience:
                logging.info("Early stopping triggered. Stopping training.")
                break

            self.scheduler.step()

        return best_val_loss

    def _evaluate(self, loader):
        """Evaluate model performance on validation/test set."""
        logging.info("Started evaluation.")
        self.model.eval()
        self.model.to(self.device)
        self.model = ipex.optimize(self.model)
        total, correct, loss_sum = 0, 0, 0.0

        with torch.no_grad(), torch.amp.autocast('xpu', enabled=True, dtype=self.dtype,
                                                 cache_enabled=False):
            for inputs, labels in tqdm(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = loss_sum / len(loader)
        accuracy = 100 * correct / total

        logging.info("Finished evaluation.")

        return avg_loss, accuracy

    def load_best_model(self, save_name="best_model.pth"):
        """Load the best saved model."""
        self.model.load_state_dict(torch.load(save_name))

    def test(self, test_loader, config):
        """Evaluate the model on test data"""
        test_loss, test_acc = self._evaluate(test_loader)

        self.writer.add_hparams(
            {'lr_clsf': config['lr_clsf'], 'lr_lrs': config['lr_lrs'],
             'weight_decay_clsf': config['weight_decay_clsf'], 'weight_decay_lrs': config['weight_decay_lrs'],
             'batch_size': config['batch_size'], 'epochs_clsf': config['epochs_clsf'],
             'epochs_lrs': config['epochs_lrs'],
             'early_stopping_patience_clsf':
                 config['early_stopping_patience_clsf'],
             'early_stopping_patience_lrs': config['early_stopping_patience_lrs']},
            {'hparam/accuracy': test_acc, 'hparam/loss': test_loss}
        )
        self.writer.flush()
        self.writer.close()

        logging.info(f"Test Accuracy: {test_acc:.2f}% - Test Loss: {test_loss:.4f}")


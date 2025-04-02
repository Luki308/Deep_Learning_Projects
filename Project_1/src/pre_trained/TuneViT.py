import torch

from Project_1.src.DataHelper import DataHelper, SimpleAugmentation
from pre_trained.FineTuneHelper import FineTuneHelper, get_run_name
import logging

logging.basicConfig(filename='runs/vit_b_16/02_vit_best_seed_coljit_3.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')

config = {
    'model': 'vit_b_16',
    'batch_size': 128,
    'subset_percentage': 0.2,
    'resize': 224,
    'lr_clsf': 1e-3,
    'weight_decay_clsf': 1e-4,
    'epochs_clsf': 10,
    'early_stopping_patience_clsf': 4,
    'lr_lrs': 1e-3,
    'weight_decay_lrs': 1e-4,
    'epochs_lrs': 10,
    'early_stopping_patience_lrs': 4,
    'augmentations': [SimpleAugmentation.ColorJitter],
    'save_name': '02_vit_best_seed_coljit_3.pth'
}


def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    data_helper = DataHelper(resize=config["resize"], batch_size=config["batch_size"],
                             subset_fraction=config["subset_percentage"], augmentations=config["augmentations"])
    train_loader, val_loader, test_loader = data_helper.get_loaders()

    logging.info(f"Device used: {device}")

    fine_tuner = FineTuneHelper(model_name=config["model"], run_name=get_run_name(config))

    # To just test a saved model, comment from here <-
    fine_tuner.freeze_main_model()

    fine_tuner.set_train_options(lr=config["lr_clsf"], weight_decay=config["weight_decay_clsf"])

    best_val_loss = fine_tuner.train(train_loader, val_loader, num_epochs=config["epochs_clsf"],
                                     patience=config["early_stopping_patience_clsf"], save_name=config["save_name"])

    fine_tuner.load_best_model(config["save_name"])

    fine_tuner.unfreeze_last_layers(lr=config["lr_lrs"], weight_decay=config["weight_decay_lrs"])

    fine_tuner.train(train_loader, val_loader, best_val_loss=best_val_loss, num_epochs=config["epochs_lrs"],
                     patience=config["early_stopping_patience_lrs"], save_name=config["save_name"])

    # To here <-, and load right model
    fine_tuner.load_best_model(config["save_name"])

    fine_tuner.test(test_loader, config)


if __name__ == '__main__':
    main()

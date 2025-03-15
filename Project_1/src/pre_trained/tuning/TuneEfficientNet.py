import torch
import intel_extension_for_pytorch as ipex
import logging
from Project_1.src.pre_trained.utils.FineTuneHelper import FineTuneHelper
from Project_1.src.pre_trained.utils.DataHelper import DataHelper, SimpleAugmentation

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_NAME = "efficientnet_b0"
RESIZE = 224  # input image size for efficient net


def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    data_helper = DataHelper(resize=RESIZE, batch_size=BATCH_SIZE, subset_fraction=0.1)
    train_loader, val_loader, test_loader = data_helper.get_loaders()

    logging.info(f"Device used: {device}")

    fine_tuner = FineTuneHelper(model_name=MODEL_NAME)

    # To just test, comment from here <-
    fine_tuner.freeze_main_model()

    fine_tuner.set_train_options(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = fine_tuner.train(train_loader, val_loader, num_epochs=6, patience=3, save_name="efficientnet_best.pth")

    fine_tuner.load_best_model("efficientnet_best2.pth")

    fine_tuner.unfreeze_last_layers()

    fine_tuner.train(train_loader, val_loader, best_val_loss=best_val_loss, num_epochs=6, patience=3, save_name="efficientnet_bestest.pth")

    # To here <-, and load right model (default: "../saved_models/efficientnet_bestest.pth")
    fine_tuner.load_best_model("efficientnet_bestest.pth")

    fine_tuner.test(test_loader)


if __name__ == '__main__':
    main()

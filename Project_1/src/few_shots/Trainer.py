import logging
import os
import random
from datetime import datetime

import torch
import intel_extension_for_pytorch as ipex
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from few_shots.Base import ConvBaseLearner
from few_shots.MetaOptimizer import MetaOptimizer

from few_shots.FewShotDataHelper import cinic_directory, cinic_mean, cinic_std, stratified_subset, build_class_index, \
    FewShotDataset

logging.basicConfig(filename=f'few_shots_01_turtle_{datetime.now().strftime("%d.%m-%H.%M")}.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')

torch.manual_seed(0)
random.seed(0)

def meta_train(base_learner, meta_opt_net, task_loader, inner_steps=5, meta_lr=1e-3,
               inner_lr_scale=0.1):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    dtype = torch.float16 if torch.xpu.is_available() else torch.bfloat16
    base_learner.to(device)
    meta_opt_net.to(device)

    meta_params = list(base_learner.parameters()) + list(meta_opt_net.parameters())
    meta_optimizer = optim.Adam(meta_params, lr=meta_lr)
    base_learner, meta_optimizer = ipex.optimize(base_learner, optimizer=meta_optimizer)

    for it, task in enumerate(task_loader):
        (x_support, y_support), (x_query, y_query) = task
        # remove batch dim added by data loader
        x_support, y_support = x_support.squeeze(0).to(device), y_support.squeeze(0).to(device)
        x_query, y_query = x_query.squeeze(0).to(device), y_query.squeeze(0).to(device)

        fast_weights = get_params(base_learner)

        # inner loop
        for step in range(inner_steps):
            support_logits = base_learner.forward(x_support, params=fast_weights)
            support_loss = torch.nn.functional.cross_entropy(support_logits, y_support)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            grad_dict = {name: grad for (name, _), grad in zip(fast_weights.items(), grads)}
            fast_weights = update_params(fast_weights, grad_dict, meta_opt_net, inner_lr_scale)

        query_logits = base_learner.forward(x_query, params=fast_weights)
        query_loss = torch.nn.functional.cross_entropy(query_logits, y_query)

        meta_optimizer.zero_grad()
        query_loss.backward()
        meta_optimizer.step()

        if (it + 1) % 100 == 0:
            with torch.no_grad(), torch.amp.autocast('xpu', enabled=True, dtype=dtype,
                                                     cache_enabled=False):
                pred = query_logits.argmax(dim=1)
                acc = (pred == y_query).float().mean().item()
            logging.info(f"Task {it + 1}: Query Loss = {query_loss.item():.4f}, Query Accuracy = {acc * 100:.2f}%")

    return base_learner, meta_opt_net


def get_params(model):
    return {name: param.clone() for name, param in model.named_parameters()}


def get_params_detached(model):
    return {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}


def update_params(params, grads, meta_optimizer, lr_scale):
    updated_params = {}
    for name, param in params.items():
        grad = grads[name]
        grad_flat = grad.view(-1)
        update_flat = meta_optimizer(grad_flat)
        update = update_flat.view_as(param)
        updated_params[name] = param + lr_scale * update
    return updated_params


def meta_evaluate(base_learner, meta_opt_net, task_loader, inner_steps=5):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    dtype = torch.float16 if torch.xpu.is_available() else torch.bfloat16
    base_learner.to(device)
    meta_opt_net.to(device)
    base_learner.eval()
    meta_opt_net.eval()

    total_loss, total_acc, total_tasks = 0.0, 0.0, 0
    with torch.amp.autocast('xpu', enabled=True, dtype=dtype,
                            cache_enabled=False):
        for task in task_loader:
            (x_support, y_support), (x_query, y_query) = task
            x_support, y_support = x_support.squeeze(0).to(device), y_support.squeeze(0).to(device)
            x_query, y_query = x_query.squeeze(0).to(device), y_query.squeeze(0).to(device)

            fast_weights = get_params_detached(base_learner)
            for step in range(inner_steps):
                support_logits = base_learner.forward(x_support, params=fast_weights)
                support_loss = F.cross_entropy(support_logits, y_support)
                grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=False)
                grad_dict = {name: grad for (name, _), grad in zip(fast_weights.items(), grads)}
                fast_weights = update_params(fast_weights, grad_dict, meta_opt_net, lr_scale=0.1)

            query_logits = base_learner.forward(x_query, params=fast_weights)
            query_loss = F.cross_entropy(query_logits, y_query)
            total_loss += query_loss.item()
            pred = query_logits.argmax(dim=1)
            acc = (pred == y_query).float().mean().item()
            total_acc += acc
            total_tasks += 1

    avg_loss = total_loss / total_tasks
    avg_acc = total_acc / total_tasks
    logging.info(f"Evaluation: Average Query Loss = {avg_loss:.4f}, Average Accuracy = {avg_acc * 100:.2f}%")
    return avg_loss, avg_acc


def main():
    config = {'train': {'subset_fr': 0.01, 'n_way': 10, 'k_shot': 10, 'query_size': 10, 'num_tasks': 500},
              'test': {'subset_fr': 0.01, 'n_way': 10, 'k_shot': 10, 'query_size': 10, 'num_tasks': 100},
              'params': {'regularization': '0.1 dropout', 'inner_steps': 10, 'meta_lr': 0.001, 'inner_lr_scale': 0.1,
                         'evaluate_steps': 10}}
    logging.info(config)

    # Prepare train data & dataloader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])
    dataset = datasets.ImageFolder(os.path.join(cinic_directory, "train"),
                                   transform=transform)
    dataset = stratified_subset(dataset, config["train"]["subset_fr"])
    class_to_indices = build_class_index(dataset)
    few_shot_dataset = FewShotDataset(dataset, class_to_indices, n_way=config["train"]["n_way"],
                                          k_shot=config["train"]["k_shot"], query_size=config["train"]["query_size"],
                                          num_tasks=config["train"]["num_tasks"])
    task_loader = DataLoader(few_shot_dataset, batch_size=1, shuffle=True)

    # Prepare test data & dataloader
    dataset_test = datasets.ImageFolder(os.path.join(cinic_directory, "test"),
                                        transform=transform)
    dataset_test = stratified_subset(dataset_test, config["test"]["subset_fr"])
    class_to_indices_test = build_class_index(dataset_test)
    few_shot_dataset_test = FewShotDataset(dataset_test, class_to_indices_test, n_way=config["test"]["n_way"],
                                               k_shot=config["test"]["k_shot"], query_size=config["test"]["query_size"],
                                               num_tasks=config["test"]["num_tasks"])
    task_loader_test = DataLoader(few_shot_dataset_test, batch_size=1, shuffle=True)

    # initialize models and start training
    base_learner = ConvBaseLearner(num_classes=config["train"]["n_way"])
    meta_opt_net = MetaOptimizer()

    logging.info("Starting meta-training...")
    trained_base_learner, trained_meta_net = meta_train(

        base_learner, meta_opt_net, task_loader, inner_steps=config["params"]["inner_steps"], meta_lr=config["params"][
            "meta_lr"], inner_lr_scale=config["params"]["inner_lr_scale"])


    logging.info("Starting evaluation on test tasks...")
    meta_evaluate(trained_base_learner, trained_meta_net, task_loader_test, inner_steps=config["params"]["evaluate_steps"])

if __name__ == '__main__':
    main()

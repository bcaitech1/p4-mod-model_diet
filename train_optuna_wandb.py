"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info

import optuna
import wandb
import pickle

class Metric(object):
    def __init__(self, macs, wandb_run=None):
        self.macs = macs
        self.wandb_run = wandb_run

    def __call__(self, f1, epoch=None):
        if self.wandb_run is not None:
            assert(epoch is not None)

        crit_macs = 13863553
        crit_f1 = 0.8342
        score_macs = self.macs / crit_macs
        score_f1 = 1 - f1 / crit_f1 if f1 < crit_f1 else 0.5 * (1 - f1 / crit_f1)
        score = score_macs + score_f1

        if self.wandb_run is not None:
            self.wandb_run.log({
                'f1':f1,
                'macs':self.macs,
                'score_f1':score_f1,
                'score_macs':score_macs,
                'score':score
            }, step=epoch)
        
        return score


def train(
        trial,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        log_dir: str,
        device: torch.device,
        wandb_run = None
    ) -> Tuple[float, float, float]:
        """Train."""
        # save model_config, data_config
        with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)

        # save path setting, load parameter
        model_instance = Model(model_config, verbose=True)
        model_path = os.path.join(log_dir, "best.pt")
        print(f"Model save path: {model_path}")
        if os.path.isfile(model_path):
            model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
        model_instance.model.to(device)

        # Create dataloader
        train_dl, val_dl, test_dl = create_dataloader(data_config)

        # Calc macs
        macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
        print(f"macs: {macs}")
        metric = Metric(macs, wandb_run)

        # Create optimizer, scheduler, criterion
        optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
                    # adamp.AdamP(model_instance.model.parameters(), lr=data_config["INIT_LR"], weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=data_config["INIT_LR"],
            steps_per_epoch=len(train_dl),
            epochs=data_config["EPOCHS"],
            pct_start=0.05,
        )
        criterion = CustomCriterion(
            samples_per_cls=get_label_counts(data_config["DATA_PATH"])
            if data_config["DATASET"] == "TACO"
            else None,
            device=device,
        )
        # Amp loss scaler
        scaler = (
            torch.cuda.amp.GradScaler() if data_config["FP16"] and device != torch.device("cpu") else None
        )

        # Create trainer
        trainer = TorchTrainer(
            model=model_instance.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            model_path=model_path,
            verbose=1,
        )
        best_score = trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            n_epoch=data_config["EPOCHS"],
            metric=metric
        )
        return best_score

class Ready(object):
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.log_dir, exist_ok=True)
        pass

    def __call__(self, trial):
        #hyper parameter trial
        self.data_config["IMG_SIZE"] = int(trial.suggest_discrete_uniform('img_size', 160, 384, 32))
        self.data_config["BATCH_SIZE"] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        self.data_config["INIT_LR"] = trial.suggest_loguniform('lr', 1e-4, 1e-2)

        #wandb init
        config = {
            'img_size':data_config['IMG_SIZE'],
            'batch_size':data_config['BATCH_SIZE'],
            'init_lr':data_config['INIT_LR'],
        }
        wandb_run = wandb.init(project='optuna', name='trial', group='sampling3', config=config, reinit=True)
        
        _, _, score = train(
            trial,
            model_config=self.model_config,
            data_config=self.data_config,
            log_dir=self.log_dir,
            device=self.device,
            wandb_run=wandb_run,
        )

        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/mobilenetv3.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    ready = Ready(model_config, data_config)
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(ready, n_trials=20)

    summary = wandb.init(project='optuna', name='summary', job_type='logging')
    trials = study.trials
    for step, trial in enumerate(trials):
        summary.log({'score': trial.value}, step=step)
        for k, v in trial.params.items():
            summary.log({k : v}, step=step)

    # save
    with open('mnist_optuna.pkl', 'wb') as f:
        pickle.dump(study, f, pickle.HIGHEST_PROTOCOL)

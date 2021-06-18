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
    '''
    평가 메트릭을 대회의 스코어와 동일하게 적용하기 위해 만들었습니다.
    스코어를 계산하면서 wandb까지 기록할 수 있도록 만들었습니다.
    baseline 코드는 train 함수 밖에서 macs를 계산하고 train 함수의 반환값으로 score를 계산하는 방식인데
    wandb로 계속해서 기록을 남기려면 train 함수 안에서 에폭이 돌아갈 때마다 score를 계산해야 하기 때문에
    구조를 크게 변화시키지 않고 깔끔하게 짜기 위해 고민한 결과 이렇게 class로 만들어서 인자로 넣어주는 방식을 택했습니다.
    더 멋있는 방법이 있는지 모르겠네요...
    '''

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

        #for param_tensor in model_instance.model.state_dict():
        #    print(param_tensor)

        #for key, item in param_dict.items():
        #    print(key)

        '''
        주석이 된 코드는 pretrained 모델의 weight를 사용하기 위해 만든 것입니다.
        구조가 timm 라이브러리 모델과는 조금 달랐지만 비슷한 레이어끼리 이름을 맞춰서
        억지로라도 weight를 적용시키면 성능이 향상될 줄 알았는데
        별로 좋은 결과가 나오지 않았습니다.
        '''

        if os.path.isfile(model_path):
            model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
        #else:
        #    param_dict = torch.hub.load_state_dict_from_url(model_config['pretrained']['url'])
        #    mapping_layer_name = model_config['pretrained']['mapping_layer_name']
        #    for key, value in mapping_layer_name.items():
        #        param_dict[value] = param_dict[key]
        #        del param_dict[key]
        #    model_instance.model.load_state_dict(param_dict, strict=False)
                
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
    '''
    optuna study에 넣어줄 클래스입니다.
    train에서 score를 계산해서 반환한 걸 __call__으로 반환하도록 만들었는데
    multi objective study를 쓰는 것과 차이가 없는 것 같기도 한데 뭐가 더 나은 방식인지는 모르겠습니다.
    '''
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.log_dir, exist_ok=True)
        pass

    def __call__(self, trial):

        #hyper parameter trial
        self.data_config["IMG_SIZE"] = int(trial.suggest_discrete_uniform('img_size', 64, 160, 32))
        self.data_config["BATCH_SIZE"] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        self.data_config["INIT_LR"] = trial.suggest_loguniform('lr', 1e-4, 1e-2)

        #wandb init
        config = {
            'img_size':data_config['IMG_SIZE'],
            'batch_size':data_config['BATCH_SIZE'],
            'init_lr':data_config['INIT_LR'],
        }
        wandb_run = wandb.init(project='optuna', name='trial', group='sampling_sqeeze3', config=config, reinit=True)
        
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
    model_config = read_yaml(cfg="configs/model/example.yaml")
    data_config = read_yaml(cfg=args.data)

    ready = Ready(model_config, data_config)
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(ready, n_trials=200)

    summary = wandb.init(project='optuna', name='summary', job_type='logging')
    trials = study.trials
    for step, trial in enumerate(trials):
        summary.log({'score': trial.value}, step=step)
        for k, v in trial.params.items():
            summary.log({k : v}, step=step)

    # save
    with open('mnist_optuna.pkl', 'wb') as f:
        pickle.dump(study, f, pickle.HIGHEST_PROTOCOL)

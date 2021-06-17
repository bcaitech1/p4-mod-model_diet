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
import torchvision.models
from pytz import timezone

from src.dataloader import create_dataloader
from src.loss import CustomCriterion, CosineAnnealingWarmupRestarts
from src.model import Model
from src.trainer_kd_wandb import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info
from efficientnet_pytorch import EfficientNet
from src.decomposer import tucker_decomposition_conv_layer, decompose

import wandb

class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes: int = 9, test:bool = False, **kwargs):
        super().__init__()
        self.my_model = EfficientNet.from_pretrained('efficientnet-b4', advprop=True, num_classes=num_classes)

    def forward(self, x):
        return self.my_model(x)
    
def train_kd(
    teacher_pretrained: str,
    model_name: str,
    model_config: Dict[str, Any],
    student_pretrained: str,
    data_config: Dict[str, Any],
    log_name: str,
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    teacher_model = EfficientNet_b4(num_classes=9)
    if os.path.isfile(teacher_pretrained):
        teacher_model.load_state_dict(torch.load(teacher_pretrained))
        print("teacher pretrained model loaded.")
    teacher_model.to(device)

    model_instance = Model(model_config, verbose=True)
    # print(model_instance.model)
#    timm.create_model(model_name='resnetv2_101x1_bitm',pretrained=True,num_classes=9)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    # if there is student pretrained, then load
    print(student_pretrained)
    print(os.path.isfile(student_pretrained))


    # decompose
    model_instance.model[0].conv = tucker_decomposition_conv_layer(model_instance.model[0].conv)

    for i in range(2, 6):
        for j in range(len(model_instance.model[i])):
            for k in range(len(model_instance.model[i][j].branch1)):
                if type(model_instance.model[i][j].branch1[k]) == nn.Conv2d and \
                        model_instance.model[i][j].branch1[k].groups == 1:
                    model_instance.model[i][j].branch1[k] = tucker_decomposition_conv_layer(
                        model_instance.model[i][j].branch1[k])
            for k in range(len(model_instance.model[i][j].branch2)):
                if type(model_instance.model[i][j].branch2[k]) == nn.Conv2d and \
                        model_instance.model[i][j].branch2[k].groups == 1:
                    model_instance.model[i][j].branch2[k] = tucker_decomposition_conv_layer(
                        model_instance.model[i][j].branch2[k])

    if os.path.isfile(student_pretrained):
        model_instance.model.load_state_dict(torch.load(student_pretrained, map_location=device))
        print("student pretrained model loaded")
    model_instance.model.to(device)
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    # sglee 브랜치 테스트.
    # sglee487 브랜치 테스트.
    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    # optimizer = torch.optim.AdamW(model_instance.model.parameters(), lr=data_config["INIT_LR"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    # first_cycle_steps = 3000
    # scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
    #                                           first_cycle_steps=first_cycle_steps,
    #                                           max_lr=data_config["INIT_LR"],
    #                                           min_lr=0.00001,
    #                                           warmup_steps=int(first_cycle_steps * 0.2),
    #                                           gamma=0.5)
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model_name=model_name,
        model=model_instance.model,
        model_macs=macs,
        log_name=log_name,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )

    wandb.log({"Model/TeacherModel": teacher_model.__class__.__name__})
    wandb.log({"Model/model_config": model_config})
    wandb.log({"Model/data_config": data_config})

    best_acc, best_f1 = trainer.train(
        teacher_model=teacher_model,
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/shufflenet_v2_05_base.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco_5_small_continue.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--teacher_pretrained", default="expsaves/efficientnetb4_pretrained_64_2021-06-12_16-01-21/best.pt",
        type=str, help="to load student pretrained weight"
    )
    parser.add_argument(
        "--student_pretrained", default="exp/shufflenet_v2_05_base_2021-06-14_15-44-41/best.pt",
        type=str, help="to load student pretrained weight"
    )
    # parser.add_argument(
    #     "--student_pretrained", default="Nope",
    #     type=str, help="to load student pretrained weight"
    # )
    args = parser.parse_args()

    model_name = args.model.split('/')[-1].split('.yaml')[0]
    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_str = datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{model_name}_{now_str}_de04"
    log_dir = os.path.join('exp', f"{model_name}_{now_str}")
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project='pstage4', reinit=False, name=log_name)

    test_loss, test_f1, test_acc = train_kd(
        teacher_pretrained=args.teacher_pretrained,
        model_name=model_name,
        model_config=model_config,
        student_pretrained=args.student_pretrained,
        data_config=data_config,
        log_name=log_name,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

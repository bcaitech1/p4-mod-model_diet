# boostcamp_pstage10
김연세, 윤석진, 이상건, 이용우, 한웅희, 황훈
<br>
<img src=https://github.com/bcaitech1/p4-mod-model_diet/blob/main/Diet.gif>
# Docker
```bash
docker run -it --gpus all --ipc=host -v $PWD:/opt/ml/code -v ${dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.1 /bin/bash
```

# Run
## 1. train
python train.py --model ${path_to_model_config} --data ${path_to_data_config}

## 2. inference(submission.csv)
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root ~/input/data/test --data_config configs/data/taco.yaml

# boostcamp_pstage10
김연세, 윤석진, 이상건, 이용우, 한웅희, 황훈
<br>
<img src=https://github.com/bcaitech1/p4-mod-model_diet/blob/main/Diet.gif>
# Docker
```bash
docker run -it --gpus all --ipc=host -v $PWD:/opt/ml/code -v ${dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.1 /bin/bash
```

```tree

```

# Run
## 1. train
### 1.1 train
```bash
python train.py --model ${path_to_model_config} --data ${path_to_data_config}
```
### 1.2 train with Knowledge Distillation (efficientnet b4)
#### 1.2.1 teacher: efficientnet b4
```bash
python train_knowdistill_efficientnetb4.py \
--model ${path_to_model_config} \
--data ${path_to_data_config} \
--teacher_pretrained ${path_to_teacher_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```
#### 1.2.2 teacher: mobilenet v3 large
```bash
python train_knowdistill_mobilenetv3.py \
--model ${path_to_model_config} \
--data ${path_to_data_config} \
--teacher_pretrained ${path_to_teacher_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```
#### 1.2.3 teacher: efficientnet b4, moblienet v3 large, shufflenet v2 x0.5
```bash
python train_knowdistill_efficientnetb4amobilenetv3largeashufflenetv205 \
--model ${path_to_model_config} \
--data ${path_to_data_config} \
--teacher1_pretrained ${path_to_teacher1_pretrained_weight} \
--teacher2_pretrained ${path_to_teacher2_pretrained_weight} \
--teacher3_pretrained ${path_to_teacher3_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```
### 1.3 train with decomposition (only shufflenet_v2_05_base.yaml)
#### 1.3.1 teacher: efficientnet b4
```bash
python train_knowdistill_efficientnetb4_shufflenetv250_decompose.py \
--model "configs/model/shufflenet_v2_05_base.yaml" \
--data ${path_to_data_config} \
--teacher_pretrained ${path_to_teacher_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```
#### 1.3.2 (VBMF) teacher: efficientnet b4
```bash
python train_knowdistill_efficientnetb4_shufflenetv250_decompose_VBMF.py \
--model "configs/model/shufflenet_v2_05_base.yaml" \
--data ${path_to_data_config} \
--teacher_pretrained ${path_to_teacher_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```
#### 1.3.3 teacher: efficientnet b4, mobilenet v3 large
```bash
python train_knowdistill_efficientnetb4amobilenetv3large_shufflenetv250_decompose.py \
--model "configs/model/shufflenet_v2_05_base.yaml" \
--data ${path_to_data_config} \
--teacher1_pretrained ${path_to_teacher1_pretrained_weight} \
--teacher2_pretrained ${path_to_teacher2_pretrained_weight} \
--student_pretrained ${path_to_student_pretrained_weight}
```

## 2. inference(submission.csv)
```bash
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root ~/input/data/test --data_config configs/data/taco.yaml
```
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)
# [CV] Document Type Classification
## 3조

| ![이현진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![권혁찬](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김소현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김태한](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문정의](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이현진](https://github.com/UpstageAILab)             |            [권혁찬](https://github.com/UpstageAILab)             |            [김소현](https://github.com/UpstageAILab)             |            [김태한](https://github.com/UpstageAILab)             |            [문정의](https://github.com/UpstageAILab)             |
|                            팀장, EDA, 피쳐 엔지니어링, 모델링                             |                            EDA, 피쳐 엔지니어링, 모델링                             |                            EDA, 피쳐 엔지니어링, 모델링                             |                            EDA, 피쳐 엔지니어링, 모델링                             |                            EDA, 피쳐 엔지니어링, 모델링                             |

## 0. Overview
### Environment
- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2

### Requirements
- albumentations==1.3.1
- numpy==1.26.0
- timm==0.9.12
- torch==2.1.0
- torchvision=0.16.0
- scikit-learn=1.3.2

## 1. Competiton Info

### Overview

- 문서는 금융, 보험, 물류, 의료 등 도메인을 가리지 않고 많이 취급됩니다. 이 대회는 다양한 종류의 문서 이미지의 클래스를 예측합니다.
- 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다.

### Timeline

- Feburary 05, 2024 - Start Date
- Feburary 07, 2024 - Mentoring1
- Feburary 16, 2024 - Mentoring2
- Feburary 19, 2024 - Mentoring3
- Feburary 19, 2024 - Final submission deadline

### Evaluation

- Macro F1

## 2. Components

### Directory
e.g.
```
├── code
│   ├── model_train.ipynb
│   └── train.py
├── docs
│   ├── [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디_3조.pptx
│   └── paper
└── data
    ├── test
    ├── train
    ├── train.csv
    ├── meta.csv
    └── sample_submission.csv
```

## 3. Data descrption

### Dataset overview

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv3/assets/79961865/641ced0f-f10e-44ff-b230-57a1a21efb94)
- 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측
- 데이터가 어떤 class를 가지고 있는지 설명하는 meta.csv와 각 이미지 파일과 label을 매치한 train.csv 제공
    - 0 계좌번호, 1 임신 의료비 지급 신청서, 2 자동차 계기판, 3 입·퇴원 확인서, 4 진단서, 5 운전면허증, 6 진료비 영수증, 7 외래 진료 증명서, 8 국민 신분증, 9 여권, 10 지불 확인서, 11 의약품 영수증, 12 처방전, 13 이력서, 14 의견 진술, 15 자동차 등록증, 16 자동차 등록판
- car_dashboard와 vehicle_registration_plate가 이질적, 나머지 classes의 이미지는 모두 문서의 형태
- test data는 flip, rotate, mixup 등이 되어 있는 문서 이미지


### EDA

- labeling이 애매한 데이터들 존재
- 대부분의 모델이 3, 7, 13번 class를 자주 혼동하는 경향을 보임
- Scikit-Learn의 Confusion Matrix의 시각화로 잘 분류되는 클래스와 자주 혼동되는 클래스 확인  

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_
- data augmentation으로는 Albumentations 라이브러리 사용
    ```python
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Resize(height=img_size, width=img_size),
        A.Flip(),
        A.GaussNoise(p=0.3),
        A.OneOf([A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
        A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=.1), A.PiecewiseAffine(p=0.3), ], p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    ```
- 애매한 labeling을 가진 데이터의 csv 파일 수정
- 16 Offline Augmentations: 다양하게 변형된 features를 학습하여 robust한 모델을 만들기 위해 offline 방법 적용.  
	+ HorizontalFlip  
	+ VerticalFlip  
	+ ShiftScaleRotate  
	+ Grayscale  
	+ ColorJitter  
	+ Blur  
	+ MedianBlur  
	+ Spatter  
	+ Defocus  
	+ ZoomBlur  
	+ OpticalDistortion 2장  
	+ Perspective 2장  
	+ Rotate 2장  
  

## 4. Modeling

### Model descrition

**김소현**
- EfficientNet : 다른 competition 등에서 활용도가 높고 성능도 좋은 모델 선택. Params 52.6
- CoAtNet : 데이터셋이 크지 않아 상대적으로 작은 모델로 학습 시도. Params 27.4
- ConvNeXt : 상대적으로 큰 CNN 모델을 선택하여 학습, 잘 구분하지 못했던 클래스에 대해 더 나은 classification 성능 기대. Params 200.1
- EfficientNet V2 : 실험의 성능이 제일 좋았던 efficienet의 상위버전인 v2 선택. Params 54.1
- HRNet : 해상도가 영향이 있는지 알기 위해 화질에 따른 feature를 추가하는 network 선택. Params 77.5
- 다른 augmentation을 적용하거나 클래스 별 학습 데이터 불균형으로 oversampling하여 실험해 보기도 함
- 각 모델이 pre-trained된 image size에 따라 size 변경

**이현진**
- EfficientNet

**김태한**
- EfficientNet V2 M: Classifier만 학습하는 Transfer Learning과 전체를 학습하는 Fine-Tuning 모두 적용
- Two Stage Model: 첫 번째 EfficientNet V2 M으로 자동차계기판, 자동차번호판, 문서를 분류. 두 번째 EfficientNet V2 M으로 나머지 15가지의 문서를 분류.

### Modeling Process

- _Write model train and test process with capture_

- transfer learning 또는 fine-tuning을 위해 미리 학습된 모델을 불러올 수 있게 제공해 주는 timm 라이브러리 사용
    ```python
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    ```

**김소현**
- scheduler로 learning rate 자동 조절
  ```python
  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
  ```
- classifier(head)만 학습할 때 다음 코드 사용(미제출)
  ```python
  for name, param in model.named_parameters() :
    if name.split('.')[0] == "head" :
        param.requires_grad = True
    else :
        param.requires_grad = False
  ```

**이현진**
- scheduler 사용 (CosineAnnealingWarmRestarts)<br>
```
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, 
                                                                 T_mult=1, eta_min= 1e-6)
```
- Focal loss 사용 (불러옴)<br>단, focal loss의 alpha는 데이터가 가장 많은 class를 기준으로, 그 class보다 데이터가 작으면 그 비율만큼 alpha를 상승시켜 loss 계산<br>
추가적으로 class 3과 class 7에서 잘 예측을 하지 못해 해당 class의 alpha를 증강 (1.3배)
```
a = train.target.value_counts().sort_index(ascending = True).reset_index().drop(columns = ['target'])
a = a['count'].apply(lambda x: round(max(a['count'])/x, 5))
a = a.to_list()
a[3] = a[3] * 1.3
a[7] = a[7] * 1.3

loss_fn = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='focal_loss',
	alpha=a,
	gamma=2,
	reduction='mean',
	device='cuda:0',
	dtype=torch.float32,
	force_reload=False
)
```

## 5. Result

### Leader Board

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv3/assets/119946138/40cf30f0-c0b9-4478-bd9b-ce71476322a0)

Public : 7th

Private : 7th

### Presentation



## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_
- [1주차 (1/30 ~ 2/2)](https://www.notion.so/1-1-30-2-2-e794e580172e4fdea9589accb0119d9e?pvs=4)
- [2주차 (2/5 ~ 2/8)](https://www.notion.so/2-02-05-02-08-c0fcfcbf73204a3bb1db27d84ec4407c?pvs=4)
- [3주차 (2/13 ~ 2/16)](https://www.notion.so/3-02-13-02-16-34b653ac62ab47cdbedd4c1dcb99827c?pvs=4)

### Reference

- https://scikit-learn.org/
- https://pytorch.org/hub/
- https://pytorch.org/docs/stable/index.html
- https://albumentations.ai/docs/api_reference/augmentations/transforms/
- https://dacon.io/codeshare/2373
- https://dacon.io/codeshare/3658
- https://ko.d2l.ai/chapter_deep-learning-basics/
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py



# -_-
# 2025-1학기 인공지능 강좌 FINAL PROJECT  
---

## 주제 선정
주제: fetal ultrasound image segmentation  
선정 배경: 초음파 영상의 경우 임상 경험이 많이 쌓이지 않은 간호, 의대 실습생의 경우 부위 판별에 어려움을 많이 겪는 편이라고 한다.(자료 첨부 예정) 특히 그중 태아 초음파 영상의 경우 실습 중엔 더더욱 접하기 힘든 것이 현실이다. 따라서 실습생들의 교육을 돕기 위한 image segmentation 모델이 있다면 도움이 될 것이라고 판단, 프로젝트를 진행하게 되었다.  

---

## 사용 모델: UNET  
좋은 선택이에요! 태아 머리 초음파 영상 데이터셋과 UNet은 의료영상 분야에서 정말 잘 맞는 조합입니다. 아래에 설명드릴게요.  
### 🧠 1. **이 데이터셋으로 만들 수 있는 딥러닝 모델 예시**

당신은 이 데이터셋을 통해 **의료영상 분할(segmentation)** 기반의 딥러닝 프로젝트를 진행할 수 있어요. 특히 **UNet**은 이 작업에 최적화된 구조입니다.

#### ✅ 예시 프로젝트 주제

**"UNet 기반의 태아 뇌 구조 분할 모델 개발 및 성능 평가"**

##### ➤ 목표

* 초음파 영상에서 뇌(BRAIN), CSP, LV 같은 구조를 자동으로 분할
* 정확도, IoU, Dice coefficient 등 평가 지표를 활용해 성능 평가

##### ➤ 기대 효과

* 자동화된 태아 생체 계측 가능
* 의료진의 수작업 부담 완화 및 진단 정확도 향상

### 🏥 2. **의료용 인공지능에서 UNet은 왜 자주 쓰일까?**

UNet은 2015년 바이오메디컬 이미지 분할을 위해 제안된 CNN 기반 구조입니다. 의료영상에 특히 잘 맞는 이유는 다음과 같아요:

| 특징                     | 설명                                |
| ---------------------- | --------------------------------- |
| **Encoder-Decoder 구조** | 컨텍스트 정보를 축소하며 추출하고, 픽셀 단위 복원까지 가능 |
| **Skip Connection**    | 저수준과 고수준 특징을 결합 → 경계가 뚜렷한 분할 가능   |
| **소량의 라벨링 데이터에도 강건함**  | 의료 데이터셋은 대체로 적은 양이므로 매우 적합        |

### 📌 3. **UNet이 쓰인 대표적인 의료 AI 연구 사례**

#### 🩺 (1) **Liver Tumor Segmentation (LiTS Challenge)**

* CT 영상에서 간과 종양을 분할
* UNet 및 변형 모델로 참가자 다수 수상

#### 🧠 (2) **Brain Tumor Segmentation (BraTS Challenge)**

* MRI 영상에서 뇌종양 부위를 픽셀 단위로 분할
* UNet 변형 (3D UNet, Attention UNet 등)이 주력으로 활용

#### 👁 (3) **Retinal Vessel Segmentation**

* 안저(fundus) 영상에서 혈관 구조 분할
* UNet으로 혈관 경계 정밀 분할 가능

#### 🩸 (4) **Skin Lesion Segmentation**

* 피부 병변 자동 경계 추출 (ISIC 데이터셋 기반)
* UNet 기반 구조가 다양한 논문에 채택됨

### 💡 Final Project 제안 요약 (학부생 수준에서 가능)

| 항목           | 설명                                                                          |
| ------------ | --------------------------------------------------------------------------- |
| **프로젝트명**    | UNet 기반 태아 머리 초음파 영상 분할 모델 개발                                               |
| **사용 데이터**   | Large-Scale Annotation Dataset for Fetal Head Biometry in Ultrasound Images |
| **목표**       | 뇌(BRAIN), CSP, LV 영역 자동 분할                                                  |
| **활용 기술**    | UNet, PyTorch or TensorFlow, Dice Loss, IoU, 시각화 (Matplotlib or Streamlit)  |
| **보너스 아이디어** | 원본 이미지로 복원해서 정확도 비교, 다양한 augmentation 기법 실험 등                               |


### 📘 도움이 될 자료

* 논문: [UNet: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
* GitHub 예제: `segmentation_models_pytorch` 또는 `keras-unet` 같은 오픈소스 활용 가능


프로젝트 아이디어를 구체화해보고 싶다면, 모델 아키텍처 설계나 학습 코드 구조도 도와줄 수 있어요. 언제든지 질문 주세요!

---

## 사용 데이터셋  
링크: https://zenodo.org/records/8265464  
이 데이터셋 **Large-Scale Annotation Dataset for Fetal Head Biometry in Ultrasound Images**는 산전 진단에 활용되는 태아 머리 초음파 이미지를 기반으로, **객관적이고 정량적인 생체 계측 알고리즘 연구를 지원**하기 위해 구축된 공개 의료영상 데이터셋입니다. 주요 특징은 다음과 같습니다:

### 🔍 **기본 정보**

* **총 이미지 수:** 3,832장
* **이미지 해상도:** 959 x 661 픽셀
* **라이선스:** CC BY 4.0 (저작자 표시 및 수정사항 명시 필요)
* **원본 이미지 보존:** 마스크나 라벨의 정밀 정렬을 위해 다양한 원본 이미지 크기도 함께 제공

### 📂 **데이터 구성**

* **4개의 주요 폴더**:

  * Trans-thalamic
  * Trans-ventricular
  * Trans-cerebellum
  * 기타 태아 머리 영상
* 각 폴더에는:

  * 원본 이미지
  * 픽셀당 mm 단위를 담은 CSV 파일
  * **11가지 포맷의 하위 폴더** (Cityscapes, YOLO, COCO 등)를 포함
* **사전 학습/검증 분할 없음**: 연구자가 직접 분할 설정 가능

### 🧠 **주석 클래스 정보**

* **Brain (뇌):** 3,794건
* **CSP (투명중격조):** 1,865건
* **LV (측뇌실):** 1,512건

### 📑 **주요 특징 및 장점**

* **윤리적 데이터 활용:** 익명화된 공공 이미지 사용, 개인 정보 보호 준수
* **재현성 보장:** 주석자, 검증 절차, 피드백 기록 문서화
* **오픈 사이언스 지향:** 자유로운 활용과 공동연구 장려
* **다양한 활용 가능성:** 분류, 객체 탐지, 분할 등 다양한 컴퓨터 비전 과제에 적합

### 🤝 **활용 시 주의사항 및 협업**

* 사용 시 **적절한 인용 및 저작자 표시** 필요
* 마스크나 예측 결과를 원본 이미지에 정밀하게 매핑할 경우, **원본 사이즈 이미지 사용 권장**
* 데이터셋 관련 문의: **[mahmood.phd@ieee.org](mailto:mahmood.phd@ieee.org)**

이 데이터셋은 의료 영상 분석 알고리즘을 연구하고 평가하는 데 최적화되어 있으며, 특히 **태아 생체 계측, 구조 분할, 객체 탐지 모델 개발에 매우 유용한 리소스**입니다.

---

## 모델 아키텍쳐  
물론이에요! 아래는 **PyTorch 기반 UNet 아키텍처 설계 예시**입니다. 학부 수준에서도 이해하고 활용할 수 있도록 **모듈화 구조**로 설계했으며, 이 데이터셋(태아 머리 초음파 영상 분할)에 맞춰 출력 채널 수는 클래스 수에 맞게 조정할 수 있어요.

### ✅ UNet 아키텍처 예시 (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((d4, e4), dim=1))

        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((d3, e3), dim=1))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((d2, e2), dim=1))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((d1, e1), dim=1))

        return self.final_conv(d1)
```

### 🧪 주요 설정

| 항목          | 값                                               |
| ----------- | ----------------------------------------------- |
| **입력 채널 수** | 1 (흑백 초음파 영상)                                   |
| **출력 채널 수** | 3 (Brain, CSP, LV 각각 클래스 마스크)                   |
| **출력 형태**   | shape = `(batch, 3, H, W)` → softmax 후 클래스 분할   |
| **손실 함수**   | CrossEntropyLoss 또는 DiceLoss (segmentation에 적합) |
| **평가지표**    | Dice Coefficient, IoU, Accuracy 등               |


### 🧱 전체 구조 요약

```
입력 (1채널) → Encoder (Downsampling) → Bottleneck → Decoder (Upsampling + Skip) → 출력 (3채널 분할맵)
```

필요하다면 다음 단계도 도와줄 수 있어요:

* 학습 루프 (`train()`, `validate()` 함수 구조)
* 전처리 파이프라인 (`Albumentations` or `torchvision.transforms`)
* 시각화 코드 (예: 마스크 overlay, IoU 그래프 등)

이 모델을 기반으로 프로젝트를 진행하면 좋은 결과가 나올 거예요. 더 구체적인 도움도 필요하면 편하게 말해줘요!


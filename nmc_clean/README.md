# NMC (Neural Multi-Class) Refactored Code

## Overview

이 저장소는 NMC (Neural Multi-Class) 시스템의 정리된 코드를 포함합니다. 기존 코드에서 불필요한 출력과 중복 코드를 제거하고, 핵심 기능만을 포함하여 깔끔하게 정리했습니다.

## 📁 Directory Structure

```
nmc_clean/
├── core/           # 핵심 NMC 모듈
├── configs/        # 설정 파일들
├── notebooks/      # 정리된 Jupyter 노트북들
├── tools/          # 평가 및 유틸리티 도구
├── scripts/        # 실행 스크립트
└── README.md       # 이 파일
```

## 🔧 Core Modules

### Models
- **EfficientNetV2**: EfficientNetV2 기반 모델들
  - `EfficientNetV2MModel`: 단일 라벨 분류용
  - `EfficientNetV2MModelMulti`: 다중 라벨 분류용
- **ResNet**: ResNet 기반 모델들
  - `ResNet50Model`: 단일 라벨 분류용
  - `ResNet50MultiHeadModel`: 다중 라벨 분류용
- **FGMaxxVit**: FGMaxxVit 기반 모델들
  - `FGMaxxVit`: 단일 라벨 분류용
  - `FGMaxxVit_Multi`: 다중 라벨 분류용
- **TestCNN**: 테스트용 간단한 CNN 모델

### Utils
- **augmentations.py**: 데이터 증강 함수들
- **losses.py**: 손실 함수들
- **metrics.py**: 평가 메트릭들
- **optimizers.py**: 최적화 알고리즘들
- **schedulers.py**: 학습률 스케줄러들
- **utils/**: 기타 유틸리티 함수들

## 📊 Configuration Files

- **NMC.yaml**: NMC 데이터셋 학습 설정
- **APTOS.yaml**: APTOS 데이터셋 학습 설정
- **ODIR.yaml**: ODIR 데이터셋 학습 설정
- **Multi_Task.yaml**: 다중 작업 학습 설정

## 📓 Jupyter Notebooks

### NMC 관련
- **NMC.ipynb**: 기본 NMC 모델 학습 및 평가
- **NMC_singlelabel.ipynb**: 단일 라벨 NMC 학습
- **NMC_labelchain.ipynb**: 라벨 체인 기반 NMC 학습
- **NMC_confusion.ipynb**: 혼동 행렬 분석

### APTOS 관련
- **APTOS.ipynb**: 기본 APTOS 모델 학습
- **APTOS_singlelabel.ipynb**: 단일 라벨 APTOS 학습
- **APTOS_NMC_finetuning.ipynb**: NMC로 APTOS 파인튜닝
- **NMC_APTOS_finetuning.ipynb**: APTOS로 NMC 파인튜닝

### 시각화 및 분석
- **NMC_APTOS_visualization.ipynb**: 시각화 도구
- **NMC_APTOS_gradcam.ipynb**: Grad-CAM 분석
- **NMC_APTOS_OSM.ipynb**: OSM (Object Saliency Map) 분석

### 특수 모델
- **NMC_APTOS_BIFPN.ipynb**: BIFPN (Bidirectional Feature Pyramid Network) 모델
- **NMC_APTOS_FPN.ipynb**: FPN (Feature Pyramid Network) 모델

## 🛠️ Tools

- **val.py**: 모델 평가 도구
- **episodic_utils.py**: 에피소딕 학습 유틸리티

## 📜 Scripts

실행 가능한 스크립트들 (구체적인 내용은 각 파일 참조)

## 🚀 Quick Start

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   ```

2. **설정 파일 확인**
   - `configs/NMC.yaml`에서 데이터 경로 및 모델 설정 확인

3. **노트북 실행**
   - `notebooks/` 폴더의 원하는 노트북 실행

## ⚠️ Important Notes

- 이 코드는 **정리된 버전**으로, 원본 코드의 출력과 불필요한 부분을 제거했습니다
- **원본 코드는 변경되지 않았습니다**
- 실행 전 설정 파일의 경로를 올바르게 설정해야 합니다
- GPU 환경이 필요합니다 (CUDA 지원)

## 🔗 Dependencies

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- PIL (Pillow)
- OpenCV
- tabulate
- tqdm
- PyYAML

## 📝 License

원본 프로젝트의 라이선스를 따릅니다.

---

**이 코드는 NMC 시스템의 핵심 기능만을 포함한 정리된 버전입니다.**

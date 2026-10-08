# CompMatte for Nuke (v3.0)
> **헐리우드 VFX 스튜디오 컴포지팅 구조를 충실히 구현한 Nuke 전용 광학 알파 매팅 툴킷**  
> **100% 순수 광학/수학 알고리즘 (PyTorch / ViTMatte 의존성 0% • 초경량 • 60fps+ 실시간)**

---

## 💡 프로젝트 개요 (Overview)

기존 Nuke에서 그린/블루스크린 키잉을 할 때 아티스트들은 다음과 같은 반복적인 고통을 겪었습니다:
1. **IBK 수작업 번거로움**: IBKColour 노드를 여러 개 연결하고 패치 레벨(1~5)을 일일이 조절하여 클린 플레이트를 만드는 데 많은 시간이 소요됨.
2. **코어 매트 내부 구멍(Holes)**: 피사체 내부 음영이나 옷감 질감으로 인해 알파에 구멍이 뚫리고 프레임마다 자글거림(Jitter) 발생.
3. **잔머리 침식**: 내부 구멍을 메우기 위해 필터를 주면 가느다란 1px 머리카락과 모션 블러가 깎여나감.

**CompMatte for Nuke**는 이러한 문제를 단 하나의 노드로 해결합니다:
- **Zero PyTorch / Zero AI 의존성**: 무거운 딥러닝(ViTMatte), CUDA VRAM 점유, 수 기가바이트 모델 다운로드가 전혀 없습니다.
- **Topological Hole-Filling**: 외곽 윤곽선을 감지하여 피사체 내부의 모든 구멍을 100% Solid Pure White (1.0)로 잠금.
- **Safe Zone Edge Detail Re-Injection**: 안전 영역(Safe Zone) 내에서 1px 미세 잔머리와 모션 블러 디테일을 비파괴 Max 결합하여 **머리카락 손실률 0% 달성**.
- **초고속 60fps+ CPU/NumPy 연산**: Nuke 내장 Python 환경에서 번개처럼 빠르게 작동합니다.

---

## ⚡ 설치 방법 (Installation)

AutoRoto와 동일하게 Nuke 사용자 디렉토리에 클론하여 간편하게 설치할 수 있습니다.

### 1) GitHub 레포지토리 다운로드 / 클론
Nuke의 사용자 플러그인 디렉토리(~/.nuke/)에 레포지토리를 클론합니다:
`ash
cd ~/.nuke
git clone https://github.com/Kiwoo0413/CompMatte_for_Nuke.git
`
*(폴더 이름은 반드시 CompMatte_for_Nuke여야 합니다.)*

### 2) Nuke init.py 설정
~/.nuke/init.py 파일(없으면 새로 생성)에 아래 1줄을 추가합니다:
`python
import nuke
nuke.pluginAddPath('CompMatte_for_Nuke')
`

---

## 🎬 Nuke에서 사용하기 (How to Use)

1. **Foundry Nuke (13.0 ~ 17.x+)**를 실행합니다.
2. 노드를 생성하는 3가지 방법:
   - **단축키**: Ctrl + Alt + M
   - **좌측 툴바**: Nodes -> CompMatte -> Create CompMatte Node 또는 Keyer -> CompMatte
   - **Tab 검색**: 노드 그래프(DAG)에서 Tab 키를 누르고 CompMatte 입력
3. 노드 파이프라인 연결:
   - **Source (Input 0)**: 원본 그린/블루스크린 영상 플레이트 연결
   - **CleanPlate (Input 1 - 선택)**: 준비된 레퍼런스 스크린이 있을 경우 연결 (미연결 시 자동 인페인팅 생성)
   - **Holdout (Input 2 - 선택)**: 추가 보호가 필요한 러프 로토 마스크가 있을 경우 연결

---

## 🎛️ 주요 노드 파라미터 탭 안내

### 1. Tab 1: CompMatte (메인 탭)
- **Screen Type**: green / lue / custom (스크린 색상 선택)
- **View Output**:
  - Final Alpha (rgba.a): 완성된 고품질 알파 마스크
  - Premultiplied RGBA: 최종 알파가 곱해진 합성용 RGBA
  - Clean Plate: 자동 균등화된 배경 레퍼런스 플레이트
  - Core Matte: 내부 구멍이 메워진 코어 알파
  - Edge Matte: 광학 색상차 기반 미세 투과 엣지
- **Red / Blue Weight**: 색상차 키잉 밸런스 가중치 (기본값: 0.5)
- **⚡ Extract Matte (Current Frame)**: 현재 프레임의 알파 마스크를 즉시 추출 및 잠금
- **🎬 Bake Frame Range...**: 지정한 타임라인 구간의 알파 시퀀스를 백그라운드 캐시로 일괄 베이크

### 2. Tab 2: Clean Plate (IBK 레퍼런스 스크린)
- **Patch Size**: 인페인팅 확장 커널 크기 (기본값: 5)
- **Patch Iterations**: 다단계 계층 확장 반복 횟수 (기본값: 4)
- **Blur Radius**: 조명 불균일 균등화 블러 반경 (기본값: 3)

### 3. Tab 3: Core & Edge Fusion (디테일 보존 및 홀 채우기)
- **Topological Hole-Filling**: 내부 구멍 자동 메움 (기본 활성화, Pure 1.0 코어 잠금)
- **Core Inset / Erode**: 코어가 외곽선 바깥으로 튀어나가지 않도록 안쪽으로 수축하는 픽셀 수 (기본값: 7)
- **Safe Zone Edge Detail Re-Injection**: 잔머리 비파괴 재주입 활성화 (머리카락 100% 보존)
- **Safe Zone Radius (px)**: 코어 주변 잔머리 재주입 안전 반경 (기본값: 40px)
- **Black Cutoff (Pure 0.0)**: 배경 잔여 노이즈를 완전한 순수 블랙으로 닫음 (기본값: 0.05)
- **White Cutoff (Pure 1.0)**: 피사체 내부를 완전한 순수 화이트로 고정 (기본값: 0.95)

---

## 🧪 유닛 테스트 실행 (TDD Validation)

`ash
# 10개 합성/수학 검증 테스트 실행 (실행 시간: ~0.02초)
python -m unittest discover -s tests
`

---

## 📄 라이선스
Apache 2.0 License

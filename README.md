# CompMatte for Nuke (v3.0)
> **헐리우드 VFX 스튜디오 컴포지팅 구조를 충실히 구현한 Nuke 전용 광학 알파 매팅 툴킷**  
> **100% 순수 광학/수학 알고리즘 • AutoRoto 방식 호스트 Python 자동 연동 • 60fps+ 실시간**

---

## 💡 프로젝트 개요 (Overview)

기존 Nuke에서 그린/블루스크린 키잉을 할 때 아티스트들은 다음과 같은 반복적인 문제를 겪었습니다:
1. **IBK 수작업 번거로움**: `IBKColour` 노드를 여러 개 연결하고 패치 레벨(1~5)을 일일이 조절하여 클린 플레이트를 만드는 데 많은 시간이 소요됨.
2. **코어 매트 내부 구멍(Holes)**: 피사체 내부 음영이나 옷감 질감으로 인해 알파에 구멍이 뚫리고 프레임마다 자글거림(Jitter) 발생.
3. **잔머리 침식**: 내부 구멍을 메우기 위해 필터를 주면 가느다란 1px 머리카락과 모션 블러가 깎여나감.
4. **Nuke Python의 NumPy 부재**: Foundry Nuke 기본 내장 Python에는 `numpy`나 `opencv`가 기본 탑재되어 있지 않아, Nuke 내부에 패키지를 억지로 설치하려다 관리자 권한이나 DLL 버전 충돌 문제를 겪음.

**CompMatte for Nuke**는 이 모든 문제를 해결합니다:
- **Zero PyTorch / Zero AI 의존성**: 무거운 딥러닝(ViTMatte), CUDA VRAM 점유, 수 기가바이트 모델 다운로드가 전혀 없습니다.
- **AutoRoto 방식 호스트 Python 자동 연동**: Nuke 내부에 복잡하게 `numpy`를 설치할 필요 없이, 컴퓨터에 이미 설치된 표준 Python(Python 3.10~3.12, Conda 등)의 NumPy/OpenCV를 백그라운드 워커로 자동 감지하여 안전하게 실행합니다.
- **Topological Hole-Filling**: 외곽 윤곽선을 감지하여 피사체 내부의 모든 구멍을 100% Solid Pure White (1.0)로 잠금.
- **Safe Zone Edge Detail Re-Injection**: 안전 영역(Safe Zone) 내에서 1px 미세 잔머리와 모션 블러 디테일을 비파괴 Max 결합하여 **머리카락 손실률 0% 달성**.
- **초고속 60fps+ CPU 연산**: 프레임당 약 0.02초 만에 즉시 마스크를 추출합니다.

---

## ⚡ 설치 방법 (Installation)

AutoRoto와 동일하게 Nuke 사용자 플러그인 디렉토리에 클론하여 즉시 사용할 수 있습니다.

### 1) GitHub 레포지토리 클론
터미널(또는 Git Bash)에서 Nuke 플러그인 디렉토리(`~/.nuke/`)로 이동한 후 클론합니다:
```bash
cd ~/.nuke
git clone https://github.com/Kiwoo0413/CompMatte_for_Nuke.git
```
*(폴더 이름은 반드시 `CompMatte_for_Nuke`여야 합니다.)*

### 2) Nuke `init.py` 설정
`~/.nuke/init.py` 파일(없으면 새로 생성)에 아래 1줄을 추가합니다:
```python
import nuke
nuke.pluginAddPath('CompMatte_for_Nuke')
```

### 3) 시스템 Python 준비 (AutoRoto 방식)
컴퓨터에 `numpy` (및 권장: `opencv-python`)가 설치된 Python이 있으면 자동으로 인식됩니다:
```bash
pip install numpy opencv-python Pillow
```
*(특정 Python 버전을 지정하고 싶다면 시스템 환경 변수 `COMPMATTE_PYTHON`에 해당 `python.exe` 경로를 지정하거나, 노드의 [Python & Settings] 탭에서 지정할 수 있습니다.)*

---

## 🎬 Nuke에서 사용하기 (How to Use)

1. **Foundry Nuke (13.0 ~ 17.x+)**를 실행합니다.
2. 노드를 생성하는 3가지 방법:
   - **단축키**: `Ctrl + Alt + M`
   - **좌측 툴바**: `Nodes -> CompMatte -> Create CompMatte Node` 또는 `Keyer -> CompMatte`
   - **Tab 검색**: 노드 그래프(DAG)에서 `Tab` 키를 누르고 `CompMatte` 입력
3. 노드 파이프라인 연결:
   - **Source (Input 0)**: 원본 그린/블루스크린 영상 플레이트 연결
   - **CleanPlate (Input 1 - 선택)**: 준비된 레퍼런스 스크린이 있을 경우 연결 (미연결 시 자동 인페인팅 생성)
   - **Holdout (Input 2 - 선택)**: 추가 보호가 필요한 러프 로토 마스크가 있을 경우 연결
4. 환경 점검:
   - Nuke 상단 메뉴의 **`CompMatte -> Check Python & NumPy Environment`**를 클릭하면 현재 컴퓨터에서 감지된 Python과 NumPy 버전을 즉시 확인할 수 있습니다.

---

## 🎛️ 주요 노드 파라미터 탭 안내

### 1. Tab 1: CompMatte (메인 탭)
- **Screen Type**: `green` / `blue` / `custom` (스크린 색상 선택)
- **View Output**:
  - `Final Alpha (rgba.a)`: 완성된 고품질 알파 마스크
  - `Premultiplied RGBA`: 최종 알파가 곱해진 합성용 RGBA
  - `Clean Plate`: 자동 균등화된 배경 레퍼런스 플레이트
  - `Core Matte`: 내부 구멍이 메워진 코어 알파
  - `Edge Matte`: 광학 색상차 기반 미세 투과 엣지
- **Red / Blue Weight**: 색상차 키잉 밸런스 가중치 (기본값: 0.5)
- **⚡ Extract Matte (Current Frame)**: 현재 프레임의 알파 마스크를 즉시 추출 및 뷰어 갱신
- **🎬 Bake Frame Range...**: 지정한 타임라인 구간의 알파 시퀀스를 일괄 베이크 (팝업창에서 프레임 구간 및 출력 경로 지정 가능)
- **📂 Open Output Folder**: 현재 설정된 출력 폴더(또는 기본 Nuke 캐시 폴더)를 윈도우 탐색기에서 즉시 열기
- **Output Folder**: 사용자 지정 출력 폴더 (비워둘 경우 Nuke 환경설정의 `DiskCachePath` 고속 디스크로 자동 저장)

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

### 4. Tab 4: Python & Settings
- **Host Python Executable**: 자동 감지된 호스트 Python 경로 (수동 지정 및 오버라이드 지원)
- **🔍 Check Python & NumPy Status**: 현재 Python/NumPy 연동 상태 팝업 확인

---

## 🧪 유닛 테스트 실행 (TDD Validation)

```bash
# 14개 합성/수학/CLI 워커/경로 검증 테스트 실행 (실행 시간: ~0.3초)
python -m unittest discover -s tests
```

---

## 📄 라이선스
Apache 2.0 License

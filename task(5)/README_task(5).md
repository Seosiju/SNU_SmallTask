# Task(5) AI-Powered Image Search System

## 📋 프로젝트 개요

이 프로젝트는 OpenAI의 GPT-4 Vision과 텍스트 임베딩 모델을 활용하여 이미지 검색 시스템을 구현한 것입니다. 자연어 쿼리를 통해 이미지 컬렉션에서 가장 관련성 높은 이미지를 찾아주는 AI 기반 검색 엔진입니다.

## 🚀 주요 기능

- **자연어 이미지 검색**: 텍스트 설명으로 이미지를 검색
- **AI 기반 캡션 생성**: GPT-4 Vision을 사용한 자동 이미지 캡션 생성
- **의미적 유사도 검색**: 텍스트 임베딩을 활용한 코사인 유사도 기반 검색
- **대화형 검색 인터페이스**: 실시간 검색 및 결과 시각화
- **다양한 이미지 형식 지원**: JPG, JPEG, PNG 등

## 🛠️ 기술 스택

- **OpenAI API**: GPT-4 Vision, GPT-4o-mini, text-embedding-3-small
- **Python**: 주요 프로그래밍 언어
- **NumPy**: 수치 계산 및 벡터 연산
- **PIL (Pillow)**: 이미지 처리
- **IPython**: Jupyter 노트북 환경

## 📁 프로젝트 구조

```
task5/
├── img/                    # 이미지 파일들
│   ├── 01.jpg
│   ├── 02.jpg
│   ├── 03.jpg
│   ├── 04.jpeg
│   ├── 05.jpg
│   └── 06.jpg
├── task(5).ipynb          # 메인 노트북 파일
└── README.md              # 프로젝트 문서
```

## 🔧 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install openai python-dotenv numpy pillow
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 이미지 준비

`img/` 폴더에 검색하고자 하는 이미지들을 추가하세요. 지원되는 형식: JPG, JPEG, PNG

## 💻 사용 방법

### 1. 기본 설정

```python
# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI()
```

### 2. 이미지 인덱싱

```python
# 이미지 캡션 생성 및 임베딩
for fname in IMAGE_FILES:
    path = os.path.join(IMAGE_DIR, fname)
    cap = caption_image(path)
    images.append({"file": fname, "caption": cap})

# 캡션 임베딩 생성
caption_texts = [d["caption"] for d in images]
caption_embs = embed_texts(caption_texts)
```

### 3. 이미지 검색

```python
# 검색 실행
query = "블랙베리 스마트폰"
results = search_images(query, top_k=3)
show_results(results)
```

### 4. 대화형 검색

노트북의 마지막 셀을 실행하여 대화형 검색 모드를 시작할 수 있습니다:

```python
# 대화형 검색 루프
while True:
    query = input("검색하고 싶은 이미지를 설명해주세요: ")
    if query.lower() in ['quit', 'exit', '종료', 'q']:
        break
    results = search_images(query, top_k=TOP_K)
    show_results(results)
```

## 🔍 핵심 함수 설명

### `caption_image(path, system_prompt=None)`

- **기능**: 이미지에 대한 한국어 캡션 생성
- **매개변수**:
  - `path`: 이미지 파일 경로
  - `system_prompt`: 선택적 시스템 프롬프트
- **반환값**: 생성된 캡션 문자열

### `search_images(query, top_k=3)`

- **기능**: 자연어 쿼리로 이미지 검색
- **매개변수**:
  - `query`: 검색 쿼리 문자열
  - `top_k`: 반환할 상위 결과 개수
- **반환값**: `[(파일명, 유사도, 캡션), ...]` 형태의 결과 리스트

### `show_results(results, k=3)`

- **기능**: 검색 결과를 시각적으로 표시
- **매개변수**:
  - `results`: 검색 결과 리스트
  - `k`: 표시할 결과 개수

## ⚙️ 설정 옵션

```python
# 주요 설정값들
IMAGE_DIR = "img"                    # 이미지 디렉토리
VISION_MODEL = "gpt-4o-mini"        # Vision 모델 (캡션 생성용)
EMB_MODEL = "text-embedding-3-small" # 임베딩 모델
TOP_K = 3                           # 상위 k개 결과 반환
```

## 📊 검색 예시

### 입력 쿼리: "블랙베리"

```
[Top-1] 05.jpg  (similarity=0.4425)
  caption: 블랙베리 스마트폰이 손에 들려 있다.

[Top-2] 06.jpg  (similarity=0.3764)
  caption: 블랙베리 스마트폰, 물리적 키패드가 특징인 디자인입니다.

[Top-3] 01.jpg  (similarity=0.2297)
  caption: 아이폰과 화이트 배터리 케이스가 나란히 있다.
```

### 입력 쿼리: "삼성"

```
[Top-1] 03.jpg  (similarity=0.3801)
  caption: 새로운 삼성 스마트폰, 두 개의 카메라와 화면.

[Top-2] 04.jpeg  (similarity=0.2019)
  caption: 스마트폰과 카메라 렌즈가 있는 테이블 위의 이미지.

[Top-3] 02.jpg  (similarity=0.1978)
  caption: 아이폰 두 대가 나란히 놓여 있습니다.
```

## 🔧 커스터마이징

### 1. 이미지 파일 추가

`IMAGE_FILES` 리스트에 새로운 이미지 파일명을 추가하세요:

```python
IMAGE_FILES = ["01.jpg", "02.jpg", "03.jpg", "04.jpeg", "05.jpg", "06.jpg", "07.png"]
```

### 2. 모델 변경

다른 OpenAI 모델을 사용하려면 설정을 변경하세요:

```python
VISION_MODEL = "gpt-4o"              # 더 정확한 캡션 생성
EMB_MODEL = "text-embedding-3-large" # 더 정확한 임베딩
```

### 3. 검색 결과 개수 조정

`TOP_K` 값을 변경하여 반환할 결과 개수를 조정할 수 있습니다.

## ⚠️ 주의사항

1. **API 비용**: OpenAI API 사용에 따른 비용이 발생합니다.
2. **이미지 크기**: 큰 이미지는 처리 시간이 오래 걸릴 수 있습니다.
3. **API 제한**: OpenAI API의 속도 제한을 고려하여 사용하세요.
4. **한국어 지원**: 현재 한국어 캡션 생성에 최적화되어 있습니다.

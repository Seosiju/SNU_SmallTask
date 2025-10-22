# Task(7-8) OpenAI Function Calling을 활용한 날씨 챗봇

이 폴더는 OpenAI의 Function Calling 기능을 활용하여 날씨 정보를 제공하는 챗봇을 구현한 프로젝트입니다.

## 📁 파일 구조

```
task(7-8)/
├── task(7).ipynb    # Function Calling 기본 구현
├── task(8).ipynb    # 대화형 날씨 챗봇 구현
└── README.md        # 프로젝트 설명서
```

## 🎯 프로젝트 목표

- OpenAI Function Calling API를 활용한 날씨 정보 제공 시스템 구현
- OpenWeatherMap API와의 연동을 통한 실시간 날씨 데이터 활용
- 대화형 인터페이스를 통한 사용자 친화적 날씨 챗봇 개발

## 📋 주요 기능

### Task 7: Function Calling 기본 구현

- **Function Calling 설정**: OpenAI API의 function calling 기능을 활용한 날씨 API 연동
- **API 연동**: OpenWeatherMap API를 통한 실시간 날씨 데이터 수집
- **단일 질의 처리**: 사용자의 날씨 질문에 대한 일회성 응답 제공

### Task 8: 대화형 날씨 챗봇

- **대화 루프**: 연속적인 대화를 지원하는 인터랙티브 챗봇
- **다국어 지원**: 한국어와 영어 질문 모두 처리 가능
- **다양한 날씨 정보**: 현재 날씨와 N일 예보 모두 제공

## 🛠️ 기술 스택

### 핵심 라이브러리

- **OpenAI API**: GPT-4o-mini 모델을 활용한 자연어 처리
- **OpenWeatherMap API**: 실시간 날씨 데이터 제공
- **Python-dotenv**: 환경 변수 관리
- **Requests**: HTTP API 호출
- **Tenacity**: API 재시도 로직
- **Termcolor**: 콘솔 출력 색상화

### 지원 기능

- **Function Calling**: OpenAI의 도구 호출 기능
- **재시도 메커니즘**: API 호출 실패 시 자동 재시도
- **에러 핸들링**: API 오류 상황 처리
- **대화 기록 관리**: 컨텍스트 유지를 위한 메시지 히스토리

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install openai python-dotenv requests tenacity termcolor

# 환경 변수 설정 (.env 파일)
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
```

### 2. 실행 방법

#### Task 7: 기본 Function Calling

```python
# Jupyter Notebook에서 task(7).ipynb 실행
# 서울의 3일간 날씨 예보를 요청하는 예제
```

#### Task 8: 대화형 챗봇

```python
# Jupyter Notebook에서 task(8).ipynb 실행
# 대화형 인터페이스로 날씨 정보 조회
```

## 🔧 구현된 함수들

### 1. `get_current_weather(location, format)`

- **기능**: 현재 날씨 정보 조회
- **매개변수**:
  - `location`: 도시명 (예: "Seoul,KR")
  - `format`: 온도 단위 ("celsius" 또는 "fahrenheit")
- **반환값**: OpenWeatherMap API의 현재 날씨 데이터

### 2. `get_n_day_weather_forecast(location, num_days, format)`

- **기능**: N일간 날씨 예보 조회
- **매개변수**:
  - `location`: 도시명
  - `num_days`: 예보 일수
  - `format`: 온도 단위
- **반환값**: OpenWeatherMap API의 예보 데이터

## 💡 주요 특징

### Function Calling 활용

- OpenAI의 Function Calling을 통해 외부 API와의 자연스러운 연동
- 사용자 질문을 분석하여 적절한 함수 자동 선택
- 함수 실행 결과를 바탕으로 자연어 응답 생성

### 대화형 인터페이스

- 연속적인 대화를 통한 다양한 날씨 정보 제공
- 컨텍스트 유지를 통한 맥락적 응답
- 사용자 친화적인 한국어 인터페이스

### 에러 처리 및 안정성

- API 호출 실패 시 자동 재시도 메커니즘
- 네트워크 오류 및 인증 오류 처리
- 사용자에게 명확한 오류 메시지 제공

## 📊 예시 사용법

### 기본 질문 예시

```
사용자: "오늘 서울 날씨는 어때?"
봇: "오늘 서울의 날씨는 맑습니다. 현재 기온은 27도 정도이며..."

사용자: "인천은 어때? 그리고 내일 날씨도 알려줘"
봇: "현재 인천의 날씨는 맑습니다... 내일 인천의 날씨는..."
```

### 지원하는 질문 유형

- 현재 날씨 조회
- 특정 도시의 날씨 정보
- N일간 날씨 예보
- 온도 단위 변경 (섭씨/화씨)
- 다국어 질문 (한국어/영어)

## 🔍 디버깅 및 모니터링

### 대화 흐름 시각화

- `pretty_print_conversation()` 함수를 통한 대화 과정 시각화
- 시스템 메시지, 사용자 입력, 어시스턴트 응답, 함수 호출을 색상으로 구분
- 디버깅 및 학습 목적으로 활용 가능

### API 상태 모니터링

- OpenWeatherMap API 연결 상태 확인
- API 키 유효성 검증
- 네트워크 연결 상태 체크

## 📈 확장 가능성

이 프로젝트는 다음과 같은 방향으로 확장 가능합니다:

1. **다른 API 연동**: 뉴스, 주식, 교통 정보 등 다양한 외부 API 추가
2. **음성 인터페이스**: 음성 인식 및 음성 합성을 통한 대화형 인터페이스
3. **웹 인터페이스**: 웹 애플리케이션으로의 확장
4. **데이터베이스 연동**: 사용자 설정 및 대화 기록 저장
5. **알림 기능**: 특정 날씨 조건에서의 자동 알림

## 📝 주의사항

- OpenAI API와 OpenWeatherMap API 키가 필요합니다
- API 사용량에 따른 비용이 발생할 수 있습니다
- 네트워크 연결이 필요합니다
- 일부 도시명은 영어로 입력해야 할 수 있습니다

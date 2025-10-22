# 실습 9: 카페 Kiosk 자연어 인터페이스 (RAG)

## 📋 개요

이 실습은 RAG(Retrieval-Augmented Generation) 기술을 활용하여 카페 키오스크용 자연어 인터페이스를 구현하는 프로젝트입니다. 프론트엔드 없이 질의 리스트를 입력하면 결과 리포트를 생성하는 시스템을 구축합니다.

## 🎯 주요 기능

- **메뉴 데이터 관리**: 음료, 디저트, 옵션 정보를 JSON 형태로 관리
- **벡터 검색**: multilingual-e5-base 모델을 사용한 의미 기반 검색
- **인텐트 분류**: 정규식 기반 의도 분류 (주문, 옵션, 추천, 복합 질의 등)
- **장바구니 관리**: 주문 항목 추가, 옵션 적용, 가격 계산
- **LLM 통합**: OpenAI API를 활용한 자연스러운 응답 생성
- **대화형 인터페이스**: 실시간 채팅 형태의 키오스크 시연

## 🛠️ 기술 스택

- **임베딩 모델**: `intfloat/multilingual-e5-base`
- **벡터 데이터베이스**: FAISS
- **LLM**: OpenAI GPT-4o-mini
- **프레임워크**: LangChain, Sentence Transformers
- **데이터 처리**: Pandas, JSON

## 📁 파일 구조

```
task(9)/
├── task(9).ipynb          # 메인 실습 노트북
├── task(9)_presentation.pdf  # 발표 자료
└── README.md              # 이 파일
```

## 🚀 실행 방법

### 1. 환경 설정

```bash
pip install sentence-transformers faiss-cpu langchain langchain-openai python-dotenv
```

### 2. 실행 순서

1. **데이터 생성**: 메뉴 데이터를 JSON 파일로 생성
2. **인덱스 구축**: 벡터 임베딩 및 FAISS 인덱스 생성
3. **로직 로딩**: 챗봇 핵심 로직 구현
4. **배치 리포트**: 질의 리스트 자동 처리
5. **시연**: 대화형 인터페이스 테스트

## 📊 메뉴 데이터 구조

### 음료 (Drinks)
- 11종의 커피/논커피 음료
- 3가지 사이즈 (S, M, L)
- 가격, 설명, 페어링, 태그 정보 포함

### 디저트 (Desserts)
- 8종의 디저트
- 가격 및 설명 정보

### 옵션 (Options)
- 샷 추가, 시럽, 우유 변경, 휘핑 등
- 적용 가능한 메뉴 타입별 분류

## 🔍 인텐트 분류

| 인텐트 | 패턴 | 설명 |
|--------|------|------|
| `order` | "주세요", "주문", "살게요" | 메뉴 주문 |
| `option` | "추가", "변경", "우유", "샷" | 옵션 관련 |
| `recommend` | "추천", "달콤", "인기" | 추천 요청 |
| `complex` | "그리고", "비교", "중에 뭐가" | 복합 질의 |
| `menu_info` | "가격", "얼마", "설명" | 메뉴 정보 |
| `cart_summary` | "장바구니" | 장바구니 조회 |

## 💡 핵심 구현 사항

### 1. 벡터 검색 시스템
```python
def retrieve(q, topk=4):
    qv = emb_model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(qv, topk)
    return [{"text": store["chunks"][i], "meta": store["meta"][i], "score": float(D[0][j])} for j, i in enumerate(I[0])]
```

### 2. 장바구니 관리
```python
@dataclass
class CartItem:
    item_id: str
    name: str
    kind: str
    size: str | None
    base_price: int
    qty: int = 1
    options: List[Dict] = field(default_factory=list)
```

### 3. LLM 통합 응답 생성
- OpenAI API를 통한 자연스러운 응답 생성
- 의도별 프롬프트 템플릿 활용
- LLM 실패 시 규칙 기반 Fallback 시스템

## 🧪 테스트 케이스

### 배치 테스트
- 메뉴 정보 조회
- 추천 요청
- 주문 처리
- 옵션 문의
- 복합 질의
- 무관한 질문 처리

### 대화형 테스트
- 실시간 채팅 인터페이스
- 대화 히스토리 관리
- 자연스러운 응답 생성

## 📈 성능 특징

- **빠른 검색**: FAISS 인덱스를 통한 고속 벡터 검색
- **정확한 분류**: 정규식 기반 인텐트 분류
- **유연한 응답**: LLM + 규칙 기반 하이브리드 시스템
- **다국어 지원**: multilingual-e5-base 모델 활용

## 🔧 설정 및 커스터마이징

### 환경 변수
```bash
OPENAI_API_KEY=your_api_key_here
```

### 메뉴 데이터 수정
`menu.json` 파일을 수정하여 메뉴 정보를 업데이트할 수 있습니다.

### 인텐트 패턴 수정
`INTENT_PATTERNS` 딕셔너리를 수정하여 새로운 의도 패턴을 추가할 수 있습니다.

## 📝 사용 예시

```python
# 기본 사용법
cart = OrderManager(menu)
intent, response = answer("아메리카노 M 사이즈 주문해주세요", cart)
print(response)

# 장바구니 확인
print(cart.summary())
```

## 🎯 학습 목표

1. **RAG 시스템 구현**: 검색과 생성의 결합
2. **인텐트 분류**: 자연어 처리의 기본
3. **벡터 검색**: 의미 기반 정보 검색
4. **대화 시스템**: 상태 관리와 맥락 유지
5. **실용적 AI**: 실제 서비스에 적용 가능한 시스템

## 📚 참고 자료

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

**실습 완료 후**: 이 시스템을 기반으로 실제 카페 키오스크나 온라인 주문 시스템에 적용할 수 있습니다.

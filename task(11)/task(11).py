#!/usr/bin/env python
# coding: utf-8

# # Task(11) ProbTutor - 확률 개념 설명 및 시각화 챗봇
# 
# ## 📋 프로젝트 개요
# 
# OpenAI Assistants API와 Code Interpreter를 활용하여 대학교 1학년 수준의 확률 개념을 설명하고, 관련 문제를 해결하며, 시각화를 제공하는 대화형 확률 튜터 챗봇 시스템입니다.
# 
# ### 주요 기능
# - **확률 개념 설명**: 베이즈 정리, 조건부 확률, 확률 분포 등 핵심 개념 설명
# - **문제 해결**: Code Interpreter를 통한 정확한 수치 계산 및 해결 과정 제시
# - **시각화**: matplotlib을 활용한 확률 분포, 차트, 그래프 자동 생성
# - **대화형 학습**: 사용자 수준에 맞춘 친절한 튜터 역할
# 
# ### 기술 스택
# - **OpenAI Assistants API**: 대화 처리 및 Code Interpreter 활용
# - **Python**: 확률 계산 및 시각화
# - **matplotlib**: 그래프 및 차트 생성
# - **python-dotenv**: 환경 변수 관리
# 

# In[1]:


# 환경 설정 및 라이브러리 import
import os
import json
import time
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import io
from IPython.display import display, Image, Markdown

# 환경 변수 로드
load_dotenv()

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 전역 설정
# 전역 설정
ASSISTANT_ID = None
THREAD_ID = None
CONVERSATION_HISTORY = []  # 빈 리스트로 명시적 초기화
MAX_CONVERSATION_LENGTH = 50

print("✅ 환경 설정 완료")
print(f"OpenAI API 키 설정: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
print(f"matplotlib 버전: {plt.matplotlib.__version__}")
print(f"numpy 버전: {np.__version__}")


# In[2]:


# 1. Assistant 생성 및 설정
def create_probability_assistant():
    """
    확률 튜터 Assistant를 생성하는 함수
    
    Returns:
        str: Assistant ID
    """
    try:
        assistant = client.beta.assistants.create(
            name="ProbTutor",
            instructions="""당신은 ProbTutor입니다. 대학교 1학년 수준의 확률 개념을 설명하는 친절한 튜터입니다.

핵심 임무:
1. 확률 개념 설명: 정의, 공식, 구성 요소, 구체적 예시 제공
2. 문제 해결: Code Interpreter를 활용한 정확한 수치 계산
3. 시각화: matplotlib을 통한 그래프, 차트, 분포 시각화
4. 대화형 학습: 사용자 수준에 맞춘 친절한 톤 유지

규칙:
- 확률 개념을 명확하고 이해하기 쉽게 설명하세요
- Python 코드를 작성하여 정확한 계산을 수행하세요
- matplotlib을 사용하여 시각화를 생성하세요
- 친절하고 교육적인 톤을 유지하세요
- 사용자의 수준에 맞춰 설명의 깊이를 조절하세요

예시 응답:
- "좋아요, 함께 계산해 봅시다."
- "이 경우엔 확률 분포를 그림으로 보는 게 도움이 될 거예요."
- "베이즈 정리는 조건부 확률을 계산하는 중요한 공식입니다."
""",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
        
        global ASSISTANT_ID
        ASSISTANT_ID = assistant.id
        print(f"✅ ProbTutor Assistant 생성 완료: {ASSISTANT_ID}")
        return assistant.id
        
    except Exception as e:
        print(f"❌ Assistant 생성 중 오류: {e}")
        return None

# Assistant 생성 테스트
print("🧪 Assistant 생성 테스트:")
assistant_id = create_probability_assistant()
if assistant_id:
    print(f"  Assistant ID: {assistant_id}")
else:
    print("  ❌ Assistant 생성 실패")


# In[3]:


# 2. 확률 계산 테스트
def test_probability_calculation():
    """
    확률 계산 기능을 테스트하는 함수
    
    Returns:
        bool: 테스트 성공 여부
    """
    if not ASSISTANT_ID:
        print("❌ Assistant가 생성되지 않았습니다.")
        return False
    
    try:
        # 새 스레드 생성
        thread = client.beta.threads.create()
        thread_id = thread.id
        
        # 테스트 메시지 전송
        test_question = "동전을 3번 던져서 앞면이 2번 나올 확률을 계산해주세요. 이항분포를 사용해서 풀어주세요."
        print(f"🔍 테스트 질문: {test_question}")
        print("-" * 60)
        
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=test_question
        )
        
        # Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # 실행 완료 대기
        while run.status in ['queued', 'in_progress', 'requires_action']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            # 응답 가져오기
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            response = messages.data[0].content[0].text.value
            
            print("✅ 확률 계산 테스트 성공")
            print(f"🤖 ProbTutor 응답: {response[:200]}...")
            return True
        else:
            print(f"❌ 실행 실패: {run.status}")
            return False
            
    except Exception as e:
        print(f"❌ 확률 계산 테스트 중 오류: {e}")
        return False

# 테스트 실행
print("🧪 확률 계산 테스트:")
test_result = test_probability_calculation()
print(f"테스트 결과: {'성공' if test_result else '실패'}")


# In[4]:


# 3. 시각화 테스트
def test_visualization():
    """
    시각화 기능을 테스트하는 함수
    
    Returns:
        bool: 테스트 성공 여부
    """
    if not ASSISTANT_ID:
        print("❌ Assistant가 생성되지 않았습니다.")
        return False
    
    try:
        # 새 스레드 생성
        thread = client.beta.threads.create()
        thread_id = thread.id
        
        # 테스트 메시지 전송
        test_question = "정규분포 그래프를 그려주세요. 평균 0, 표준편차 1인 표준정규분포로 해주세요."
        print(f"🔍 테스트 질문: {test_question}")
        print("-" * 60)
        
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=test_question
        )
        
        # Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # 실행 완료 대기
        while run.status in ['queued', 'in_progress', 'requires_action']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            # 응답 가져오기
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            
            # 텍스트와 이미지를 구분해서 처리
            text_response = ""
            has_image = False
            
            for content in messages.data[0].content:
                if hasattr(content, 'text') and content.text:
                    text_response = content.text.value
                elif hasattr(content, 'image_file') and content.image_file:
                    has_image = True
            
            print("✅ 시각화 테스트 성공")
            if text_response:
                print(f"🤖 ProbTutor 응답: {text_response[:200]}...")
            if has_image:
                print("📊 시각화 이미지가 생성되었습니다.")
            
            return True
        else:
            print(f"❌ 실행 실패: {run.status}")
            return False
            
    except Exception as e:
        print(f"❌ 시각화 테스트 중 오류: {e}")
        return False

# 테스트 실행
print("🧪 시각화 테스트:")
viz_result = test_visualization()
print(f"테스트 결과: {'성공' if viz_result else '실패'}")


# In[ ]:


# 3. 핵심 함수들
def create_new_thread():
    """새 대화 스레드 생성"""
    global THREAD_ID
    thread = client.beta.threads.create()
    THREAD_ID = thread.id
    return thread.id

def send_message_to_assistant(message: str) -> str:
    """
    Assistant에게 메시지를 전송하고 응답을 받는 함수
    
    Args:
        message: 사용자가 입력한 메시지
        
    Returns:
        str: Assistant의 응답
    """
    global CONVERSATION_HISTORY
    
    if not ASSISTANT_ID:
        return "❌ Assistant가 초기화되지 않았습니다."
    
    if not THREAD_ID:
        create_new_thread()
    
    try:
        print(f"📤 메시지 전송 중... (Thread: {THREAD_ID})")
        
        # 메시지 전송
        client.beta.threads.messages.create(
            thread_id=THREAD_ID,
            role="user",
            content=message
        )
        
        print("🔄 Assistant 실행 중...")
        
        # Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID
        )
        
        print(f"🆔 실행 ID: {run.id}")
        
        # 실행 완료 대기
        max_wait_time = 60
        wait_time = 0
        
        while run.status in ['queued', 'in_progress', 'requires_action']:
            print(f"⏳ 실행 상태: {run.status} (대기 시간: {wait_time}초)")
            time.sleep(2)
            wait_time += 2
            
            if wait_time > max_wait_time:
                return f"❌ 실행 시간 초과 (최대 {max_wait_time}초 대기)"
            
            run = client.beta.threads.runs.retrieve(
                thread_id=THREAD_ID,
                run_id=run.id
            )
        
        print(f"✅ 실행 완료: {run.status}")
        
        if run.status == 'completed':
            # 응답 가져오기
            print("📥 응답 가져오는 중...")
            messages = client.beta.threads.messages.list(thread_id=THREAD_ID)
            
            if not messages.data:
                return "❌ 응답 메시지를 찾을 수 없습니다."
            
            latest_message = messages.data[0]
            print(f"📨 메시지 역할: {latest_message.role}")
            print(f"📄 컨텐츠 개수: {len(latest_message.content)}")
            
            # 텍스트 응답 찾기
            response_text = ""
            for content in latest_message.content:
                if hasattr(content, 'text') and content.text:
                    response_text = content.text.value
                    break
            
            if not response_text:
                return "❌ 텍스트 응답을 찾을 수 없습니다."
            
            print(f"📝 응답 길이: {len(response_text)}자")
            
            # 대화 히스토리에 추가
            if CONVERSATION_HISTORY is not None:
                CONVERSATION_HISTORY.append({"role": "user", "content": message})
                CONVERSATION_HISTORY.append({"role": "assistant", "content": response_text})
                
                if len(CONVERSATION_HISTORY) > MAX_CONVERSATION_LENGTH:
                    CONVERSATION_HISTORY = CONVERSATION_HISTORY[-MAX_CONVERSATION_LENGTH:]
            
            return response_text
        else:
            return f"❌ 실행 실패: {run.status}"
            
    except Exception as e:
        print(f"❌ 상세 오류: {type(e).__name__}: {str(e)}")
        return f"❌ 메시지 전송 중 오류: {e}"

def display_assistant_response(response: str):
    """
    Assistant 응답을 표시하는 함수 (텍스트 + 이미지) - 개선 버전
    
    Args:
        response: Assistant의 응답 텍스트
    """
    # 1. 텍스트 응답 먼저 표시
    display(Markdown(f"### 🤖 ProbTutor\n{response}"))
    
    # 2. 이미지가 있으면 표시
    if not THREAD_ID:
        print("⚠️ Thread ID가 설정되지 않았습니다.")
        return
    
    try:
        print("\n🔍 이미지 검색 중...")
        messages = client.beta.threads.messages.list(thread_id=THREAD_ID)
        
        if not messages.data:
            print("⚠️ 메시지를 찾을 수 없습니다.")
            return
        
        latest_message = messages.data[0]
        print(f"📨 최신 메시지 역할: {latest_message.role}")
        print(f"📄 컨텐츠 개수: {len(latest_message.content)}")
        
        image_found = False
        for i, content in enumerate(latest_message.content):
            print(f"🔍 컨텐츠 {i} 타입: {type(content).__name__}")
            
            if hasattr(content, 'image_file') and content.image_file:
                file_id = content.image_file.file_id
                print(f"📊 이미지 파일 발견! (File ID: {file_id})")
                
                try:
                    # 이미지 다운로드
                    print("⬇️ 이미지 다운로드 중...")
                    image_data = client.files.content(file_id)
                    image_bytes = image_data.read()
                    
                    print(f"📏 이미지 크기: {len(image_bytes)} bytes")
                    
                    # 이미지 표시
                    display(Image(image_bytes))
                    print("✅ 이미지가 성공적으로 표시되었습니다.")
                    image_found = True
                    
                except Exception as img_error:
                    print(f"❌ 이미지 다운로드 실패: {type(img_error).__name__}: {str(img_error)}")
                    import traceback
                    traceback.print_exc()
        
        if not image_found:
            print("ℹ️ 이 응답에는 이미지가 없습니다.")
            
    except Exception as e:
        print(f"⚠️ 이미지 처리 오류: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

def show_conversation_history():
    """저장된 대화 히스토리를 표시"""
    if not CONVERSATION_HISTORY:
        print("📭 대화 기록이 없습니다.")
        return
    
    print("\n📜 대화 히스토리:")
    print("=" * 60)
    for msg in CONVERSATION_HISTORY:
        role = "👤 사용자" if msg["role"] == "user" else "🤖 ProbTutor"
        print(f"\n{role}:")
        print(msg["content"][:200] + ("..." if len(msg["content"]) > 200 else ""))
        print("-" * 60)

print("✅ 핵심 함수 구현 완료")


# In[ ]:


# 4. 대화 인터페이스
def chat_with_probtutor():
    """
    ProbTutor와의 메인 대화 루프
    """
    print("🎓 ProbTutor - 확률 개념 설명 및 시각화 챗봇에 오신 것을 환영합니다!")
    print("📊 확률과 통계에 대한 질문을 자유롭게 해주세요.")
    print("💡 종료하려면 'quit', 'exit', '종료' 중 하나를 입력하세요.")
    print("=" * 60)
    
    # 추천 주제 표시
    print("\n📚 추천 주제:")
    print("  • 베이즈 정리란 무엇인가요?")
    print("  • 조건부 확률을 설명해주세요")
    print("  • 정규분포의 특징은 무엇인가요?")
    print("  • 동전을 3번 던져서 앞면이 2번 나올 확률을 계산해주세요")
    print("  • 이항분포 그래프를 그려주세요")
    print("=" * 60)
    
    # 초기화
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []
    create_new_thread()
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n👤 질문을 입력하세요: ").strip()
            
            if not user_input:
                print("❌ 입력이 비어있습니다. 다시 입력해주세요.")
                continue
            
            # 종료 조건 확인
            if user_input.lower() in ['quit', 'exit', '종료', 'q', '끝']:
                print("\n👋 ProbTutor와의 대화를 종료합니다. 확률 학습에 도움이 되었길 바랍니다!")
                break
            
            # 입력 검증
            if len(user_input.strip()) < 3:
                print("❌ 좀 더 구체적인 질문을 해주세요.")
                continue
            
            if len(user_input) > 1000:
                print("❌ 질문이 너무 깁니다. 1000자 이내로 입력해주세요.")
                continue
            
            # 사용자 질문을 명시적으로 표시
            display(Markdown(f"### 👤 사용자\n{user_input}"))
            print("-" * 60)
            
            # Assistant에게 메시지 전송 및 응답 받기
            response = send_message_to_assistant(user_input)
            
            # 응답 표시 (텍스트 + 이미지)
            display_assistant_response(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

print("✅ 대화 인터페이스 구현 완료")
print("💡 chat_with_probtutor() 함수를 호출하여 대화를 시작하세요!")


# In[ ]:


# 5. 이미지 생성 테스트 함수
def test_image_generation():
    """이미지 생성 및 표시를 테스트하는 함수"""
    print("🧪 이미지 생성 테스트 시작")
    print("=" * 60)
    
    if not ASSISTANT_ID:
        print("❌ Assistant가 생성되지 않았습니다.")
        return False
    
    try:
        # 새 스레드 생성
        thread = client.beta.threads.create()
        thread_id = thread.id
        print(f"✅ 새 스레드 생성: {thread_id}")
        
        # 시각화 요청
        test_question = "정규분포 그래프를 그려주세요. 평균 0, 표준편차 1로 해주세요."
        print(f"📤 테스트 질문: {test_question}")
        print("-" * 60)
        
        # 메시지 전송
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=test_question
        )
        
        # Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        print(f"🆔 실행 ID: {run.id}")
        print("⏳ 실행 완료 대기 중...")
        
        # 실행 완료 대기
        max_wait = 60
        wait_time = 0
        
        while run.status in ['queued', 'in_progress', 'requires_action']:
            time.sleep(2)
            wait_time += 2
            
            if wait_time % 10 == 0:
                print(f"   상태: {run.status} (대기 시간: {wait_time}초)")
            
            if wait_time > max_wait:
                print(f"❌ 실행 시간 초과 (최대 {max_wait}초 대기)")
                return False
            
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
        
        print(f"✅ 실행 완료: {run.status}")
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            
            if not messages.data:
                print("❌ 응답 메시지를 찾을 수 없습니다.")
                return False
            
            latest_message = messages.data[0]
            print(f"\n📨 메시지 역할: {latest_message.role}")
            print(f"📄 컨텐츠 개수: {len(latest_message.content)}")
            
            # 텍스트 응답 찾기
            response_text = ""
            for content in latest_message.content:
                if hasattr(content, 'text') and content.text:
                    response_text = content.text.value
                    break
            
            if response_text:
                print(f"📝 텍스트 응답 길이: {len(response_text)}자")
                print("-" * 60)
                display(Markdown(f"### 🤖 ProbTutor\n{response_text}"))
                print("-" * 60)
            
            # 이미지 확인 및 표시
            print("\n🖼️ 이미지 확인:")
            image_found = False
            
            for i, content in enumerate(latest_message.content):
                print(f"  컨텐츠 {i}: {type(content).__name__}")
                
                if hasattr(content, 'image_file') and content.image_file:
                    file_id = content.image_file.file_id
                    print(f"  📊 이미지 파일 발견: {file_id}")
                    
                    try:
                        print("  ⬇️ 이미지 다운로드 중...")
                        image_data = client.files.content(file_id)
                        image_bytes = image_data.read()
                        
                        print(f"  📏 이미지 크기: {len(image_bytes)} bytes")
                        print("  🖼️ 이미지 표시:")
                        display(Image(image_bytes))
                        print("  ✅ 이미지가 성공적으로 표시되었습니다!")
                        image_found = True
                        
                    except Exception as img_error:
                        print(f"  ❌ 이미지 처리 실패: {type(img_error).__name__}: {str(img_error)}")
                        import traceback
                        traceback.print_exc()
            
            if not image_found:
                print("  ℹ️ 이미지가 생성되지 않았습니다.")
                print("  💡 Assistant가 Code Interpreter를 사용했는지 확인해주세요.")
                return False
            
            print("\n" + "=" * 60)
            print("✅ 이미지 생성 및 표시 테스트 성공!")
            return True
            
        else:
            print(f"❌ 실행 실패: {run.status}")
            if hasattr(run, 'last_error') and run.last_error:
                print(f"오류 세부사항: {run.last_error}")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 중 오류: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

print("✅ 이미지 생성 테스트 함수 구현 완료")
print("💡 test_image_generation() 함수로 이미지 생성을 테스트할 수 있습니다.")


# In[ ]:


# 6. 사용법 안내
print("=" * 60)
print("🎉 ProbTutor - 확률 개념 설명 및 시각화 챗봇")
print("=" * 60)

print("\n📋 주요 기능:")
print("  • 확률 개념 설명 (베이즈 정리, 조건부 확률 등)")
print("  • 문제 해결 (Code Interpreter를 통한 정확한 계산)")
print("  • 시각화 (matplotlib을 활용한 그래프 생성)")
print("  • 대화형 학습 (사용자 수준에 맞춘 친절한 튜터)")

print("\n💡 사용 가능한 함수:")
print("  1. chat_with_probtutor() - 대화 시작")
print("  2. test_probability_calculation() - 확률 계산 테스트")
print("  3. test_visualization() - 시각화 테스트")
print("  4. test_image_generation() - 이미지 생성 테스트 (개선 버전)")
print("  5. show_conversation_history() - 대화 기록 확인")

print("\n🔧 현재 설정:")
print(f"  • Assistant ID: {ASSISTANT_ID or '미설정'}")
print(f"  • Thread ID: {THREAD_ID or '미설정'}")
print(f"  • 대화 히스토리: {len(CONVERSATION_HISTORY)}개 메시지")

print("\n🚀 빠른 시작:")
print("  • 대화 시작: chat_with_probtutor()")
print("  • 이미지 테스트: test_image_generation()")

print("\n" + "=" * 60)
print("✅ 모든 준비가 완료되었습니다!")
print("=" * 60)


# In[ ]:


# 7. 대화 시작 (이 셀을 실행하여 ProbTutor와 대화하세요)
# chat_with_probtutor()


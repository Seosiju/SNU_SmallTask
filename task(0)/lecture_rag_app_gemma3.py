import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain 
import os

# --- 1. 모델 및 RAG 체인 설정 (캐싱 사용) ---
# @st.cache_resource 데코레이터를 사용하여 RAG 체인을 캐싱
@st.cache_resource
def get_rag_chain(pdf_path):
    # PDF 경로에서 문서를 로드합니다.
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    
    # 문서를 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    # 임베딩 모델을 설정합니다.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 벡터 스토어를 생성합니다.
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # LLM을 설정합니다.
    llm = ChatOllama(model="gemma3:4b")
    
    # 프롬프트를 정의합니다.
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    # RAG 체인을 생성하고 반환합니다.
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 2. Streamlit UI 구성 ---
st.title("🦙 Ollama RAG: gemma3:4b")
st.markdown("강의 PDF 문서의 내용에 대해 질문해보세요!")

# 분석할 PDF 파일 경로를 여기에 지정합니다.
PDF_FILE_PATH = "/Users/snu.sim/git/RAG_test/The Ghost in the Machine.pdf"

# 메인 화면 구성
if not os.path.exists(PDF_FILE_PATH):
    st.error(f"지정된 파일을 찾을 수 없습니다: {PDF_FILE_PATH}")
    st.info("코드의 PDF_FILE_PATH 변수에 올바른 파일 경로를 입력했는지 확인해주세요.")
else:
    try:
        # 지정된 경로의 파일로 RAG 체인을 생성합니다.
        rag_chain = get_rag_chain(PDF_FILE_PATH)

        # 사용자 질문 입력란
        question = st.text_input("질문을 입력하세요:", placeholder="문서의 주요 내용은 무엇인가요?")

        if question:
            # 로딩 스피너와 함께 RAG 체인 실행
            with st.spinner("답변을 생성하는 중입니다..."):
                response = rag_chain.invoke({"input": question})
                
                # 결과 출력
                st.write("### 🤖 AI 답변:")
                st.write(response["answer"])

                # 근거 문서(Context) 출력 (확장/축소 가능)
                with st.expander("RAG Context 확인하기"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**문서 #{i+1}**")
                        st.write(doc.page_content)
                        st.markdown("---")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
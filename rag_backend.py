import os 
import json
from dotenv import load_dotenv 
import sqlite3 
from typing import Annotated, List, TypedDict, Optional, Any, Dict 
import tempfile
# from langchain.retrievers import BM25Retriever 
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_community.vectorstores import FAISS 
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage 
from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langgraph.checkpoint.sqlite import SqliteSaver 
from langgraph.graph import START, END, StateGraph 
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode, tools_condition 
import requests 
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging 
import sys 
from datetime import datetime 
from langchain_core.messages import ToolMessage
import time 
load_dotenv()

# LLM + embeddings
llm = ChatOpenAI(model = "gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Logging 
logging.basicConfig(
    level=logging.INFO, 
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    handlers = [
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler("rag.log", encoding = "utf-8"), 
    ], 
)

logger = logging.getLogger("RAG")

# PDF retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Dict[str, Any]] = {} 
_THREAD_METADATA: Dict[str, Any] = {} 

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> Dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    logger.info(f"Ingesting PDF | thread_id={thread_id} | filename={filename}")
    if not file_bytes:
        logger.error("Empty PDF upload")
        raise ValueError("No bytes received for ingestion") 

    with tempfile.NamedTemporaryFile(delete = False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name 
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load() 
        logger.info(f"Loaded PDF pages | pages={len(docs)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800, chunk_overlap = 200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs) 
        logger.info(f"Split into chunks | chunks={len(chunks)}")
        # FAISS retriever
        vector_store = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = vector_store.as_retriever(
            search_type = "similarity", search_kwargs = {"k":20}
        )
        # BM25 Retriever 
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 20

        _THREAD_RETRIEVERS[str(thread_id)] = {
            "faiss": faiss_retriever, 
            "bm25": bm25_retriever,
        }
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path), 
            "documents": len(docs), 
            "chunks": len(chunks)
        }

        logger.info(f"PDF indexed successfully | thread_id={thread_id}")
        return {
            "filename": filename or os.path.basename(temp_path), 
            "documents": len(docs), 
            "chunks": len(chunks)            
        }

    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

def hybrid_retrieve_and_rerank(query: str, thread_id: str, top_k: int = 4):
    logger.info(f"Hybrid retrieval | thread_id={thread_id} | query='{query}'")
    retrievers = _THREAD_RETRIEVERS.get(thread_id)
    if not retrievers:
        logger.warning("No retriever found for thread")
        return [] 
    
    # 1. Retrieve 
    faiss_docs = retrievers["faiss"].invoke(query) 
    bm25_docs = retrievers["bm25"].invoke(query)

    logger.info(
        f"Retrieved candidates | FAISS={len(faiss_docs)} | BM25={len(bm25_docs)}"
    )

    # 2. Merge + Deduplicate 
    all_docs = {(doc.page_content.strip(), doc.metadata.get("page")): doc 
            for doc in faiss_docs + bm25_docs}
    docs = list(all_docs.values()) 
    logger.info(f"Deduplicated docs | total={len(docs)}")
    if not docs:
        logger.warning("No documents after deduplication")
        return []

    # 3. Rerank 
    pairs = [[query, doc.page_content[:800]] for doc in docs] 
    # scores = reranker.predict(pairs)
    try:
        scores = reranker.predict(pairs)
    except Exception:
        logger.exception("Reranker failed, falling back to top docs")
        return docs[:top_k]
    
    # 4. Sort by score 
    ranked_docs = sorted(
        zip(docs, scores), 
        key = lambda x: x[1] , reverse = True
    )
    logger.info("Reranking complete")

    # Return top-k docs
    return [
    {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": float(score),
    }
    for doc, score in ranked_docs[:top_k]
    ]

# Tools 
class LoggedToolNode(ToolNode):
    def invoke(self, state, config=None):
        messages = state["messages"]
        last_message = messages[-1]

        # If no tool calls, pass through
        if not hasattr(last_message, "tool_calls"):
            return super().invoke(state, config)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            logger.info(f"TOOL CALL → {tool_name}")
            logger.info(f"TOOL INPUT → {tool_args}")

        start = time.time()

        try:
            result = super().invoke(state, config)
            duration = int((time.time() - start) * 1000)

            logger.info(f"TOOL EXECUTED SUCCESSFULLY")
            logger.info(f"TOOL EXEC TIME → {duration} ms")

            return result

        except Exception as e:
            logger.exception("TOOL EXECUTION FAILED")
            raise

search_tool = DuckDuckGoSearchRun(region = "us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

@tool 
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url) 
    return r.json()

@tool 
def summarize_pdf(thread_id: str, max_chunks: int = 20) -> dict:
    """
    Summarize the uploaded PDF for the given thread.
    """
    logger.info(f"PDF summarizer called | thread_id={thread_id}")
    retrievers = _THREAD_RETRIEVERS.get(thread_id) 
    if not retrievers:
        logger.warning("PDF summarizer called without indexed PDF")
        return {"error": "No PDF Uploaded"} 
    ## Use FAISS retriever to get representative chunks
    docs = retrievers["faiss"].invoke("Summarize this document")[:max_chunks]
    
    text = "\n\n".join(doc.page_content for doc in docs) 

    summary_prompt = f"""
    Summarize the following document clearly and concisely.
    Focus on key ideas, structure, and conclusions.

    Document:
    {text}
    """

    response = llm.invoke([HumanMessage(content = summary_prompt)])
    logger.info("PDF summarization complete")

    return {
        "summary": response.content, 
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }
# Answers a question, Returns citations with page numbers, Uses reranked chunks, Enterprise-grade feature
# @tool 
# def cite_answer(query: str, thread_id: str) -> dict:
#     """
#     Answer a query with citations from the PDF.
#     """
#     logger.info(f"Citation tool called | thread_id={thread_id} | query='{query}'")

#     docs = hybrid_retrieve_and_rerank(query, thread_id) 

#     if not docs:
#         return {"error": "No relevant context found"}
    
#     citations = []
#     context_text = "" 

#     for i, doc in enumerate(docs):
#         page = doc["metadata"].get("page", "N/A")
#         citations.append(f"[{i+1}] Page {page}")
#         context_text += f"\n[{i+1}] {doc['content']}"
    
#     prompt = f"""
#     Answer the question using ONLY the sources below.
#     Cite sources using [1], [2], etc.

#     Question:
#     {query}

#     Sources:
#     {context_text}
#     """

#     response = llm.invoke([HumanMessage(content = prompt)])

#     logger.info("Citation-based answer generated")

#     return {
#         "answer": response.content,
#         "citations": citations,
#         "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
#     }

@tool 
def cite_answer(query: str, thread_id: str) -> dict:
    """
    Answer a query with citations from the PDF.
    """
    logger.info(f"Citation tool called | thread_id={thread_id} | query='{query}'")

    # hybrid_retrieve_and_rerank returns list of dicts
    docs = hybrid_retrieve_and_rerank(query, thread_id) 

    if not docs:
        logger.warning("No relevant context found for citation")
        return {"error": "No relevant context found"}
    
    citations = []
    context_text = "" 

    for i, doc in enumerate(docs):
        # doc is a dict with keys: content, metadata, score
        page = doc["metadata"].get("page", "N/A")
        score = doc.get("score", 0.0)
        
        citations.append({
            "id": i + 1,
            "page": page,
            "relevance_score": round(score, 3)
        })
        
        context_text += f"\n[{i+1}] (Page {page}, Score: {score:.2f})\n{doc['content']}\n"
    
    prompt = f"""
Answer the question using ONLY the sources below.
Cite sources using [1], [2], etc. in your answer.

Question:
{query}

Sources:
{context_text}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    logger.info("Citation-based answer generated")

    return {
        "answer": response.content,
        "citations": citations,
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
        "num_sources": len(docs)
    }

# Improves vague / weak queries, Boosts retrieval quality before RAG, Used internally by the agent
@tool 
def rewrite_query(query: str) -> dict:
    """
    Rewrite a user query to improve retrieval quality.
    """
    logger.info(f"Query rewriter called | query='{query}'")

    prompt = f"""
    Rewrite the following query to be more precise,
    unambiguous, and suitable for document retrieval.

    Original query:
    {query}

    Rewritten query:
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    rewritten = response.content.strip() 
    logger.info(f"Query rewritten → {rewritten}")
    return {
        "original_query": query,
        "rewritten_query": rewritten,
    }

# @tool 
# def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
#     """
#     Retrieve relevant information from the uploaded PDF for this chat thread.
#     Always include the thread_id when calling this tool.
#     """
#     retriever = _get_retriever(thread_id)
#     logger.info(f"RAG tool called | thread_id={thread_id} | query='{query}'")
#     if retriever is None:
#         logger.warning("RAG tool called without indexed PDF")
#         return {
#             "error": "No document indexed for this chat. Upload a PDF first.",
#             "query": query,
#         }
#     # result = retriever.invoke(query) 
#     docs = hybrid_retrieve_and_rerank(query, thread_id)
#     # context = [doc.page_content for doc in docs] 
#     # metadata = [doc.metadata for doc in docs] 
#     logger.info(f"Returning {len(docs)} context chunks")
#     return {
#         "query": query,
#         "context": [doc.page_content for doc in docs],
#         "metadata": [doc.metadata for doc in docs],
#         "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
#     }

@tool 
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    logger.info(f"RAG tool called | thread_id={thread_id} | query='{query}'")

    if retriever is None:
        logger.warning("RAG tool called without indexed PDF")
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # hybrid_retrieve_and_rerank returns list of dicts with keys: content, metadata, score
    docs = hybrid_retrieve_and_rerank(query, thread_id)
    
    if not docs:
        logger.warning("No documents retrieved")
        return {
            "error": "No relevant content found for this query.",
            "query": query,
        }
    
    logger.info(f"Returning {len(docs)} context chunks")
    
    # Extract content and metadata from the dict format
    return {
        "query": query,
        "context": [doc["content"] for doc in docs],
        "metadata": [doc["metadata"] for doc in docs],
        "scores": [doc["score"] for doc in docs],  # Include relevance scores
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool, summarize_pdf, cite_answer, rewrite_query]
llm_with_tools = llm.bind_tools(tools)

# State Management
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class EvalResult(TypedDict):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    citation_accuracy: float
    overall_score: float

# Agent Nodes 
# def chat_node(state: ChatState, config = None):
#     """LLM node that may answer or request a tool call."""
#     thread_id = None 
#     if config and isinstance(config, dict):
#         thread_id = config.get("configurable", {}).get("thread_id")

#     system_message = SystemMessage(
#         content=(
#             "You are a document reasoning agent.\n\n"
#             "Tool usage rules:\n"
#             "- Use `rewrite_query` if the question is vague or unclear.\n"
#             "- Use `rag_tool` for factual lookup from the PDF.\n"
#             "- Use `cite_answer` when the user asks for citations or evidence.\n"
#             "- Use `summarize_pdf` when the user asks for a summary.\n\n"
#             "Always include the thread_id when required."
#         )
#     )

#     messages = [system_message, *state["messages"]]
#     response = llm_with_tools.invoke(messages, config = config) 
#     return {"messages": [response]}

# def planner_node(state: ChatState, config = None):
#     thread_id = config.get("configurable", {}).get("thread_id")

#     planner_prompt = SystemMessage(
#         content="""
#         You are a planning agent.

#         CRITICAL RULE:
#         - If NO PDF is indexed for the given thread_id,
#         DO NOT call summarize_pdf or cite_answer.
#         - Instead, respond with a plan that asks the user to upload a PDF.

#         Tool rules:
#         - If query asks for summary/citations AND no document exists →
#         return:
#         {
#             "steps": [
#             {
#                 "tool": "none",
#                 "args": {
#                 "message": "Please upload a PDF first"
#                 }
#             }
#             ]
#         }

#         Otherwise:
#         - If vague → rewrite_query
#         - If summary → summarize_pdf
#         - If citations → cite_answer
#         - Else → rag_tool

#         Return ONLY valid JSON.
#         """
#         )

#     messages = [planner_prompt, *state["messages"]]
#     plan = llm.invoke(messages) 
#     logger.info(f"PLANNER OUTPUT → {plan.content}")

#     return {
#         "messages": [AIMessage(content=plan.content)]
#     }

def planner_node(state: ChatState, config = None):
    thread_id = config.get("configurable", {}).get("thread_id")
    
    # CHECK IF PDF EXISTS
    has_pdf = thread_id in _THREAD_RETRIEVERS
    pdf_info = ""
    if has_pdf:
        metadata = _THREAD_METADATA.get(thread_id, {})
        pdf_info = f"\nPDF available: {metadata.get('filename', 'Unknown')} ({metadata.get('chunks', 0)} chunks)"
    
    planner_prompt = SystemMessage(
        content=f"""
        You are a planning agent.

        CURRENT STATE:
        - Thread ID: {thread_id}
        - PDF indexed: {has_pdf}{pdf_info}

        CRITICAL RULES:
        1. If has_pdf is False AND user asks for PDF-related tasks:
           - Return a plan that tells user to upload a PDF
           - Do NOT call summarize_pdf, cite_answer, or rag_tool
        
        2. If has_pdf is True:
           - You MAY use summarize_pdf, cite_answer, rag_tool
           - Choose appropriate tool based on query
        
        Tool selection when PDF exists:
        - Vague query → rewrite_query first
        - Summary request → summarize_pdf
        - Citation request → cite_answer
        - Factual lookup → rag_tool
        - Web search → search_tool
        - Stock price → get_stock_price
        - Math → calculator

        Return ONLY valid JSON in this format:
        {{
            "steps": [
                {{
                    "tool": "tool_name",
                    "args": {{}}
                }}
            ]
        }}

        If no PDF and user needs one, return:
        {{
            "steps": [
                {{
                    "tool": "none",
                    "args": {{
                        "message": "Please upload a PDF document first to enable document-based queries."
                    }}
                }}
            ]
        }}
        """
    )

    messages = [planner_prompt, *state["messages"]]
    plan = llm.invoke(messages) 
    logger.info(f"PLANNER OUTPUT → {plan.content}")
    logger.info(f"PDF indexed for thread {thread_id}: {has_pdf}")

    return {
        "messages": [AIMessage(content=plan.content)]
    }

# def executor_node(state: ChatState, config = None):
#     last_message = state["messages"][-1]
#     user_query = state["messages"][0].content # Original User Input 
#     thread_id = config.get("configurable", {}).get("thread_id")
#     try:
#         plan = json.loads(last_message.content)
#     except Exception:
#         logger.error("Planner did not return valid JSON")
#         return state 
    
#     results = [] 

#     for step in plan.get("steps", []):
#         tool_name = step["tool"]
#         tool_args = step["args"]
        
#         # Saftey Injection
#         if tool_name in ["rag_tool", "cite_answer", "rewrite_query"]:
#             tool_args.setdefault("query", user_query)

#         if tool_name in ["summarize_pdf", "cite_answer", "rag_tool"]:
#             if thread_id not in _THREAD_RETRIEVERS:
#                 logger.warning("EXECUTOR → Skipping tool, no PDF indexed")
#                 results.append({
#                     "error": "No PDF uploaded. Please upload a document first."
#                 })
#                 continue

#         logger.info(f"EXECUTOR → Running {tool_name}")

#         tool_fn = {t.name: t for t in tools}.get(tool_name) 
#         if not tool_fn:
#             continue 

#         output = tool_fn.invoke(tool_args) 
#         results.append(output)
#     return {
#         'messages': [AIMessage(content = str(results))]
#     }

def executor_node(state: ChatState, config = None):
    last_message = state["messages"][-1]
    user_query = state["messages"][0].content if state["messages"] else ""
    thread_id = config.get("configurable", {}).get("thread_id")
    
    try:
        plan = json.loads(last_message.content)
    except Exception as e:
        logger.error(f"Planner did not return valid JSON: {e}")
        return {
            'messages': [AIMessage(content=json.dumps({
                "error": "Planning failed - invalid plan format"
            }))]
        }
    
    results = [] 

    for step in plan.get("steps", []):
        tool_name = step.get("tool")
        tool_args = step.get("args", {})
        
        # Handle "none" tool (no-op with message)
        if tool_name == "none":
            logger.info("EXECUTOR → No tool needed, passing message")
            results.append(tool_args)
            continue
        
        # Safety: inject query if missing
        if tool_name in ["rag_tool", "cite_answer", "rewrite_query"]:
            tool_args.setdefault("query", user_query)
        
        # Safety: inject thread_id if missing
        if tool_name in ["summarize_pdf", "cite_answer", "rag_tool"]:
            tool_args.setdefault("thread_id", thread_id)
            
            # Double-check PDF exists
            if thread_id not in _THREAD_RETRIEVERS:
                logger.warning(f"EXECUTOR → Skipping {tool_name}, no PDF indexed")
                results.append({
                    "error": f"Cannot execute {tool_name}: No PDF uploaded for this thread."
                })
                continue

        logger.info(f"EXECUTOR → Running {tool_name} with args {tool_args}")

        # Find and execute tool
        tool_fn = {t.name: t for t in tools}.get(tool_name) 
        if not tool_fn:
            logger.warning(f"EXECUTOR → Unknown tool: {tool_name}")
            results.append({"error": f"Unknown tool: {tool_name}"})
            continue 

        try:
            output = tool_fn.invoke(tool_args) 
            results.append(output)
            logger.info(f"EXECUTOR → {tool_name} completed successfully")
        except Exception as e:
            logger.exception(f"EXECUTOR → {tool_name} failed")
            results.append({"error": f"Tool {tool_name} failed: {str(e)}"})
    
    return {
        'messages': [AIMessage(content=json.dumps(results, indent=2))]
    }

def answer_node(state: ChatState, config = None):
    system = SystemMessage(
            content="""
            You are an answer agent.

            STRICT RULES:
            - If tool output contains an error, DO NOT answer from general knowledge
            - Clearly inform the user what is missing
            - NEVER fabricate citations
            """
            )

    
    messages = [system, *state["messages"]]
    response = llm.invoke(messages) 

    return {"messages": [response]}

# 1. Context Relevance: Do retrieved chunks actually match the question?
# 2. Faithfulness (Groundedness): Is the answer supported by retrieved text?
# 3. Answer Relevance: Did the model actually answer the question?
# 4. Citation Accuracy: Are citations actually used correctly?
def evaluation_node(state: ChatState, config=None):
    """
    Evaluates RAG quality after answer generation
    """
    messages = state["messages"]
    
    # Find user query (first HumanMessage)
    user_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    if not user_query:
        logger.warning("No user query found for evaluation")
        return state
    
    # Find final answer (last AIMessage from answer_node)
    answer = None
    if messages and isinstance(messages[-1], AIMessage):
        answer = messages[-1].content
    
    if not answer:
        logger.warning("No answer found for evaluation")
        return state
    
    # Extract tool output (executor result) - look for the executor's output
    tool_output = None
    for msg in messages:
        if isinstance(msg, AIMessage):
            try:
                parsed = json.loads(msg.content)
                # Check if this looks like executor output (has context and scores)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict) and "context" in parsed[0]:
                        tool_output = parsed[0]
                        break
            except (json.JSONDecodeError, TypeError, KeyError):
                # Not JSON or not the right format, skip
                continue
    
    if not tool_output:
        logger.info("No RAG tool output found, skipping evaluation")
        return state
    
    # Extract data safely
    context_list = tool_output.get("context", [])
    scores = tool_output.get("scores", [])
    citations = tool_output.get("citations", [])
    
    if not context_list:
        logger.warning("No context found in tool output")
        return state
    
    context = "\n".join(context_list)
    
    # 1. Context relevance (based on reranker scores)
    context_relevance = 0.0
    if scores:
        # Normalize scores to 0-1 range (reranker scores can vary)
        avg_score = sum(scores) / len(scores)
        context_relevance = min(max(avg_score, 0.0), 1.0)
    
    # 2. Faithfulness - check if answer is grounded in context
    faith_prompt = f"""
Score how well the answer is supported by the context below.
Return ONLY a number between 0.0 and 1.0 (e.g., 0.85)
- 1.0 = Fully supported by context
- 0.5 = Partially supported
- 0.0 = Not supported at all

Answer:
{answer}

Context:
{context[:2000]}

Score:"""
    
    try:
        faith_response = llm.invoke([HumanMessage(content=faith_prompt)]).content.strip()
        # Extract first number found
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', faith_response)
        faithfulness = float(numbers[0]) if numbers else 0.5
        faithfulness = min(max(faithfulness, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Faithfulness scoring failed: {e}")
        faithfulness = 0.5
    
    # 3. Answer relevance - check if answer addresses the question
    relevance_prompt = f"""
Score how well the answer addresses the question.
Return ONLY a number between 0.0 and 1.0 (e.g., 0.85)
- 1.0 = Directly answers the question
- 0.5 = Partially answers
- 0.0 = Doesn't answer

Question:
{user_query}

Answer:
{answer}

Score:"""
    
    try:
        relevance_response = llm.invoke([HumanMessage(content=relevance_prompt)]).content.strip()
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', relevance_response)
        answer_relevance = float(numbers[0]) if numbers else 0.5
        answer_relevance = min(max(answer_relevance, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Answer relevance scoring failed: {e}")
        answer_relevance = 0.5
    
    # 4. Citation accuracy - check if citations are present when expected
    has_citations = bool(citations)
    citation_markers = ["[1]", "[2]", "[3]", "(Page", "page"] 
    has_citation_markers = any(marker in answer for marker in citation_markers)
    citation_accuracy = 1.0 if (has_citations and has_citation_markers) else 0.0
    
    overall = round(
        (context_relevance + faithfulness + answer_relevance + citation_accuracy) / 4,
        3
    )
    
    eval_result = {
        "context_relevance": round(context_relevance, 3),
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "citation_accuracy": citation_accuracy,
        "overall_score": overall,
        "num_chunks": len(context_list),
        "avg_reranker_score": round(sum(scores) / len(scores), 3) if scores else 0.0
    }
    
    logger.info(f"RAG EVALUATION → {eval_result}")
    
    # Return state with evaluation appended
    return state  # Don't add evaluation to messages - it's just for logging


tool_node = LoggedToolNode(tools)

# Graph 
conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node) 

# graph.add_edge(START, "chat_node")
# graph.add_conditional_edges("chat_node", tools_condition)
# graph.add_edge("tools", "chat_node") 

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node) 
graph.add_node("answer", answer_node) 
graph.add_node("evaluation", evaluation_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "answer")
graph.add_edge("answer", "evaluation")
graph.add_edge("evaluation", END)

chatbot = graph.compile(checkpointer=checkpointer)
"""LangGraph workflow: START -> retrieve -> generate -> END."""
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str
    score: float


def build_rag_graph(index_name: str):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(
            state["question"], k=config.TOP_K
        )
        return {
            "context": [doc.page_content for doc, _ in results],
            "scores": [float(s) for _, s in results],
        }

    def generate_node(state: AgentState):
        best = max(state["scores"], default=0.0)

        # Out-of-scope guard: nothing relevant retrieved -> refuse without calling the LLM
        if not state["context"] or best < config.MIN_RELEVANCE:
            return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
Do not use outside knowledge. If the context does not contain enough information,
reply exactly: '{config.REFUSAL_MESSAGE}'

Context:
{context_str}

Question: {state['question']}"""

        answer = llm.invoke(prompt).content.strip()

        # Confidence = best retrieval similarity (cosine); 0 if the model refused
        refused = config.REFUSAL_MESSAGE.lower() in answer.lower()
        confidence = 0.0 if refused else round(min(max(best, 0.0), 1.0), 4)
        return {"answer": answer, "score": confidence}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

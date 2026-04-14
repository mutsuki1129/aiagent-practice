from __future__ import annotations

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """你是一個專業的文件問答助手。根據以下從 PDF 中擷取的相關段落來回答使用者的問題。

相關段落：
{context}

使用者問題：{question}

回答規則：
1. 只根據提供的段落內容回答，不要憑空捏造
2. 如果段落中沒有相關資訊，請明確說「根據文件內容，我找不到相關資訊」
3. 回答請使用繁體中文
4. 如果可以，請標註資訊來源的頁碼
"""


def build_qa_chain(retriever, llm) -> RetrievalQA:
    """Build a RetrievalQA chain with a Traditional Chinese grounded-answer prompt."""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

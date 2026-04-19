from typing import List, TypedDict

class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: bool
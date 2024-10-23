from dataclasses import dataclass

@dataclass
class Setting:
    persona: str
    goal: str
    description: str
    obstacle: str
    question: str
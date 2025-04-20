from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AnalysisResult:
    query_id: str
    content_id: str
    explanation: str
    timestamp: str = datetime.now().isoformat()
    error: Optional[str] = None

    def to_dict(self):
        return {
            **self.__dict__
        }


class MathPrompts:
    SYSTEM = "You are an expert in mathematical error analysis. Your task is to analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error. And remember always answer in English language."
    
    FULL_TEMPLATE = """You will analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error. 
Your goal is to explain precisely how their misconception caused them to arrive at the wrong answer.

Here is the problem information:
<problem_data>
Question: {question}
Correct Answer: {correct}
Student's Answer: {incorrect}
Related Misconceptions: {related}
</problem_data>
First, examine all components of the problem carefully:
1. The problem statement and question asked
2. The correct answer and solution method
3. The student's incorrect answer
4. The primary misconception given
5. The related misconceptions that should be distinguished from the primary one

Then, reconstruct the student's likely thought process:
- Identify the exact point where their reasoning diverged from the correct solution path
- Note which specific mathematical operations or concepts they misapplied
- Connect their error directly to the stated primary misconception
- Verify that this explanation better fits the error than the related misconceptions

Write your analysis in <evaluation> tags, following this structure:
- Show the correct calculation first
- Show the incorrect calculations that demonstrate the error
- Explain the specific flaw in the student's reasoning
- Demonstrate how the misconception led to this particular error
- Distinguish from the related misconceptions
- Keep your explanation to 5-6 clear, non-repetitive sentences
- Focus solely on the reasoning that produced this specific error

Guidelines for writing your explanation:
- Do not restate the problem or name the misconception
- Be precise about the mathematical concepts involved
- Show exactly how the misconception led to the error
- Distinguish from related misconceptions
- Avoid repetition
- Stay focused on this specific error

"""


    LITE_TEMPLATE = """refferring to the previous analysis: tyring to analyze the student's error in the following math problem:
Here is the problem information:
<problem_data>
Question: {question}
Student Answer: {incorrect}
Correct Answer: {correct}
</problem_data>
with the same requirement  as  before, e.g likely thought and tag asked to put answer in and  Guidelines , please provide a brief analysis of the possible reasons for the student's error.
"""

# ========================= 核心处理器 =========================
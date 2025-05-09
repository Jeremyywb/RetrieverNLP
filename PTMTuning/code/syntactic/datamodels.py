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
    CURATE_TEMPLATE = """## Your Role
You are a "Verificationer" - an expert evaluator of mathematical reasoning. Your task is to critically examine chain-of-thought explanations for mathematical error analysis and determine if the analysis is sound, complete, and provides an accurate diagnosis of student errors.

## Current Case
You will evaluate a chain-of-thought analysis for the following student math error:

```
Question: {question}
Correct Answer: {correct}
Student's Answer: {incorrect}
```

## Analysis to Evaluate
```
{rationale}
```

## Evaluation Criteria
Evaluate the above chain-of-thought analysis on the following criteria:

1. **Mathematical Accuracy**: Are all mathematical steps and calculations correctly stated?
2. **Logical Coherence**: Does the analysis follow a clear logical progression?
3. **Error Diagnosis**: Does it correctly identify the likely causes of the student's error?
4. **Reasonableness of Assumptions**: Are the assumptions about the student's thinking process reasonable?
5. **Relevance**: Does the analysis avoid introducing irrelevant concepts or analogies?
6. **Mathematical Principles**: Does it correctly explain the mathematical principles involved?

## Instructions
1. Examine each step of the analysis against the criteria above.
2. Identify any discrepancies, errors, or unwarranted assumptions.
3. Check if alternative explanations for the student's error were missed.
4. Assess whether the numerical work in the analysis is consistent with the student's actual answer.
5. Provide a detailed evaluation with specific examples from the analysis.
6. Assign an overall rating: Correct, Partially Correct, or Incorrect.
7. Suggest improvements to the analysis if needed.

## Important Note
Focus on the quality of reasoning and explanation in the analysis, not just whether the final conclusion is correct. An analysis can reach the right conclusion through faulty reasoning, which should be identified."""

# ========================= 核心处理器 =========================


# 基于TRL的SFT训练代码，用于CoT能力蒸馏
# 支持DeepSpeed、Hydra配置、混合精度训练和训练过程可视化，后面是相关数据的详细描述

# 输入数据是一个csv，格式如下所示:
#     {fold_id,QuestionText，CorrectAnswerText，InCorrectAnswerText,Explanation}
# 注释：Explanation为cot的文本

# 闭源模型获取cot的prompt如下所示:

# ```python
# class MathPrompts:
#     SYSTEM = "You are an expert in mathematical error analysis. Your task is to analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error. And remember always answer in English language."
    
#     FULL_TEMPLATE = """You will analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error. 
# Your goal is to explain precisely how their misconception caused them to arrive at the wrong answer.

# Here is the problem information:
# <problem_data>
# Question: {question}
# Correct Answer: {correct}
# Student's Answer: {incorrect}
# Related Misconceptions: {related}
# </problem_data>
# First, examine all components of the problem carefully:
# 1. The problem statement and question asked
# 2. The correct answer and solution method
# 3. The student's incorrect answer
# 4. The primary misconception given
# 5. The related misconceptions that should be distinguished from the primary one

# Then, reconstruct the student's likely thought process:
# - Identify the exact point where their reasoning diverged from the correct solution path
# - Note which specific mathematical operations or concepts they misapplied
# - Connect their error directly to the stated primary misconception
# - Verify that this explanation better fits the error than the related misconceptions

# Write your analysis in <evaluation> tags, following this structure:
# - Show the correct calculation first
# - Show the incorrect calculations that demonstrate the error
# - Explain the specific flaw in the student's reasoning
# - Demonstrate how the misconception led to this particular error
# - Distinguish from the related misconceptions
# - Keep your explanation to 5-6 clear, non-repetitive sentences
# - Focus solely on the reasoning that produced this specific error

# Guidelines for writing your explanation:
# - Do not restate the problem or name the misconception
# - Be precise about the mathematical concepts involved
# - Show exactly how the misconception led to the error
# - Distinguish from related misconceptions
# - Avoid repetition
# - Stay focused on this specific error

# """

# def _build_full_prompt(row: Dict) -> str:
#     """构建完整数据提示"""
#     return MathPrompts.FULL_TEMPLATE.format(
#         question=row['QuestionText'],
#         correct=row['CorrectAnswerText'],
#         incorrect=row['InCorrectAnswerText'],
#         related=", ".join(row['related_misconceptions'].split(';'))
#     )

# ```

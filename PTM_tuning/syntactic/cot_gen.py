import os
import json
import asyncio
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pyrate_limiter import Duration, RequestRate, Limiter
from prometheus_client import start_http_server, Counter, Histogram
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import platform



# 加载环境变量
load_dotenv()

# ========================= 配置项 =========================
class Config:
    OUTPUT_ROOT = Path("../analysis")
    INPUT_ROOT = Path("../input")
    
    
    # 子目录定义
    FULL_INPUT_DIR = INPUT_ROOT / "synthetic.csv"            # Full任务输入数据
    LITE_INPUT_DIR = INPUT_ROOT / "lite.csv"           # Lite任务输入数据

    FULL_OUTPUT_DIR = OUTPUT_ROOT / "full"          # Full任务成功结果
    LITE_OUTPUT_DIR = OUTPUT_ROOT / "lite"          # Lite任务成功结果
    ERROR_DIR = OUTPUT_ROOT / "errors"              # 所有失败结果（按任务类型细分）
    
    @classmethod
    def init_dirs(cls):
        """安全创建所有需要的目录"""
        for d in [cls.FULL_OUTPUT_DIR, cls.LITE_OUTPUT_DIR, cls.ERROR_DIR]:
            d.mkdir(parents=True, exist_ok=True)
# ========================= 监控配置 =========================
start_http_server(8000)
API_CALL_COUNTER = Counter('api_calls', 'API调用统计', ['status', 'task_type'])
API_RESPONSE_TIME = Histogram('response_time', '响应时间分布', ['task_type'])

# # ========================= 日志配置 =========================
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(Config.LOG_FILE),
#         logging.StreamHandler(),
#     ]
# )
# logger = logging.getLogger("MathAnalyzer")
# logger.setLevel(logging.INFO)

# ========================= 数据结构 =========================
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


class ProgressManager:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.progress_file = Config.ERROR_DIR / f"{task_type}_progress.txt"
        self._processed = self._load_existing()

    def _load_existing(self) -> set:
        """加载进度文件并校验有效性"""
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                pass
            return set()
            
        processed = set()
        with open(self.progress_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # 格式: query_id,status,timestamp
                if len(parts) >=2 and parts[1] == "success":
                    processed.add(parts[0])
        return processed

    def is_processed(self, query_id: str) -> bool:
        return query_id in self._processed

    def mark_success(self, query_id: str):
        """标记成功记录"""
        with open(self.progress_file, 'a') as f:
            f.write(f"{query_id},success,{datetime.now().isoformat()}\n")
        self._processed.add(query_id)

    def mark_failed(self, query_id: str, error: str):
        """标记失败记录"""
        with open(self.progress_file, 'a') as f:
            f.write(f"{query_id},failed,{datetime.now().isoformat()},{error}\n")



class BailianMathEvaluator:
    def __init__(self, task_type: str):
        assert task_type in ["full", "lite"], "Invalid task type"
        self.task_type = task_type
        if task_type == "full":
            self.full_df = pd.read_csv(Config.FULL_INPUT_DIR)

        self.full_analysis_pool = self._get_full_files() if task_type == "lite" else None
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.rate_limiter = Limiter(RequestRate(10, Duration.MINUTE))
        self.semaphore = asyncio.Semaphore(10)

        # 初始化输出目录
        Config.init_dirs()

    def _preview_full_analysis(self, full_file) -> str:
        """预览Full任务分析结果 full_file 为文件名 不含路径"""
        full_file_path = Config.FULL_OUTPUT_DIR / full_file
        with open(full_file_path, 'r') as f:
            data = json.load(f)
            return data.get('explanation', '')

    def _get_full_files(self) -> List[dict]:
            """加载所有Full任务分析结果"""
            analysis_files = [c for c in os.listdir(Config.FULL_OUTPUT_DIR) if c.endswith("_full.json")]

            if not analysis_files:
                raise FileNotFoundError("No full analysis results found")
            return analysis_files

    async def process_row(self, row: Dict, progress: ProgressManager) -> AnalysisResult:
        """处理单行数据"""
        query_id = row['query_id']
        content_id = row['content_id']
        if progress.is_processed(query_id):
            return AnalysisResult(
                query_id=query_id,
                content_id="",
                explanation="[SKIPPED] This query has been processed",
                error="SKIPPED"  # 特殊标识
            )
        try:
            if self.task_type == "full":
                result = await self._process_full(row)
            else:
                result = await self._process_lite(row)
                
            # 标记成功
            if hasattr(result, 'error') and result.error is None:
                progress.mark_success(query_id)
                self._save_result(Config.FULL_OUTPUT_DIR if self.task_type == "full" else Config.LITE_OUTPUT_DIR, result)
            else:
                progress.mark_failed(query_id, result.error)
            return result## 依旧存在 parse error 的问题
        except Exception as e:
            error_result = AnalysisResult(
                query_id=query_id,
                content_id=content_id,
                explanation="",
                error=str(e)
            )
            progress.mark_failed(query_id, str(e))
            self._save_error_result(error_result)
            return error_result

    async def _process_full(self, row: Dict) -> AnalysisResult:
        """处理完整数据任务"""
        prompt = self._build_full_prompt(row)
        messages =  [
            {"role": "system", "content": MathPrompts.SYSTEM},
            {"role": "user", "content": prompt}
            ]
        result = await self._safe_api_call(messages)
        return AnalysisResult(
            query_id=row['query_id'],
            content_id=row['content_id'],
            explanation=result.explanation,
            error=result.error
        )

    async def _process_lite(self, row: Dict) -> AnalysisResult:
        """处理简化数据任务（多轮对话）"""
        # 第一轮对话
        import random
        seed_analysis_file = random.choice(self.full_analysis_pool)
        seed_analysis = self._preview_full_analysis(seed_analysis_file)
        try_count = 0
        while seed_analysis == "" and try_count < 3:
            seed_analysis_file = random.choice(self.full_analysis_pool)
            seed_analysis = self._preview_full_analysis(seed_analysis_file)
            try_count += 1
        if seed_analysis == "":
            raise ValueError("Failed to get a valid seed analysis")
        
        query_id = seed_analysis_file.split("_")[1]
        full_row = self.full_df[self.full_df['query_id'].map(str) == query_id].iloc[0].to_dict()
        full_prompt = self._build_full_prompt(full_row)

        messages =  [
            {"role": "system", "content": MathPrompts.SYSTEM},
            {"role": "user", "content": full_prompt},
            {"role": "assistant", "content": seed_analysis}
            ]
        
        prompt = self._build_lite_prompt(row)
        messages.append({"role": "user", "content": prompt})
        result = await self._safe_api_call(messages)
        return AnalysisResult(
            query_id=row['query_id'],
            content_id=row['content_id'],
            explanation=result.explanation,
            error=result.error
        )

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OSError, TimeoutError))
    )
    async def _safe_api_call(self, messages: str) -> AnalysisResult:
        """带重试机制的API调用"""
        async with self.semaphore:
            with self.rate_limiter.ratelimit("api_call", delay=True):
                start_time = datetime.now()
                try:
                    response = await self.client.chat.completions.create(
                        model="deepseek-r1",
                        messages=messages,
                        temperature=0.3,
                        timeout=15
                    )
                    print(f"API调用成功: {response}")
                    # 记录指标
                    api_duration = (datetime.now() - start_time).total_seconds()
                    API_RESPONSE_TIME.labels(task_type=self.task_type).observe(api_duration)
                    API_CALL_COUNTER.labels(status="success", task_type=self.task_type).inc()

                    return self._parse_response(response)
                except Exception as e:
                    API_CALL_COUNTER.labels(status="error", task_type=self.task_type).inc()
                    raise e
    def validate_response(self, response) -> bool:
        """验证API响应的有效性"""
        if not hasattr(response, "choices") or len(response.choices) == 0:
            return "Empty response or no choices"
        if not hasattr(response.choices[0], "message"):
            return "False response format, no message"
        else:
            content = response.choices[0].message.content
            if not content.__contains__("<evaluation>"): 
                return "False response format, no <evaluation>"
        return "True"
    def _parse_response(self, response) -> AnalysisResult:
        """解析API响应"""
        valid_res = self.validate_response(response)
        if valid_res != "True":
            return AnalysisResult(
                query_id="",
                content_id="",
                explanation="",
                error=valid_res
            )
        content = response.choices[0].message.content
        try:
            import re
            evaluation_match = re.search(r'<evaluation>(.*?)</evaluation>', content, re.DOTALL)
            if evaluation_match:
                explanation = evaluation_match.group(1).strip()
            else:
                explanation = content.strip()
            return AnalysisResult(
                query_id="",
                content_id="",
                explanation=explanation
                # 其他字段由上层填充
            )
        except Exception as e:
            return AnalysisResult(
                query_id="",
                content_id="",
                explanation=content,
                error=str(e)
            )

    def _save_result(self, output_dir ,result: AnalysisResult):
        """实时保存结果"""
        output_path = output_dir / f"example_{result.query_id}_{self.task_type}.json"
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_error_result(self, result: AnalysisResult):
        """错误文件隔离存储"""
        error_dir = Config.ERROR_DIR / self.task_type
        error_dir.mkdir(exist_ok=True)
        filename = f"{result.query_id}_{result.content_id}_error.json"
        with open(error_dir / filename, 'w') as f:
            json.dump(result.to_dict(), f)

    def _build_full_prompt(self, row: Dict) -> str:
        """构建完整数据提示"""
        return MathPrompts.FULL_TEMPLATE.format(
            question=row['QuestionText'],
            correct=row['CorrectAnswerText'],
            incorrect=row['InCorrectAnswerText'],
            related=", ".join(row['related_misconceptions'].split(';'))
        )

    def _build_lite_prompt(self, row: Dict) -> str:
        """构建简化数据提示"""
        return MathPrompts.LITE_TEMPLATE.format(
            question=row['QuestionText'],
            correct=row['CorrectAnswerText'],
            incorrect=row['InCorrectAnswerText']
        )



            
# ========================= 命令行接口 =========================
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["full", "lite"], required=True)
    args = parser.parse_args()
    filemap = {
        "full": Config.FULL_INPUT_DIR,
        "lite": Config.LITE_INPUT_DIR
    }

    try:

        evaluator = BailianMathEvaluator(args.task)

        progress = ProgressManager(args.task)
    
        df = pd.read_csv( filemap[args.task]) 

        rows_to_process = [row for _, row in df.iterrows() if not progress.is_processed(row['query_id'])]
        tasks = [evaluator.process_row(row, progress) for row in rows_to_process]
        with tqdm(total=len(tasks), desc=f"Processing {args.task} tasks") as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                if result.error is not None and result.error != "SKIPPED":
                    print(f"处理失败: {result.query_id} | 错误信息: {result.error}")
                pbar.update(1)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 运行主协程
    asyncio.run(main())

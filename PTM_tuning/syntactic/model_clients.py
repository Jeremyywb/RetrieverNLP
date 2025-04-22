import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class BaseModelClient(ABC):
    """Base abstract class for model clients"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize client"""
        pass
    
    @abstractmethod
    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """Get completion from the model"""
        pass
    
    @abstractmethod
    async def get_streaming_completion(self, messages: List[Dict[str, str]], **kwargs):
        """Get streaming completion from the model"""
        pass


class DeepseekClient(BaseModelClient):
    """Client for Deepseek models"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.model = model or "deepseek-r1"
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = None
    
    async def initialize(self):
        """Initialize OpenAI-compatible client"""
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OSError, TimeoutError))
    )
    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """Get completion from Deepseek"""
        if not self.client:
            await self.initialize()
            
        temperature = kwargs.get("temperature", 0.3)
        timeout = kwargs.get("timeout", 15)
        
        if not stream:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=timeout
            )
            return response
        else:
            return await self.get_streaming_completion(messages, **kwargs)
    
    async def get_streaming_completion(self, messages: List[Dict[str, str]], **kwargs):
        """Get streaming completion from Deepseek"""
        if not self.client:
            await self.initialize()
            
        temperature = kwargs.get("temperature", 0.3)
        timeout = kwargs.get("timeout", 15)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            stream=True
        )
        
        full_content = ""
        done_reasoning = False
        
        async for chunk in response:
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning_chunk = chunk.choices[0].delta.reasoning_content or ''
                if reasoning_chunk:
                    print(reasoning_chunk, end='', flush=True)
                    # You could also collect reasoning in a separate variable if needed
            
            answer_chunk = chunk.choices[0].delta.content or ''
            if answer_chunk:
                if not done_reasoning:
                    print('\n\n === Final Answer ===\n')
                    done_reasoning = True
                print(answer_chunk, end='', flush=True)
                full_content += answer_chunk
        
        # Create a simulated response object that matches the expected format
        simulated_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': full_content
                    })
                })
            ]
        })
        
        return simulated_response

class OpenAIClient(BaseModelClient):
    """Client for OpenAI models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
    
    async def initialize(self):
        """Initialize OpenAI client"""
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OSError, TimeoutError))
    )
    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """Get completion from OpenAI"""
        if not self.client:
            await self.initialize()
            
        model = kwargs.get("model", "gpt-4o")
        temperature = kwargs.get("temperature", 0.3)
        timeout = kwargs.get("timeout", 15)
        
        if not stream:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout
            )
            return response
        else:
            return await self.get_streaming_completion(messages, **kwargs)
    
    async def get_streaming_completion(self, messages: List[Dict[str, str]], **kwargs):
        """Get streaming completion from OpenAI"""
        if not self.client:
            await self.initialize()
            
        model = kwargs.get("model", "gpt-4o")
        temperature = kwargs.get("temperature", 0.3)
        timeout = kwargs.get("timeout", 15)
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            stream=True
        )
        
        full_content = ""
        done_reasoning = False
        
        async for chunk in response:
            # OpenAI doesn't have reasoning_content, but if you want to
            # add support for extended reasoning mode in future:
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning_chunk = chunk.choices[0].delta.reasoning_content or ''
                if reasoning_chunk:
                    print(reasoning_chunk, end='', flush=True)
            
            # Normal content handling
            if hasattr(chunk.choices[0].delta, 'content'):
                answer_chunk = chunk.choices[0].delta.content or ''
                if answer_chunk:
                    if not done_reasoning and hasattr(chunk.choices[0].delta, 'reasoning_content'):
                        print('\n\n === Final Answer ===\n')
                        done_reasoning = True
                    print(answer_chunk, end='', flush=True)
                    full_content += answer_chunk
        
        # Create a simulated response object that matches the expected format
        simulated_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': full_content
                    })
                })
            ]
        })
        
        return simulated_response



# 修复ZhipuAI客户端类，实现get_streaming_completion抽象方法
class ZhipuAIClient(BaseModelClient):
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        self.client = None
        
    async def initialize(self):
        """Lazy initialization of client"""
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=self.api_key)
    
    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OSError, TimeoutError, ConnectionError))
    )
    async def get_completion(self, messages: List[Dict], stream: bool = False, **kwargs):
        """Get completion from ZhipuAI with polling mechanism"""
        if not self.client:
            await self.initialize()
            
        model = kwargs.get("model", "glm-z1-airx")
        temperature = kwargs.get("temperature", 0.3)
        timeout = kwargs.get("timeout", 60)
        max_tokens = kwargs.get("max_tokens", 1024)
        
        if stream:
            return await self.get_streaming_completion(messages, **kwargs)
        
        # Initial request to get task ID
        response = await self._make_async_call(
            lambda: self.client.chat.asyncCompletions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[{"type": "web_search", "web_search": {"search_result": False}}],
                do_sample=True
            )
        )
        
        task_id = response.id
        task_status = ''
        poll_count = 0
        # 使用timeout和最大轮询次数限制之间的较小值
        max_polls = min(timeout // 2, 40)  # 最多等待80秒
        
        # Poll until completion or timeout
        while task_status != 'SUCCESS' and task_status != 'FAILED' and poll_count <= max_polls:
            await asyncio.sleep(2)  # Sleep for 2 seconds between polls
            
            result_response = await self._make_async_call(
                lambda: self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            )
            
            task_status = result_response.task_status
            poll_count += 1
            
            # Check if we have a result
            if task_status == 'SUCCESS':
                return result_response
            elif task_status == 'FAILED':
                raise Exception(f"ZhipuAI task failed: {result_response}")
                
        if poll_count > max_polls:
            raise TimeoutError(f"ZhipuAI task timed out after {timeout} seconds")
            
        return result_response
    
    async def get_streaming_completion(self, messages: List[Dict[str, str]], **kwargs):
        """
        获取流式输出的完成结果 - 因为智普AI当前的异步API不支持真正的流式输出，
        所以这里我们模拟一个流式输出的行为，实际上是先完整获取结果再返回
        """
        if not self.client:
            await self.initialize()
            
        # 使用非流式API获取完整结果
        complete_response = await self.get_completion(messages, stream=False, **kwargs)
        
        # 获取完整内容
        if hasattr(complete_response, "choices") and len(complete_response.choices) > 0:
            if hasattr(complete_response.choices[0], "message"):
                full_content = complete_response.choices[0].message.content
            else:
                full_content = complete_response.choices[0].get("content", "")
        else:
            full_content = "无法获取有效内容"

        # 打印完整结果以模拟流式输出效果
        print(full_content)
        
        # 创建与非流式输出相同格式的响应对象
        return complete_response
    
    async def _make_async_call(self, func):
        """Wrapper to make synchronous API calls asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)




# Factory function to get appropriate client
def get_model_client(client_type: str = "deepseek", **kwargs) -> BaseModelClient:
    """
    Get model client by type
    
    Args:
        client_type: Type of client ('deepseek', 'openai')
        **kwargs: Additional arguments for client initialization
    
    Returns:
        BaseModelClient: Initialized model client
    """

        


    

    clients = {
        "deepseek": DeepseekClient,
        "qwq_plus": DeepseekClient,  # Assuming DeepseekClient is used for both
        "openai": OpenAIClient,
        "zhipu": ZhipuAIClient,
        "doubao_think_pro": DeepseekClient,  # Assuming ZhipuAIClient is used for both
        "doubao_r1_distil_qwen": DeepseekClient,
        "doubao_15_pro": DeepseekClient,
        "doubao_r1": DeepseekClient,
        "deepseek_dis_llama": DeepseekClient,
        "qwen_math72": DeepseekClient,
        "tencent_r1": DeepseekClient
    }
    api_keys = {
        "doubao_think_pro": os.getenv("DOUBAO_API_KEY"),
        "doubao_r1_distil_qwen": os.getenv("DOUBAO_API_KEY"),
        "doubao_r1": os.getenv("DOUBAO_API_KEY"),
        "doubao_15_pro": os.getenv("DOUBAO_API_KEY"),
        "tencent_r1": os.getenv("TENC_API_KEY"),
        "qwq_plus":os.getenv("DASHSCOPE_API_KEY"),
        "deepseek_dis_llama": os.getenv("DASHSCOPE_API_KEY"),
        "qwen_math72": os.getenv("DASHSCOPE_API_KEY"),
    }
    api_urls = {
        "doubao_think_pro": "https://ark.cn-beijing.volces.com/api/v3",
        "doubao_r1_distil_qwen": "https://ark.cn-beijing.volces.com/api/v3",
        "doubao_r1": "https://ark.cn-beijing.volces.com/api/v3",
        "doubao_15_pro": "https://ark.cn-beijing.volces.com/api/v3",
        "qwq_plus": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen_math72": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek_dis_llama": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "tencent_r1": "https://api.lkeap.cloud.tencent.com/v1",
        "openai": None,
        "zhipu": None
    }

    model_names = {
        "doubao_think_pro": "doubao-1-5-thinking-pro-250415",
        "doubao_r1_distil_qwen": "deepseek-r1-distill-qwen-32b-250120",
        "doubao_15_pro": "doubao-1-5-pro-256k-250115",
        "doubao_r1": "deepseek-r1-250120",
        "zhipu": "glm-z1-airx",
        "tencent_r1": "deepseek-r1",
        "qwq_plus": "qwq-plus",
        "deepseek_dis_llama": "deepseek-r1-distill-llama-70b",
        "qwen_math72": "qwen2.5-math-72b-instruct",
    }
    new_kwargs = {
        "api_key": api_keys.get(client_type, None),
        "base_url": api_urls.get(client_type, None),
        "model": model_names.get(client_type, None),
    }
    kwargs.update(new_kwargs)
        
    if client_type not in clients:
        raise ValueError(f"Unsupported client type: {client_type}")
    
    return clients[client_type](**kwargs)

# llama-4-maverick-17b-128e-instruct
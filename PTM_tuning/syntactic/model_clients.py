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


class BaseModelClient(ABC):
    """Base abstract class for model clients"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize client"""
        pass
    
    @abstractmethod
    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """Get completion from the model
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional parameters for the completion
        
        Returns:
            Response object or simulated response object (if streaming)
        """
        pass



class DeepseekClient(BaseModelClient):
    """Client for Deepseek models"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
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
            
        model = kwargs.get("model", "deepseek-r1")
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
        """Get streaming completion from Deepseek"""
        if not self.client:
            await self.initialize()
            
        model = kwargs.get("model", "deepseek-r1")
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
        "openai": OpenAIClient,
    }
    
    if client_type not in clients:
        raise ValueError(f"Unsupported client type: {client_type}")
    
    return clients[client_type](**kwargs)
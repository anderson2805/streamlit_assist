import os
import json
import asyncio
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI
from openai import RateLimitError
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type
import httpx
from urllib.parse import urlparse

# Try to import streamlit for secrets support
try:
    import streamlit as st
    _streamlit_available = True
except ImportError:
    _streamlit_available = False

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value from st.secrets if available, otherwise from environment variables.
    
    Args:
        key: The secret key to retrieve
        default: Default value if key is not found
        
    Returns:
        The secret value or default
    """
    # Try st.secrets first if streamlit is available
    if _streamlit_available:
        try:
            # Check if we're in a streamlit context and secrets are available
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            # If there's any error accessing st.secrets, fall back to env vars
            pass
    
    # Fall back to environment variables
    return os.getenv(key, default)

class OpenAIClient:
    """
    Client for interacting with the OpenAI API with web search capabilities.
    
    Documentation: https://platform.openai.com/docs/api-reference/chat/create
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI API client.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from st.secrets or environment variables.
            model: Default model to use. If None, it will use gpt-4.1 as default.
        """
        load_dotenv()  # Load environment variables from .env file
        
        self.api_key = api_key or _get_secret("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in secrets.toml (for Streamlit Cloud) or .env file (for local development) or pass it as argument.")
        
        self.model = model or _get_secret("OPENAI_MODEL", "gpt-4.1")
        
        # Initialize the official OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize the async OpenAI client
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion with the OpenAI API.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (defaults to instance default)
            temperature: Temperature for sampling
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            logit_bias: Logit bias to apply
            user: User identifier
            stream: Whether to stream the response
            tools: List of tool definitions to use (including web search)
            tool_choice: Specifies which tool to use
            response_format: Response format (e.g. {"type": "json_object"})
            
        Returns:
            API response as a dictionary
        """
        # Create parameters dictionary
        params = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
        }
        
        # Add optional parameters if provided
        if max_tokens:
            params["max_tokens"] = max_tokens
        if presence_penalty != 0:
            params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0:
            params["frequency_penalty"] = frequency_penalty
        if logit_bias:
            params["logit_bias"] = logit_bias
        if user:
            params["user"] = user
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
        if response_format:
            params["response_format"] = response_format
        
        # Call the OpenAI API using the SDK
        if stream:
            response = self.client.chat.completions.create(**params)
            return self._process_streaming_response(response)
        else:
            response = self.client.chat.completions.create(**params)
            return self._process_chat(response)
    
    @retry(
        wait=wait_exponential(multiplier=5, min=1, max=60),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException, asyncio.TimeoutError))
    )
    async def get_chat_completion_async(
        self,
        messages: List[dict],
        max_tokens: int = 3000,
        temperature: float = 0,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None
    ):
        """
        Asynchronously get chat completion
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Optional stop sequence(s)
            stream: Whether to stream the response
            response_format: Optional response format
            model: Model to use (defaults to instance default)
            
        Returns:
            ChatCompletion object containing the response (or generator if streaming)
        """
        response = await self.async_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            response_format=response_format
        )
        
        if stream:
            # Handle streaming response
            full_content = ""
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_content += chunk.choices[0].delta.content
            
            # Create a mock response object for consistency
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockResponse(full_content)
        else:
            return response
    
    @retry(
        wait=wait_exponential(multiplier=5, min=1, max=60),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException, asyncio.TimeoutError))
    )
    async def get_chat_completion_stream_async(
        self,
        messages: List[dict],
        max_tokens: int = 3000,
        temperature: float = 0,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None
    ):
        """
        Asynchronously get streaming chat completion that yields chunks
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Optional stop sequence(s)
            response_format: Optional response format
            model: Model to use (defaults to instance default)
            
        Yields:
            Content chunks as they arrive
        """
        response = await self.async_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=True,
            response_format=response_format
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    @retry(
        wait=wait_exponential(multiplier=5, min=1, max=60),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException, asyncio.TimeoutError))
    )
    async def get_parsed_completion_async(
        self,
        messages: List[dict],
        max_completion_tokens: int = None, 
        temperature: float = 0,
        top_p: float = 0.95,
        model: Optional[str] = None,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        response_format: Optional[Any] = None,
        effort: Optional[Literal["low", "medium", "high"]] = None
    ):
        """
        Asynchronously get parsed chat completion with JSON validation and retry logic.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Optional stop sequence(s)
            response_format: Optional response format (a Pydantic model)
            
        Returns:
            ChatCompletion object containing the parsed response
        """
        
        # Clone messages to avoid modifying the original
        messages_copy = messages.copy()


        # Prepare parameters for parse method
        parse_params = {
            "model": model or self.model,
            "messages": messages_copy,
            "max_completion_tokens": max_completion_tokens,
            "response_format": response_format
        }
        
        
        # if self.model contain 3o or 1o, then add effort to the messages
        if self.model in ["3o", "1o", "4o"]:
            parse_params['reasoning_effort'] = effort

            
        response = await self.async_client.beta.chat.completions.parse(**parse_params)
            
        # Validate JSON by dumping and parsing it
        dumped_json = response.choices[0].message.parsed.model_dump_json()
        json.loads(dumped_json)
        return response
    
    def get_parsed_completion(
        self,
        messages: List[dict],
        max_tokens: int = 800,
        temperature: float = 0,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None
    ):
        """
        Synchronously get parsed chat completion
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Optional stop sequence(s)
            response_format: Optional response format (a Pydantic model)
            tools: Optional list of tools
            tool_choice: Optional tool choice
            
        Returns:
            ChatCompletion object containing the parsed response
        """
        stop_sequences = stop or ["<END>"]
        
        # Clone messages to avoid modifying the original
        messages_copy = messages.copy()
        
        # Prepare the response_format parameter for the API call
        api_response_format = None
        if response_format is not None:
            # If response_format is a dict with a "type" key, use it as is
            if isinstance(response_format, dict) and "type" in response_format:
                api_response_format = response_format
            # Otherwise, set simple JSON format
            else:
                # For Pydantic models, create a proper json_schema format
                schema = response_format.model_json_schema()['$defs']
                
                # Remove unsupported fields from schema
                unsupported_top_fields = ["title"]
                for field in unsupported_top_fields:
                    if field in schema:
                        del schema[field]
                
                # Ensure additionalProperties is set to false for strict schema validation
                if "additionalProperties" not in schema:
                    schema["additionalProperties"] = False
                        
                # Create proper response_format with json_schema type
                api_response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__ if hasattr(response_format, "__name__") else "ResponseSchema",
                        "description": schema['description'],
                        "schema": schema,
                        "strict": True
                    }
                }
        
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages_copy,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            response_format=api_response_format,
            tools=tools,
            tool_choice=tool_choice
        )
        
        # Validate JSON by dumping and parsing it
        dumped_json = response.choices[0].message.parsed.model_dump_json()
        json.loads(dumped_json)        
        
        return response
    
    def _process_response(self, response):
        """
        Convert the API response object to a dictionary.
        
        Args:
            response: The API response object
            
        Returns:
            The response as a dictionary
        """
        return response.model_dump()
    
    def _process_streaming_response(self, response):
        """
        Process a streaming response.
        
        Args:
            response: The streaming API response
            
        Yields:
            Processed chunks of the response
        """
        for chunk in response:
            yield chunk.model_dump()

    @retry(
        wait=wait_exponential(multiplier=5, min=1, max=60),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException, asyncio.TimeoutError))
    )
    def _process_chat(self, response: Any) -> dict:
        """
        Process the response from the OpenAI API.
        """
        # Handle potential errors or empty responses
        if not response or not response.choices:
            return {"content": "", "citations": []}
            
        # Extract the text content
        candidate = response.choices[0]
        text_completion = candidate.message.content if candidate.message.content else ""
        citations = []
        print(candidate.message.annotations)
        if candidate.message.annotations:
            for annotation in candidate.message.annotations:
                if annotation.type == "url_citation":
                    citations.append(annotation.url_citation.url)

        
        return {"content": text_completion, "citations": citations}


    def _process_response_object(self, response: Any) -> dict:
        """
        Process the Response object from the OpenAI responses.create API.
        """
        text_completion = ""
        citations = []

        # Check if it's a Response object from responses.create
        if hasattr(response, 'output') and isinstance(response.output, list):
            for item in response.output:
                # Look for the message containing the text and annotations
                if hasattr(item, 'type') and item.type == 'message' and hasattr(item, 'content') and isinstance(item.content, list):
                    for content_item in item.content:
                        if hasattr(content_item, 'type') and content_item.type == 'output_text':
                            text_completion = getattr(content_item, 'text', "")
                            # Extract citations from annotations
                            if hasattr(content_item, 'annotations') and isinstance(content_item.annotations, list):
                                for annotation in content_item.annotations:
                                    # Check specifically for url_citation and extract url
                                    if hasattr(annotation, 'type') and annotation.type == 'url_citation' and hasattr(annotation, 'url'):
                                        url = annotation.url
                                        parsed_url = urlparse(url)
                                        domain = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                                        citations.append({'domain': domain, 'url': url})
                            # Found the main text, break inner loops
                            break
                    # Found the message, break outer loop
                    break
        # Handle cases where the structure might be dumped to dict before processing
        elif isinstance(response, dict):
             if 'output' in response and isinstance(response['output'], list):
                 for item in response['output']:
                     if item.get('type') == 'message' and 'content' in item and isinstance(item['content'], list):
                         for content_item in item['content']:
                             if content_item.get('type') == 'output_text':
                                 text_completion = content_item.get('text', "")
                                 if 'annotations' in content_item and isinstance(content_item['annotations'], list):
                                     for annotation in content_item['annotations']:
                                         if annotation.get('type') == 'url_citation' and 'url' in annotation:
                                             url = annotation['url']
                                             parsed_url = urlparse(url)
                                             domain = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                                             citations.append({'domain': domain, 'url': url})
                                 break
                         break

        # Ensure return dictionary format is consistent
        return {"content": text_completion, "citations": citations} # Return list of citation dicts



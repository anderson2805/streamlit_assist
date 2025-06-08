import os
import json
import requests
from typing import List, Dict, Any, Literal, Optional, Generator
from dotenv import load_dotenv
import aiohttp
import asyncio
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

class PerplexityClient:
    """
    Client for interacting with the Perplexity API.
    
    Documentation: https://docs.perplexity.ai/api-reference/chat-completions-post
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, search_context_size: Literal["low", "medium", "high"] = "high"):
        """
        Initialize the Perplexity API client.
        
        Args:
            api_key: Perplexity API key. If None, it will be loaded from st.secrets or environment variables.
            model: Default model to use. If None, it will be loaded from st.secrets or environment variables.
            search_context_size: Default search context size to use ("low", "medium", or "high").
        """
        load_dotenv(override = True)  # Load environment variables from .env file
        
        self.api_key = api_key or _get_secret("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key is required. Set it in secrets.toml (for Streamlit Cloud) or .env file (for local development) or pass it as argument.")
        
        self.model = model or _get_secret("PERPLEXITY_MODEL", "sonar-pro")
        self.search_context_size = search_context_size
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        response_format: Optional[Dict[str, str]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        search_domain_filter: Optional[List[str]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: Optional[Literal["month", "week", "day", "hour"]] = None,
        top_k: int = 0,
        return_citations: bool = False,
        cite_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Create a chat completion with the Perplexity API.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (defaults to instance default)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            response_format: Response format (e.g. {"type": "json_object"})
            temperature: Temperature for sampling (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            search_domain_filter: List of domains to filter search results
            return_images: Whether to include images in the response
            return_related_questions: Whether to include related questions
            search_recency_filter: Filter search results by recency
            top_k: Top-k sampling parameter
            return_citations: Whether to return citations
            cite_sources: Whether to cite sources in the response
            
        Returns:
            API response as a dictionary
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
        }
        
        # Add optional parameters if provided
        if frequency_penalty != 0:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0:
            payload["presence_penalty"] = presence_penalty
        if response_format:
            payload["response_format"] = response_format
        if temperature != 1.0:
            payload["temperature"] = temperature
        if top_p != 1.0:
            payload["top_p"] = top_p
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = stream
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if return_images:
            payload["return_images"] = return_images
        if return_related_questions:
            payload["return_related_questions"] = return_related_questions
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if top_k > 0:
            payload["top_k"] = top_k
        if return_citations:
            payload["return_citations"] = return_citations
        if cite_sources:
            payload["cite_sources"] = cite_sources
            
        # Handle streaming responses
        if stream:
            return self._stream_response(payload)
        else:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    def _stream_response(self, payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the response from the API.
        
        Args:
            payload: Request payload
            
        Yields:
            Parsed chunks of the streaming response
        """
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        pass
    
    async def search_async(self, query: str, search_context_size: Literal["low", "medium", "high"] = "high", search_domain_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform an asynchronous search query using Perplexity.
        
        Args:
            query: Search query string
            search_context_size: Size of search context ("low", "medium", or "high")
            search_domain_filter: List of domains to filter search results (defaults to gov.sg domains)
            
        Returns:
            Processed response with content and citations
        """
          
        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [{"role": "user", "content": query}],
            "return_citations": True,
            "cite_sources": True,
            "search_domain_filter": search_domain_filter
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                api_response = await response.json()
                processed_response = self._process_response(api_response)
                return processed_response

    
    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the response from the Perplexity API according to the latest API structure.
        
        Args:
            response: Raw API response
            
        Returns:
            Processed response with content, citations, search_results, and usage info
        """
        # Handle potential errors or empty responses
        if not response or 'choices' not in response or not response['choices']:
            return {
                "content": "", 
                "citations": [], 
                "search_results": [],
                "usage": {},
                "id": response.get('id', ''),
                "model": response.get('model', ''),
                "created": response.get('created', 0)
            }
            
        # Extract the text content from the first choice
        choice = response['choices'][0]
        content = choice['message']['content']
        
        # Process citations
        citations = response.get('citations', [])
        processed_citations = []
        
        if citations:
            for citation in citations:
                if isinstance(citation, str):
                    try:
                        parsed_url = urlparse(citation)
                        domain = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                        processed_citations.append({
                            'domain': domain, 
                            'url': citation,
                            'title': None  # Title not available in URL-only citations
                        })
                    except Exception: 
                        processed_citations.append({
                            'domain': None, 
                            'url': citation,
                            'title': None
                        })
                elif isinstance(citation, dict):
                    # Handle structured citation objects if they exist
                    processed_citations.append(citation)

        # Process search results
        search_results = response.get('search_results', [])
        processed_search_results = []
        
        for result in search_results:
            if isinstance(result, dict):
                processed_search_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'date': result.get('date', ''),
                    'snippet': result.get('snippet', '')
                })

        # Process usage information
        usage = response.get('usage', {})
        processed_usage = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'search_context_size': usage.get('search_context_size', ''),
            'citation_tokens': usage.get('citation_tokens', 0),
            'num_search_queries': usage.get('num_search_queries', 0),
            'reasoning_tokens': usage.get('reasoning_tokens', 0)
        }

        return {
            "content": content,
            "citations": processed_citations,
            "search_results": processed_search_results,
            "usage": processed_usage,
            "id": response.get('id', ''),
            "model": response.get('model', ''),
            "created": response.get('created', 0),
            "object": response.get('object', ''),
            "finish_reason": choice.get('finish_reason', '')
        }

    def get_available_models(self) -> List[str]:
        """
        Get list of available models (based on common Perplexity models).
        
        Returns:
            List of available model names
        """
        return [
            "sonar",
            "sonar-pro", 
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online"
        ]


# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = PerplexityClient()
    
    # Search example with simpler parameters
    async def main():
        query = "What has been the response of Britain's Defence Ministry to the FSB's allegation about special forces operations?"
        search_result = await client.search_async(query)
        print("Search Result:")
        print(f"Content: {search_result['content'][:200]}...")
        print(f"Citations: {len(search_result['citations'])} found")
        print(f"Usage: {search_result}")
        
        print(search_result['citations'])
        
        # # Example of regular chat completion
        # messages = [
        #     {"role": "system", "content": "Be precise and concise."},
        #     {"role": "user", "content": "How many stars are there in our galaxy?"}
        # ]
        
        # completion = client.chat_completion(
        #     messages=messages,
        #     temperature=0.7,
        #     return_citations=True
        # )
        
        # processed = client._process_response(completion)
        # print(f"\nChat Completion: {processed['content'][:200]}...")
        
    asyncio.run(main())

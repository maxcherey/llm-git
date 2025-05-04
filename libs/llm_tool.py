import argparse
import json
import logging
import os
import requests
import sys
from typing import List

# Configuration
PREFERRED_MODELS = [
    'gpt-4',           # OpenAI
    'gpt-4-turbo',     # OpenAI
    'gpt-3.5-turbo',   # OpenAI
    'llama3',          # Llama variants
    'llama-2',
    'mistral',         # Mistral variants
    'mixtral',
    'qwen',            # Qwen variants
    'nemotron',        # Nvidia
    'claude',          # Anthropic
]

DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:1234",      # LM Studio
    "http://localhost:1234",       # LM Studio alternative
    "http://127.0.0.1:11434",     # Ollama
    "http://localhost:11434",      # Ollama alternative
]


class LLMToolBase:
    def __init__(self, temperature: float = 0.5, max_tokens: int = 16384,
                 api_endpoints: List[str] = None, model: str = None, verbose: int = 0,
                 quiet: bool = False):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.quiet = quiet

        # Configure logging based on verbosity
        self.setup_logging(verbose)

        self.api_endpoints = api_endpoints or DEFAULT_ENDPOINTS

        logging.info(f"Initialized with {len(self.api_endpoints)} endpoints")
        logging.debug(f"Endpoints: {self.api_endpoints}")

        # Load API keys from environment variables
        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        }

    def setup_logging(self, verbose: int):
        """Configure logging based on verbosity level."""
        if verbose >= 2:
            log_level = logging.DEBUG
        elif verbose >= 1:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def discover_models(self, endpoint: str) -> List[str]:
        """Discover available models from the endpoint."""
        try:
            logging.info(f"Discovering models from {endpoint}/v1/models")
            response = requests.get(f"{endpoint}/v1/models",
                                 headers=self._get_headers_for_endpoint(endpoint),
                                 timeout=(3, 5))

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    models = [model["id"] for model in data["data"]]
                else:
                    models = [m["id"] if isinstance(m, dict) else m for m in data]

                logging.info(f"Found {len(models)} models: {', '.join(models[:5])}" +
                           ("..." if len(models) > 5 else ""))
                return models
            else:
                logging.warning(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logging.debug(f"Error discovering models: {str(e)}")
            return []

    def discover_all_models(self) -> dict:
        """Discover all available models from all endpoints."""
        all_models = {}

        print("\nDiscovering available models...")
        for endpoint in self.api_endpoints:
            try:
                models = self.discover_models(endpoint)
                if models:
                    all_models[endpoint] = models
                    print(f"✓ Found {len(models)} models at {endpoint}")
            except Exception as e:
                print(f"✗ Failed to discover models at {endpoint}: {str(e)}")

        return all_models

    def discover_endpoints(self):
        """Test each endpoint and return available ones."""
        available_endpoints = []
        default_models = {}  # Store default models for each endpoint

        # If specific model requested, try to find it first
        if self.model:
            all_models = self.discover_all_models()
            if not all_models:
                print("\nNo endpoints with models found!")
                sys.exit(1)

            # Try to find the requested model
            model_found = False
            print(f"\nLooking for model '{self.model}'...")
            for endpoint, models in all_models.items():
                if self.model in models:
                    model_found = True
                    print(f"✓ Found requested model at {endpoint}")
                    available_endpoints.append(endpoint)
                    default_models[endpoint] = self.model
                    break

            if not model_found:
                print(f"\n✗ Model '{self.model}' not found in any endpoint.")
                print("\nAvailable models by endpoint:")
                for endpoint, models in all_models.items():
                    print(f"\n{endpoint}:")
                    for model in models:
                        print(f"  - {model}")
                print("\nPlease choose one of the available models using the -m option.")
                sys.exit(1)

            return available_endpoints

        # If no specific model requested, try endpoints one by one
        print("\nTesting endpoints...")
        for endpoint in self.api_endpoints:
            try:
                print(f"\nTrying endpoint: {endpoint}")

                # Discover models
                models = self.discover_models(endpoint)
                if not models:
                    print(f"✗ No models found at {endpoint}")
                    continue

                # Try to find the best model from preferred list
                for preferred in PREFERRED_MODELS:
                    matching_model = next(
                        (m for m in models if preferred in m.lower()),
                        None
                    )
                    if matching_model:
                        default_models[endpoint] = matching_model
                        print(f"✓ Selected model: {matching_model}")
                        break
                else:
                    # If no preferred model found, use the first available
                    default_models[endpoint] = models[0]
                    print(f"✓ Using default model: {models[0]}")

                # Test endpoint with selected model
                test_model = default_models[endpoint]
                test_payload = {
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "model": test_model
                }

                response = requests.post(
                    f"{endpoint}/v1/chat/completions",
                    headers=self._get_headers_for_endpoint(endpoint),
                    json=test_payload,
                    timeout=(3, 5)
                )

                if response.status_code == 200:
                    print(f"✓ Endpoint working: {endpoint}")
                    available_endpoints.append(endpoint)
                    self.model = default_models[endpoint]
                    break
                else:
                    error_msg = self._parse_error_response(response)
                    print(f"✗ Endpoint failed: {endpoint}\n  Status: {response.status_code}\n  Error: {error_msg}")
            except requests.exceptions.ConnectTimeout:
                print(f"✗ Endpoint connection timeout: {endpoint}")
            except requests.exceptions.ReadTimeout:
                print(f"✗ Endpoint read timeout: {endpoint}")
            except requests.exceptions.ConnectionError:
                print(f"✗ Endpoint connection failed: {endpoint}")
            except Exception as e:
                print(f"✗ Endpoint error: {endpoint}\n  Error: {str(e)}")

        if not available_endpoints:
            print("\nNo working endpoints found. Please check that one of these is running:")
            print("  - LM Studio (http://127.0.0.1:1234)")
            print("  - Ollama (http://127.0.0.1:11434)")
            sys.exit(1)

        return available_endpoints

    def make_api_request(self, endpoint: str, messages: List[dict]) -> str:
        """Make API request to LLM endpoint."""
        headers = self._get_headers_for_endpoint(endpoint)

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model
        }

        logging.debug(f"Making request to {endpoint}/v1/chat/completions")
        logging.debug(f"Request payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(
                f"{endpoint}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(3, 30)
            )
            logging.debug(f"Response status: {response.status_code}")
            logging.debug(f"Response headers: {dict(response.headers)}")
            logging.debug(f"Response body: {response.text[:1000]}...")  # Truncate long responses

            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                if isinstance(data, dict):
                    if "choices" in data:
                        # OpenAI-compatible format
                        if isinstance(data["choices"], list) and len(data["choices"]) > 0:
                            if "message" in data["choices"][0]:
                                return data["choices"][0]["message"]["content"]
                            elif "text" in data["choices"][0]:
                                return data["choices"][0]["text"]
                    elif "response" in data:
                        # Ollama format
                        return data["response"]
                    elif "content" in data:
                        # Anthropic format
                        return data["content"][0]["text"]

                logging.error(f"Unexpected response format: {data}")
                raise Exception("Unexpected response format")

            error_msg = self._parse_error_response(response)
            raise Exception(f"API request failed ({response.status_code}): {error_msg}")
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
            raise

    def _get_headers_for_endpoint(self, endpoint: str) -> dict:
        """Get appropriate headers for the endpoint."""
        headers = {"Content-Type": "application/json"}
        if "openai" in endpoint:
            headers["Authorization"] = f"Bearer {self.api_keys['OPENAI_API_KEY']}"
        elif "anthropic" in endpoint:
            headers["x-api-key"] = self.api_keys['ANTHROPIC_API_KEY']
        return headers

    def _parse_error_response(self, response):
        """Parse error response to get meaningful message."""
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                return error_data.get('error', {}).get('message') or error_data.get('detail') or response.text
            return response.text
        except:
            return response.text


def parse_args(additional_groups=None) -> argparse.Namespace:
    """
    Parse command line arguments with grouped options.

    Args:
        additional_groups: Optional list of (group_name, arguments) tuples
            where arguments is a list of (args, kwargs) tuples for add_argument

    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Git commit helper with LLM-generated commit messages"
    )

    # Create option group for LLM parameters
    llm_group = parser.add_argument_group('LLM Options')
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="LLM temperature parameter"
    )
    llm_group.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for LLM response"
    )
    llm_group.add_argument(
        "--api-url",
        type=str,
        help="Custom API endpoint URL"
    )
    llm_group.add_argument(
        "-m", "--model",
        type=str,
        help="LLM model to use (e.g., gpt-4, gpt-3.5-turbo, qwen)"
    )

    # Create option group for output control
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG)"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - don't show patch"
    )

    # Add any additional option groups
    if additional_groups:
        for group_name, arguments in additional_groups:
            group = parser.add_argument_group(group_name)
            for args, kwargs in arguments:
                group.add_argument(*args, **kwargs)

    args = parser.parse_args()
    return args

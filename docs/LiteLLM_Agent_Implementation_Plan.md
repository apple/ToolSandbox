# LiteLLM Agent Implementation Plan

## Overview

This document outlines the implementation plan for replacing the current provider-specific agent architecture with a unified LiteLLM-based approach. This will consolidate 12+ separate agent classes into a single, extensible implementation.

## Current Architecture Problems

### Code Duplication Analysis
```
Current Structure:
├── openai_api_agent.py (172 lines)
├── anthropic_api_agent.py (468 lines)
├── cohere_agent.py (303 lines)
├── gemini_agent.py (411 lines)
├── mistral_api_agent.py (405 lines)
└── ... (8+ more provider agents)

Total: ~2000+ lines of mostly duplicated logic
```

### Key Issues
1. **Provider-specific clients**: Each agent imports different SDKs (openai, anthropic, vertexai, etc.)
2. **Custom message conversion**: Each has unique `to_X_messages()` functions
3. **Different tool schemas**: OpenAI vs Anthropic vs Gemini tool formats
4. **Inconsistent error handling**: Different retry mechanisms and exception types
5. **Maintenance overhead**: Adding new models requires new agent classes

## Proposed Solution Architecture

### High-Level Design
```
┌─────────────────────────────────────────────────────────────┐
│                    LiteLLMAgent                             │
├─────────────────────────────────────────────────────────────┤
│  • Single agent class for all providers                    │
│  • Model name determines provider (e.g., "gpt-4", "claude")│
│  • Unified error handling and retry logic                  │
│  • Optional LiteLLM Router for advanced features           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LiteLLM Library                       │
├─────────────────────────────────────────────────────────────┤
│  • Handles provider-specific API calls                     │
│  • Standardizes request/response formats                   │
│  • Built-in fallbacks, retries, caching                   │
│  • Supports 100+ models from 20+ providers                │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Implementation Plan

### Phase 1: Core LiteLLMAgent Implementation

#### 1.1 Base Agent Class

```python
# tool_sandbox/roles/litellm_agent.py

class LiteLLMAgent(BaseRole):
    """Unified agent using LiteLLM for all LLM providers."""

    def __init__(self, model_name: str, **config):
        """
        Args:
            model_name: LiteLLM model identifier (e.g., "gpt-4", "claude-3-opus")
            **config: Additional LiteLLM configuration
        """
        self.model_name = model_name
        self.config = self._validate_and_setup_config(config)
        self.router = self._setup_router_if_needed()

    def _validate_and_setup_config(self, config: dict) -> dict:
        """
        Pseudocode:
        1. Extract LiteLLM-specific parameters (api_key, base_url, etc.)
        2. Set default timeouts and retries
        3. Validate model name format
        4. Setup provider-specific configurations
        """
        pass

    def _setup_router_if_needed(self) -> Optional[litellm.Router]:
        """
        Pseudocode:
        1. Check if fallbacks or load balancing configured
        2. Create Router instance with model list
        3. Configure retry/timeout policies
        4. Return Router or None for simple cases
        """
        pass

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Main response method - unified across all providers."""
        # Pseudocode implementation below
```

#### 1.2 Core Response Logic

```python
def respond(self, ending_index: Optional[int] = None) -> None:
    """
    Pseudocode:
    1. Get messages from context using inherited method
    2. Validate messages are directed to agent
    3. Filter messages for relevant ones only
    4. Skip if last message is from SYSTEM
    5. Get available tools and convert to OpenAI format (reuse existing)
    6. Convert messages to OpenAI format (reuse existing)
    7. Call LiteLLM with unified parameters
    8. Parse response and create tool calls or user messages
    9. Add response messages to context
    """

    # Step 1-4: Reuse existing BaseRole logic
    messages = self.get_messages(ending_index=ending_index)
    self.messages_validation(messages=messages)
    messages = self.filter_messages(messages=messages)
    if messages[-1].sender == RoleType.SYSTEM:
        return

    # Step 5-6: Reuse existing conversion utilities
    available_tools = self.get_available_tools()
    available_tool_names = set(available_tools.keys())

    if should_provide_tools(messages[-1]):
        openai_tools = convert_to_openai_tools(available_tools)
    else:
        openai_tools = None

    openai_messages, _ = to_openai_messages(messages)

    # Step 7: Unified LiteLLM call
    response = self._make_llm_call(
        messages=openai_messages,
        tools=openai_tools,
        available_tool_names=available_tool_names
    )

    # Step 8-9: Unified response handling
    response_messages = self._parse_response_to_messages(
        response, available_tool_names
    )
    self.add_messages(response_messages)
```

#### 1.3 LLM Call Implementation

```python
def _make_llm_call(self, messages, tools, available_tool_names):
    """
    Pseudocode:
    1. Prepare LiteLLM parameters from instance config
    2. Add tools if provided
    3. Use Router if configured, otherwise direct litellm.completion()
    4. Handle streaming vs non-streaming
    5. Apply consistent error handling
    6. Return standardized ModelResponse
    """

    llm_params = {
        'model': self.model_name,
        'messages': messages,
        'tools': tools,
        **self._get_generation_params()
    }

    try:
        if self.router:
            response = self.router.completion(**llm_params)
        else:
            response = litellm.completion(**llm_params)
        return response
    except Exception as e:
        # Unified error handling
        raise self._convert_to_standard_exception(e)

def _get_generation_params(self) -> dict:
    """
    Pseudocode:
    1. Extract generation parameters from config
    2. Apply defaults for temperature, max_tokens, etc.
    3. Return dict of parameters for LiteLLM
    """
    return {
        'temperature': self.config.get('temperature', 0.0),
        'max_tokens': self.config.get('max_tokens', 1024),
        'timeout': self.config.get('timeout', 30),
        'num_retries': self.config.get('num_retries', 3)
    }
```

#### 1.4 Response Parsing

```python
def _parse_response_to_messages(self, response, available_tool_names):
    """
    Pseudocode:
    1. Check if response contains tool calls
    2. If no tool calls: create user-directed message
    3. If tool calls: create execution environment messages
    4. Reuse existing openai_tool_call_to_python_code()
    5. Handle tool name scrambling via context
    6. Return list of Message objects
    """

    response_messages = []
    llm_message = response.choices[0].message

    if llm_message.tool_calls is None:
        # Regular text response
        response_messages.append(Message(
            sender=self.role_type,
            recipient=RoleType.USER,
            content=llm_message.content
        ))
    else:
        # Tool calls - reuse existing conversion logic
        current_context = get_current_context()
        for tool_call in llm_message.tool_calls:
            execution_facing_name = current_context.get_execution_facing_tool_name(
                tool_call.function.name
            )
            response_messages.append(Message(
                sender=self.role_type,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=openai_tool_call_to_python_code(
                    tool_call, available_tool_names, execution_facing_name
                ),
                openai_tool_call_id=tool_call.id,
                openai_function_name=tool_call.function.name
            ))

    return response_messages
```

### Phase 2: Advanced Features

#### 2.1 Router Configuration

```python
class LiteLLMAgentWithRouter(LiteLLMAgent):
    """Enhanced agent with LiteLLM Router for advanced features."""

    def __init__(self, model_config: list[dict], **router_config):
        """
        Args:
            model_config: List of model configurations for Router
            **router_config: Router-specific settings
        """
        self.router = self._setup_router(model_config, router_config)
        # No single model_name since Router manages multiple models
        super().__init__(model_name="router", **router_config)

    def _setup_router(self, model_config, router_config):
        """
        Pseudocode:
        1. Validate model configuration list
        2. Setup fallback chains
        3. Configure load balancing strategy
        4. Setup caching if enabled
        5. Configure retry/timeout policies
        6. Return configured Router instance
        """

        # Example configuration:
        router_models = [
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "claude-3-opus",
                    "api_key": os.getenv("ANTHROPIC_API_KEY")
                }
            }
        ]

        return litellm.Router(
            model_list=router_models,
            fallbacks=[{"primary": ["fallback"]}],
            num_retries=3,
            timeout=30,
            **router_config
        )
```

#### 2.2 Caching Integration

```python
def _setup_caching(self, cache_config: dict):
    """
    Pseudocode:
    1. Determine cache backend (redis, disk, memory)
    2. Setup cache instance with TTL
    3. Configure cache for specific call types
    4. Handle cache errors gracefully
    """

    if cache_config.get('enabled'):
        cache_type = cache_config.get('type', 'memory')
        cache_ttl = cache_config.get('ttl', 3600)

        if cache_type == 'redis':
            litellm.cache = litellm.Cache(
                type="redis",
                host=cache_config.get('host', 'localhost'),
                port=cache_config.get('port', 6379),
                ttl=cache_ttl
            )
        elif cache_type == 'disk':
            litellm.cache = litellm.Cache(
                type="disk",
                directory=cache_config.get('directory', './llm_cache'),
                ttl=cache_ttl
            )
        # ... other cache types
```

#### 2.3 Cost Tracking

```python
def _setup_cost_tracking(self):
    """
    Pseudocode:
    1. Enable LiteLLM cost calculation callbacks
    2. Setup budget limits if configured
    3. Track costs per model/user
    4. Log cost information
    """

    # Setup cost tracking callback
    def cost_callback(kwargs, response_obj, start_time, end_time):
        cost = litellm.completion_cost(response_obj)
        # Log or store cost information
        self._log_cost(cost, kwargs.get('model'), kwargs.get('metadata'))

    litellm.success_callback = [cost_callback]

    # Setup budget limits
    if self.config.get('max_budget'):
        litellm.max_budget = self.config['max_budget']
```

### Phase 3: Integration and Migration

#### 3.1 Factory Function Updates

```python
# In tool_sandbox/cli/utils.py

# New LiteLLM-based factory functions
def create_litellm_agent(model_name: str, **config) -> LiteLLMAgent:
    """Factory function for creating LiteLLM agents."""
    return LiteLLMAgent(model_name=model_name, **config)

# Updated AGENT_TYPE_TO_FACTORY
AGENT_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    # Legacy agents (deprecated)
    RoleImplType.GPT_4_0125: lambda: create_litellm_agent("gpt-4-0125-preview"),
    RoleImplType.Claude_3_Opus: lambda: create_litellm_agent("claude-3-opus-20240229"),
    RoleImplType.Gemini_1_5: lambda: create_litellm_agent("gemini-1.5-pro"),

    # New model types (easy to add)
    RoleImplType.GPT_4_Turbo: lambda: create_litellm_agent("gpt-4-turbo"),
    RoleImplType.Claude_3_5_Sonnet: lambda: create_litellm_agent("claude-3-5-sonnet-20241022"),
    RoleImplType.Llama_3_70B: lambda: create_litellm_agent("groq/llama3-70b-8192"),
    RoleImplType.Mistral_Large: lambda: create_litellm_agent("mistral/mistral-large-latest"),

    # Router-based configurations
    RoleImplType.Multi_Provider: lambda: create_router_agent([
        {"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")},
        {"model": "claude-3-opus", "api_key": os.getenv("ANTHROPIC_API_KEY")}
    ]),
}
```

#### 3.2 Configuration Management

```python
# New configuration system
class LiteLLMConfig:
    """Configuration management for LiteLLM agents."""

    @classmethod
    def from_env(cls, model_name: str) -> dict:
        """
        Pseudocode:
        1. Detect provider from model name
        2. Load appropriate environment variables
        3. Set provider-specific defaults
        4. Return configuration dict
        """

        config = {'model': model_name}

        # Auto-detect provider and load credentials
        if model_name.startswith('gpt') or model_name.startswith('openai/'):
            config['api_key'] = os.getenv('OPENAI_API_KEY')
            config['api_base'] = os.getenv('OPENAI_API_BASE')
        elif model_name.startswith('claude') or model_name.startswith('anthropic/'):
            config['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        elif model_name.startswith('gemini') or model_name.startswith('vertex_ai/'):
            config['api_key'] = os.getenv('GOOGLE_API_KEY')
            config['vertex_project'] = os.getenv('VERTEX_PROJECT')
            config['vertex_location'] = os.getenv('VERTEX_LOCATION')
        # ... other providers

        return config

    @classmethod
    def with_fallbacks(cls, primary_model: str, fallback_models: list[str]) -> dict:
        """Create configuration with fallback models."""
        return {
            'model_list': [
                {'model_name': 'primary', 'litellm_params': cls.from_env(primary_model)}
            ] + [
                {'model_name': f'fallback_{i}', 'litellm_params': cls.from_env(model)}
                for i, model in enumerate(fallback_models)
            ],
            'fallbacks': [{'primary': [f'fallback_{i}' for i in range(len(fallback_models))]}]
        }
```

#### 3.3 Migration Strategy

```python
# Backward compatibility wrappers
class GPT_4_0125_Agent(LiteLLMAgent):
    """Backward compatibility wrapper."""
    def __init__(self):
        super().__init__(model_name="gpt-4-0125-preview")

class ClaudeOpusAgent(LiteLLMAgent):
    """Backward compatibility wrapper."""
    def __init__(self):
        super().__init__(model_name="claude-3-opus-20240229")

# Similar wrappers for other legacy agents...
```

### Phase 4: Testing and Validation

#### 4.1 Test Suite Updates

```python
# Enhanced testing for unified agent
class TestLiteLLMAgent:
    """Test suite for LiteLLM agent."""

    def test_basic_completion(self):
        """Test basic completion across providers."""
        for model in ["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"]:
            agent = LiteLLMAgent(model_name=model)
            # Test basic functionality

    def test_tool_calling(self):
        """Test tool calling standardization."""
        # Verify tool calls work consistently across providers

    def test_fallback_behavior(self):
        """Test fallback mechanisms."""
        # Test Router fallbacks work correctly

    def test_error_handling(self):
        """Test unified error handling."""
        # Verify consistent error behavior across providers
```

#### 4.2 Performance Benchmarking

```python
def benchmark_agent_performance():
    """
    Pseudocode:
    1. Compare LiteLLMAgent vs existing agents
    2. Measure latency, memory usage, error rates
    3. Test with various model types and scenarios
    4. Validate cost tracking accuracy
    5. Test fallback performance impact
    """
    pass
```

## Benefits Summary

### Immediate Benefits
1. **Code Reduction**: ~2000 lines → ~200 lines (90% reduction)
2. **Unified Interface**: Same API for all providers
3. **Easy Model Addition**: New models require zero code changes
4. **Consistent Behavior**: Same retry/timeout/error handling everywhere

### Long-term Benefits
1. **Advanced Routing**: Load balancing, latency optimization
2. **Cost Management**: Built-in cost tracking and budgets
3. **Better Reliability**: Professional fallback mechanisms
4. **Extensibility**: Easy to add new features across all models

### Risk Mitigation
1. **Backward Compatibility**: Legacy agent classes remain as wrappers
2. **Gradual Migration**: Can migrate scenarios one by one
3. **Fallback to Direct APIs**: Can always bypass LiteLLM if needed
4. **Comprehensive Testing**: Validate behavior matches existing agents

## Implementation Timeline

- **Week 1**: Phase 1 - Core LiteLLMAgent implementation
- **Week 2**: Phase 2 - Advanced features (Router, caching)
- **Week 3**: Phase 3 - Integration and migration setup
- **Week 4**: Phase 4 - Testing, validation, documentation

## Conclusion

This unified LiteLLM approach will dramatically simplify the codebase while making it more extensible and maintainable. The implementation leverages LiteLLM's battle-tested provider integrations while maintaining backward compatibility and adding powerful new features like automatic fallbacks and cost tracking.

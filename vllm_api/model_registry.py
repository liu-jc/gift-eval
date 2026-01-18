"""
Model Registry - centralized LLM/VLM configuration management

Supported model types:
- Text-only LLM (Qwen, GPT, LLaMA, etc.)
- Vision-Language Model (Qwen-VL, GPT-4V, LLaMA-Vision, etc.)

Usage:
    from model_registry import ModelRegistry
    
    # Get model config
    model_config = ModelRegistry.get("qwen2.5-7b")
    
    # Register new model
    ModelRegistry.register("custom-model", CustomModelConfig)
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np

from utils import get_readable_frequency


# ============================================================================
# Base Model Configuration
# ============================================================================

class BaseModelConfig(ABC):
    """Base class for model configurations"""
    
    # Model info
    model_name: str = ""
    model_type: str = "text"  # "text" or "vlm"
    api_endpoint: str = "/generate"
    
    # Default parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 5000
    default_top_p: float = 1.0
    default_top_k: int = -1
    
    @abstractmethod
    def create_prompt(
        self, 
        historical_data: np.ndarray, 
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create model prompt"""
        pass
    
    @abstractmethod
    def create_api_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        image_base64: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create API request payload"""
        pass
    
    @abstractmethod
    def extract_response(self, response_json: Dict[str, Any], prompt: str = "") -> str:
        """Extract text from API response"""
        pass
    
    def format_historical_data(self, data: np.ndarray, max_points: int = 4000) -> str:
        """Format historical data as string (can be overridden by subclasses)"""
        if data.ndim == 1:
            context = data[-max_points:]
            return ", ".join([f"{x:.4f}" for x in context])
        else:
            # Multivariate
            if data.shape[0] > data.shape[1]:
                data = data.T
            context = data[:, -max_points:]
            formatted_steps = []
            for t in range(context.shape[1]):
                values = [f"{context[v, t]:.4f}" for v in range(context.shape[0])]
                formatted_steps.append(f"[{', '.join(values)}]")
            return ", ".join(formatted_steps)


# ============================================================================
# Qwen Models
# ============================================================================

class QwenTextConfig(BaseModelConfig):
    """Qwen2.5-Instruct text model configuration"""
    
    model_name = "Qwen2.5-7B-Instruct"
    model_type = "text"
    api_endpoint = "/generate"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen text prompt"""
        is_multivariate = num_variates > 1
        total_values_needed = prediction_length * num_variates
        
        history_str = self.format_historical_data(historical_data)
        
        # Statistics (univariate only)
        if not is_multivariate:
            recent_data = historical_data[-min(100, len(historical_data)):]
            mean_val = np.mean(recent_data)
            std_val = np.std(recent_data)
            trend = "increasing" if recent_data[-1] > recent_data[0] else "decreasing"
            stats_info = f"\nMean: {mean_val:.4f}, Std: {std_val:.4f}, Trend: {trend}"
        else:
            stats_info = ""
        
        # Context info
        domain_info = f"in the {domain} domain" if domain else ""
        freq_readable = get_readable_frequency(freq) if freq else ""
        freq_info = f"with {freq_readable} data" if freq_readable else ""
        context_str = f"{domain_info} {freq_info}".strip()
        
        # Build prompt
        if enable_reasoning:
            system_content = (
                "You are an expert in time series forecasting. "
                "Analyze the data step by step and provide reasoning before predictions."
            )
            
            if is_multivariate:
                user_content = f"""Time series {context_str} with {num_variates} variables: {history_str}

Forecast the next {prediction_length} time steps for all variables.

Format:
Rationale: [Your analysis]
Prediction: [v1_t1, v2_t1, ...], ..., [v1_t{prediction_length}, v2_t{prediction_length}, ...]

Output exactly {total_values_needed} numbers."""
            else:
                user_content = f"""Time series {context_str}: {history_str}{stats_info}

Predict the next {prediction_length} values.

Format:
Rationale: [Your analysis]
Prediction: [val1, val2, ..., val{prediction_length}]

Output exactly {prediction_length} numbers."""
        else:
            system_content = "You are an expert in time series forecasting. Predict future values."
            
            if is_multivariate:
                user_content = f"""Time series {context_str} ({num_variates} variables): {history_str}

Forecast {prediction_length} steps. Output {total_values_needed} numbers:"""
            else:
                user_content = f"""Time series {context_str}: {history_str}

Predict {prediction_length} values. Output {prediction_length} numbers:"""
        
        # Qwen chat template
        return (
            f"<|im_start|>system\n{system_content}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def create_api_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Create vLLM API payload"""
        return {
            "prompt": prompt,
            "n": 1,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stop_token_ids": [],
        }
    
    def extract_response(self, response_json: Dict[str, Any], prompt: str = "") -> str:
        """Extract vLLM response"""
        full_text = response_json["text"][0]
        return full_text[len(prompt):] if prompt else full_text


class QwenVL3BConfig(BaseModelConfig):
    """Qwen2.5-VL-3B-Instruct vision-language model (image only)"""
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"  # Full path matching server model ID
    model_type = "vlm"
    api_endpoint = "/v1/chat/completions"
    include_data_in_prompt = False  # No data points in prompt
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen-VL multimodal prompt (image only)"""
        is_multivariate = num_variates > 1
        total_values_needed = prediction_length * num_variates
        
        domain_info = f"in the {domain} domain" if domain else ""
        freq_readable = get_readable_frequency(freq) if freq else ""
        freq_info = f"with {freq_readable} data" if freq_readable else ""
        context_str = f"{domain_info} {freq_info}".strip()
        
        if enable_reasoning:
            if is_multivariate:
                text = f"""📊 CAREFULLY ANALYZE THE TIME SERIES PLOT IN THE IMAGE ABOVE.

The image shows a multivariate time series {context_str} with {num_variates} variables.

**Task**: Based on the visual patterns, trends, seasonality, and data points you observe in the plot, forecast the next {prediction_length} time steps.

**Output Format**:
Rationale: [Detailed visual analysis of trends, patterns, and relationships between variables]
Prediction: [v1_t1, v2_t1, ...], ..., [v1_t{prediction_length}, v2_t{prediction_length}, ...]

Output exactly {total_values_needed} numbers."""
            else:
                text = f"""📊 CAREFULLY ANALYZE THE TIME SERIES PLOT IN THE IMAGE ABOVE.

The image shows a time series {context_str}.

**Task**: Based on the visual patterns, trends, seasonality, and data points you observe in the plot, predict the next {prediction_length} values.

**Output Format**:
Rationale: [Detailed visual analysis of trends, patterns, and key features]
Prediction: [val1, val2, ..., val{prediction_length}]

Output exactly {prediction_length} numbers."""
        else:
            if is_multivariate:
                text = f"""📊 ANALYZE THE TIME SERIES PLOT IN THE IMAGE ABOVE.

The plot shows a multivariate time series {context_str} with {num_variates} variables.

Based on the visual patterns, trends, and data points you can see in the plot, forecast the next {prediction_length} time steps.

Output exactly {total_values_needed} numbers (as comma-separated values):"""
            else:
                text = f"""📊 ANALYZE THE TIME SERIES PLOT IN THE IMAGE ABOVE.

The plot shows time series data {context_str}.

Based on the visual patterns, trends, and data points you can see in the plot, predict the next {prediction_length} values.

Output exactly {prediction_length} numbers (as comma-separated values):"""
        
        return text
    
    def create_api_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        image_base64: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create Qwen-VL API payload (OpenAI format)"""
        content = []
        
        # Add image
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })
        
        # Add text
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
    
    def extract_response(self, response_json: Dict[str, Any], prompt: str = "") -> str:
        """Extract OpenAI format response"""
        return response_json["choices"][0]["message"]["content"]


class QwenVL7BConfig(QwenVL3BConfig):
    """Qwen2.5-VL-7B-Instruct vision-language model (image only)"""
    
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # 7B model path


class QwenVL3BWithDataConfig(QwenVL3BConfig):
    """Qwen2.5-VL-3B-Instruct vision-language model (image + data)"""
    
    include_data_in_prompt = True  # Include data points in prompt
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen-VL multimodal prompt (image + data)"""
        is_multivariate = num_variates > 1
        total_values_needed = prediction_length * num_variates
        
        domain_info = f"in the {domain} domain" if domain else ""
        freq_readable = get_readable_frequency(freq) if freq else ""
        freq_info = f"with {freq_readable} data" if freq_readable else ""
        context_str = f"{domain_info} {freq_info}".strip()
        
        # Format historical data
        if is_multivariate:
            if historical_data.shape[0] < historical_data.shape[1]:
                data = historical_data
            else:
                data = historical_data.T
            
            data_str_parts = []
            for i in range(min(data.shape[0], num_variates)):
                var_data = data[i][-100:]  # Show max 100 points
                data_str_parts.append(f"Variable {i+1}: [{', '.join(map(str, var_data.tolist()))}]")
            data_str = "\n".join(data_str_parts)
        else:
            recent_data = historical_data[-100:] if len(historical_data) > 100 else historical_data
            data_str = f"[{', '.join(map(str, recent_data.tolist()))}]"
        
        if enable_reasoning:
            if is_multivariate:
                text = f"""📊 ANALYZE BOTH THE TIME SERIES PLOT IN THE IMAGE ABOVE AND THE NUMERICAL DATA BELOW.

The image shows a multivariate time series {context_str} with {num_variates} variables.

**Historical Data**:
{data_str}

**Task**: Combine your visual analysis of the plot with the numerical data to forecast the next {prediction_length} time steps.

**Output Format**:
Rationale: [Combined visual and numerical analysis of trends, patterns, and values]
Prediction: [v1_t1, v2_t1, ...], ..., [v1_t{prediction_length}, v2_t{prediction_length}, ...]

Output exactly {total_values_needed} numbers."""
            else:
                text = f"""📊 ANALYZE BOTH THE TIME SERIES PLOT IN THE IMAGE ABOVE AND THE NUMERICAL DATA BELOW.

The image shows a time series {context_str}.

**Historical Data**: {data_str}

**Task**: Combine your visual analysis of the plot with the numerical data to predict the next {prediction_length} values.

**Output Format**:
Rationale: [Combined visual and numerical analysis of trends, patterns, and values]
Prediction: [val1, val2, ..., val{prediction_length}]

Output exactly {prediction_length} numbers."""
        else:
            if is_multivariate:
                text = f"""📊 ANALYZE BOTH THE TIME SERIES PLOT IN THE IMAGE ABOVE AND THE NUMERICAL DATA BELOW.

The plot shows a multivariate time series {context_str} with {num_variates} variables.

**Historical Data**:
{data_str}

Based on BOTH the visual patterns in the plot AND the numerical data points provided, forecast the next {prediction_length} time steps.

Output exactly {total_values_needed} numbers (as comma-separated values):"""
            else:
                text = f"""📊 ANALYZE BOTH THE TIME SERIES PLOT IN THE IMAGE ABOVE AND THE NUMERICAL DATA BELOW.

The plot shows time series data {context_str}.

**Historical Data**: {data_str}

Based on BOTH the visual patterns in the plot AND the numerical data points provided, predict the next {prediction_length} values.

Output exactly {prediction_length} numbers (as comma-separated values):"""
        
        return text


class QwenVL7BWithDataConfig(QwenVL7BConfig):
    """Qwen2.5-VL-7B-Instruct vision-language model (image + data)"""
    
    include_data_in_prompt = True  # Include data points in prompt
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen-VL multimodal prompt (image + data)"""
        # Reuse 3B logic (same prompt format)
        config_3b = QwenVL3BWithDataConfig()
        return config_3b.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=enable_reasoning,
            domain=domain,
            freq=freq,
            **kwargs
        )


# ============================================================================
# Other Models (Examples)
# ============================================================================

class GPT4VisionConfig(BaseModelConfig):
    """GPT-4 Vision configuration example"""
    
    model_name = "gpt-4-vision-preview"
    model_type = "vlm"
    api_endpoint = "/v1/chat/completions"
    
    def create_prompt(self, historical_data, prediction_length, num_variates=1, **kwargs):
        # Can implement GPT-4 prompt style
        return "Analyze this time series plot and predict the next values..."
    
    def create_api_payload(self, prompt, temperature, max_tokens, top_p, top_k, image_base64=None, **kwargs):
        # GPT-4 API format
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    
    def extract_response(self, response_json, prompt=""):
        return response_json["choices"][0]["message"]["content"]


# ============================================================================
# Qwen3 Series (235B MoE)
# ============================================================================

class Qwen3TextConfig(BaseModelConfig):
    """Qwen3-235B-A22B-Instruct text model"""
    
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    model_type = "text"
    api_endpoint = "/v1/chat/completions"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3 text prompt (reuse Qwen2.5 logic)"""
        config_qwen25 = QwenTextConfig()
        return config_qwen25.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=enable_reasoning,
            domain=domain,
            freq=freq,
            **kwargs
        )
    
    def create_api_payload(self, prompt, temperature, max_tokens, top_p, top_k, image_base64=None, **kwargs):
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
    
    def extract_response(self, response_json, prompt=""):
        return response_json["choices"][0]["message"]["content"]


class Qwen3VLConfig(BaseModelConfig):
    """Qwen3-VL-235B-A22B-Instruct VLM (image only)"""
    
    model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    model_type = "vlm"
    api_endpoint = "/v1/chat/completions"
    include_data_in_prompt = False
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL multimodal prompt (image, reuse 7B logic)"""
        config_qwen25_7b = QwenVL7BConfig()
        return config_qwen25_7b.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=enable_reasoning,
            domain=domain,
            freq=freq,
            **kwargs
        )
    
    def create_api_payload(self, prompt, temperature, max_tokens, top_p, top_k, image_base64=None, **kwargs):
        messages = [{"role": "user", "content": []}]
        
        if image_base64:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })
        
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
    
    def extract_response(self, response_json, prompt=""):
        return response_json["choices"][0]["message"]["content"]


class Qwen3VLWithDataConfig(Qwen3VLConfig):
    """Qwen3-VL-235B-A22B-Instruct VLM (image + data)"""
    
    include_data_in_prompt = True
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL multimodal prompt (image + data)"""
        config_qwen25_7b_data = QwenVL7BWithDataConfig()
        return config_qwen25_7b_data.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=enable_reasoning,
            domain=domain,
            freq=freq,
            **kwargs
        )


class Qwen3ThinkingConfig(Qwen3TextConfig):
    """Qwen3-235B-A22B-Thinking enhanced text model"""
    
    model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-Thinking prompt (force reasoning)"""
        # Thinking models always enable reasoning
        config_qwen25 = QwenTextConfig()
        return config_qwen25.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=True,  # Force enable reasoning
            domain=domain,
            freq=freq,
            **kwargs
        )


class Qwen3VLThinkingConfig(Qwen3VLConfig):
    """Qwen3-VL-235B-A22B-Thinking enhanced VLM (image only)"""
    
    model_name = "Qwen/Qwen3-VL-235B-A22B-Thinking"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL-Thinking prompt (force reasoning)"""
        # Thinking models always enable reasoning
        config_qwen25_7b = QwenVL7BConfig()
        return config_qwen25_7b.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=True,  # Force enable reasoning
            domain=domain,
            freq=freq,
            **kwargs
        )


class Qwen3VLThinkingWithDataConfig(Qwen3VLWithDataConfig):
    """Qwen3-VL-235B-A22B-Thinking enhanced VLM (image + data)"""
    
    model_name = "Qwen/Qwen3-VL-235B-A22B-Thinking"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL-Thinking prompt (image + data, force reasoning)"""
        # Thinking models always enable reasoning
        config_qwen25_7b_data = QwenVL7BWithDataConfig()
        return config_qwen25_7b_data.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=True,  # Force enable reasoning
            domain=domain,
            freq=freq,
            **kwargs
        )


class Qwen3VL8BThinkingConfig(Qwen3VLConfig):
    """Qwen3-VL-8B-Thinking enhanced VLM (image only)"""
    
    model_name = "Qwen/Qwen3-VL-8B-Thinking"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL-8B-Thinking prompt (force reasoning)"""
        # Thinking models always enable reasoning
        config_qwen25_7b = QwenVL7BConfig()
        return config_qwen25_7b.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=True,  # Force enable reasoning
            domain=domain,
            freq=freq,
            **kwargs
        )


class Qwen3VL8BThinkingWithDataConfig(Qwen3VLWithDataConfig):
    """Qwen3-VL-8B-Thinking enhanced VLM (image + data)"""
    
    model_name = "Qwen/Qwen3-VL-8B-Thinking"
    
    def create_prompt(
        self,
        historical_data: np.ndarray,
        prediction_length: int,
        num_variates: int = 1,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        **kwargs
    ) -> str:
        """Create Qwen3-VL-8B-Thinking prompt (image + data, force reasoning)"""
        # Thinking models always enable reasoning
        config_qwen25_7b_data = QwenVL7BWithDataConfig()
        return config_qwen25_7b_data.create_prompt(
            historical_data=historical_data,
            prediction_length=prediction_length,
            num_variates=num_variates,
            enable_reasoning=True,  # Force enable reasoning
            domain=domain,
            freq=freq,
            **kwargs
        )


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Global model registry"""
    
    _registry: Dict[str, BaseModelConfig] = {}
    
    @classmethod
    def register(cls, model_id: str, config_class: BaseModelConfig):
        """Register new model"""
        cls._registry[model_id] = config_class
    
    @classmethod
    def get(cls, model_id: str) -> BaseModelConfig:
        """Get model configuration"""
        if model_id not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {available}"
            )
        return cls._registry[model_id]
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all registered models"""
        return {
            model_id: config.model_name 
            for model_id, config in cls._registry.items()
        }
    
    @classmethod
    def get_by_type(cls, model_type: str) -> Dict[str, BaseModelConfig]:
        """Get models by type"""
        return {
            model_id: config 
            for model_id, config in cls._registry.items()
            if config.model_type == model_type
        }


# ============================================================================
# Pre-registered Models
# ============================================================================

# Qwen2.5 - Text
ModelRegistry.register("qwen2.5-7b", QwenTextConfig())

# Qwen2.5 - VLM (image only)
ModelRegistry.register("qwen2.5-vl-3b", QwenVL3BConfig())
ModelRegistry.register("qwen2.5-vl-7b", QwenVL7BConfig())

# Qwen2.5 - VLM (image + data)
ModelRegistry.register("qwen2.5-vl-3b-with-data", QwenVL3BWithDataConfig())
ModelRegistry.register("qwen2.5-vl-7b-with-data", QwenVL7BWithDataConfig())

# Qwen3 - Text (235B MoE)
ModelRegistry.register("qwen3-235b", Qwen3TextConfig())
ModelRegistry.register("qwen3-235b-thinking", Qwen3ThinkingConfig())

# Qwen3 - VLM image only (235B MoE)
ModelRegistry.register("qwen3-vl-235b", Qwen3VLConfig())
ModelRegistry.register("qwen3-vl-235b-thinking", Qwen3VLThinkingConfig())

# Qwen3 - VLM image + data (235B MoE)
ModelRegistry.register("qwen3-vl-235b-with-data", Qwen3VLWithDataConfig())
ModelRegistry.register("qwen3-vl-235b-thinking-with-data", Qwen3VLThinkingWithDataConfig())

# Qwen3 - VLM (8B Thinking)
ModelRegistry.register("qwen3-vl-8b-thinking", Qwen3VL8BThinkingConfig())
ModelRegistry.register("qwen3-vl-8b-thinking-with-data", Qwen3VL8BThinkingWithDataConfig())

# GPT series (example)
ModelRegistry.register("gpt-4-vision", GPT4VisionConfig())

# Add more models here...


# ============================================================================
# Model ID to Path Mapping (for vLLM server launch)
# ============================================================================

MODEL_PATH_MAPPING = {
    # Qwen2.5 text models
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    
    # Qwen2.5 VLM (3B and 7B map to respective paths)
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-3b-with-data": "Qwen/Qwen2.5-VL-3B-Instruct",  # Same model, different prompt strategy
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b-with-data": "Qwen/Qwen2.5-VL-7B-Instruct",
    
    # Qwen3 text models (235B MoE)
    "qwen3-235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen3-235b-thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    
    # Qwen3 VLM (235B MoE)
    "qwen3-vl-235b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen3-vl-235b-with-data": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen3-vl-235b-thinking": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "qwen3-vl-235b-thinking-with-data": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    
    # Qwen3 VLM (8B Thinking)
    "qwen3-vl-8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3-vl-8b-thinking-with-data": "Qwen/Qwen3-VL-8B-Thinking",
    
    # GPT series (requires OpenAI API, no local path)
    "gpt-4-vision": "gpt-4-vision-preview",
}


def get_model_path(model_id: str) -> str:
    """Get model path for vLLM server by model ID"""
    if model_id not in MODEL_PATH_MAPPING:
        raise ValueError(
            f"Model ID '{model_id}' not found in MODEL_PATH_MAPPING.\n"
            f"Available models: {list(MODEL_PATH_MAPPING.keys())}"
        )
    return MODEL_PATH_MAPPING[model_id]


def get_model_mode(model_id: str) -> str:
    """Get vLLM server mode (llm/vlm) by model ID"""
    config = get_model_config(model_id)
    return "vlm" if config.model_type == "vlm" else "llm"


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_config(model_id: str) -> BaseModelConfig:
    """Get model configuration"""
    return ModelRegistry.get(model_id)


def list_available_models() -> None:
    """Print all available models"""
    print("\n" + "="*80)
    print("📋 Available Models")
    print("="*80)
    
    models = ModelRegistry.list_models()
    text_models = {k: v for k, v in models.items() if ModelRegistry.get(k).model_type == "text"}
    vlm_models = {k: v for k, v in models.items() if ModelRegistry.get(k).model_type == "vlm"}
    
    if text_models:
        print("\n📝 Text-only Models:")
        for model_id, model_name in text_models.items():
            print(f"  - {model_id:20s} → {model_name}")
    
    if vlm_models:
        print("\n🎨 Vision-Language Models:")
        for model_id, model_name in vlm_models.items():
            print(f"  - {model_id:20s} → {model_name}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test model registry
    list_available_models()
    
    # Get model example
    qwen_text = ModelRegistry.get("qwen2.5-7b")
    print(f"Loaded: {qwen_text.model_name}")
    print(f"Type: {qwen_text.model_type}")
    print(f"Endpoint: {qwen_text.api_endpoint}")



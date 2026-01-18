"""
Common Utility Functions

Includes:
- Frequency conversion
- Image generation
- Response parsing
- Process management
"""

import re
import sys
import base64
import subprocess
import signal
from io import BytesIO
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt


# ============================================================================
# Frequency Mapping (based on actual dataset frequencies)
# ============================================================================

FREQ_MAPPING = {
    # Second-level
    "10S": "every 10 seconds",
    "S": "second",
    
    # Hour-level
    "H": "hourly",
    
    # Day-level
    "D": "daily",
    
    # Week-level
    "W": "weekly",
    "W-WED": "weekly (Wednesday)",
    "W-SUN": "weekly (Sunday)",
    "W-MON": "weekly (Monday)",
    "W-TUE": "weekly (Tuesday)",
    "W-THU": "weekly (Thursday)",
    "W-FRI": "weekly (Friday)",
    "W-SAT": "weekly (Saturday)",
    
    # Month-level
    "M": "monthly",
    
    # Quarter-level
    "Q": "quarterly",
    
    # Year-level
    "A": "yearly",
    "Y": "yearly",
}


def get_readable_frequency(freq: str) -> str:
    """
    Convert frequency abbreviation to human-readable description
    
    Args:
        freq: Frequency abbreviation (e.g. "D", "H", "W-WED")
        
    Returns:
        Readable frequency description (e.g. "daily", "hourly", "weekly (Wednesday)")
    
    Examples:
        >>> get_readable_frequency("D")
        'daily'
        >>> get_readable_frequency("W-WED")
        'weekly (Wednesday)'
        >>> get_readable_frequency("10S")
        'every 10 seconds'
    """
    if not freq:
        return ""
    
    # Direct match
    freq_upper = freq.upper()
    if freq_upper in FREQ_MAPPING:
        return FREQ_MAPPING[freq_upper]
    
    # Fallback: return original value
    return freq.lower() + " frequency"


# ============================================================================
# Image Generation Tools
# ============================================================================

def create_time_series_plot(
    historical_data: np.ndarray,
    domain: str = None,
    prediction_length: int = None,
    save_path: str = None
) -> Optional[str]:
    """
    Visualize time series data as image, return base64-encoded PNG
    
    Args:
        historical_data: Historical data (univariate: 1D array, multivariate: 2D array)
        domain: Data domain name (for title)
        prediction_length: Prediction length (for annotation)
        save_path: Optional save path
        
    Returns:
        base64-encoded PNG image string, None on failure
    """
    try:
        # create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Check if univariate or multivariate
        if historical_data.ndim == 1:
            # Univariate time series
            ax.plot(historical_data, linewidth=2, color='#2E86AB', label='Historical Data')
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Time Series - {domain or "Unknown Domain"}', fontsize=14, fontweight='bold')
        else:
            # Multivariate time series
            if historical_data.shape[0] > historical_data.shape[1]:
                data = historical_data.T  # convert to (num_variates, time_steps)
            else:
                data = historical_data
            
            num_variates = data.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, min(num_variates, 10)))
            
            for i in range(num_variates):
                ax.plot(data[i], linewidth=1.5, color=colors[i % 10], 
                       label=f'Variable {i+1}', alpha=0.8)
            
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Multivariate Time Series ({num_variates} variables) - {domain or "Unknown"}', 
                       fontsize=14, fontweight='bold')
            
            if num_variates <= 10:
                ax.legend(loc='best', fontsize=9)
        
        # General settings
        ax.set_xlabel('Time Step', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add prediction area hint
        if prediction_length:
            last_idx = len(historical_data[0] if historical_data.ndim > 1 else historical_data) - 1
            ax.axvline(x=last_idx, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                      label=f'Forecast {prediction_length} steps ahead')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Optional: save image to file
        if save_path:
            fig_save, ax_save = plt.subplots(figsize=(12, 6))
            
            if historical_data.ndim == 1:
                ax_save.plot(historical_data, linewidth=2, color='#2E86AB')
            else:
                data = historical_data.T if historical_data.shape[0] > historical_data.shape[1] else historical_data
                for i in range(data.shape[0]):
                    ax_save.plot(data[i], linewidth=1.5, alpha=0.8)
            
            ax_save.set_xlabel('Time Step')
            ax_save.set_ylabel('Value')
            ax_save.set_title('Time Series')
            ax_save.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig_save)
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None


# ============================================================================
# Response parsing tools
# ============================================================================

def parse_llm_response(
    response_text: str,
    prediction_length: int,
    num_variates: int = 1,
    enable_reasoning: bool = False
) -> np.ndarray:
    """
    Parse LLM response, extract prediction values
    
    Args:
        response_text: LLM response text
        prediction_length: Prediction length
        num_variates: Number of variates
        enable_reasoning: Whether reasoning mode is enabled (will try to extract content after "Prediction:")
        
    Returns:
        Prediction array, shape (prediction_length,) or (prediction_length, num_variates)
    """
    try:
        text = response_text.strip()
        
        if not text:
            if num_variates > 1:
                return np.zeros((prediction_length, num_variates))
            return np.zeros(prediction_length)
        
        # If reasoning enabled, try to extract content after "Prediction:"
        if enable_reasoning and ("Prediction:" in text or "prediction:" in text.lower()):
            parts = text.lower().split("prediction:")
            if len(parts) > 1:
                text = parts[1]
        
        # Remove code block markers
        if "```" in text:
            code_blocks = text.split("```")
            if len(code_blocks) > 1:
                text = code_blocks[1].strip()
        
        # Extract all numbers (including scientific notation)
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        
        total_values_needed = prediction_length * num_variates
        
        # Handle number extraction results
        if len(numbers) >= total_values_needed:
            predictions = [float(x) for x in numbers[:total_values_needed]]
        elif len(numbers) > 0:
            # Not enough numbers, pad with last value
            predictions = [float(x) for x in numbers]
            while len(predictions) < total_values_needed:
                predictions.append(predictions[-1] if predictions else 0.0)
        else:
            # No numbers found, return zero array
            predictions = [0.0] * total_values_needed
        
        predictions = np.array(predictions[:total_values_needed])
        
        # Reshape to multivariate format
        if num_variates > 1:
            predictions = predictions.reshape(prediction_length, num_variates)
        
        return predictions
        
    except Exception as e:
        print(f"Parse error: {e}")
        if num_variates > 1:
            return np.zeros((prediction_length, num_variates))
        return np.zeros(prediction_length)


# ============================================================================
# Process management tools
# ============================================================================

def cleanup_vllm_servers(ports=None):
    """
    Clean up vLLM server processes
    
    Args:
        ports: List of ports to clean up. If None, cleans up ALL vLLM processes.
               If specified, only cleans processes on those ports.
    """
    print("\n" + "="*80)
    if ports:
        print(f"🧹 Cleaning up vLLM servers on ports: {ports}")
    else:
        print("🧹 Cleaning up ALL vLLM servers")
    print("="*80)
    
    try:
        import time
        
        if ports:
            # Smart cleanup: only kill processes on specific ports
            pids_to_kill = []
            for port in ports:
                result = subprocess.run(
                    ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        # Verify it's a Python/vLLM process
                        check = subprocess.run(
                            ["ps", "-p", pid, "-o", "cmd="],
                            capture_output=True,
                            text=True
                        )
                        if "vllm" in check.stdout.lower():
                            pids_to_kill.append(pid)
            
            if pids_to_kill:
                print(f"Terminating vLLM processes: {pids_to_kill}")
                for pid in pids_to_kill:
                    subprocess.run(["kill", pid], capture_output=True)
                
                time.sleep(2)
                
                # Force kill if still alive
                still_alive = []
                for pid in pids_to_kill:
                    check = subprocess.run(
                        ["ps", "-p", pid],
                        capture_output=True
                    )
                    if check.returncode == 0:
                        still_alive.append(pid)
                
                if still_alive:
                    print(f"Force terminating remaining processes: {still_alive}")
                    for pid in still_alive:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                    time.sleep(1)
                
                print("✓ Cleanup completed")
            else:
                print("✓ No vLLM processes found on specified ports")
        else:
            # Global cleanup: kill all vLLM processes
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                print("Terminating ALL vLLM processes...")
                subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
                
                time.sleep(2)
                
                # Check for remaining processes
                result2 = subprocess.run(
                    ["pgrep", "-f", "vllm.entrypoints"],
                    capture_output=True,
                    text=True
                )
                
                if result2.returncode == 0 and result2.stdout.strip():
                    print("Force terminating remaining processes...")
                    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True)
                    time.sleep(1)
                
                print("✓ Cleanup completed")
            else:
                print("✓ No running processes")
            
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")
    
    print("="*80 + "\n")


def setup_signal_handlers(cleanup_callback=None):
    """
    Setup signal handlers for graceful exit
    
    Args:
        cleanup_callback: Optional cleanup callback function
    """
    def signal_handler(signum, frame):
        print(f"\n\n{'='*80}")
        print(f"⚠️  Received signal {signum}, preparing to exit...")
        print("="*80)
        
        if cleanup_callback:
            try:
                cleanup_callback()
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        print("✓ Exit completed")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command


# ============================================================================
# Load balancing tools
# ============================================================================

class PortRotator:
    """Port rotator for load balancing"""
    
    def __init__(self, ports: list):
        """
        Initialize port rotator
        
        Args:
            ports: Port list
        """
        import itertools
        self.ports = ports
        self.port_cycle = itertools.cycle(ports)
        self._lock = None
        
        try:
            import threading
            self._lock = threading.Lock()
        except:
            pass
    
    def get_next_port(self) -> int:
        """Get next port (thread-safe)"""
        if self._lock:
            with self._lock:
                return next(self.port_cycle)
        else:
            return next(self.port_cycle)


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("🧪 Utils Module Test")
    print("="*80 + "\n")
    
    # Test 1: Frequency conversion
    print("1️⃣  Frequency conversion:")
    test_freqs = ["10S", "H", "D", "W-WED", "M", "Q"]
    for freq in test_freqs:
        readable = get_readable_frequency(freq)
        print(f"  {freq:10s} → {readable}")
    
    # Test 2: Image generation
    print("\n2️⃣  Image generation:")
    test_data = np.random.randn(100).cumsum()
    image_b64 = create_time_series_plot(test_data, domain="Test", prediction_length=10)
    if image_b64:
        print(f"  ✓ Generated successfully, base64 length: {len(image_b64)}")
    else:
        print("  ✗ Generation failed")
    
    # Test 3: Response parsing
    print("\n3️⃣  Response parsing:")
    test_responses = [
        "The predictions are: 1.5, 2.3, 3.1, 4.2, 5.0",
        "Rationale: ... Prediction: [10.5, 11.2, 12.0]",
        "```\n1.0 2.0 3.0\n```",
    ]
    for i, resp in enumerate(test_responses):
        parsed = parse_llm_response(resp, prediction_length=5, num_variates=1, enable_reasoning=(i==1))
        print(f"  Test {i+1}: {len(parsed)} values → {parsed[:3]}...")
    
    # Test 4: Port rotation
    print("\n4️⃣  Port rotation:")
    rotator = PortRotator([8000, 8001, 8002])
    ports = [rotator.get_next_port() for _ in range(6)]
    print(f"  Rotation result: {ports}")
    
    print("\n" + "="*80)
    print("✅ All tests completed")
    print("="*80 + "\n")


#!/bin/bash
# Clean up all running vLLM servers (supports both LLM + VLM modes)

echo ""
echo "================================================================================"
echo "🧹 Cleaning up vLLM server processes"
echo "================================================================================"

# Find vLLM processes (both entrypoints)
FOUND=0
if pgrep -f "vllm.entrypoints.api_server" > /dev/null; then FOUND=1; fi
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then FOUND=1; fi

if [ $FOUND -eq 1 ]; then
    echo "found running vLLM processes..."
    echo ""
    
    # Display process info
    echo "Running processes:"
    ps aux | grep "vllm.entrypoints" | grep -v grep
    echo ""
    
    # Graceful termination
    echo "terminatingprocesses (SIGTERM)..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    pkill -f "vllm.entrypoints.api_server" || true
    
    # waiting for processes to exit
    sleep 3
    
    # Check for remaining processes
    REMAIN=0
    if pgrep -f "vllm.entrypoints.api_server" > /dev/null; then REMAIN=1; fi
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then REMAIN=1; fi
    
    if [ $REMAIN -eq 1 ]; then
        echo "⚠️  some processes not responding，force terminating (SIGKILL)..."
        pkill -9 -f "vllm.entrypoints.openai.api_server" || true
        pkill -9 -f "vllm.entrypoints.api_server" || true
        sleep 2
    fi
    
    # Final check
    FINAL=0
    if pgrep -f "vllm.entrypoints.api_server" > /dev/null; then FINAL=1; fi
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then FINAL=1; fi
    
    if [ $FINAL -eq 1 ]; then
        echo "❌ Cleanup failed, some processes still running"
        exit 1
    else
        echo "✅ All vLLM processes cleaned up"
    fi
else
    echo "✓ No running vLLM processes found"
fi

echo "================================================================================"
echo ""

exit 0

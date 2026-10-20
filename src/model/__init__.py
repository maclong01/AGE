from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "2b": "Qwen/Qwen3.5-2B",
    "08b": "Qwen/Qwen3.5-0.8B",
}

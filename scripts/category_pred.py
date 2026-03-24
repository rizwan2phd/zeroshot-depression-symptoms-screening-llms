from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from huggingface_hub import login
import pandas as pd


class TextClassifier:
    """Optimized text classifier supporting multiple latest LLM architectures"""
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        "llama-3": {"style": "instruct", "stop_tokens": ["<|eot_id|>"]},
        "llama-2": {"style": "chat", "stop_tokens": ["</s>"]},
        "mistral": {"style": "instruct", "stop_tokens": ["</s>"]},
        "gemma": {"style": "instruct", "stop_tokens": ["<end_of_turn>"]},
        "qwen": {"style": "instruct", "stop_tokens": ["<|im_end|>"]},
        "phi": {"style": "instruct", "stop_tokens": ["<|end|>"]},
    }
    
    def __init__(self, model_name, device=None):
        """
        Initialize classifier with any HuggingFace model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Explicitly move model to device (avoids device_map issues)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Auto-detect model configuration
        self.config = self._detect_model_config()
    
    def _detect_model_config(self):
        """Auto-detect model type and return appropriate configuration"""
        name_lower = self.model_name.lower()
        
        for key, config in self.MODEL_CONFIGS.items():
            if key in name_lower:
                return config
        
        # Default fallback
        return {"style": "instruct", "stop_tokens": []}
    
    def create_prompt(self, text, categories, domain_instruction=None):
        """
        Create classification prompt optimized for latest LLMs with strict single-category output
        
        Args:
            text: Text to classify
            categories: List of valid categories
            domain_instruction: Domain-specific instruction (e.g., DSM-5 expertise)
        """
        categories_str = ", ".join(categories)
        
        # Build system message with strict constraints
        if domain_instruction:
            system_message = f"""{domain_instruction}

TASK: Classify the following text into EXACTLY ONE category from the list below.

RULES:
1. Output ONLY the category label - no explanations, no punctuation, no additional text
2. Choose the MOST PROMINENT symptom if multiple are present
3. Choose "NONE" if the text shows NO depressive symptoms
4. Your response must be a single word matching one category exactly

CATEGORIES: {categories_str}"""
        else:
            system_message = f"""You are a precise classification system.

TASK: Classify the text into EXACTLY ONE category from the list below.

RULES:
1. Output ONLY the category label - no explanations, no punctuation, no additional text
2. Your response must be a single word matching one category exactly

CATEGORIES: {categories_str}"""
        
        categories_list = "\n".join([f"- {cat}" for cat in categories])
        style = self.config["style"]
        
        # Llama-3+ format
        if "llama-3" in self.model_name.lower():
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

Text to classify: "{text}"

Category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Llama-2 chat format
        elif "llama-2" in self.model_name.lower() and "chat" in self.model_name.lower():
            return f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

Text to classify: "{text}"

Category: [/INST]"""
        
        # Mistral/Mixtral format
        elif "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
            return f"""<s>[INST] {system_message}

Text to classify: "{text}"

Category: [/INST]"""
        
        # Gemma format
        elif "gemma" in self.model_name.lower():
            return f"""<bos><start_of_turn>user
{system_message}

Text to classify: "{text}"

Category:<end_of_turn>
<start_of_turn>model
"""
        
        # Qwen format
        elif "qwen" in self.model_name.lower():
            return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
Text to classify: "{text}"

Category:<|im_end|>
<|im_start|>assistant
"""
        
        # Phi format
        elif "phi" in self.model_name.lower():
            return f"""<|system|>
{system_message}<|end|>
<|user|>
Text to classify: "{text}"

Category:<|end|>
<|assistant|>
"""
        
        # Generic instruct format (fallback)
        else:
            return f"""### System:
{system_message}

### User:
Text to classify: "{text}"

Category:

### Assistant:
"""
    
    def get_short_model_name(self):
        """Extract short model name for column naming"""
        # Simply take everything after the last '/'
        return self.model_name.split('/')[-1]
    
    def classify(self, text, categories, domain_instruction=None, max_new_tokens=15, temperature=0.0):
        """
        Classify text into one of the given categories (zero-shot)
        
        Args:
            text: Text to classify
            categories: List of valid category labels
            domain_instruction: Optional domain expertise instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic zero-shot)
            
        Returns:
            Valid category string or None if no valid category found
        """
        prompt = self.create_prompt(text, categories, domain_instruction)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=None if temperature == 0.0 else temperature,
                do_sample=False if temperature == 0.0 else True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return self._extract_valid_category(generated, categories)
    
    def _extract_valid_category(self, response, categories):
        """
        Extract valid category from response or return None
        
        Args:
            response: Model output text
            categories: List of valid categories
            
        Returns:
            Valid category string or None
        """
        if not response:
            return None
        
        # Clean the response - remove common artifacts
        response_clean = response.strip().upper()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["CATEGORY:", "ANSWER:", "LABEL:", "CLASSIFICATION:"]
        for prefix in prefixes_to_remove:
            if response_clean.startswith(prefix):
                response_clean = response_clean[len(prefix):].strip()
        
        # Remove punctuation at the end
        response_clean = response_clean.rstrip('.,;:!?')
        
        # Try exact match (case-insensitive)
        for cat in categories:
            if cat.upper() == response_clean:
                return cat
        
        # Try partial match in first 30 characters (stricter than before)
        response_head = response_clean[:30]
        for cat in categories:
            if cat.upper() in response_head:
                return cat
        
        # Try word boundary match - first word only
        words = response_clean.split()
        if words:
            first_word = words[0]
            for cat in categories:
                if cat.upper() == first_word:
                    return cat
        
        # Try checking if any category is a substring of first word
        if words:
            first_word = words[0]
            for cat in categories:
                if cat.upper() in first_word or first_word in cat.upper():
                    return cat
        
        # No valid category found
        return None


if __name__ == "__main__":
    # DSM-5 Depression Categories (including NONE for no symptoms)
    categories = [
        "DEPRESSED_MOOD",
        "WORTHLESSNESS",
        "ANHEDONIA",
        "SUICIDAL_THOUGHTS",
        "APPETITE_CHANGE",
        "SLEEP_ISSUES",
        "FATIGUE",
        "COGNITIVE_ISSUES",
        "PSYCHOMOTOR",
        "NONE"  # For texts with no depressive symptoms
    ]
    
    # Enhanced domain-specific instruction with clear guidance for NONE
    domain_instruction = (
        "You are an expert clinical psychologist trained in DSM-5 diagnostic criteria for Major Depressive Disorder. "
        "Analyze the text carefully for indicators of depressive symptoms. "
        "If the text describes a depressive symptom, classify it into the one category. "
        "If the text shows NO depressive symptoms (e.g., normal mood, neutral content, unrelated topics), classify it as NONE."
    )

    # Load dataset
    dsm5_data = pd.read_csv("dsm5_control_all.csv")
 
    # Model selection - supports latest models
    model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    # Other supported models:
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"

   

    # HuggingFace login
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Initializing classifier with: {model_name}")
    classifier = TextClassifier(model_name)
    
    # Auto-generate column name from model
    col_name_model = classifier.get_short_model_name()
    print(f"Column name: {col_name_model}")
    
    # Create column for predictions
    dsm5_data[col_name_model] = None
    
    print(f"Using prompt style: {classifier.config['style']}\n")
    print("Starting zero-shot classification...\n")

    # Classification loop
    valid_predictions = 0
    invalid_predictions = 0
    
    for idx, text in dsm5_data["sentence_text"].items():
        if pd.isna(text):
            continue

        predicted_category = classifier.classify(
            text,
            categories,
            domain_instruction=domain_instruction,
            max_new_tokens=15,  # Increased slightly to handle longer category names
            temperature=0.0  # Zero-shot deterministic prediction
        )

        dsm5_data.at[idx, col_name_model] = predicted_category
        
        if predicted_category is None:
            invalid_predictions += 1
        else:
            valid_predictions += 1

        # Progress logging
        if idx % 50 == 0:
            print(f"Processed: {idx} | Valid: {valid_predictions} | Invalid: {invalid_predictions} | Last: {predicted_category}")

    # Save results
    output_path = "dsm5_control_all.csv"
    dsm5_data.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Classification complete!")
    print(f"Total processed: {valid_predictions + invalid_predictions}")
    print(f"Valid predictions: {valid_predictions}")
    print(f"Invalid predictions (None): {invalid_predictions}")
    print(f"Accuracy rate: {valid_predictions/(valid_predictions + invalid_predictions)*100:.2f}%")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")
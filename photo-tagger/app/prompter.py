import os
import time
import json
from openai import OpenAI

BASE_URL = {
    "togetherai": "https://api.together.xyz/v1",
    "openai": None
}

def auth():
    from metaflow.metaflow_config_funcs import init_config
    conf = init_config()
    if conf:
        headers = {'x-api-key': conf['METAFLOW_SERVICE_AUTH_KEY']}
    else:
        headers = json.loads(os.environ['METAFLOW_SERVICE_HEADERS'])
    return headers

class Prompter():

    def __init__(self, provider, model, obhost=None):
        if provider == 'outerbounds':
            base_url = os.path.join(obhost, 'v1')
            key = 'none'
            self.extra_headers = auth()
        elif provider == 'finetuned':
            self._load_finetuned_model(model)
            return
        else:
            base_url = BASE_URL[provider]
            key = os.environ[f"{provider}_api"]
            self.extra_headers = {}
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=key)
    
    def _load_finetuned_model(self, model_id):
        import torch
        import tempfile
        import json
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        from metaflow import load_model

        # Create a temporary directory for model files (don't auto-delete)
        temp_dir = tempfile.mkdtemp()
        self._temp_model_dir = temp_dir  # Keep reference for cleanup
        
        print(f"🔍 DEBUG: Loading model {model_id} to {temp_dir}")
        
        # Use load_model with the model object and output directory
        load_model(model_id, temp_dir)
        model_dir = temp_dir
        
        # Debug: Check what files were actually downloaded
        print(f"🔍 DEBUG: Contents of {model_dir}:")
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Read adapter config to get base model info
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"🔍 DEBUG: Found adapter_config.json")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            print(f"🔍 DEBUG: Adapter config: {adapter_config}")
            base_model = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-VL-3B-Instruct')
            print(f"🔍 DEBUG: Using base model: {base_model}")
        else:
            print('🚨 DEBUG: Adapter config not found, falling back to default base model')
            base_model = 'Qwen/Qwen2.5-VL-3B-Instruct'
        
        # Load the processor from the base model to ensure compatibility
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # Quantize the base model to 4-bit (NF4) to fit in <8 GB GPU memory
        print(f"🔍 DEBUG: Loading base model {base_model} with 4-bit quantization")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            low_cpu_mem_usage=True,
        )
        print(f"🔍 DEBUG: Base model loaded & quantized successfully")
        
        # Attach the fine-tuned LoRA adapter from the downloaded path
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            print(f"🔍 DEBUG: Loading LoRA adapter from {model_dir}")
            self.local_model = PeftModel.from_pretrained(model, model_dir)
            print(f"🔍 DEBUG: LoRA adapter loaded successfully")
            self.model = f"finetuned_model"
        else:
            print(f"🚨 DEBUG: No LoRA adapter found, using base model only")
            self.local_model = model
            self.model = f"base_model_fallback"

    def _format_messages(self, prompt, image_url=None, encoded_image=None):
        if image_url is not None:
            url = {'url': image_url}
        else:
            url = {'url': f'data:image/jpeg;base64,{encoded_image}'}
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": url},
                ],
            }
        ]

    def prompt(self, prompt, image_url=None, encoded_image=None):
        if hasattr(self, 'local_model'):
            return self._prompt_local(prompt, image_url, encoded_image)
        
        t = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._format_messages(prompt,
                                           image_url=image_url,
                                           encoded_image=encoded_image),
            extra_headers=self.extra_headers
        )
        latency = int((time.time() - t) * 1000)
        resp = response.choices[0].message.content
        try:
            tags = [t.lower() for t in json.loads(resp)["tags"]]
            validity = "✅"
        except:
            tags = []
            validity = "❌"
        return resp, tags, validity, latency
    
    def _prompt_local(self, prompt, image_url=None, encoded_image=None):
        import torch
        from PIL import Image
        import requests
        import io
        import base64
        
        t = time.time()
        
        # Load image for local processing
        if image_url is not None:
            # Download image from URL
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        elif encoded_image is not None:
            # Decode base64 image
            image_data = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            raise ValueError("Either image_url or encoded_image must be provided")
        
        # Use the correct processor for multimodal input
        messages = self._format_messages(prompt, image_url=image_url, encoded_image=encoded_image)
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=image,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            
        with torch.no_grad():
            gen_out = self.local_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False,
            )

        # Remove the prompt tokens so we only keep newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = gen_out.sequences[0][prompt_len:]
        resp = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        latency = int((time.time() - t) * 1000)
        
        try:
            clean_resp = resp.strip()
            # Find the first JSON object in the generated text
            start = clean_resp.find('{')
            end = clean_resp.find('}', start) + 1
            if start != -1 and end > start:
                candidate = clean_resp[start:end]
                tags = [t.lower() for t in json.loads(candidate)["tags"]]
                validity = "✅"
            else:
                tags = []
                validity = "❌"
        except (json.JSONDecodeError, IndexError, ValueError):
            tags = []
            validity = "❌"
        
        return resp, tags, validity, latency

if __name__ == '__main__':
    prompter = Prompter('outerbounds', 'Qwen/Qwen2.5-VL-3B-Instruct', obhost=open('obhost.txt').read().strip())
    prompt = open('prompt.txt').read()
    image_url = 'https://images.unsplash.com/photo-1742832599361-7aa7decd73b4?q=80&w=1587&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
    print('\n'.join(map(str, prompter.prompt(prompt, image_url=image_url))))


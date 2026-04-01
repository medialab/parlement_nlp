import argparse
import os
os.environ["VLLM_USE_FLEX_ATTENTION"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Use your API key from environment variable for security
os.environ["OPENAI_API_KEY"] = 'Add your OpenAI API key here'

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig
from vllm import LLM, SamplingParams

# ---------------------------
# Load Prompt
# ---------------------------
def load_prompt(prompt_file):
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------
# Truncate context prompts
# ---------------------------

LIMITS = {
    "speech": 500,
    "amendment_summary": 300,
    "speaker_name": 100,
    "speaker_group": 100,
    "amendment_content": 300,
}

def truncate(text, key):
    limit = LIMITS.get(key, 150)
    if not isinstance(text, str):
        text = ""
    if len(str(text)) <= limit :
        return text
    return str(text)[:limit]


def truncate_middle(text, key):
    limit = LIMITS.get(key, 150)
    text = text if isinstance(text, str) else ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "..." + text[-half:]


# ---------------------------
# Inference Function
# ---------------------------
def classify_batch(prompts, llm, sampling_params):
    max_len = llm.llm_engine.model_config.max_model_len

    # Truncate each prompt if it's too long
    truncated_prompts = []
    for p in prompts:
        if len(p) > max_len:
            print(f"/!\\ Truncating long prompt, length {len(p)}")
            truncated_prompts.append(p[:max_len])
        else:
            truncated_prompts.append(p)

    outputs = llm.generate(truncated_prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]

def classify_batch_openai(client, prompts, model_name, temperature=0.0, max_tokens=5):
    responses = []
    for prompt in prompts:
        # Truncate long prompts if needed (OpenAI models support up to ~128k for gpt-4o)
        if len(prompt) > 120000:
            print(f"/!\\ Truncating long prompt, length {len(prompt)}")
            prompt = prompt[:120000]

        # Create chat completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful text classification assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        responses.append(response.choices[0].message.content.strip())
    return responses


# ---------------------------
# Main Eval Loop
# ---------------------------
def run_inference(df, model_name, llm, sampling_params, prompt_file, batch_size=8):
    results = []
    client = None
    if "openai" in model_name or "gpt" in model_name:
        client = OpenAI()

    PROMPT_TEMPLATE = load_prompt(prompt_file)
    print(f"Using PROMPT_TEMPLATE: {PROMPT_TEMPLATE}\n")
    if "mistral" in model_name.lower():
        PROMPT_TEMPLATE =  "<s>[INST] " + PROMPT_TEMPLATE + " [/INST]</s>"
    if "qwen" in model_name.lower():
         PROMPT_TEMPLATE =  "[INST] " + PROMPT_TEMPLATE + " [/INST]"
    for i in tqdm(range(0, len(df), batch_size), desc="Running inference"):
        batch_speech = df["speech"].iloc[i:i+batch_size].tolist()
        if "context" in prompt_file:
            batch_amendment = df["amendment_summary"].iloc[i:i+batch_size].tolist()
            #batch_amendment_content = df["amendment_content"].iloc[i:i+batch_size].tolist() // amendment_content=truncate(ac, "amendment_content")
            batch_speaker_name = df["speaker_name"].iloc[i:i+batch_size].tolist()
            batch_speaker_group = df["speaker_group"].iloc[i:i+batch_size].tolist()

            batch_prompts = [
                PROMPT_TEMPLATE.format(speech=truncate(s, "speech"), speaker_name=truncate(sn, "speaker_name"), speaker_group=truncate(sg, "speaker_group"), amendment_summary=truncate(a, "amendment_summary")) 
                for s, sn, sg, a, ac in zip(batch_speech, batch_speaker_name, batch_speaker_group, batch_amendment, batch_amendment_content)
            ]
        elif "amend" in prompt_file:
            batch_amendment = df["amendment_summary"].iloc[i:i+batch_size].tolist()
            batch_prompts = [
                PROMPT_TEMPLATE.format(speech=truncate_middle(s, "speech"), amendment_summary=truncate_middle(a, "amendment_summary"))
                for s, a in zip(batch_speech, batch_amendment)
            ]
        else:
            batch_prompts = [PROMPT_TEMPLATE.format(speech=truncate_middle(s, "speech")) for s in batch_speech]
        print(f"BATCH TEXTS: {batch_speech}")
        preds = []
        if "openai" in model_name or "gpt" in model_name:
            preds = classify_batch_openai(client, batch_prompts, model_name)
        else:
            preds = classify_batch(batch_prompts, llm, sampling_params)
        print(f"PREDS: {preds}")
        results.extend(preds)
    return results

def normalize_label(pred: str, prompt_file: str) -> int:
    """Convert raw model output to {1,0,-1}."""
    if not isinstance(pred, str):
        return -1

    p = pred.strip().lower()  # normalize once

    if "_pour_contre" in prompt_file:  # Yes/No prompts
        if "pour" in p and "contre" in p:
            return -1
        if "pour" in p:
            return 1
        elif "contre" in p:
            return 0
        else:
            return -1  

    else:  # 0/1 prompts
        if p.startswith("1"):
            return 1
        elif p.startswith("0"):
            return 0
        else:
            return -1


# ---------------------------
# Call function
# ---------------------------
def run_llm(args):
    print(f"Opening data path: {args.data_path}")
    model_name = args.model_name
    print(f"Model name is: {model_name}")
   
    df = pd.read_csv(args.data_path, low_memory=False)
    #df = pd.read_csv(args.data_path, sep=sep, low_memory=False, quotechar='"')
  
    if "speech" not in df.columns:
        raise ValueError("CSV must contain 'speech' column.")
    df["speech"] = df["speech"].fillna("").astype(str)
    predictions = []

    if "gpt" in model_name or "openai" in model_name:
        predictions = run_inference(
            df,
            model_name=args.model_name,  # e.g., "gpt-4o-mini"
            llm=None, 
            sampling_params=None,
            prompt_file=args.prompt_file,
            batch_size=args.batch_size,
        )
    else :

        # vLLM sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic outputs
            max_tokens=5,
        )

        dtype = "auto"
        print(f"Loading {model_name}, dtype={dtype}")

        gpu_mem_use = 0.75
        if "llama" in model_name.lower():
            gpu_mem_use = 0.75
        if "qwen" in model_name.lower():
            gpu_mem_use = 0.85

        llm = LLM(
            model=model_name,
            tokenizer_mode='auto',
            dtype='float16',
            max_model_len=4096,
            tensor_parallel_size=args.num_gpu, 
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_use,  # (optional) squeeze more memory
        )

        predictions = run_inference(df, args.model_name, llm, sampling_params, args.prompt_file, args.batch_size)

    df["predicted_label"] = [normalize_label(p, args.prompt_file) for p in predictions]
    df["llm_output"] = predictions

    print(df)
    # Save if needed
    print(f"Writing to output path: {args.output_path}")
    df.to_csv(args.output_path, index=False)




def parse_args(parser):
    # Data args
    parser.add_argument("--data_path", type=str, default="data/error_analysis_testset_fake.csv") #"data/interro1_Manon_tous.csv")
    parser.add_argument('--size', type=str, default='large', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')

    # Model args
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help='the model to use') 
    # mistralai/Mistral-7B-Instruct-v0.2
    # meta-llama/Llama-3.2-3B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
    # openai/gpt-oss-20b
    # gpt-4o-mini 

    # Hyper params
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpu", type=int, default=1)

    # Inference argument
    parser.add_argument('--inference-only', action='store_true', help='Run inference only and write predictions to CSV')
    parser.add_argument('--output_path', type=str, default="error_analysis_testset_qwen.csv", help='Path to save inference results (CSV with predicted scores and labels)')
    parser.add_argument("--prompt_file", type=str, default="models/prompts/prompt_llama.txt", help="Filename of the prompt template in prompts/")
    return parser.parse_args()

# ---------------------------
# Main call
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for point d'arrêt detection")
    args = parse_args(parser)
    run_llm(args)

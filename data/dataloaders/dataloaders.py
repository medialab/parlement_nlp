from unicodedata import name

from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader, DistributedSampler


DATA_SPLIT_SEED = 42
# Define context segment limits
LIMITS = {
    "speech": 500,
    "amendment_summary": 300,
    "speaker_name": 100,
    "speaker_group": 100,
    "amendment_content": 300,
}

def truncate_text(text, limit):
    """Truncate text by character length safely."""
    if not text or not isinstance(text, str):
        return ""
    return text.strip()[:limit]


def get_dataloads(
    train_data_path=None, test_data_path=None, size="small", validation=False, test_size=0.2,
    tokenizer_name="camembert-base", batch_size=8, seed=DATA_SPLIT_SEED,
    distributed=False, rank=None, world_size=None,
    model_name="text_only",  # options: text_only, post_text_concat, post_text_embed
    inference_only=False, tokenize=True
):
    
    print(f"Loading train dataset from {train_data_path}, and test dataset from {test_data_path}, size is {size}, validation is {validation}...")
    
    train_df = pd.read_csv(train_data_path, low_memory=False) if train_data_path else None
    test_df = pd.read_csv(test_data_path, low_memory=False) if test_data_path else None
    
    if train_df is not None and "speech" not in train_df.columns:
        raise ValueError(f"CSV train {train_data_path} must contain 'speech' columns.")
    if train_df is not None:
        train_df["speech"] = train_df["speech"].fillna("").astype(str)
    if test_df is not None and "speech" not in test_df.columns:
        raise ValueError(f"CSV test {test_data_path} must contain 'speech' columns.")
    if test_df is not None:
        test_df["speech"] = test_df["speech"].fillna("").astype(str)
    
    if inference_only: #inference_only:
        if test_df is None:
            raise ValueError("Test data path must be provided for inference mode.")
        test_df['labels'] = -1
    else: # eval mode should have labels, i.e. 'label' column 
        if test_df is None:
            raise ValueError("Test data path must be provided for train/eval mode.")  
        if train_df is None:
            raise ValueError("Train data path must be provided for train/eval mode.")
        if "label" not in test_df.columns:
            raise ValueError(f"CSV test {test_data_path} must contain 'label' columns.")
        test_df['labels'] = test_df['label'].apply(lambda x: 0 if x == 'CONTRE' else 1)
        if "label" not in train_df.columns:
            raise ValueError(f"CSV train {train_data_path} must contain 'label' columns.")
        train_df['labels'] = train_df['label'].apply(lambda x: 0 if x == 'CONTRE' else 1)

    
    if "_all" in model_name:
        if "amendment_content" not in test_df.columns or "amendment_content" not in train_df.columns:
            raise ValueError("CSV must contain 'amendment_content' column for this model_name.")
        test_df["amendment_content"] = test_df["amendment_content"].fillna("").astype(str)
        train_df["amendment_content"] = train_df["amendment_content"].fillna("").astype(str)
        if "amendment_summary" not in test_df.columns or "amendment_summary" not in train_df.columns:
            raise ValueError("CSV must contain 'amendment_summary' column for this model_name.")
        test_df["amendment_summary"] = test_df["amendment_summary"].fillna("").astype(str)
        train_df["amendment_summary"] = train_df["amendment_summary"].fillna("").astype(str)
        if "speaker_group" not in test_df.columns or "speaker_group" not in train_df.columns:
            raise ValueError("CSV must contain 'speaker_group' column for this model_name.")
        test_df["speaker_group"] = test_df["speaker_group"].fillna("").astype(str)
        train_df["speaker_group"] = train_df["speaker_group"].fillna("").astype(str)
        if "speaker_name" not in test_df.columns or "speaker_name" not in train_df.columns:
            raise ValueError("CSV must contain 'speaker_name' column for this model_name.")
        test_df["speaker_name"] = test_df["speaker_name"].fillna("").astype(str)
        train_df["speaker_name"] = train_df["speaker_name"].fillna("").astype(str)
        if "amendment_author_name" not in test_df.columns or "amendment_author_name" not in train_df.columns:
            raise ValueError("CSV must contain 'amendment_author_name' column for this model_name.")
        test_df["amendment_author_name"] = test_df["amendment_author_name"].fillna("").astype(str)
        train_df["amendment_author_name"] = train_df["amendment_author_name"].fillna("").astype(str)
        if "amendment_author_group" not in test_df.columns or "amendment_author_group" not in train_df.columns:
            raise ValueError("CSV must contain 'amendment_author_group' column for this model_name.")
        test_df["amendment_author_group"] = test_df["amendment_author_group"].fillna("").astype(str)
        train_df["amendment_author_group"] = train_df["amendment_author_group"].fillna("").astype(str)
    if "amendment" in model_name: #OK
        if "amendment_summary" not in test_df.columns or "amendment_summary" not in train_df.columns:
            raise ValueError("CSV must contain 'amendment_summary' column for this model_name.")
        test_df["amendment_summary"] = test_df["amendment_summary"].fillna("").astype(str)
        train_df["amendment_summary"] = train_df["amendment_summary"].fillna("").astype(str)

    # create the index column, start with 0
    train_df["index"] = range(len(train_df))
    test_df["index"] = range(len(test_df))
    
    if size == "small":
        train_df = train_df.sample(n=100, random_state=seed)
    elif size == "medium":
        train_df = train_df.sample(n=1000, random_state=seed)
    elif size == "large":
        pass
    else:
        raise ValueError("Size must be one of: small, medium, large")
    
    if inference_only and test_size == 1.0:
        test_df = test_df.copy()
        train_df = pd.DataFrame(columns=test_df.columns)
        val_df = pd.DataFrame(columns=test_df.columns)
    else:
        if validation:
            train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=seed)

    train_dataset, test_dataset, val_dataset = None, None, None
    class_counts = []
    tokenizer = None
    if tokenize and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_text_only(batch):
            tokens = tokenizer(batch["speech"], truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens

        def tokenize_amendment_concat(batch):
            sep_token = tokenizer.sep_token or "[SEP]"
            concat_text = [amendment + f" {sep_token} " + speech for speech, amendment in zip(batch["speech"], batch["amendment_summary"])]
            tokens = tokenizer(concat_text, truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens
        
        def tokenize_speaker_concat(batch):
            sep_token = tokenizer.sep_token or "[SEP]"
            
            concat_text = [str(name) + f" {sep_token} " + str(group) + f" {sep_token} " + str(speech) for name, group, speech in zip(batch["speaker_name"], batch["speaker_group"], batch["speech"])]
            tokens = tokenizer(concat_text, truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens
 
        
        def tokenize_all_concat(batch): # for ContextConcat
            sep = tokenizer.sep_token or "[SEP]"
            concat_texts = []

            for speech, s_name, s_group, amendment_sum, amendment_cont, a_name, a_group in zip(
                batch["speech"],
                batch["speaker_name"],
                batch["speaker_group"],
                batch["amendment_summary"],
                batch["amendment_content"],
                batch["amendment_author_name"],
                batch["amendment_author_group"],
            ):
                # Truncate fields
                speech = truncate_text(speech, LIMITS["speech"])
                amendment_sum = truncate_text(amendment_sum, LIMITS["amendment_summary"])
                amendment_cont = truncate_text(amendment_cont, LIMITS["amendment_content"])
                a_name = truncate_text(a_name, LIMITS["speaker_name"])
                a_group = truncate_text(a_group, LIMITS["speaker_group"])
                s_name = truncate_text(s_name, LIMITS["speaker_name"])
                s_group = truncate_text(s_group, LIMITS["speaker_group"])

                # Build sections dynamically (only if not empty)
                sections = [speech]
                if amendment_sum:
                    sections.append(f"[AMENDMENT_SUMMARY] {amendment_sum}")
                if amendment_cont:
                    sections.append(f"[AMENDMENT_CONTENT] {amendment_cont}")
                if a_name:
                    sections.append(f"[AMENDMENT_AUTHOR_NAME] {a_name}")
                if a_group:
                    sections.append(f"[AMENDMENT_AUTHOR_GROUP] {a_group}")
                if s_name:
                    sections.append(f"[SPEAKER_NAME] {s_name}")
                if s_group:
                    sections.append(f"[SPEAKER_GROUP] {s_group}")

                # Join all with SEP token
                concat = f" {sep} ".join(sections)
                concat_texts.append(concat.strip())
            print(concat_texts)

            # Tokenize everything with truncation and padding
            tokens = tokenizer(
                concat_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )

            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]

            return tokens
 
        def tokenize_dual(batch): # for ContextEmbed
            output = {}
            sep_token = tokenizer.sep_token or "[SEP]"
            ctx_text = batch["speech"]
            if 'amendment_summary' in batch.keys() and 'amendment_content' in batch.keys() and 'speaker_name' in batch.keys() and 'speaker_group' in batch.keys() and 'amendment_author_name' in batch.keys() and 'amendment_author_group' in batch.keys():
                concat_texts = []
                for amendment_sum, amendment_cont, s_name, s_group, a_name, a_group in zip(
                    batch["amendment_summary"],
                    batch["amendment_content"],
                    batch["speaker_name"],
                    batch["speaker_group"],
                    batch["amendment_author_name"],
                    batch["amendment_author_group"],
                ):
                    # Truncate fields
                    amendment_sum = truncate_text(amendment_sum, LIMITS["amendment_summary"])
                    amendment_cont = truncate_text(amendment_cont, LIMITS["amendment_content"])
                    s_name = truncate_text(s_name, LIMITS["speaker_name"])
                    s_group = truncate_text(s_group, LIMITS["speaker_group"])
                    a_name = truncate_text(a_name, LIMITS["speaker_name"])
                    a_group = truncate_text(a_group, LIMITS["speaker_group"])

                    # Build sections dynamically (only if not empty)
                    sections = []
                    if amendment_sum:
                        sections.append(f"[AMENDMENT_SUMMARY] {amendment_sum}")
                    if amendment_cont:
                        sections.append(f"[AMENDMENT_CONTENT] {amendment_cont}")
                    if s_name:
                        sections.append(f"[SPEAKER_NAME] {s_name}")
                    if s_group:
                        sections.append(f"[SPEAKER_GROUP] {s_group}")
                    if a_name:
                        sections.append(f"[AMENDMENT_AUTHOR_NAME] {a_name}")
                    if a_group:
                        sections.append(f"[AMENDMENT_AUTHOR_GROUP] {a_group}")

                    # Join all with SEP token
                    concat = f" {sep_token} ".join(sections)
                    concat_texts.append(concat.strip())
                ctx_text = concat_texts # will do the code after for all 
            elif 'speaker_name' in batch.keys() and 'speaker_group' in batch.keys() and 'amendment_summary' in batch.keys():
                ctx_text = [f"[SPEAKER_NAME] {name} {sep_token} [SPEAKER_GROUP] {group} {sep_token} [AMENDMENT_SUMMARY] {summary}" for name, group, summary in zip(batch["speaker_name"], batch["speaker_group"], batch["amendment_summary"])]

            elif 'amendment_summary' in batch.keys():
                ctx_text = batch['amendment_summary']
            elif 'speaker_name' in batch.keys() and 'speaker_group' in batch.keys():
                ctx_text = [f"[SPEAKER_NAME] {name} {sep_token} [SPEAKER_GROUP] {group}" for name, group in zip(batch["speaker_name"], batch["speaker_group"])]

            
            text_tok = tokenizer(batch["speech"], truncation=True, padding="max_length")
            ctx_tok = tokenizer(ctx_text, truncation=True, padding="max_length")
            for k in text_tok:
                output[f"text_{k}"] = text_tok[k]
            for k in ctx_tok:
                output[f"ctx_{k}"] = ctx_tok[k]
            output["labels"] = batch["labels"]
            output["index"] = batch["index"]
            print(output.keys())
            return output

        def tokenize_four(batch):
            output = {}
            sep_token = tokenizer.sep_token or "[SEP]"

            # Tokenize text column
            text_tok = tokenizer(batch["speech"], truncation=True, padding="max_length")

            # Concatenate speaker and amendment, and amendment speaker strings element-wise
            speaker_strings = [f"[SPEAKER_NAME] {name} {sep_token} [SPEAKER_GROUP] {group}" for name, group in zip(batch["speaker_name"], batch["speaker_group"])]
            amendment_strings = [f"[AMENDMENT_SUMMARY] {summary}" for summary in batch["amendment_summary"]]
            amendment_author_strings = [f"[AMENDMENT_AUTHOR_NAME] {a_name} {sep_token} [AMENDMENT_AUTHOR_GROUP] {a_group}" for a_name, a_group in zip(batch["amendment_author_name"], batch["amendment_author_group"])]

            # Concatenate url strings element-wise
            speaker_tok = tokenizer(speaker_strings, truncation=True, padding="max_length")
            amendment_tok = tokenizer(amendment_strings, truncation=True, padding="max_length")
            amendment_author_tok = tokenizer(amendment_author_strings, truncation=True, padding="max_length")

            # Add tokenized outputs with prefixes
            for k in text_tok:
                output[f"text_{k}"] = text_tok[k]
            for k in speaker_tok:
                output[f"speaker_{k}"] = speaker_tok[k]
            for k in amendment_tok:
                output[f"amendment_{k}"] = amendment_tok[k]
            for k in amendment_author_tok:
                output[f"amendment_author_{k}"] = amendment_author_tok[k]

            # Add labels and index
            output["labels"] = batch["labels"]
            output["index"] = batch["index"]

            return output


        if model_name == "text_only":
            cols = ['speech', 'labels', 'index']
            tok_func = tokenize_text_only
        elif model_name == "amendment_text_concat":
            cols = ['speech', 'amendment_summary', 'labels', 'index']
            tok_func = tokenize_amendment_concat
        elif model_name == "speaker_text_concat":
            cols = ['speech', 'speaker_name', 'speaker_group', 'labels', 'index']
            tok_func = tokenize_speaker_concat
       
        # all context models
        elif model_name == "context_all_text_embed":
            cols = ['speech', 'amendment_summary', 'amendment_content', 'speaker_name', 'speaker_group', 'amendment_author_name', 'amendment_author_group' 'labels', 'index']
            tok_func = tokenize_dual
        elif model_name == "context_all_text_concat":
            cols = ['speech', 'amendment_summary', 'amendment_content', 'speaker_name', 'speaker_group', 'amendment_author_name', 'amendment_author_group', 'labels', 'index']
            tok_func = tokenize_all_concat
        elif model_name == "context_all_embed":
            cols = ['speech', 'amendment_summary', 'amendment_content', 'speaker_name', 'speaker_group', 'amendment_author_name', 'amendment_author_group', 'labels', 'index']
            tok_func = tokenize_four
        else :
            cols = ['speech', 'labels', 'index']

    # Convert to HuggingFace Datasets
    if not train_df.empty:
        train_dataset = Dataset.from_pandas(train_df[cols])
        if tokenize:
            train_dataset = train_dataset.map(tok_func, batched=True)
        class_counts = train_df['labels'].value_counts(sort=False).tolist()
        
    if not test_df.empty:
        test_dataset = Dataset.from_pandas(test_df[cols])
        if tokenize:
            test_dataset = test_dataset.map(tok_func, batched=True)
    if validation and not val_df.empty:
        val_dataset = Dataset.from_pandas(val_df[cols])
        if tokenize:
            val_dataset = val_dataset.map(tok_func, batched=True)

    # --- Case 1: tokenized mode ---
    if tokenize:    
    # Set output format
        if model_name == "text_only" or "_concat" in model_name or "llama" in model_name:
            columns = ["input_ids", "attention_mask", "labels", "index"]
        elif model_name == "context_all_embed" or model_name == "graph_context_all":
            columns = [
                "text_input_ids", "text_attention_mask",
                "speaker_input_ids", "speaker_attention_mask",
                "amendment_input_ids", "amendment_attention_mask",
                "amendment_author_input_ids", "amendment_author_attention_mask",
                "labels", "index"
            ]            
        elif "_embed" in model_name :  # context_embed
            columns = [
                "text_input_ids", "text_attention_mask",
                "ctx_input_ids", "ctx_attention_mask",
                "labels", "index"
            ]
        else:
            columns = ["input_ids", "attention_mask", "labels", "index"]

        if train_dataset:
            train_dataset.set_format("torch", columns=columns)
        if test_dataset:
            test_dataset.set_format("torch", columns=columns)
        if val_dataset:
            val_dataset.set_format("torch", columns=columns)

        # Collator logic
        if "_embed" in model_name and model_name != "context_all_embed":
            def collate_dual(batch):
                return {
                    'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'ctx_input_ids': torch.stack([x['ctx_input_ids'] for x in batch]),
                    'ctx_attention_mask': torch.stack([x['ctx_attention_mask'] for x in batch]),
                    'labels': torch.stack([x['labels'] for x in batch]),
                    'index': torch.stack([x['index'] for x in batch]),
                }
            data_collator = collate_dual
        elif model_name == "context_all_embed":
            def collate_four(batch):
                return {
                    'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'speaker_input_ids': torch.stack([x['speaker_input_ids'] for x in batch]),
                    'speaker_attention_mask': torch.stack([x['speaker_attention_mask'] for x in batch]),
                    'amendment_input_ids': torch.stack([x['amendment_input_ids'] for x in batch]),
                    'amendment_attention_mask': torch.stack([x['amendment_attention_mask'] for x in batch]),
                    'amendment_author_input_ids': torch.stack([x['amendment_author_input_ids'] for x in batch]),
                    'amendment_author_attention_mask': torch.stack([x['amendment_author_attention_mask'] for x in batch]),
                    'labels': torch.stack([x['labels'] for x in batch]),
                    'index': torch.stack([x['index'] for x in batch]),
                }
            data_collator = collate_four
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    else:
        # Just return the dicts with raw text + labels + index
        def collate_raw(batch):
            return {
                'text': [x['text'] for x in batch],
                'labels': torch.tensor([x['labels'] for x in batch]),
                'index': torch.tensor([x['index'] for x in batch])
            }
        data_collator = collate_raw


    train_loader, test_loader, val_loader = None, None, None

    if train_dataset:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=data_collator)
    
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator) if validation else None

    return train_loader, test_loader, val_loader, class_counts

import torch
import torch_geometric.transforms as T
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, to_hetero
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertModel,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaTokenizer,
    PretrainedConfig,
    RobertaModel
)

#from llama_cpp import Llama

all_model_names = [
    "text_only", # Text classifier over the comment text
    "amendment_text_concat", # Text classifier over the speech and the amendment summary texts
    "speaker_text_concat", # Text classifier over the speaker name, speaker group [SEP] speech text

    "context_all_text_concat",  # Text classifier over the speech text [AMENDMENT_SUMMARY] amendment summary [AMENDMENT_CONTENT] amendment content [AMENDMENT_AUTHOR_NAME] amendment author name [AMENDMENT_AUTHOR_GROUP] amendment author group [SPEAKER_NAME] speaker name [SPEAKER_GROUP] speaker group

    "context_all_text_embed",  # Generates one embedding for the speech text, and one embdding for the amendement, one for the speaker, one of the amendment speaker, combine them with a FC layer, then classify
    "context_all_embed", # Generates one embedding for the speech text, and one embdding per contextual element (amendment, amendment author name and group, speaker name and group), combine them with a FC layer, then classify
    "graph_context_all",
    ]
    
all_base_pretrained_models = [
    "almanach/camembertv2-base", # CamemBERT v2 (French)
    "almanach/camembert-base", # CamemBERT v1 (French
]


class DualTextEmb(nn.Module):
    def __init__(self, pretrained_model_name='camembert-base', num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3):
        super().__init__()
        self.device = get_device()
        self.model_name = pretrained_model_name

        # Load config with specified dropout values
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load full model to get both encoder and classifier
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        # Extract encoder and classifier head
        self.encoder = full_model.base_model  # works for BERT, RoBERTa, CamemBERT, etc.

        self.reduce_fc = nn.Linear(1536, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_text; CLS_title] → 768
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, self.config.num_labels)
        ).to(self.device)
        
    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids
            - text_attention_mask
            - ctx_input_ids
            - ctx_attention_mask
        """
        print(f'inside forward, batch keys; {batch.keys()}')
        text_input_ids = batch["text_input_ids"].to(self.device)
        text_attention_mask = batch["text_attention_mask"].to(self.device)
        ctx_input_ids = batch["ctx_input_ids"].to(self.device)
        ctx_attention_mask = batch["ctx_attention_mask"].to(self.device)
        
        # Encode both text and title independently
        text_outputs = self.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        ctx_outputs = self.encoder(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask
        )

        # Extract [CLS] token embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]   # (B, 768)
        ctx_cls = ctx_outputs.last_hidden_state[:, 0, :] # (B, 768)

        # Concatenate and reduce
        combined = torch.cat([text_cls, ctx_cls], dim=1)   # (B, 1536)
        reduced = self.reduce_fc(combined)                   # (B, 768)

        # Final classification (reusing pretrained classifier head)
        logits = self.classifier(reduced)                    # (B, num_classes)

        return logits
    

class AllContextEmb(nn.Module):
    def __init__(self, pretrained_model_name='camembert-base', num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3):
        super().__init__()
        self.device = get_device()
        self.model_name = pretrained_model_name

        # Load config with specified dropout values
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load full model to get both encoder and classifier
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        # Extract encoder and classifier head
        self.encoder = full_model.base_model  # works for BERT, RoBERTa, CamemBERT, etc.
        self.reduce_ctx = nn.Linear(2304, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_speaker; CLS_amendment; CLS_amendment_speaker] → 768

        self.reduce_fc = nn.Linear(1536, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_text; CLS_ctx] → 768
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, self.config.num_labels)
        ).to(self.device)
        
    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids
            - text_attention_mask
            - speaker_input_ids
            - speaker_attention_mask
            - amendment_input_ids
            - amendment_attention_mask
            - amendment_author_input_ids
            - amendment_author_attention_mask
        """

        text_input_ids = batch["text_input_ids"].to(self.device)
        text_attention_mask = batch["text_attention_mask"].to(self.device)
        speaker_input_ids = batch["speaker_input_ids"].to(self.device)
        speaker_attention_mask = batch["speaker_attention_mask"].to(self.device)
        amendment_input_ids = batch["amendment_input_ids"].to(self.device)
        amendment_attention_mask = batch["amendment_attention_mask"].to(self.device)
        amendment_author_input_ids = batch["amendment_author_input_ids"].to(self.device)
        amendment_author_attention_mask = batch["amendment_author_attention_mask"].to(self.device)

        # Encode both text, speaker, amendment and amendment_author independently
        text_outputs = self.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        speaker_outputs = self.encoder(
            input_ids=speaker_input_ids,
            attention_mask=speaker_attention_mask
        )
        amendment_outputs = self.encoder(
            input_ids=amendment_input_ids,
            attention_mask=amendment_attention_mask
        )
        amendment_author_outputs = self.encoder(
            input_ids=amendment_author_input_ids,
            attention_mask=amendment_author_attention_mask
        )

        # Extract [CLS] token embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]   # (B, 768)
        speaker_cls = speaker_outputs.last_hidden_state[:, 0, :] # (B, 768)
        amendment_cls = amendment_outputs.last_hidden_state[:, 0, :] # (B, 768)
        amendment_author_cls = amendment_author_outputs.last_hidden_state[:, 0, :] # (B, 768)

        combined_ctx = torch.cat([speaker_cls, amendment_cls, amendment_author_cls], dim=1)   # (B, 2304)
    
        # Concatenate and reduce the context
        reduced_ctx = self.reduce_ctx(combined_ctx)         # (B, 768)

        # Concatenate and reduce text + context
        combined = torch.cat([text_cls, reduced_ctx], dim=1)   # (B, 1536)
        reduced = self.reduce_fc(combined)                  # (B, 768)

        # Final classification (reusing pretrained classifier head)
        logits = self.classifier(reduced)                    # (B, num_classes)

        return logits

class GraphContextModel(nn.Module):
    def __init__(self, pretrained_model_name="camembert-base", num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3,
                 hidden_size=768, num_heads=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load encoder backbone
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        self.encoder = full_model.base_model  # CamemBERT, BERT, etc.

        # Graph Attention Layer (1 layer + ELU)
        self.gat = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.3)
        self.act = nn.ELU()

        # Reduction and classifier
        self.reduce_fc = nn.Linear(hidden_size + hidden_size * num_heads, hidden_size).to(self.device)  # concat(text_cls, graph_text_emb)
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_classes)
        ).to(self.device)

    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids, text_attention_mask
            - speaker_input_ids, speaker_attention_mask
            - amendment_input_ids, amendment_attention_mask
            - amendment_speaker_input_ids, amendment_speaker_attention_mask
        """
        # Encode each input separately
        text_outputs = self.encoder(input_ids=batch["text_input_ids"], attention_mask=batch["text_attention_mask"])
        speaker_outputs = self.encoder(input_ids=batch["speaker_input_ids"], attention_mask=batch["speaker_attention_mask"])
        amendment_outputs = self.encoder(input_ids=batch["amendment_input_ids"], attention_mask=batch["amendment_attention_mask"])
        amendment_speaker_outputs = self.encoder(input_ids=batch["amendment_speaker_input_ids"], attention_mask=batch["amendment_speaker_attention_mask"])

        # CLS embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]    # (B, 768)
        speaker_cls = speaker_outputs.last_hidden_state[:, 0, :]  # (B, 768)
        amendment_cls = amendment_outputs.last_hidden_state[:, 0, :] # (B, 768)
        amendment_speaker_cls = amendment_speaker_outputs.last_hidden_state[:, 0, :] # (B, 768)

        # Build graph per batch element
        batch_size = text_cls.size(0)
        logits_list = []

        for i in range(batch_size):
            # 4 nodes: [text, speaker, amendment, amendment_speaker]
            node_feats = torch.stack([text_cls[i], speaker_cls[i], amendment_cls[i], amendment_speaker_cls[i]], dim=0)  # (4, 768)

            # Edges: connect text (0) with others (1,2,3), bidirectional
            edge_index = torch.tensor([
                [0, 0, 0, 1, 2, 3],
                [1, 2, 3, 0, 0, 0]
            ], dtype=torch.long, device=self.device)  # (2, E)

            # Apply GAT
            g_emb = self.gat(node_feats, edge_index)  # (4, hidden_size * heads)
            g_emb = self.act(g_emb)

            # Take contextualized embedding of main text node (index 0)
            g_text_emb = g_emb[0]

            # Concatenate with original text_cls
            combined = torch.cat([text_cls[i], g_text_emb], dim=-1)  # (1536,)
            reduced = self.reduce_fc(combined)  # (768,)
            logit = self.classifier(reduced)   # (num_classes,)
            logits_list.append(logit.unsqueeze(0))

        logits = torch.cat(logits_list, dim=0)  # (B, num_classes)
        return logits


def get_model(args):
    model_name = args.model_name
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    pretrained_model_name = args.pretrained_model_name
    assert pretrained_model_name in all_base_pretrained_models, "Invalid model name: {}".format(pretrained_model_name)
    # Instantiate your model
    model = ""

    if model_name == "text_only" or "_concat" in model_name:
        custom_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=2,                    
            hidden_dropout_prob=args.hidden_dropout_prob,         
            attention_probs_dropout_prob=args.attention_probs_dropout_prob 
        )     
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=custom_config
        )
    elif "text_embed" in model_name:
        model = DualTextEmb(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob
        )
    elif model_name == "context_all_embed":
        model = AllContextEmb(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        )
    elif model_name == "graph_context_all":
        model = GraphContextModel(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob
        )
    else:
        raise NameError(f"Invalid model name: {model_name}")

    return model


def get_device(rank=None):
    if rank is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        print(f"Using CUDA device {rank} for distributed training")
        return device

    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device('mps')

    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device('cuda')

    print("Using CPU")
    return torch.device('cpu')

def log_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")

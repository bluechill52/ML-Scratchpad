import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import createDataloaderV1, DatasetV1
from tokenizers import SimpleTokenizer
import tiktoken
from gpt import Model



def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter=1):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_batch(ip, target, model):
    logits = model(ip)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    return loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train(model, train_dataloader, test_dataloader, settings):
    device = torch.device(settings['device'])
    optim = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    for epoch in range(settings['num_epochs']):
        model.train()
        
        for idx, (ip, target) in enumerate(train_dataloader):
            ip = ip.to(device)
            target = target.to(device)
            
            optim.zero_grad()
            loss = calc_loss_batch(ip, target, model)
            loss.backward()
            optim.step()
            
            tokens_seen += ip.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % settings['eval_freq'] == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, test_dataloader, device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
    
    return train_losses, val_losses, track_tokens_seen



if __name__ == '__main__':
    # Configuration
    cfg = {
        'batch_size' : 4,
        'stride' : 1,
        'context_length' : 5,
        'embedding_dim' : 10,
        'qkv_dim' : 256,
        'ff_hidden_dim' : 64,
        'num_heads' : 4,
        'num_transformer_layers' : 4,
        'dropout_rate' : 0.1,
    }
    
    settings = {
        'num_epochs' : 100,
        'lr' : 5e-4,
        'weight_decay' : 0.2,
        'device' : 'cuda',
        'eval_freq' : 50
    }
    
    
    fname = 'the-verdict.txt'

    # tokenizer = SimpleTokenizer(fname)
    tokenizer = tiktoken.get_encoding('gpt2')
    
    with open(os.path.join(os.getcwd(), fname), 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    train_ratio = 0.9
    split_idx = int(len(raw_text) * train_ratio)
    train_text = raw_text[:split_idx]
    test_text = raw_text[split_idx:]
    
    # dataset = DatasetV1(raw_text, tokenizer, cfg)
    train_dataloader = createDataloaderV1(train_text, tokenizer, cfg)
    test_dataloader = createDataloaderV1(test_text, tokenizer, cfg)
    
    data_iter = iter(train_dataloader)
    ip, target = next(data_iter)
    
    cfg['vocab_size'] = tokenizer.n_vocab
    
    M = Model(cfg)
    M.to(torch.device(settings['device']))
    
    train_losses, val_losses, track_tokens_seen = train(M, train_dataloader, test_dataloader, settings)
    
    
    # Evaluate model
    M.eval() # disable dropout
    start_context = "Money's only"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=M,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg['context_length']
    )

    print("Output:", out)
    print("Output length:", len(out[0]))
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
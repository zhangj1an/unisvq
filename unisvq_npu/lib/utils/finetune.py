import torch
from torch import nn

import glog

def extract_susv_params(module):
    susv_params = []
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'SU' in name or 'SV' in name:
                susv_params.append(param)
            else:
                params.append(param)
            # glog.info(f"tuneable: {name}")
    return susv_params, params


def get_susv_adam(susv_params, params, args):
    return torch.optim.Adam([
        {
            'params': susv_params,
            'lr': args.ft_susv_lr
        },
        {
            'params': params,
            'lr': args.ft_lr
        },
    ])


def save_susv(module, path, save_tuneable=False):
    saved_layer = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = module.SV.data.to(torch.float16)
    if save_tuneable:
        saved_layer['codebook_class.codebook.linear_proj.weight'] = module.codebook_class.codebook.linear_proj.weight.data.to(dtype=torch.float16)
        saved_layer['codebook_class.codebook.linear_proj.bias'] = module.codebook_class.codebook.linear_proj.bias.data.to(dtype=torch.float16)
    torch.save(saved_layer, path)


def calculate_mse_loss(layer, dataloader, device, position_embeddings, position_ids):
    layer.eval()
    total_loss = 0
    ct = 0
    batch_position_embeddings = None
    batch_position_ids = None
    with torch.no_grad():
        for _, (source, target) in enumerate(dataloader):
            layer_kwargs = {
                "hidden_states": source.to(device)
            }
            batch_size = source.shape[0]
            if position_embeddings is not None:
                if batch_position_embeddings is None:
                    batch_position_embeddings = (position_embeddings[0][:batch_size], position_embeddings[1][:batch_size])
                layer_kwargs["position_embeddings"] = batch_position_embeddings
            else:
                if batch_position_ids is None:
                    batch_position_ids = position_ids[:batch_size]
                layer_kwargs["position_ids"] = batch_position_ids
            total_loss += nn.MSELoss()(layer(**layer_kwargs)[0], target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss(layer, position_embeddings, attention_mask, dataloader):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            output = layer(
                source,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask.float())[:, :-1].contiguous()
            total_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                target.to(0).view(-1, target.shape[-1]),
            )
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

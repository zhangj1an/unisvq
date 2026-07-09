import gc
import pdb
import sys

import glog
import torch
from tqdm import tqdm


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def show_metrics(hatW, W_orig, H, msg):
    err_frob = (hatW - W_orig).square().sum() / W_orig.square().sum()
    err_proxy = (((hatW - W_orig) @ H) * (hatW - W_orig)).sum() / ((W_orig @ H) * W_orig).sum()
    glog.info(f"{msg} frob  error: {err_frob}")
    glog.info(f"{msg} proxy error: {err_proxy}")


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    use as:
    from lib import utils
    utils.ForkedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


'''


    for i in tqdm(range(args.devset_size // args.batch_size), desc=f"calculating dev emb for layer {idx}"):
        dev_emb[(args.batch_size * i):(args.batch_size * (i + 1)), :, :] = transformer_layer(
            dev_emb[(args.batch_size * i):(args.batch_size * (i + 1)), :, :].to(device),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False)[0].cpu()

    act_error = (dev_emb.to(dtype_) - orig_emb.to(dtype_)).square().sum() / (orig_emb.to(dtype_) -
                                                                             orig_emb.to(dtype_).mean(
                                                                                 (0, 1))).square().sum()
    glog.info(f"layer {idx} act error: {act_error}")

    glog.info(f"saving activations for layer {idx}")

    torch.save(
        {
            'devset': devset,
            'dev_emb': dev_emb,
            'orig_emb': orig_emb,
            'after_layer': idx,
            'timestamp': str(datetime.datetime.now())
        }, f"{args.save_path}/dev_activations.new.pt")
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        os.remove(f"{args.save_path}/dev_activations.pt")
    os.rename(f"{args.save_path}/dev_activations.new.pt", f"{args.save_path}/dev_activations.pt")

    glog.info(f'"done processing layer {idx}, total time {time.time() - st_time} seconds')


'''

from . import (
    index_codebook, 
    index_guassian_codebook,
)

codebook_id = {
    'identical': (20, index_codebook.Index_codebook),
    'linear_guassian': (21, index_guassian_codebook.LinearGuassian_codebook),
}

quantized_class = {
    20: index_codebook.IndexLinear,
    21: index_guassian_codebook.LinearGuassianLinear,
}

cache_permute_set = {}

def get_codebook(name, codebook_bit=None):
    return codebook_id[name][1](codebook_bit=codebook_bit)

def get_id(name):
    return codebook_id[name][0]

def get_quantized_class(id):
    return quantized_class[id]

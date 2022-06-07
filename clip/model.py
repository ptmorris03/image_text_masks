import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["CLIP", "CLIP_ENCODE_IMAGE", "CLIP_ENCODE_TEXT", "CLIP_SIMILARITY", "similarity"]


def QuickGELU(x):
    return x * jax.nn.sigmoid(1.702 * x)

def zscore(x, axis=-1, eps=1e-5):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)
    inverse_std = jax.lax.rsqrt(variance + eps)
    return inverse_std * (x - mean)

def layernorm(x, scale, offset, axis=-1, eps=1e-5):
    x = zscore(x, axis=axis, eps=eps)
    return x * scale + offset

def linear(x, weight, bias):
    return jnp.dot(x, weight.T) + bias

def CLIP_MLP(params, x, name=''):
    scale = params[name + '.ln_2.weight']
    offset = params[name + '.ln_2.bias']
    weight1 = params[name + '.mlp.c_fc.weight']
    bias1 = params[name + '.mlp.c_fc.bias']
    weight2 = params[name + '.mlp.c_proj.weight']
    bias2 = params[name + '.mlp.c_proj.bias']

    x = layernorm(x, scale, offset)
    x = linear(x, weight1, bias1)
    x = QuickGELU(x)
    return linear(x, weight2, bias2)

def CLIP_ATTN(params, x, heads: int, mask=None, name=''):
    scale = params[name + '.ln_1.weight']
    offset = params[name + '.ln_1.bias']
    weight_in = params[name + '.attn.in_proj_weight']
    bias_in = params[name + '.attn.in_proj_bias']
    weight_out = params[name + '.attn.out_proj.weight']
    bias_out = params[name + '.attn.out_proj.bias']

    x = layernorm(x, scale, offset)
    x = linear(x, weight_in, bias_in)
    x = x.reshape((*x.shape[:2], 3, heads, -1))
    q, k, v = x[...,0,:,:], x[...,1,:,:], x[...,2,:,:]

    sqrt_k = np.sqrt(x.shape[-1]).astype(k.dtype)
    attn_logits = jnp.einsum("tbhd,Tbhd->bhtT", q, k) / sqrt_k
    
    if mask is not None:
        attn_logits += mask

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("bhtT,Tbhd->tbhd", attn_weights, v)
    attn = jnp.reshape(attn, (*q.shape[:2], -1))

    return linear(attn, weight_out, bias_out)

def CLIP_RESBLOCK(params, x, heads: int, mask=None, name=''):
    x = CLIP_ATTN(params, x, heads, mask, name)
    return CLIP_MLP(params, x, name)

def CLIP_TRANSFORMER(params, x, layers: int, heads: int, mask=None, name=''):
    for i in range(layers):
        print(jnp.linalg.norm(x[0] - x[1]))
        x = CLIP_RESBLOCK(params, x, heads, mask=mask, name=name + F'.resblocks.{i}')
    return x

def CLIP_ENCODE_IMAGE(params, image, name='visual'):
    conv_kernel = params[name + '.conv1.weight']
    class_emb = params[name + '.class_embedding']
    pos_emb = params[name + '.positional_embedding']
    scale_pre = params[name + '.ln_pre.weight']
    offset_pre = params[name + '.ln_pre.bias']
    scale_post = params[name + '.ln_post.weight']
    offset_post = params[name + '.ln_post.bias']
    proj = params[name + '.proj']

    layers = len([k for k in params.keys() if k.startswith(name) and k.endswith(".attn.in_proj_weight")])
    heads = conv_kernel.shape[0] // 64
    
    x = jax.lax.conv(image, conv_kernel, conv_kernel.shape[-2:], 'SAME')
    x = x.reshape(x.shape[0], x.shape[1], -1).transpose((0, 2, 1))
    class_emb = class_emb + jnp.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype)
    x = jnp.concatenate([class_emb, x], axis=1)
    x = x + pos_emb

    x = layernorm(x, scale_pre, offset_pre)
    x = x.transpose((1, 0, 2))
    x = CLIP_TRANSFORMER(params, x, layers, heads, name=name + '.transformer')
    x = x.transpose((1, 0, 2))
    x = layernorm(x[:,0,:], scale_post, offset_post)

    return x @ proj

def CLIP_ENCODE_TEXT(params, text, name=''):
    token_emb = params[name + 'token_embedding.weight']
    pos_emb = params[name + 'positional_embedding']
    scale_final = params[name + 'ln_final.weight']
    offset_final = params[name + 'ln_final.bias']
    proj = params[name + 'text_projection']

    layers = len(set(k.split(".")[2] for k in params if k.startswith(f"transformer.resblocks")))
    heads = scale_final.shape[0] // 64
    
    mask = jnp.full((text.shape[-1], text.shape[-1]), -10e10)
    mask = jnp.triu(mask, 1)

    x = jnp.asarray(token_emb)[(text,)]
    x = x + pos_emb

    x = x.transpose((1, 0, 2))
    x = CLIP_TRANSFORMER(params, x, layers, heads, mask=mask, name=name + 'transformer')
    x = x.transpose((1, 0, 2))
    x = layernorm(x, scale_final, offset_final)
    x = x[jnp.arange(x.shape[0]), text.argmax(axis=-1)]

    return x @ proj

def similarity(image, text, logit_scale):
    image = image / jnp.linalg.norm(image, axis=-1, keepdims=True)
    text = text / jnp.linalg.norm(text, axis=-1, keepdims=True)

    logit_scale = jnp.exp(logit_scale)
    image_logits = logit_scale * image @ text.transpose()
    text_logits = logit_scale * text @ image.transpose()

    return image_logits, text_logits

def CLIP_SIMILARITY(params, image, text):
    logit_scale = params['logit_scale']

    return similarity(image, text, logit_scale)

def CLIP(params, image, text):
    image = CLIP_ENCODE_IMAGE(params, image)
    text = CLIP_ENCODE_TEXT(params, text)
    return CLIP_SIMILARITY(params, image, text)


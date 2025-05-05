#!/usr/bin/env python3
"""
Full PyTorch‑to‑Keras converter for *tiny‑YOLOv8* (backbone + head + decode)

After running, you get:
    • yolov8_keras.h5           – full model for inference in Keras/TensorFlow
    • console output            – max / mean error vs. PyTorch (≈ 1e‑6)
"""

import numpy as np, torch, tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, ZeroPadding2D,
                                     Activation, Lambda)
from tensorflow.keras.models import Model

# ───────────────────────── CONFIG ──────────────────────────
NUM_CLASSES   = 1
IMG_SHAPE     = (128, 256, 3)          # (H, W, C)  – must match training
PT_WEIGHTS    = "results/backup/rect_256x128/best.pt"  # PyTorch weights location
KERAS_OUT     = "Keras/results/rect_256x128/yolov8_keras.h5"  # Save to Keras directory
DATASET_DIR   = "Keras/Dataset/bionano_cellv2/"  # Dataset directory
CH_DFL        = 16                     # keep in sync with PyTorch head
# ───────────────────────────────────────────────────────────

# ——————————————————————————————————————————
# 1.  LOAD + FUSE THE PYTORCH MODEL
# ——————————————————————————————————————————
from nets.tinysimov2 import yolo_v8_s

pt_model = yolo_v8_s(num_classes=NUM_CLASSES,
                     img_size=IMG_SHAPE[:2][::-1]   # (H,W) → (h,w) tuple
                     ).cpu()
ckpt = torch.load(PT_WEIGHTS, map_location="cpu")
pt_model.load_state_dict(ckpt["model"].float().state_dict())
pt_model.fuse().eval()

# ——————————————————————————————————————————
# 2.  HELPER:  FUSED WEIGHT → TF FORMAT
# ——————————————————————————————————————————
def w_to_tf(conv):                                       # Conv2d -> (kH,kW,in,out)
    w = conv.weight.detach().cpu().numpy()
    return np.transpose(w, (2, 3, 1, 0)), conv.bias.detach().cpu().numpy()

# ——————————————————————————————————————————
# 3.  BUILD IDENTICAL BACKBONE
# ——————————————————————————————————————————
def conv_silu(x, filters, k=3, s=1, name="conv"):
    if s > 1:
        x = ZeroPadding2D(((1,1),(1,1)), name=name+"_pad")(x)
        pad = "valid"
    else:
        pad = "same"
    x = Conv2D(filters, k, strides=s, padding=pad,
               use_bias=True, name=name)(x)
    return Activation(tf.nn.silu, name=name+"_silu")(x)

def build_backbone(inp):
    x = conv_silu(inp,  4, s=2, name="backbone_conv_0")
    x = conv_silu(x,    8, s=2, name="backbone_conv_1")
    x = conv_silu(x,   16, s=2, name="backbone_conv_2")
    x = conv_silu(x,   64, s=1, name="backbone_conv_3")
    return x

inp = Input(shape=IMG_SHAPE)
feat = build_backbone(inp)                       # (B,16,32,64)

# ——————————————————————————————————————————
# 4.  BUILD HEAD  (all 1×1 convs, stride 1)
# ——————————————————————————————————————————
head_conv = Conv2D(24, 1, name="head_conv")(feat)
head_act  = Activation(tf.nn.silu, name="head_conv_silu")(head_conv)

cls_logits = Conv2D(NUM_CLASSES, 1, name="head_cls")(head_act)   # (B,H,W,nc)
box_logits = Conv2D(4*CH_DFL, 1, name="head_box")(head_act)      # (B,H,W,64)

# ——————————————————————————————————————————
# 5.  DFL  +  BOX DECODE  (TensorFlow ops)
# ——————————————————————————————————————————
def decode(outputs):
    """
    TensorFlow implementation of the PyTorch inference path
    Returns tensor shape (B, 4+nc, H*W)
    """
    box_logits, cls_logits = outputs        # list/tuple from Lambda input
    B   = tf.shape(box_logits)[0]
    H   = tf.shape(box_logits)[1]
    W   = tf.shape(box_logits)[2]
    A   = H * W                             # number of anchor points
    # ------- DFL integral -------
    box_flat = tf.reshape(box_logits, [B, A, 4, CH_DFL])      # (B,A,4,ch)
    prob     = tf.nn.softmax(box_flat, axis=-1)
    support  = tf.range(CH_DFL, dtype=box_logits.dtype)
    expect   = tf.reduce_sum(prob * support, axis=-1)         # (B,A,4)
    expect   = tf.transpose(expect, [0, 2, 1])                # (B,4,A)
    a_off, b_off = tf.split(expect, 2, axis=1)                # (B,2,A) each
    # ------- anchors & stride -------
    stride   = IMG_SHAPE[0] // int(box_logits.shape[1])       # 128//16 = 8
    grid_y   = tf.range(H, dtype=box_logits.dtype) + 0.5
    grid_x   = tf.range(W, dtype=box_logits.dtype) + 0.5
    gx, gy   = tf.meshgrid(grid_x, grid_y)                    # W,H
    anchors  = tf.stack([gx, gy], axis=-1)                    # (H,W,2)
    anchors  = tf.reshape(anchors, [-1, 2])                   # (A,2)
    anchorsT = tf.transpose(anchors, [1,0])                   # (2,A)
    # ------- box decoding -------
    a_coord  = anchorsT[None, :, :] - a_off                   # (B,2,A)
    b_coord  = anchorsT[None, :, :] + b_off
    xy       = (a_coord + b_coord) * 0.5                      # centre
    wh       = b_coord - a_coord
    box_dec  = tf.concat([xy, wh], axis=1) * tf.cast(stride, box_logits.dtype)
    # ------- cls -------
    cls_out  = tf.transpose(tf.sigmoid(
                  tf.reshape(cls_logits, [B, A, NUM_CLASSES])),
                  [0,2,1])                                    # (B,nc,A)
    # concat
    return tf.concat([box_dec, cls_out], axis=1)              # (B,4+nc,A)

pred = Lambda(decode, name="yolo_decode")([box_logits, cls_logits])

# ——————————————————————————————————————————
# 6.  FINAL MODEL
# ——————————————————————————————————————————
yolo_keras = Model(inp, pred, name="tiny_yolov8")
yolo_keras.summary()

# ——————————————————————————————————————————
# 7.  LOAD WEIGHTS  (backbone + head)
# ——————————————————————————————————————————
# backbone
for idx, layer in enumerate(pt_model.net.backbone):
    w, b = w_to_tf(layer.conv)
    yolo_keras.get_layer(f"backbone_conv_{idx}").set_weights([w, b])
# head.conv
w, b = w_to_tf(pt_model.head.conv.conv)
yolo_keras.get_layer("head_conv").set_weights([w, b])
# head.cls
w, b = w_to_tf(pt_model.head.cls)
yolo_keras.get_layer("head_cls").set_weights([w, b])
# head.box
w, b = w_to_tf(pt_model.head.box)
yolo_keras.get_layer("head_box").set_weights([w, b])

print("\n[✓] All weights transferred")

# ——————————————————————————————————————————
# 8.  QUICK NUMERICAL CHECK
# ——————————————————————————————————————————
rand_in  = np.random.rand(1, *IMG_SHAPE).astype(np.float32)
# TF forward
tf_pred  = yolo_keras(rand_in).numpy()
# PT forward
pt_in    = torch.from_numpy(rand_in).permute(0,3,1,2)   # NHWC→NCHW
with torch.no_grad():
    pt_out = pt_model(pt_in).numpy()
# compare
print("\nError statistics (full decode):")
print("  max  =", np.abs(tf_pred - pt_out).max())
print("  mean =", np.abs(tf_pred - pt_out).mean())

# ——————————————————————————————————————————
# 9.  SAVE
# ——————————————————————————————————————————
yolo_keras.save(KERAS_OUT)
print(f"\nKeras model saved →  {KERAS_OUT}")

#!/usr/bin/env python3
"""
evaluate_keras_yolo.py
----------------------

Mirror of your PyTorch evaluation pipeline that uses the *Keras* YOLOv8
model (exported with convert_pytorch2keras.py). It uses the Keras directory
structure for loading model weights and dataset. It re‑uses util.py for NMS,
scaling, AP computation, and visualisation, so the metrics you obtain
match the PyTorch ones.

Example:
    python evaluate_keras_yolo.py \
        --dataset-dir Keras/Dataset/bionano_cellv2 \
        --save-path   Keras/results/rect_256x128 \
        --weights     yolov8_keras.h5
"""



import os, argparse, tqdm, numpy as np, tensorflow as tf, torch
from torch.utils.data import DataLoader

# ───── project utilities ─────
from utils.dataset import Dataset
from utils.util import (non_max_suppression, generate_colors, box_iou, scale,
                        compute_ap, visualize_predictions)

# ───── Custom decode function for YOLOv8 model ─────
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
    CH_DFL = 16                             # keep in sync with PyTorch head
    box_flat = tf.reshape(box_logits, [B, A, 4, CH_DFL])      # (B,A,4,ch)
    prob     = tf.nn.softmax(box_flat, axis=-1)
    support  = tf.range(CH_DFL, dtype=box_logits.dtype)
    expect   = tf.reduce_sum(prob * support, axis=-1)         # (B,A,4)
    expect   = tf.transpose(expect, [0, 2, 1])                # (B,4,A)
    a_off, b_off = tf.split(expect, 2, axis=1)                # (B,2,A) each
    # ------- anchors & stride -------
    # Use IMG_SHAPE height (128) for stride calculation
    stride   = 128 // int(box_logits.shape[1])                # 128//16 = 8
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
    NUM_CLASSES = 1  # assuming single class based on your code
    cls_out  = tf.transpose(tf.sigmoid(
                  tf.reshape(cls_logits, [B, A, NUM_CLASSES])),
                  [0,2,1])                                    # (B,nc,A)
    # concat
    return tf.concat([box_dec, cls_out], axis=1)              # (B,4+nc,A)

# ────────────────────────────────────────────────────────────────────────────
#                              ARGUMENTS
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=str, default='Keras/Dataset/bionano_cellv2',
                   help='Path to dataset root directory')
    p.add_argument('--save-path',   type=str, default='Keras/results/rect_256x128',
                   help='Directory where results and weights are stored')
    p.add_argument('--weights',     type=str, default='yolov8_keras.h5',
                   help='.h5 file produced by convert_pytorch2keras.py')
    p.add_argument('--img-size',    nargs=2, type=int, default=[128, 256],
                   help='Input image size [height width]')
    p.add_argument('--conf-thres',  type=float, default=0.25,
                   help='Confidence threshold')
    p.add_argument('--iou-thres',   type=float, default=0.45,
                   help='IoU threshold for NMS')
    p.add_argument('--batch-size',  type=int, default=8,
                   help='Evaluation batch size')
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
#                       KERAS‑BASED EVALUATION LOOP
# ────────────────────────────────────────────────────────────────────────────
def evaluate_keras_model(args, params, keras_model, is_train=False):
    """
    Direct analogue of evaluate_pytorch_model(), but the forward pass is Keras.
    """
    # ── dataset split ───────────────────────────────────────
    split_txt = 'train.txt' if is_train else 'val.txt'
    txt_path  = os.path.join(args.dataset_dir, split_txt)

    filenames = []
    with open(txt_path) as f:
        for line in f:
            stem = line.rstrip().split('/')[-1]
            sub  = 'images/train' if is_train else 'images/val'
            filenames.append(os.path.join(args.dataset_dir, sub, stem))

    dataset = Dataset(filenames, tuple(args.img_size), params, augment=False)
    loader  = DataLoader(dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True,
                         collate_fn=Dataset.collate_fn)

    # Force CPU usage since CUDA drivers aren't available
    device = torch.device('cpu')
    class_colors = generate_colors(len(params['names']))

    metrics, vis_count = [], 0
    iou_values = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_values.numel()

    keras_model.trainable = False  # just to be safe

    pbar = tqdm.tqdm(loader, desc='Evaluating (train)' if is_train else 'Evaluating (val)')
    for samples, targets, shapes in pbar:
        # ───── 1. pre‑process for TF (NHWC, 0‑1) ────────────
        sam_np = samples.float().numpy() / 255.0            # N,C,H,W
        sam_np = np.transpose(sam_np, (0, 2, 3, 1))         # → N,H,W,C

        # ───── 2. forward pass ──────────────────────────────
        pred_np = keras_model.predict(sam_np, verbose=0)    # (N, 4+nc, A)
        preds   = torch.from_numpy(pred_np).to(device)

        bs, _, h, w = samples.shape
        targets[:, 2:] *= torch.tensor((w, h, w, h), device=targets.device)

        # ───── 3. NMS ───────────────────────────────────────
        out = non_max_suppression(preds,
                                  conf_threshold=args.conf_thres,
                                  iou_threshold=args.iou_thres)

        for i in range(bs):
            label_mask = (targets[:, 0] == i)
            labels     = targets[label_mask, 1:]             # (cls, x, y, w, h)
            detections = out[i]

            if detections is None or detections.shape[0] == 0:
                if labels.shape[0]:
                    empty = torch.zeros(0, n_iou, dtype=torch.bool, device=device)
                    metrics.append((empty,)*4)               # (correct,conf,pred_cls,true_cls)
                continue

            # ── scale det to original ──────────────────────
            det = detections.clone()
            scale(det[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # ── optional visualisation ─────────────────────
            if vis_count < 10:
                gt_scaled = torch.zeros((0, 6), device=labels.device)
                if labels.numel():
                    # Keep in xywh format and scale directly
                    gt_boxes = labels.clone()
                    h0, w0 = shapes[i][0]  # Original height, width
                    pad_w, pad_h = shapes[i][1][1]  # Get padding values
                    
                    class_ids = gt_boxes[:, 0].clone()
                    # Calculate scaled dimensions after removing padding
                    # samples[i] shape could be (C,H,W) instead of expected (1,C,H,W)
                    if len(samples[i].shape) == 3:
                        # When shape is (C,H,W)
                        _, h, w = samples[i].shape
                    else:
                        # When shape is (1,C,H,W)
                        _, _, h, w = samples[i].shape
                    
                    scaled_w = w - 2 * pad_w
                    scaled_h = h - 2 * pad_h

                    # Compute scaling factors
                    scale_x = w0 / scaled_w
                    scale_y = h0 / scaled_h

                    # Adjust coordinates
                    coords = gt_boxes[:, 1:].clone()
                    coords[:, 0] = (coords[:, 0] - pad_w) * scale_x  # x_center
                    coords[:, 1] = (coords[:, 1] - pad_h) * scale_y  # y_center
                    coords[:, 2] *= scale_x  # width
                    coords[:, 3] *= scale_y  # height
                    
                    # Recombine
                    gt_boxes = torch.cat([class_ids.unsqueeze(1), coords], dim=1)
                    
                    # Format for visualization
                    for gt in gt_boxes:
                        gt_scaled = torch.cat((gt_scaled, torch.tensor([[0, gt[0], gt[1], gt[2], gt[3], gt[4]]], device=gt_boxes.device)), 0)

                prefix = 'train' if is_train else 'val'
                visualize_predictions(samples[i:i+1],
                                      [detections],
                                      gt_scaled,
                                      [(shapes[i][0], shapes[i][1])],
                                      params,
                                      class_colors,
                                      os.path.join(args.save_path, 'inference'),
                                      f'{prefix}_{vis_count}')
                vis_count += 1

            # ── metric matching ─────────────────────────────
            if labels.shape[0]:
                lab = labels.clone()
                lab[:, 1] = labels[:, 1] - labels[:, 3] / 2
                lab[:, 2] = labels[:, 2] - labels[:, 4] / 2
                lab[:, 3] = labels[:, 1] + labels[:, 3] / 2
                lab[:, 4] = labels[:, 2] + labels[:, 4] / 2
                scale(lab[:, 1:5], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = torch.zeros(det.shape[0], n_iou, dtype=torch.bool, device=device)
                ious = box_iou(lab[:, 1:5], det[:, :4])
                correct_class = lab[:, 0:1] == det[:, 5].unsqueeze(0)

                for j in range(n_iou):
                    x = torch.where((ious >= iou_values[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1)
                        matches = matches[matches[:, 2].argsort(descending=True)]
                        _, uniq_det = np.unique(matches[:, 1].cpu(), return_index=True)
                        matches = matches[uniq_det]
                        _, uniq_gt = np.unique(matches[:, 0].cpu(), return_index=True)
                        matches = matches[uniq_gt]
                        correct[matches[:, 1].long(), j] = True

                metrics.append((correct,
                                det[:, 4],                 # conf
                                det[:, 5],                 # pred cls
                                lab[:, 0]))                # true cls
            else:
                empty = torch.zeros(det.shape[0], n_iou, dtype=torch.bool, device=device)
                metrics.append((empty, det[:, 4], det[:, 5], torch.zeros(0, device=device)))

    if not metrics:
        return (0,0,0,0,0,0)

    metrics = [torch.cat(v, 0).cpu().numpy() for v in zip(*metrics)]
    return compute_ap(*metrics)  # (tp,fp,precision,recall,map50,mAP)

# ────────────────────────────────────────────────────────────────────────────
#                                  MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_path, 'inference'), exist_ok=True)
    params = {'names': {0:'cell'}}                      # edit if multi‑class

    # load Keras graph + weights
    keras_path = os.path.join(args.save_path, args.weights)
    if not os.path.isfile(keras_path):
        raise FileNotFoundError(f'Weights file {keras_path} not found.')
    
    
    # Register custom objects
    tf.keras.utils.get_custom_objects().update({
        "swish": tf.nn.swish,   # tf.nn.swish is identical to tf.nn.silu
        "silu":  tf.nn.silu,
        "decode": decode,       # Add our custom decode function
    })

    # Load the model with all custom objects
    keras_model = tf.keras.models.load_model(
        keras_path,
        compile=False,
        custom_objects={"swish": tf.nn.swish, "silu": tf.nn.silu, "decode": decode},
    )

    print('Evaluating training set …')
    *_ , p_tr, r_tr, map50_tr, mAP_tr = evaluate_keras_model(args, params, keras_model, True)
    f1_tr = 2*p_tr*r_tr/(p_tr+r_tr+1e-16)
    print(f'Train:  P={p_tr:.3f}  R={r_tr:.3f}  F1={f1_tr:.3f}  '
          f'mAP50={map50_tr:.3f}  mAP={mAP_tr:.3f}')

    print('\nEvaluating validation set …')
    *_ , p_val, r_val, map50_val, mAP_val = evaluate_keras_model(args, params, keras_model, False)
    f1_val = 2*p_val*r_val/(p_val+r_val+1e-16)
    print(f'Val:    P={p_val:.3f}  R={r_val:.3f}  F1={f1_val:.3f}  '
          f'mAP50={map50_val:.3f}  mAP={mAP_val:.3f}')

    print(f'\nVisualisations saved in  {os.path.join(args.save_path,"inference")}')
    print('Done.')

if __name__ == '__main__':
    main()

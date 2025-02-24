import argparse
import copy
import csv
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from torch.utils import data

from nets import pico
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']
    return fn


def train(args, params):
    """
    Main training loop with mosaic handling, warmup, gradient scaling, and
    final model saving. After each epoch, calls `test(...)` for real mAP
    measurement and visualization of first 5 images from validation set.
    """
    # -----------------------------------------------
    # 1) Build model
    # -----------------------------------------------
    model = pico.yolo_v8_p(len(params['names'].values())).cuda()

    # -----------------------------------------------
    # 2) Optimizer & Scheduler
    # -----------------------------------------------
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    # Separate parameters (conv.weight, bn.weight, bn.bias, etc.)
    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(
        p[2],
        params['lr0'],
        params['momentum'],
        nesterov=True
    )
    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    lr_func = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func, last_epoch=-1)

    # -----------------------------------------------
    # 3) EMA
    # -----------------------------------------------
    ema = util.EMA(model) if args.local_rank == 0 else None

    # -----------------------------------------------
    # 4) Datasets & DataLoaders
    # -----------------------------------------------
    train_filenames = []
    with open('./Dataset/Yeast/train.txt') as ftrain:
        for line in ftrain.readlines():
            line = line.rstrip().split('/')[-1]
            train_filenames.append('./Dataset/Yeast/images/train/' + line)

    dataset = Dataset(train_filenames, args.input_size, params, True)
    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(
        dataset,
        args.batch_size,
        sampler is None,
        sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    # If using multi-GPU
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    # -----------------------------------------------
    # 5) Training Loop
    # -----------------------------------------------
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)

    # step.csv for logging
    with open(os.path.join(args.save_path, 'step.csv'), 'w', newline='') as csv_f:
        if args.local_rank == 0:
            writer = csv.DictWriter(csv_f, fieldnames=['epoch', 'mAP@50', 'mAP', 'Precision', 'Recall', 'F1'])
            writer.writeheader()

        # ---------------------
        #   For each epoch
        # ---------------------
        for epoch in range(args.epochs):
            model.train()

            # Turn off mosaic for last 10 epochs
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)

            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(p_bar, total=num_batch)

            optimizer.zero_grad()

            # -----------------------------------------------
            # 6) Batch iteration
            # -----------------------------------------------
            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch
                samples = samples.cuda().float() / 255.0
                targets = targets.cuda()

                # Warmup logic
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, np.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            # bias lr
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr_func(epoch)]
                        else:
                            # normal lr
                            fp = [0.0, y['initial_lr'] * lr_func(epoch)]
                        y['lr'] = np.interp(x, xp, fp)

                        # momentum
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = np.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)

                loss = criterion(outputs, targets)
                m_loss.update(loss.item(), samples.size(0))

                # Scale loss for multi-GPU
                loss *= args.batch_size
                loss *= args.world_size

                # Backprop
                amp_scale.scale(loss).backward()

                # Gradient accumulation
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)
                    util.clip_gradients(model)
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # For local_rank=0, print progress
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                    p_bar.set_description(s)

                del loss, outputs

            # Scheduler step after each epoch
            scheduler.step()

            # -----------------------------------------------
            # 7) Validation & Logging
            # -----------------------------------------------
            if args.local_rank == 0:
                # Evaluate (test) -> returns (tp, fp, precision, recall, map50, mean_ap)
                tp, fp, precision, recall, map50, mean_ap = test(args, params, ema.ema)

                # Compute F1
                f1 = 2 * precision * recall / (precision + recall + 1e-16)

                # Write row in step.csv
                writer.writerow({
                    'epoch': str(epoch + 1).zfill(3),
                    'mAP@50': f'{map50:.3f}',
                    'mAP': f'{mean_ap:.3f}',
                    'Precision': f'{precision:.3f}',
                    'Recall': f'{recall:.3f}',
                    'F1': f'{f1:.3f}'
                })
                csv_f.flush()

                # Update best
                if mean_ap > best:
                    best = mean_ap

                # Save model: last & best
                ckpt = {'model': copy.deepcopy(ema.ema).half()}
                torch.save(ckpt, os.path.join(args.save_path, 'last.pt'))
                if best == mean_ap:
                    torch.save(ckpt, os.path.join(args.save_path, 'best.pt'))
                del ckpt

    # Strip optimizers
    if args.local_rank == 0:
        util.strip_optimizer(os.path.join(args.save_path, 'best.pt'))
        util.strip_optimizer(os.path.join(args.save_path, 'last.pt'))

    torch.cuda.empty_cache()


def generate_colors(num_classes):
    """Helper function to generate random colors for each class."""
    return {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(num_classes)}


def visualize_predictions(samples, outputs, gt_boxes, shapes, params, class_colors, results_dir, index):
    """
    Visualize first 5 images from the val set: draws ground-truth boxes on the
    left, predicted boxes on the right, and saves side-by-side images.
    """
    # Convert single image to CPU numpy
    img = samples[0].cpu().float().numpy()
    img = img.transpose((1, 2, 0))  # CHW -> HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    # Remove padding, resize back
    original_shape = shapes[0][0]
    pad_w, pad_h = shapes[0][1][1]
    h, w = img.shape[:2]
    img = img[int(pad_h):int(h - pad_h), int(pad_w):int(w - pad_w)]
    img = cv2.resize(img, (int(original_shape[1]), int(original_shape[0])))

    img_gt = img.copy()
    img_pred = img.copy()

    # Draw GT
    for gt in gt_boxes:
        cls_gt = int(gt[1])
        coords = gt[2:].cpu().numpy()
        x_c, y_c = coords[0], coords[1]
        w_b, h_b = coords[2], coords[3]

        # Convert to corner coordinates
        x1 = int(x_c - w_b / 2)
        y1 = int(y_c - h_b / 2)
        x2 = int(x_c + w_b / 2)
        y2 = int(y_c + h_b / 2)
        color = tuple(map(int, class_colors[cls_gt]))
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
        if cls_gt in params['names']:
            cls_name = params['names'][cls_gt]
        else:
            cls_name = str(cls_gt)
        cv2.putText(img_gt, cls_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw predictions
    if outputs[0] is not None:
        
        # Scale predictions using util.scale like in test function
        det_clone = outputs[0].clone()
        util.scale(det_clone[:, :4], samples[0].shape[1:], shapes[0][0], shapes[0][1])
        
        for det in det_clone.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            color = tuple(map(int, class_colors[cls_id]))
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if cls_id in params['names']:
                cls_name = params['names'][cls_id]
            else:
                cls_name = str(cls_id)
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save side-by-side
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.title('Predictions')
    plt.axis('off')

    save_path = os.path.join(results_dir, f'result_{index}.png')
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def test(args, params, model=None):
    """
    Similar to the 'old version' test function, but also includes code for
    visualizing the first 5 images with bounding boxes side-by-side.

    Returns: (tp, fp, precision, recall, map50, mean_ap)
    """
    # Directory to save results
    results_dir = os.path.join(args.save_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Read validation files
    val_filenames = []
    with open('./Dataset/Yeast/val.txt') as fval:
        for line in fval.readlines():
            line = line.rstrip().split('/')[-1]
            val_filenames.append('./Dataset/Yeast/images/val/' + line)

    # Build val dataset/loader
    dataset = Dataset(val_filenames, args.input_size, params, False)
    loader = data.DataLoader(
        dataset,
        batch_size=8,      # same as old version (for faster eval)
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    # If no model provided, load best
    if model is None:
        ckpt_path = os.path.join(args.save_path, 'best.pt')
        model = torch.load(ckpt_path, map_location='cuda')['model'].float()

    model.half()
    model.eval()

    class_colors = generate_colors(len(params['names']))

    # iou vector for mAP@0.5:0.95
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou = iou_v.numel()

    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    metrics = []

    # We'll track a small counter to do visualization for the first 5 images only
    vis_count = 0

    p_bar = tqdm.tqdm(loader, desc='Evaluating')
    for samples, targets, shapes in p_bar:
        samples = samples.cuda().half() / 255.0  # normalize
        bs = samples.size(0)

        # Forward
        outputs = model(samples)

        # For each batch item, we do NMS
        # But first, scale targets to pixel coords
        _, _, h, w = samples.shape
        targets[:, 2:] *= torch.tensor((w, h, w, h), device=targets.device)
        out = util.non_max_suppression(outputs, 0.25, 0.45)  # Increased confidence threshold, decreased NMS threshold

        # We'll evaluate metrics per item in the batch
        for i in range(bs):
            # All labels for image i
            label_mask = (targets[:, 0] == i)
            labels = targets[label_mask, 1:]  # (class, x, y, w, h)
            detections = out[i]

            # If we want to visualize, do it for up to 5 images only
            if vis_count < 5:
                # Format single-sample versions for visualization
                single_sample = samples[i:i+1]
                single_out = [detections]  # wrap in list for index usage
                single_shapes = [(shapes[i][0], shapes[i][1])]
                
                # Scale ground truth boxes to original image size
                single_gt = []
                if labels.shape[0] > 0:
                    # Scale coordinates back to original image size
                    gt_boxes = labels.clone()
                    h0, w0 = shapes[i][0]  # Original height, width
                    gt_boxes[:, 1:] *= torch.tensor([w0/w, h0/h, w0/w, h0/h], device=gt_boxes.device)
                    for gt in gt_boxes:
                        single_gt.append(torch.tensor([0, gt[0], gt[1], gt[2], gt[3], gt[4]], device=gt.device))
                    single_gt = torch.stack(single_gt)
                else:
                    single_gt = torch.zeros((0, 6), device=labels.device)

                visualize_predictions(
                    single_sample,
                    single_out,
                    single_gt,      # Pass the correctly scaled ground truth boxes
                    single_shapes,
                    params,
                    class_colors,
                    results_dir,
                    vis_count
                )
                vis_count += 1

            # Evaluate metrics (like old code):
            if detections is None or detections.shape[0] == 0:
                # No predictions
                if labels.shape[0]:
                    # if we have labels but no detections
                    correct = torch.zeros(0, n_iou, dtype=torch.bool).cuda()
                    # metrics.append((correct, *[torch.zeros((3, 0), device='cuda')]))
                    metrics.append((
                        correct,
                        torch.zeros(0, device='cuda'),  # conf
                        torch.zeros(0, device='cuda'),  # pred_cls
                        torch.zeros(0, device='cuda')   # target_cls
                    ))
                continue

            # Scale detection boxes back to original shape (like old code)
            det_clone = detections.clone()
            util.scale(det_clone[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Convert label xywh -> xyxy
            if labels.shape[0]:
                # labels: (cls, x, y, w, h)
                # convert to xyxy
                label_boxes = labels.clone()
                label_boxes[:, 1] = labels[:, 1] - labels[:, 3] / 2.0
                label_boxes[:, 2] = labels[:, 2] - labels[:, 4] / 2.0
                label_boxes[:, 3] = labels[:, 1] + labels[:, 3] / 2.0
                label_boxes[:, 4] = labels[:, 2] + labels[:, 4] / 2.0
                util.scale(label_boxes[:, 1:5], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                # Now compute IoU matching
                correct = np.zeros((det_clone.shape[0], iou_v.shape[0]), dtype=bool)
                # Re-check classes
                t_tensor = label_boxes[:, :5].clone().cuda()  # (class, x1, y1, x2, y2)
                # t_tensor: col0=class, col1..4= xyxy
                # For old code: t_tensor[:, 1:] are boxes, t_tensor[:, 0:1] is class
                iou_vals = util.box_iou(t_tensor[:, 1:].cuda(), det_clone[:, :4].cuda())
                correct_class = t_tensor[:, 0:1] == det_clone[:, 5].unsqueeze(0)
                for j in range(len(iou_v)):
                    x = torch.where((iou_vals >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        # shape is (#matches)
                        # we want to keep the highest IoU match for each detection
                        # replicate old logic
                        matches = torch.cat((torch.stack(x, dim=1), iou_vals[x[0], x[1]][:, None]), dim=1)
                        matches = matches.cpu().numpy()  # each row => label_idx, detection_idx, iou
                        # Sort by iou desc
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        # unique detection_idx
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # unique label_idx
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)

                # gather conf, pred_cls, true_cls
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                true_cls = t_tensor[:, 0]
                # store: (correct, conf, pred_cls, target_cls)
                # but we only store repeated for each detection
                # old code aggregated them after the loop
                metrics.append((correct, conf, pred_cls, true_cls))
            else:
                print(f'No labels for image {i}')
                # no labels => no matches
                correct = torch.zeros(det_clone.shape[0], n_iou, dtype=torch.bool).cuda()
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                target_cls = torch.zeros(0, device=pred_cls.device)  # no ground truths
                metrics.append((correct, conf, pred_cls, target_cls))

    # Now compute final metrics
    # metrics is a list of (correct, conf, pred_cls, target_cls) for each image
    # unify them
    if len(metrics):
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        # metrics => [correct, conf, pred_cls, target_cls]
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    else:
        # fallback no data
        tp = 0
        fp = 0
        m_pre = 0
        m_rec = 0
        map50 = 0
        mean_ap = 0

    print(f'Precision: {m_pre:.3f}, Recall: {m_rec:.3f}, mAP50: {map50:.3f}, mAP: {mean_ap:.3f}')
    return (tp, fp, m_pre, m_rec, map50, mean_ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--yaml_file', type=str, default='utils/args_yeast.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--save-path', type=str, default='./results/weights_yeast',
                        help='Directory to save model weights and logs')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Prepare directories
    if args.local_rank == 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    util.setup_seed()
    util.setup_multi_processes()

    with open(args.yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == '__main__':
    main()

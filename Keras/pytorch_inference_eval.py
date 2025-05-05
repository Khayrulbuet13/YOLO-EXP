import os
import torch
import argparse
import tqdm
import numpy as np
import yaml
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.util import non_max_suppression, generate_colors, box_iou, scale, compute_ap, visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='Keras/Dataset/bionano_cellv2',
                        help='Path to dataset root directory')
    parser.add_argument('--save-path', type=str, default='Keras/results/rect_256x128',
                        help='Directory where model weights are saved')
    parser.add_argument('--weights', type=str, default='best.pt',
                        help='Model weights filename')
    parser.add_argument('--img-size', nargs=2, type=int, default=[128, 256],
                        help='Input image size [height width]')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Evaluation batch size')
    return parser.parse_args()

@torch.no_grad()
def evaluate_pytorch_model(args, params, model=None, is_train=False):
    """
    Similar to your main.py test(...) function. Computes (tp, fp, precision, recall, map50, mean_ap).
    Also saves visualizations of bounding boxes to the 'inference' directory.
    """
    # Create inference directory if it doesn't exist
    inference_dir = os.path.join(args.save_path, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    # If is_train=True, we evaluate on 'train.txt', else on 'val.txt'.
    split_txt = 'train.txt' if is_train else 'val.txt'
    txt_path = os.path.join(args.dataset_dir, split_txt)

    # Build dataset
    filenames = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.rstrip().split('/')[-1]
            if is_train:
                # images/train subfolder
                filenames.append(os.path.join(args.dataset_dir, 'images/train', line))
            else:
                # images/val subfolder
                filenames.append(os.path.join(args.dataset_dir, 'images/val', line))

    dataset = Dataset(filenames, (args.img_size[0], args.img_size[1]), params, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.half().to(device)
    model.eval()

    class_colors = generate_colors(len(params['names']))

    metrics = []
    iou_values = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_values.numel()
    
    # Counter for visualization (we'll visualize up to 10 images)
    vis_count = 0
    
    pbar = tqdm.tqdm(loader, desc='Evaluating (train)' if is_train else 'Evaluating (val)')

    for samples, targets, shapes in pbar:
        samples = samples.to(device, non_blocking=True).half() / 255.0
        bs = samples.size(0)
        # Forward
        preds = model(samples)

        # Convert from (class, x, y, w, h) -> scale
        _, _, h, w = samples.shape
        targets[:, 2:] *= torch.tensor((w, h, w, h), device=targets.device)

        # NMS
        out = non_max_suppression(preds, conf_threshold=args.conf_thres, iou_threshold=args.iou_thres)

        for i in range(bs):
            # All labels for image i
            label_mask = (targets[:, 0] == i)
            labels = targets[label_mask, 1:]  # (class, x, y, w, h)
            detections = out[i]

            if detections is None or detections.shape[0] == 0:
                # If no detection
                if labels.shape[0]:
                    # If we have GT
                    correct = torch.zeros(0, n_iou, dtype=torch.bool, device=device)
                    conf = torch.zeros(0, device=device)
                    pred_cls = torch.zeros(0, device=device)
                    true_cls = torch.zeros(0, device=device)
                    metrics.append((correct, conf, pred_cls, true_cls))
                continue

            # Scale detection to original shape
            det = detections.clone()
            scale(det[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])
            
            # Save visualization if we haven't reached the limit (10 images)
            if vis_count < 10:
                # Format single-sample versions for visualization
                single_sample = samples[i:i+1]
                single_out = [detections]  # wrap in list for index usage
                single_shapes = [(shapes[i][0], shapes[i][1])]
                
                # Prepare ground truth boxes for visualization
                single_gt = []
                if labels.shape[0] > 0:
                    # Scale coordinates back to original image size
                    gt_boxes = labels.clone()
                    h0, w0 = shapes[i][0]  # Original height, width
                    pad_w, pad_h = shapes[i][1][1]  # Get padding values
                    
                    class_ids = gt_boxes[:, 0].clone()
                    # Calculate scaled dimensions after removing padding
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
                    
                    for gt in gt_boxes:
                        single_gt.append(torch.tensor([0, gt[0], gt[1], gt[2], gt[3], gt[4]], device=gt.device))
                    single_gt = torch.stack(single_gt)
                else:
                    single_gt = torch.zeros((0, 6), device=labels.device)

                # Generate a unique filename based on dataset split and count
                prefix = 'train' if is_train else 'val'
                visualize_predictions(
                    single_sample,
                    single_out,
                    single_gt,      # Pass the correctly scaled ground truth boxes
                    single_shapes,
                    params,
                    class_colors,
                    inference_dir,
                    f"{prefix}_{vis_count}"
                )
                vis_count += 1

            # Convert GT label xywh->xyxy
            if labels.shape[0]:
                label_boxes = labels.clone()
                # x-> x - w/2, etc
                label_boxes[:, 1] = labels[:, 1] - labels[:, 3] / 2
                label_boxes[:, 2] = labels[:, 2] - labels[:, 4] / 2
                label_boxes[:, 3] = labels[:, 1] + labels[:, 3] / 2
                label_boxes[:, 4] = labels[:, 2] + labels[:, 4] / 2
                # scale to original
                scale(label_boxes[:, 1:5], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = torch.zeros(det.shape[0], n_iou, dtype=torch.bool, device=device)
                # Move label_boxes to the same device as det
                label_boxes = label_boxes.to(device)
                
                # iou matching
                ious = box_iou(label_boxes[:, 1:5], det[:, :4])  # shape (#gt, #det)
                correct_class = label_boxes[:, 0:1] == det[:, 5].unsqueeze(0)

                for j in range(n_iou):
                    # find matches
                    x = torch.where((ious >= iou_values[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat(
                            (torch.stack(x, dim=1), ious[x[0], x[1]][:, None]),
                            dim=1
                        )
                        # each row => [gt_idx, det_idx, iou_val]
                        # sort by iou desc
                        matches = matches[matches[:, 2].argsort(descending=True)]
                        # unique detection_idx
                        _, unique_idx = np.unique(matches[:, 1].cpu().numpy(), return_index=True)
                        matches = matches[unique_idx]
                        # unique label_idx
                        _, unique_idx = np.unique(matches[:, 0].cpu().numpy(), return_index=True)
                        matches = matches[unique_idx]
                        correct[matches[:, 1].long(), j] = True

                conf = det[:, 4]
                pred_cls = det[:, 5]
                true_cls = label_boxes[:, 0]  # Already on the device from earlier move
                metrics.append((correct, conf, pred_cls, true_cls))
            else:
                # no GT
                correct = torch.zeros(det.shape[0], n_iou, dtype=torch.bool, device=device)
                conf = det[:, 4]
                pred_cls = det[:, 5]
                true_cls = torch.zeros(0, device=device)
                metrics.append((correct, conf, pred_cls, true_cls))

    if len(metrics) == 0:
        return (0, 0, 0, 0, 0, 0)

    # unify
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
    # metrics => [correct, conf, pred_cls, target_cls]
    tp, fp, precision, recall, map50, mean_ap = compute_ap(*metrics)
    return (tp, fp, precision, recall, map50, mean_ap)


def main():
    args = parse_args()
    # load your YAML params or define them inline
    # if you have a param file e.g. hyperparams.yaml, load it:
    # with open('data/hyperparams.yaml') as f:
    #     params = yaml.safe_load(f)
    #
    # or define minimal structure for 'names' used in dataset
    params = {
        'names': {0: 'cell'}  # or however many classes you have
    }

    # Create inference directory
    inference_dir = os.path.join(args.save_path, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {inference_dir}")
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(args.save_path, args.weights)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model file {ckpt_path} not found.")
    
    # Load with weights_only=False since we're working with a trusted checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ckpt['model'].float()

    # Evaluate on training
    print("Evaluating training set metrics...")
    (tp_tr, fp_tr, prec_tr, rec_tr, map50_tr, mAP_tr) = evaluate_pytorch_model(args, params, model, is_train=True)
    f1_tr = 2 * prec_tr * rec_tr / (prec_tr + rec_tr + 1e-16)
    print(f"Training set: P={prec_tr:.3f}, R={rec_tr:.3f}, F1={f1_tr:.3f}, mAP@50={map50_tr:.3f}, mAP={mAP_tr:.3f}")

    # Evaluate on validation
    print("\nEvaluating validation set metrics...")
    (tp_val, fp_val, prec_val, rec_val, map50_val, mAP_val) = evaluate_pytorch_model(args, params, model, is_train=False)
    f1_val = 2 * prec_val * rec_val / (prec_val + rec_val + 1e-16)
    print(f"Validation set: P={prec_val:.3f}, R={rec_val:.3f}, F1={f1_val:.3f}, mAP@50={map50_val:.3f}, mAP={mAP_val:.3f}")
    
    print(f"\nBounding box visualizations have been saved to: {inference_dir}")

if __name__ == "__main__":
    main()

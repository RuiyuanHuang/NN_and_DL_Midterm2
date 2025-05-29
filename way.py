import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# 常用库
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch 
import torchvision.ops as ops 

# detectron2 相关库
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes


image_names_to_process = ["new01.jpg", "new02.jpg", "new03.jpg"]

# --- 辅助函数 ---
def visualize_predictions(image, predictions, metadata, title="Predictions"):
    """
    可视化模型的预测结果 (bounding box, instance mask, 类别标签和得分)
    """
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    plt.figure(figsize=(14, 10)) # 稍微调整图像大小以便看得更清楚
    plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_proposals(image, proposals_list: list, metadata, title="Proposals",
                        nms_thresh=0.7, max_proposals_after_nms=15): # 稍微增加显示的proposal数量
    """
    可视化 Mask R-CNN 第一阶段的 proposal box (应用NMS以减少混乱).
    image: 原始图像 (OpenCV BGR HWC format).
    proposals_list: RPN的输出 (List[Instances]), 每个 Instances 对象包含 'proposal_boxes' 和 'objectness_logits'.
    metadata: 模型的元数据.
    title: 可视化图像的标题.
    nms_thresh: NMS 的 IoU 阈值.
    max_proposals_after_nms: NMS 后最多显示的 proposal 数量.
    """
    drawn_image = image.copy()

    if not proposals_list:
        print(f"警告: 未提供 proposal 列表用于 '{title}'")
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (无 proposal 列表)")
        plt.axis("off")
        plt.show()
        return

    proposals_inst = proposals_list[0] 

    if not hasattr(proposals_inst, 'proposal_boxes') or \
       not hasattr(proposals_inst, 'objectness_logits') or \
       len(proposals_inst.proposal_boxes) == 0:
        print(f"警告: 在 Instances 对象中未找到 'proposal_boxes' 或 'objectness_logits'，或数量为0，用于 '{title}'")
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (Instances 对象中无 proposal boxes/logits)")
        plt.axis("off")
        plt.show()
        return

    proposal_boxes_tensor = proposals_inst.proposal_boxes.tensor.to("cpu") 
    objectness_logits = proposals_inst.objectness_logits.to("cpu")   
    scores = objectness_logits 

    keep_indices = ops.nms(proposal_boxes_tensor, scores, nms_thresh)
    
    print(f"Proposals before NMS: {len(proposal_boxes_tensor)}, After NMS (IoU > {nms_thresh}): {len(keep_indices)}")

    nms_boxes = proposal_boxes_tensor[keep_indices]
    nms_scores = scores[keep_indices]

    if len(nms_boxes) > max_proposals_after_nms:
        top_k_indices_after_nms = torch.topk(nms_scores, max_proposals_after_nms).indices
        final_boxes_to_draw = nms_boxes[top_k_indices_after_nms]
    else:
        final_boxes_to_draw = nms_boxes
        
    print(f"Drawing top {len(final_boxes_to_draw)} proposals after NMS and score filtering.")

    for box_tensor in final_boxes_to_draw:
        box = box_tensor.long().numpy() 
        x1, y1, x2, y2 = box
        h, w = drawn_image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x1 < x2 and y1 < y2:
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), (0, 255, 0), 1) 

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    print("Detectron2 version:", detectron2.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available()) 

    # --- 1. 配置和加载 Mask R-CNN 模型 ---
    print("\n--- Setting up Mask R-CNN ---")
    cfg_mask_rcnn = get_cfg()
    cfg_mask_rcnn.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_mask_rcnn.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_mask_rcnn.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg_mask_rcnn.MODEL.DEVICE = "cpu" 
    mask_rcnn_predictor = DefaultPredictor(cfg_mask_rcnn)
    
    try:
        mask_rcnn_metadata = MetadataCatalog.get(cfg_mask_rcnn.DATASETS.TRAIN[0])
    except IndexError:
        print("Warning: Could not get metadata from cfg_mask_rcnn.DATASETS.TRAIN[0]. Using coco_2017_val metadata as a fallback.")
        # Fallback, in case an environment doesn't have the default train dataset registered for this cfg
        try:
            mask_rcnn_metadata = MetadataCatalog.get("coco_2017_val") # Common fallback
        except KeyError:
            print("Error: Fallback metadata 'coco_2017_val' not found. Visualization might lack class names.")
            # Create a dummy metadata if all else fails, to prevent crashes
            from detectron2.data import _get_builtin_metadata
            mask_rcnn_metadata = MetadataCatalog.get("__dummy_coco")
            mask_rcnn_metadata.thing_classes = [f"class_{i}" for i in range(80)] # COCO has 80 classes


    print(f"\n\n--- Processing Images: {image_names_to_process} ---")
    for image_name in image_names_to_process:
        if not os.path.exists(image_name):
            print(f"Warning: Image {image_name} not found. Skipping.")
            continue

        print(f"\n--- Processing Image: {image_name} ---")
        img = cv2.imread(image_name)
        if img is None:
            print(f"Error: Unable to load image {image_name}")
            continue


        print("Running Mask R-CNN for final predictions...")
        mask_rcnn_outputs = mask_rcnn_predictor(img) # DefaultPredictor takes BGR numpy image
        
        print(f"Mask R-CNN found {len(mask_rcnn_outputs['instances'])} instances in {image_name}")
        visualize_predictions(img.copy(), mask_rcnn_outputs, mask_rcnn_metadata, title=f"Mask R-CNN - Predictions on {image_name}")

    print("\n--- All processing finished ---")
import time
import matplotlib.pyplot as plt
import warnings

import torch

from uav_search.api_test import generate_object_description

# Hyperparameters
box_threshold = 0.3
text_threshold = 0.3

def grounded_sam(qwen_processor, qwen_model, dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64, object_description):

    device = "cuda:7"
    warnings.filterwarnings("ignore")  # 关闭所有警告
    
    time_start = time.time()
        
    prompt = f"""
                You are an aerial navigation AI agent's decision module. Your task is to process an input consisting of an image observation and a text description of a target object being searched for. You must perform two specific functions:

                1. Analyze the image to identify objects potentially related to the search target. Generate an object string containing concrete, detectable object descriptions (e.g., "red house", "road", "windmill") separated by commas. Avoid vague or non-detecable terms like "rural place". Be lenient in association - include any object with even minimal relevance to the target.However, don't include any objects that can easily cause errors in segmentation, such as power lines.
                2. After generating the object string, output a semicolon ";", followed by a comma-separated sequence of numbers between [0,1] indicating each object's relevance to the search target. Maintain the same order as the object string. Avoid exact zeros; use small values like 0.1 for weak associations.

                Output format: object1,object2,object3;score1,score2,score3

                Example:
                
                Input: <image> TrafficLight: Geometric vertical structure, dark green metallic texture with signal lights, three stacked circular lenses in standard red-yellow-green arrangement, used for controlling vehicle and pedestrian traffic.
                Output: road,a red house,a blue car;0.6,0.2,0.5

                Process the current input accordingly.Don't output any additional explanation or notes, only provide the formatted response as specified.The input is as follows:{object_description}
            """
    
    prompt_cot = f"""
                You are an aerial navigation AI agent's decision module. Your goal is to identify objects in an image that are relevant to finding a search target and score their relevance.
                Follow these steps precisely:

                Step 1: Analyze the provided image from an aerial perspective. List all concrete, detectable objects you see. Exclude objects that are difficult to segment, like power lines. Call this the "Object List".
                Step 2: The search target is:{object_description} Understand what this target is and where it is typically found.
                Step 3: Compare the "Object List" from Step 1 with the context of the search target from Step 2. For each object in the "Object List", assign a relevance score between 0.1 (very weak association) and 1.0 (very strong association). A high score (e.g., 0.9) means the object is a primary indicator for the target. A low score (e.g., 0.2) means it's a weak, secondary indicator.
                Step 4: Format your final output as a single line.Output format is just like this: object1,object2,object3;score1,score2,score3

                Process the current input accordingly. Don't output your thought process, only provide the final formatted response as specified in Step 5.Remember to keep the object numbers and score numbers consistent and in the same order.
            """
    
    '''
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{prompt_cot}"}
            ]
        }
    ]

    qwen_time_start = time.time()

    inputs = qwen_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    
    # 生成描述
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    qwen_time_end = time.time()
    print("[Planning] MLLM Time taken:", qwen_time_end - qwen_time_start, "seconds")
    print("[Planning] MLLM output:", output_text[0])
    '''
    
    result_dict = {
        "success": False,
        "masks": None,
        "scores": None,
    }
    
    mllm_result = generate_object_description(rgb_base64, object_description)
    if not mllm_result["success"]:
        print("[Planning] MLLM生成对象描述失败")
        return result_dict, []

    try:
        object_part, score_part = mllm_result["response"].split(";")
        #object_part, score_part = output_text[0].split(";")
        text_labels = [[obj.strip() for obj in object_part.split(",")]]
        attraction_scores = [float(score.strip()) for score in score_part.split(",")]
    except Exception as e:
        print(f"[Planning] 解析MLLM响应失败: {str(e)}")
        return result_dict, []

    if len(text_labels[0]) != len(attraction_scores):
        print("[Planning] 解析MLLM响应失败: 对象数量与分数数量不匹配")
        return result_dict, []

    #print("[Planning] text_labels:", text_labels)
    #print("[Planning] attraction_scores:", attraction_scores)

    time_dino_start = time.time()
    
    inputs = dino_processor(images=pil_image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]]
    )

    time_dino_end = time.time()
    print(f"[Planning] GroundingDINO Time taken: {time_dino_end - time_dino_start} seconds")

    result = results[0]

    all_masks = [[] for _ in range(len(text_labels[0]))]
    all_scores = [[] for _ in range(len(text_labels[0]))]

    # 创建一个字典来映射标签到其在text_labels中的索引
    label_to_index = {label: idx for idx, label in enumerate(text_labels[0])}

    time_sam_start = time.time()

    # 遍历检测结果
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        # 检查当前标签是否在text_labels中
        if label in label_to_index:
            if box[2] - box[0] < 20 and box[3] - box[1] < 20:
                continue
            
            idx = label_to_index[label]

            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            input_point = [[(center_x.item(), center_y.item())]]
            
            #print(f"[Planning] Detected {label} (score={score:.3f}) at {box.tolist()}")
            
            inputs = sam_processor(
                images=pil_image,
                input_points=input_point,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = sam_model(**inputs)
            
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            )
            scores = outputs.iou_scores
            
            masks_tensor = masks[0][0]  # shape: [3, H, W]
            scores_tensor = scores.squeeze(1)[0]  # shape: [3]
            
            # 选择最高分的mask
            best_idx = torch.argmax(scores_tensor).item()
            final_mask = masks_tensor[best_idx]
            final_score = scores_tensor[best_idx].item()
            
            # 将结果存储到对应的位置
            all_masks[idx].append(final_mask.cpu().numpy())
            all_scores[idx].append(final_score)
            
            # 保存mask图像
            '''
            plt.figure(figsize=(5, 5))
            plt.imshow(final_mask.cpu().numpy(), cmap='Reds', alpha=0.7)
            plt.title(f"{label}\nIoU: {final_score:.3f}")
            plt.axis('off')
            filename = f"test_images/{label.replace(' ', '_')}_iou_{final_score:.3f}.png"
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()  # 关闭图形，避免内存泄漏
            print(f"[Planning] Saved mask image to {filename}")
            '''
    
    time_end = time.time()
    time_sam_end = time_end
    print(f"[Planning] SAM Time taken: {time_sam_end - time_sam_start} seconds")
    print(f"[Planning] Grounded SAM Time taken: {time_end - time_start} seconds")

    result_dict.update({
        "success": True,
        "masks": all_masks,
        "scores": all_scores
    })
    
    return result_dict, attraction_scores
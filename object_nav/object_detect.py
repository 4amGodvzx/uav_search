from groundingdino.util.inference import load_model, predict, annotate
import cv2
import time

def object_detect(image_np, image_transformed):
    model = load_model("/home/godv/miniconda3/envs/airsim/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/godv/miniconda3/envs/airsim/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    TEXT_PROMPT = "windmill"
    BOX_TRESHOLD = 0.50
    TEXT_TRESHOLD = 0.70

    image_source, image = image_np, image_transformed

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    image_name = str(int(time.time()))
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f"/home/godv/miniconda3/envs/airsim/uav_test/annotated_nav/image_{image_name}.jpg", annotated_frame)
    print("Detected boxes ,logits and phrases:", boxes, logits, phrases)
    return boxes, logits, phrases
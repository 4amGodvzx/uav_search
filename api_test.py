from dashscope import MultiModalConversation
import airsim
import time

def generate_object_description(rgb_base64, object_description) -> dict:
    system_prompt = f"""
                You are an aerial navigation AI agent's decision module. Your task is to process an input consisting of an image observation and a text description of a target object being searched for. You must perform two specific functions:

                1. Analyze the image to identify objects potentially related to the search target. Generate an object string containing concrete, detectable object descriptions (e.g., "red house", "road", "windmill") separated by commas. Avoid vague or non-detecable terms like "rural place". Be lenient in association - include any object with even minimal relevance to the target.However, don't include any objects that can easily cause errors in segmentation, such as power lines.
                2. After generating the object string, output a semicolon ";", followed by a comma-separated sequence of numbers between [0,1] indicating each object's relevance to the search target. Maintain the same order as the object string. Avoid exact zeros; use small values like 0.1 for weak associations.

                Output format: "object1,object2,object3;score1,score2,score3"

                Example:
                
                Input: "<image> TrafficLight: Geometric vertical structure, dark green metallic texture with signal lights, three stacked circular lenses in standard red-yellow-green arrangement, used for controlling vehicle and pedestrian traffic."
                Output: "road,a red house,a blue car;0.6,0.2,0.5"

                Process the current input accordingly.
            """

    system_prompt_cot = f"""
                    You are a sophisticated AI decision module for an autonomous search drone. Your mission is to analyze an aerial surveillance image and a description of a search target, then decide which objects warrant closer investigation. Follow these steps precisely:

                    Step 1: Scene Analysis & Object Identification
                    First, meticulously analyze the provided image. Identify and list all concrete, clearly visible objects.
                        -Criteria: Focus on objects that are suitable for standard object detection and segmentation models.
                        -Exclusions: Explicitly exclude ambiguous or hard-to-segment elements like power lines, wires, or thin branches.
                        -Key Principle: This step is a purely objective inventory of what is present in the scene. Do not consider the search target yet.

                    Step 2: Strategic Relevance Scoring 
                    Next, evaluate each object identified in Step 1 against the search target description: "{object_description}". Assign an "interest score" from 0.1 to 1.0 to each object. This score represents the strategic value of dispatching the drone to investigate that object more closely.
                        -High Score (0.8 - 1.0): Means approaching this object has the highest probability of locating the target or a critical clue (e.g., a road when searching for a truck).
                        -Medium Score (0.4 - 0.7): Indicates the object provides important context, and the target is likely to be nearby (e.g., a cabin or trail when searching for a lost person).
                        -Low Score (0.1 - 0.3): Means the object is part of the general environment with only a weak or indirect connection to the search (e.g., trees when searching for a boat).

                    Step 3: Formatted Output Generation
                    Finally, compile your results into a single string with the specified format. Do not add any other text, explanations, or notes.
                    Output Format: object1,object2,object3;score1,score2,score3

                    Example:
                    Input: <image> A lost hiker, last seen wearing a blue jacket near the Eagle Peak trail. Output: dirt trail,wooden cabin,river,trees;0.9,0.8,0.5,0.2

                    Process the current input accordingly. The input is as follows:
                """
    
    messages=[
                {"role": "system", "content": [system_prompt_cot]},
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/png;base64,{rgb_base64}"},
                        {"text": object_description} # From dataset
                    ]
                }
            ]

    time_start = time.time()

    try:
        response = MultiModalConversation.call(
            api_key="sk-cfccd77d667c4147a11f6e293c1edb62",
            model="qwen-vl-max",
            messages=messages
        )

        if not response or "output" not in response:
            raise RuntimeError("DashScope API response is None or missing 'output'")
        else:
            time_end = time.time()
            print("[Planning] API Time taken:", time_end - time_start, "seconds")
            
            return {
                "success": True,
                "response": response.output.choices[0].message.content[0]["text"]
            }
    except Exception as e:
        print(f"[Planning] Error occurred: {e}")
        return {
            "success": False,
            "response": str(e)
        }
    
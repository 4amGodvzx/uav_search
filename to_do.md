# To Do List

1. Action output of the model
- (Not be used)forward, backward, ascend, descend, rotate left, rotate right, rotate backward, left, right, stop(decided by model stage 2)
- (Will be used)forward, rotate left, rotate right, rotate backward, ascend, descend
    - (Done)How to choose?
    - (Done)Can the step size of the action be decided by the model?
    - (Done)Maybe we should consider fixing the start orientation of the uav?
- (not necessary)the smoothness of the action
2. (Done)The model stage 2: When the object is visible
3. (Done)Obstacle avoidance
    - (Done)Should this be designed in the model or in a standalone module?
4. (Done)The problem of boundary
    - (Done)How to process the input data when uav is near the boundary?
    - (Done)Should we design a mechanism to prevent from overstepping the boundary?
5. (Done)The process logic for the exploration map
    - (Done)Punish observing the same place for too long only when the observed area is near enough from uav.

7. Efficiency
    - (Done)Maybe a smaller model?
    - (Done)Discard detected objects that are too small?(those low-quality detections)
    - (Done)Don't forget to test the used time of each module
    - (Done)If the airsim_lock is necessary? I think it is not necessary.
    - (Done)Multithreading -> Multiprocessing
    - (Done)GroundingDINO in detection is too slow
    - (Done)Getting images from airsim is too slow
    - (Done)Improve the efficiency of map updating
    - (Done)Maybe a smaller SAM model
8. (Done)Multi Environment training
9. (Done)The problem of uav_pose updating
10. (Done)Get the ground truth dataset of the map
11. (Done)Warning: Logical error in exploration map updating
12. (Done)Warning: The error in trace_rays_vectorized()
13. Problem about the deciding LLM : episodical hallucination
14. (not so neccesary)Curriculum learning
15. (Done)Warning: Set the timeout time of airsim api in test code
16. (Done)Warning: The uav prefers to get more attraction reward rather than reach the target quickly
17. (Done)Warning: The attraction reward scale is different in different maps
18. (Done)More training logs
19. (not neccesary)EvalCallBack()
20. (Done) Catastrophic forgetting in RL training
21. Whole test for Reward shaping

# The Policies
1. Map input: Relative location (Alt: absolute location)
2. Navigation policy: Asynchronous (Alt: Synchronous)
3. Action output: Discrete (Alt: Continuous)
4. Obstacle input: Obstacle map (Alt: Depth image)
5. Obstacle avoidance: RL-based (Alt: Rule-based)
6. Training method: Synchronous

# Hyperparameters or method to decide(to be completed)

0. settings.json
    - RGB camera resolution
    - Depth camera resolution
1. multiprocess_test.py
    - attraction/exploration/obstacle map size
    - time ratio between action/planning/detection
    - max action steps in evaluation
2. grounded_sam_test.py
    - threshold for GroundingDINO
    - threshold for low-quality detection filtering
3. map_updating_test.py
    - attraction/exploration map grid size
    - the rule to attraction map replacement
    - exploration map max depth/decay rate
    - exploration map gain
    - (Optional) the forgetting factor of exploration map
4. api_test.py
    - MLLM system prompt
5. detection_test.py
    - detection threshold for GroundingDINO
6. action_model_inputs_test.py
    - input crop size
    - padding value
7. hyperparameters and model choice for MLLM, GroundingDINO, SAM
8. RL training hyperparameters
    - reward design
        - distance reward weight
            - distance reward distance threshold
            - distance reward decay rate
            - step penalty weight
        - attraction reward weight
            - attraction reward distance threshold
            - attraction reward decay rate
            - attraction key point extra reward
        - exploration reward weight
            - exploration reward distance threshold
            - exploration reward/punishment boundary
            - exploration reward decay rate
        - collision/boundary punishment weight
        - success reward weight
            - success distance threshold
    - environment clockspeed
    - data input policy
        - environment number
        - episodes per task
    - curriculum policy
    - max steps per task in training   
    - training policy and hyperparameters
    - policy model structure and hyperparameters

- Warning: RGB image size must be 2 times the depth image size

# Reward Shape testing

1. ppo_num_3:
- TIMESTEPS=300000, max_steps = 100
- REWARD_DISTANCE_THRESHOLD = 30.0 KEY_THRESHOLD = 0.9 KEY_REWARD = 5.0 RATE_CENTER = 0.4
- VIEW_DEPTH = 50.0 VIEW_HEIGHT = 20.0 EXPLORATION_GAIN = 1.0
- W_ATTRACTION = 0.0, W_EXPLORATION = 0.5, W_DISTANCE = 0.0, W_SPARSE = 1.0, STEP_PENALTY = 0.0, ,success reward = 0.0, termination_reward = -50.0
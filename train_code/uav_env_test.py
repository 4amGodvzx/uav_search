from uav_search.train_code.uav_env import AirSimDroneEnv
import traceback
import time

def run_test():
    print("--- Starting UAV Search Environment Test ---")
    
    # 1. 创建环境实例
    # 将在 try...finally 块中创建，以确保无论发生什么错误都能调用 env.close()
    env = None
    try:
        env = AirSimDroneEnv()

        num_episodes_to_test = 3
        max_steps_per_episode = 10 # 每个 episode 只模拟几步，快速检查

        # 2. 循环运行指定数量的 episodes
        for episode in range(num_episodes_to_test):
            print(f"\n==================== Episode {episode}/{num_episodes_to_test} ====================")
            
            # 重置环境，这将加载一个新的任务
            observation, info = env.reset()

            # 3. 在单个 episode 内模拟几个步骤
            for step in range(max_steps_per_episode):
                # 从动作空间中随机采样一个动作
                action = env.action_space.sample()
                
                print(f"--- Step {step}/{max_steps_per_episode}, Action: {action} ---")
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                
                # 如果 episode 结束，则提前退出 step 循环
                if terminated or truncated:
                    print("Episode finished early.")
                    break
            
            print(f"==================== Episode {episode} Finished ====================")
            time.sleep(1) # 短暂暂停，方便查看日志

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        traceback.print_exc()
    finally:
        # 4. 确保在测试结束或出错时关闭环境，从而终止 AirSim 进程
        if env:
            print("\n--- Test finished. Closing environment. ---")
            env.close()

if __name__ == "__main__":
    run_test()

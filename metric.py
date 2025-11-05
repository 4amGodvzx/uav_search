import json
import os
import numpy as np
import argparse

# <<< MODIFIED: 函数签名改变，接收 episode_data 而不是 summaries
def print_and_calculate_metrics(episode_data, group_name, geodesic_distances, start_index=0):
    """
    计算并打印指定摘要列表的各项性能指标。
    
    :param episode_data: 一个包含 {'summary': ..., 'duration': ...} 字典的列表。
    :param group_name: 字符串，用于在打印时标识该组（如 "Seen Tasks"）。
    :param geodesic_distances: 与摘要列表对应的测地线距离列表。
    :param start_index: 该组在全局任务列表中的起始索引，用于计算全局任务序号。
    """
    if not episode_data:
        print(f"\n--- {group_name}: No data to analyze ---")
        return

    # <<< MODIFIED: 从 episode_data 中提取 summaries 和 durations
    summaries = [item['summary'] for item in episode_data]
    durations = [item['duration'] for item in episode_data]

    # --- 1. 数据预处理和逻辑修正 ---
    for s in summaries:
        if s.get("success", False):
            s["oracle_success"] = True

    # --- 指标计算 ---
    total_episodes = len(summaries)
    
    # Success Rate (SR)
    successful_episodes_count = sum(1 for s in summaries if s.get("success", False))
    success_rate = successful_episodes_count / total_episodes if total_episodes > 0 else 0

    collision_count = sum(1 for s in summaries if s.get("termination_reason") == "collision" or s.get("termination_reason") == "out_of_bounds")
    collision_rate = collision_count / total_episodes if total_episodes > 0 else 0

    # Oracle Success Rate (在修正后计算)
    oracle_successful_episodes_count = sum(1 for s in summaries if s.get("oracle_success", False))
    oracle_success_rate = oracle_successful_episodes_count / total_episodes if total_episodes > 0 else 0

    all_path_lengths = [s.get("path_length_actual") for s in summaries if s.get("path_length_actual") is not None]
    avg_path_length = np.mean(all_path_lengths) if all_path_lengths else 0.0
    
    # Navigation Error (计算所有episodes的平均值)
    all_nav_errors = [s.get("navigation_error") for s in summaries if s.get("navigation_error") is not None]
    if all_nav_errors:
        avg_nav_error = np.mean(all_nav_errors)
    else:
        print(f"Warning for '{group_name}': 'navigation_error' key not found. Cannot calculate Average Navigation Error.")
        avg_nav_error = float('nan')
        
    # <<< MODIFIED: 计算平均 episode 时长
    avg_step_duration = np.mean(durations) if durations else 0.0

    # --- SPL (Success weighted by Path Length) - 使用传入的 geodesic_distance 重新计算 ---
    spl_scores = []
    if geodesic_distances and len(geodesic_distances) == len(summaries):
        for i, s in enumerate(summaries):
            if s.get("success", False):
                actual_path_length = s.get("path_length_actual", float('inf'))
                optimal_path_length = geodesic_distances[i]
                spl_score = optimal_path_length / max(optimal_path_length, actual_path_length)
                spl_scores.append(spl_score)
            else:
                spl_scores.append(0.0)
        
        avg_spl = np.mean(spl_scores) if spl_scores else 0.0
    else:
        print(f"Warning for '{group_name}': Geodesic distances not provided or length mismatch. Cannot calculate SPL.")
        avg_spl = float('nan')

    successful_task_indices = [start_index + i + 1 for i, s in enumerate(summaries) if s.get("success", False)]

    # --- 结果打印 ---
    print("\n" + "="*45)
    print(f"      Analysis for: {group_name}")
    print("="*45)
    print(f"Total Episodes:             {total_episodes}")
    print("-" * 45)
    print(f"Success Rate (SR):          {success_rate:.4f} ({successful_episodes_count}/{total_episodes})")
    print(f"Oracle Success Rate:        {oracle_success_rate:.4f} ({oracle_successful_episodes_count}/{total_episodes})")
    print(f"Average Navigation Error:   {avg_nav_error:.4f} (meters, over ALL episodes)")
    print(f"Average SPL:                {avg_spl:.4f} (re-calculated with geodesic distance)")
    print(f"Collision Rate:             {collision_rate:.4f} ({collision_count}/{total_episodes})")
    print(f"Average Path Length:        {avg_path_length:.4f} (meters)")
    # <<< MODIFIED: 打印新的指标
    print(f"Average Step Duration:   {avg_step_duration:.4f} (seconds)")
    print("-" * 45)
    print(f"Successful Task Indices:    {successful_task_indices}")
    print("="*45)


def analyze_logs(log_dir, tasks_file):
    """
    分析日志文件，并使用外部任务文件中的测地线距离来计算SPL。
    """
    try:
        with open(tasks_file, 'r') as f:
            val_tasks = json.load(f)
        all_geodesic_distances = [task['geodesic_distance'] for task in val_tasks]
        print(f"Successfully loaded {len(all_geodesic_distances)} geodesic distances from '{tasks_file}'.")
    except FileNotFoundError:
        print(f"Error: Tasks file '{tasks_file}' not found. Cannot calculate SPL correctly.")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing '{tasks_file}': {e}. Check file format.")
        return

    # <<< MODIFIED: 创建一个新的列表来存储 summary 和 duration
    all_episode_data = []
    filenames = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
    
    if not filenames:
        print(f"No .json log files found in '{log_dir}'.")
        return
        
    print(f"Found {len(filenames)} log files. Reading summaries and step durations...")
    for filename in filenames:
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # 假设包含详细步骤的键是 'step_history'，如果不是请修改这里
                step_history_key = "step_data" 
                
                if "episode_summary" in data and data["episode_summary"] and step_history_key in data:
                    summary = data["episode_summary"]
                    
                    # <<< MODIFIED: 计算当前 episode 的平均每步时长
                    step_durations = [step.get("step_duration", 0) for step in data[step_history_key]]
                    avg_step_duration = np.mean(step_durations) if step_durations else 0.0

                    # <<< MODIFIED: 将 summary 和 duration 一起存储
                    all_episode_data.append({
                        "summary": summary,
                        "duration": avg_step_duration
                    })
                else:
                    print(f"Warning: Skipping {filename}, 'episode_summary' or '{step_history_key}' is missing or empty.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process {filename}. Error: {e}")

    if not all_episode_data:
        print("No valid experiment data found to analyze.")
        return

    # <<< MODIFIED: 使用 all_episode_data 进行后续操作
    total_tasks_from_logs = len(all_episode_data)
    total_tasks_from_file = len(all_geodesic_distances)
    min_tasks = min(total_tasks_from_logs, total_tasks_from_file)
    if total_tasks_from_logs != total_tasks_from_file:
        print(f"\nCRITICAL WARNING: Mismatch between log files ({total_tasks_from_logs}) and tasks in JSON file ({total_tasks_from_file}).")
        min_tasks = min(total_tasks_from_logs, total_tasks_from_file)
        # <<< MODIFIED: 截断 all_episode_data
        all_episode_data = all_episode_data[:min_tasks]
        all_geodesic_distances = all_geodesic_distances[:min_tasks]
        print(f"Proceeding with the first {min_tasks} tasks.")
    
    num_seen = 89
    if min_tasks < num_seen:
        print("Warning: Total tasks are less than 90, 'Unseen' group will be empty.")

    # <<< MODIFIED: 分割新的数据结构
    seen_data = all_episode_data[:num_seen]
    unseen_data = all_episode_data[num_seen:]
    
    seen_geodesic_distances = all_geodesic_distances[:num_seen]
    unseen_geodesic_distances = all_geodesic_distances[num_seen:]

    # <<< MODIFIED: 传递新的数据结构给分析函数
    print_and_calculate_metrics(seen_data, "Seen Tasks", seen_geodesic_distances, start_index=0)
    print_and_calculate_metrics(unseen_data, "Unseen Tasks", unseen_geodesic_distances, start_index=num_seen)
    print_and_calculate_metrics(all_episode_data, "Overall", all_geodesic_distances, start_index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AirSim UAV search experiment logs, with Seen/Unseen split and geodesic distances for SPL.")
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default="all_experiment_logs", 
        help="Directory containing the experiment log files."
    )
    parser.add_argument(
        '--tasks-file',
        type=str,
        default="uav_search/task_map/val_tasks.json",
        help="Path to the JSON file containing task details, including 'geodesic_distance'."
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.log_dir):
        print(f"Error: Log directory not found at '{args.log_dir}'")
    else:
        analyze_logs(args.log_dir, args.tasks_file)


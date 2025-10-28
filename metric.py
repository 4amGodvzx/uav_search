import json
import os
import numpy as np
import argparse

def print_and_calculate_metrics(summaries, group_name, geodesic_distances, start_index=0):
    """
    计算并打印指定摘要列表的各项性能指标。
    
    :param summaries: 一个包含 episode_summary 字典的列表。
    :param group_name: 字符串，用于在打印时标识该组（如 "Seen Tasks"）。
    :param geodesic_distances: 与摘要列表对应的测地线距离列表。
    :param start_index: 该组在全局任务列表中的起始索引，用于计算全局任务序号。
    """
    if not summaries:
        print(f"\n--- {group_name}: No data to analyze ---")
        return

    # --- 1. 数据预处理和逻辑修正 ---
    for s in summaries:
        if s.get("success", False):
            s["oracle_success"] = True

    # --- 指标计算 ---
    total_episodes = len(summaries)
    
    # Success Rate (SR)
    successful_episodes_count = sum(1 for s in summaries if s.get("success", False))
    success_rate = successful_episodes_count / total_episodes if total_episodes > 0 else 0

    # Oracle Success Rate (在修正后计算)
    oracle_successful_episodes_count = sum(1 for s in summaries if s.get("oracle_success", False))
    oracle_success_rate = oracle_successful_episodes_count / total_episodes if total_episodes > 0 else 0
    
    # Navigation Error (计算所有episodes的平均值)
    all_nav_errors = [s.get("navigation_error") for s in summaries if s.get("navigation_error") is not None]
    if all_nav_errors:
        avg_nav_error = np.mean(all_nav_errors)
    else:
        print(f"Warning for '{group_name}': 'navigation_error' key not found. Cannot calculate Average Navigation Error.")
        avg_nav_error = float('nan')

    # --- SPL (Success weighted by Path Length) - 使用传入的 geodesic_distance 重新计算 ---
    spl_scores = []
    # 确保传入的测地线距离列表与摘要列表长度匹配
    if geodesic_distances and len(geodesic_distances) == len(summaries):
        for i, s in enumerate(summaries):
            if s.get("success", False):
                # 从日志中获取智能体实际行走的路径长度
                actual_path_length = s.get("path_length_actual", float('inf'))
                # 从传入的列表中获取最优路径长度（测地线距离）
                optimal_path_length = geodesic_distances[i]
                
                # 计算SPL： Si * (Li / max(Li, Pi))
                # 其中Si是成功标志, Li是最优路径, Pi是实际路径
                spl_score = optimal_path_length / max(optimal_path_length, actual_path_length)
                spl_scores.append(spl_score)
            else:
                # 失败的任务，SPL为0
                spl_scores.append(0.0)
        
        avg_spl = np.mean(spl_scores) if spl_scores else 0.0
    else:
        print(f"Warning for '{group_name}': Geodesic distances not provided or length mismatch. Cannot calculate SPL.")
        avg_spl = float('nan')

    # 获取成功任务的序号
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
    print("-" * 45)
    print(f"Successful Task Indices:    {successful_task_indices}")
    print("="*45)


def analyze_logs(log_dir, tasks_file):
    """
    分析日志文件，并使用外部任务文件中的测地线距离来计算SPL。
    """
    # --- 1. 加载外部任务文件以获取测地线距离 ---
    try:
        with open(tasks_file, 'r') as f:
            val_tasks = json.load(f)
        # 提取所有任务的测地线距离到一个列表中
        all_geodesic_distances = [task['geodesic_distance'] for task in val_tasks]
        print(f"Successfully loaded {len(all_geodesic_distances)} geodesic distances from '{tasks_file}'.")
    except FileNotFoundError:
        print(f"Error: Tasks file '{tasks_file}' not found. Cannot calculate SPL correctly.")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing '{tasks_file}': {e}. Check file format.")
        return

    # --- 2. 加载所有实验日志摘要 ---
    all_summaries = []
    filenames = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
    
    if not filenames:
        print(f"No .json log files found in '{log_dir}'.")
        return
        
    print(f"Found {len(filenames)} log files. Reading summaries...")
    for filename in filenames:
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if "episode_summary" in data and data["episode_summary"]:
                    all_summaries.append(data["episode_summary"])
                else:
                    print(f"Warning: Skipping {filename}, 'episode_summary' is missing or empty.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process {filename}. Error: {e}")

    if not all_summaries:
        print("No valid experiment summaries found to analyze.")
        return

    # --- 3. 检查数据一致性并分割数据集 ---
    total_tasks_from_logs = len(all_summaries)
    total_tasks_from_file = len(all_geodesic_distances)
    min_tasks = min(total_tasks_from_logs, total_tasks_from_file)
    if total_tasks_from_logs != total_tasks_from_file:
        print(f"\nCRITICAL WARNING: Mismatch between log files ({total_tasks_from_logs}) and tasks in JSON file ({total_tasks_from_file}).")
        # 决定是继续还是中止，这里选择继续并使用较小的值作为任务总数
        min_tasks = min(total_tasks_from_logs, total_tasks_from_file)
        all_summaries = all_summaries[:min_tasks]
        all_geodesic_distances = all_geodesic_distances[:min_tasks]
        print(f"Proceeding with the first {min_tasks} tasks.")
    
    num_seen = 89
    if min_tasks < num_seen:
        print("Warning: Total tasks are less than 90, 'Unseen' group will be empty.")

    seen_summaries = all_summaries[:num_seen]
    unseen_summaries = all_summaries[num_seen:]
    
    seen_geodesic_distances = all_geodesic_distances[:num_seen]
    unseen_geodesic_distances = all_geodesic_distances[num_seen:]

    # --- 4. 分别计算和打印结果 ---
    print_and_calculate_metrics(seen_summaries, "Seen Tasks", seen_geodesic_distances, start_index=0)
    print_and_calculate_metrics(unseen_summaries, "Unseen Tasks", unseen_geodesic_distances, start_index=num_seen)
    print_and_calculate_metrics(all_summaries, "Overall", all_geodesic_distances, start_index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AirSim UAV search experiment logs, with Seen/Unseen split and geodesic distances for SPL.")
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default="random_experiment_logs", 
        help="Directory containing the experiment log files."
    )
    # 新增命令行参数，用于指定任务文件
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

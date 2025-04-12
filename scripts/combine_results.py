import json
import glob

def combine_results(log_dir="./experiments/logs"):
    result_files = glob.glob(f"{log_dir}/results_*.json")
    total_episodes = 0
    total_successes = 0

    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            total_episodes += data["total_episodes"]
            total_successes += data["total_successes"]
            print(f"Loaded results from {file}: {data['total_successes']} successes out of {data['total_episodes']} episodes.")

    overall_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes > 0 else 0.0
    print(f"\nOverall success rate: {overall_success_rate * 100:.2f}% ({total_successes}/{total_episodes})")

if __name__ == "__main__":
    combine_results()

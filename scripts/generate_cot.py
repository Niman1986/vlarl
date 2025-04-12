import os
import json
import time
import pickle
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.primitive_movements import get_move_primitives_episode


class Gemini:
    def __init__(self):
        api_key = "AIzaSyB56l5kPvNK3IbegqE9jiRrBgFU0E8DG2k"
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def safe_call(self, f):
        while True:
            try:
                res = f()
                return res
            except ResourceExhausted:
                time.sleep(5)

    def generate(self, prompt):
        chat = self.safe_call(lambda: self.model.start_chat(history=[]))
        response = self.safe_call(lambda: chat.send_message(prompt).text)

        for i in range(8):
            if response.endswith("}") or response.endswith("```"):
                print(f"n_retries: {i}")
                return response

            response = response + self.safe_call(lambda: chat.send_message("Truncated, please continue.").text)

        return response


def build_prompt(features, language_instruction, image_path=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features = structured_features + f'    {i}: {"{"}\n'

            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation'
        )

    assert image_path is not None
    image = genai.upload_file(image_path)

    text_prompt = f"""# Annotate the Training Trajectory with Reasoning

## Scene Description

The robot is operating in the environment visualized in the provided image.

## Annotate the Training Trajectory with Reasoning

### Specification of the Experimental Setup

As an expert in reinforcement learning, you've trained an optimal policy for controlling a robotic arm. The robot successfully completed a task specified by the instruction: "{language_instruction}". To accomplish this, the robotic arm executed a sequence of actions as follows:

{features_desc}

{structured_features}

### Objective

Your task is to annotate the given trajectory with detailed reasoning. For each step, explain not only which action should be chosen but also why that action is justified. Be descriptive and include all relevant information.

- **Task to complete:** Describe the activity and objects involved, including their locations.
- **High-level steps:** Explain the necessary high-level movements and their intervals.
- **Justification:** Provide reasoning for each movement, considering object locations, obstacles, and other factors.

### Overview of the Task

Begin by thoroughly describing the task. Include:

- The activity and objects the robotic arm interacts with.
- Relative locations of these objects.
- High-level movements executed and their step intervals.
- Justification for each high-level movement.

This description should align with the trajectory specified by the `trajectory_features` dictionary.

### Reasoning for Each Step

For each step, describe:

- Current progress.
- Relevant objects and plan for next actions.
- Gradual breakdown from high-level to finer details.

Ensure consistency with the task and executed trajectory.

### Task Summary

Break down the task as follows:

- **Task Description**
- **High-level Movements and Plan**
- **Step-by-step Reasoning**

### Output Format

Make sure to provide a JSON-formatted dictionary. Combine steps with the same movement for simplicity. Use the following structure:

```json
{{
    \"task\": \"<task description>\",
    "plan": [
        {{
            \"stage\": <stage number starting from 0>,
            \"high_level_step\": \"<high level step description>\",
            \"interval\": [<start step>, <end step>],
            \"reason\": \"<reason for executing the high level step>\"
        }},
        ...
        {{
            \"stage\": <stage number starting from 0>,
            \"high_level_step\": \"<high level step description>\",
            \"interval\": [<start step>, <end step>],
            \"reason\": \"<reason for executing the high level step>\"
        }}
    ],
    \"steps\": [
        {{
            \"interval\": [<start step>, <end step>],
            \"stage\": <stage number corresponding to the high level step>,
            \"move\": \"<primitive movement>\",
            \"justification\": \"<reason for executing the primitive movement and other relevant information>\"
        }},
        ...
        {{
            \"interval\": [<start step>, <end step>],
            \"stage\": <stage number corresponding to the high level step>,
            \"move\": \"<primitive movement>\",
            \"justification\": \"<reason for executing the primitive movement and other relevant information>\"
        }}
    ],
}}
```
"""

    return [image, text_prompt]


def get_reasoning_dict(features, metadata, lm):
    language_instruction = metadata["language_instruction"]
    # caption = metadata["caption"] if "caption" in metadata.keys() else None
    image_path = metadata["image_path"]

    prompt = build_prompt(features, language_instruction, image_path=image_path, list_only_moves=True)

    reasoning_output = lm.generate(prompt)

    os.makedirs("reasoning_workdir", exist_ok=True)
    reasoning_text_path = f"reasoning_workdir/reasoning_{metadata['episode_id']}.txt"
    with open(reasoning_text_path, "w") as f:
        f.write(reasoning_output)
    
    try:
        reasoning_output = json.loads(reasoning_output.strip('```json\n').strip('```'))
        print("################################SUCCESS###############################################")
        print("Generated response for episode_id:", metadata["episode_id"])
        print("######################################################################################\n")
    except json.JSONDecodeError:
        reasoning_output = dict(path=reasoning_text_path)
        print("################################FAILURE###############################################")
        print("Failed to generate response for episode_id:", metadata["episode_id"])
        print("Saved response to file:", reasoning_text_path)
        print("######################################################################################\n")
    return reasoning_output


def build_single_reasoning(episode_id, builder, lm):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    ft = dict()

    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]

    move_primitives = get_move_primitives_episode(episode)
    ft["move_primitive"] = [move[0] for move in move_primitives]

    mt = {
        "episode_id": episode_id,
        "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
        "n_steps": len(episode["steps"]),
        "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
    }
    image = next(iter(episode["steps"]))["observation"]["image"].numpy()
    image_tmp_path = f"reasoning_workdir/image_{mt['episode_id']}.png"
    os.makedirs("reasoning_workdir", exist_ok=True)
    Image.fromarray(image).save(image_tmp_path)
    mt["image_path"] = image_tmp_path

    reasoning = get_reasoning_dict(ft, mt, lm)
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}

    return entry


def generate_reasonings(builder, episode_ids, reasonings, save_path="reasonings.pkl"):
    lm = Gemini()

    pbar = tqdm(episode_ids)
    for i in pbar:
        pbar.set_description(f"Processing episode_id: {i}")
        try:
            entry = build_single_reasoning(i, builder, lm)
        except Exception as e:
            print(f"Failed to process episode_id: {i}")
            continue
        reasonings.append(entry)
        
        reasonings = sorted(reasonings, key=lambda x: x["metadata"]["episode_id"])
        with open(save_path, "wb") as out_f:
            pickle.dump(reasonings, out_f)


def format_reasonings(reasonings):
    formatted_reasonings = {}
    valid_num, total_num = 0, 0
    valid_ids = []
    for episode_idx, entry in enumerate(reasonings):
        metadata = entry["metadata"]
        file_path = metadata["file_path"]
        n_steps = metadata["n_steps"]

        reasoning = entry["reasoning"]
        task = reasoning.get("task", None)
        plans = reasoning.get("plan", None)
        steps = reasoning.get("steps", None)
        
        per_step_reasonings = dict()
        if task is None or plans is None or steps is None:
            valid = False
            print(f"Epsiode_id: {metadata['episode_id']}. Missing task, plans, or steps.")
            formatted_reasonings[file_path] = dict()
        
        else:
            plans_str = ""
            for subplan in plans:
                plans_str += f"Stage{subplan['stage']}: {subplan['high_level_step']}\n"
                if isinstance(subplan["interval"], list) and len(subplan["interval"]) == 2:
                    subplan["interval"] = range(subplan["interval"][0], subplan["interval"][1] + 1)
            
            for step in steps:
                if isinstance(step["interval"], list) and len(step["interval"]) == 2:
                    step["interval"] = range(step["interval"][0], step["interval"][1] + 1)
            
            total_steps = []
            for step in steps:
                for i in step["interval"]:
                    total_steps.append(i)
                    per_step_reasoning = dict()
                    per_step_reasoning["task"] = task
                    per_step_reasoning["plan"] = plans_str
                    subplan_stage = step["stage"]
                    high_level_step = plans[subplan_stage]["high_level_step"]
                    subplan_str = f"Current stage: {subplan_stage}, High-level step: {high_level_step}"
                    subplan_reason = plans[subplan_stage]["reason"]
                    per_step_reasoning["subtask"] = subplan_str
                    per_step_reasoning["subtask_reasoning"] = subplan_reason
                    per_step_reasoning["move"] = step["move"]
                    per_step_reasoning["move_reason"] = step["justification"]
                    per_step_reasonings[i] = per_step_reasoning
            total_steps = sorted(list(set(total_steps)))
            if len(total_steps) != n_steps:
                valid = False
                print(f"Epsiode_id: {metadata['episode_id']}. Missing reasoning for some steps.")
                print(f"Total steps: {n_steps}, Reasoned steps: {len(total_steps)}")
                formatted_reasonings[file_path] = dict()
            else:
                valid = True
                valid_ids.append(episode_idx)
                formatted_reasonings[file_path] = {"reasoning": per_step_reasonings}
        print(f"Valid: {valid}, episode_id: {metadata['episode_id']}")
        valid_num += valid
        total_num += 1
    
    print(f"Valid: {valid_num}, Total: {total_num}")
    return formatted_reasonings, valid_ids


if __name__ == "__main__":
    builder = tfds.builder("libero_10_no_noops", data_dir="./data/modified_libero_rlds/")
    len_train = builder.info.splits["train"].num_examples
    print("len_train:", len_train)
    
    if os.path.exists("reasonings.pkl"):
        with open("reasonings.pkl", "rb") as in_f:
            reasonings_old = pickle.load(in_f)
            # Remove entries that failed to generate reasoning
            reasonings = []
            for i, entry in enumerate(reasonings_old):
                if not "path" in entry["reasoning"].keys():
                    reasonings.append(entry)
            existing_ids = [entry["metadata"]["episode_id"] for entry in reasonings]
            print("Existing reasonings:", len(existing_ids))
            episode_ids = list(set(range(len_train)) - set(existing_ids))
    else:
        reasonings = []
        episode_ids = list(range(len_train))
    
    generate_reasonings(builder, episode_ids=episode_ids, reasonings=reasonings)
    
    with open("reasonings.pkl", "rb") as f:
        reasonings = pickle.load(f)
    reasonings_formatted, valid_ids = format_reasonings(reasonings)
    reasonings = [reasonings[i] for i in valid_ids]
    with open("reasonings.pkl", "wb") as f:
        pickle.dump(reasonings, f)
    with open("reasonings_formatted.pkl", "wb") as f:
        pickle.dump(reasonings_formatted, f)
    with open("reasonings_formatted.json", "w") as f:
        json.dump(reasonings_formatted, f, indent=4)

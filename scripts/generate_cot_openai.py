import argparse
import base64
import os
import json
import pickle
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds
from openai import OpenAI

from scripts.primitive_movements import get_move_primitives_episode


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class GPT:
    def __init__(self, model, base_url):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )
        self.model = model

    def generate(self, prompt):
        if isinstance(prompt, list):
            image_path, text_prompt = prompt
        else:
            image_path = None
            text_prompt = prompt
    
        if image_path is not None:
            base64_image = encode_image(image_path)
            response = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":  f"data:image/jpeg;base64,{base64_image}"
                                    },
                            }
                        ]
                    },
                ],
            )
        else:
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_prompt,
                            }
                        ]
                    },
                ],
            )
        
        return response.to_dict()['choices'][0]['message']['content']


def build_prompt(features, language_instruction, image_path=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

    print(f"Lenght: {len(features[keys[0]])}")
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

    return [image_path, text_prompt]


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


def generate_reasonings(lm, builder, episode_ids, reasonings, save_path="reasonings.pkl"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-vl-plus")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    args = parser.parse_args()
    
    lm = GPT(model=args.model, base_url=args.base_url)
    
    builder = tfds.builder("libero_10_no_noops", data_dir="./data/modified_libero_rlds/")
    len_train = builder.info.splits["train"].num_examples
    print("len_train:", len_train)
    save_path = f"reasonings-{args.model}.pkl"
    if os.path.exists(save_path):
        with open(save_path, "rb") as in_f:
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
    
    generate_reasonings(lm, builder, episode_ids=episode_ids, reasonings=reasonings, save_path=save_path)

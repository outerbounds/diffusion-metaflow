import random
from metaflow import Task, Flow, Step, namespace
import os
from utils import create_prompt
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    pass

def img_reshape(img, width=300, height=300):
    img = img.resize((width,height))
    img = np.asarray(img)
    return img
    
def create_image_grid(run_metadata, prompt, style, rows=4, cols=4,width=150, height=150,randomly_selected=False):
    
    selected_values = []
    
    for val in run_metadata:
        
        if prompt is not None:
            if prompt.lower() in val['prompt'].lower():
                selected_values.append(val)
        if style is not None:
            if style.lower() in val['style'].lower():
                selected_values.append(val)
    
    if len(selected_values) == 0:
        print("No Images could be filtered for prompt:%s style:%s" % (prompt, style))
        return 
    img_count = 0
    if randomly_selected:
        selected_values = random.choices(
            selected_values, k=min(len(selected_values), rows*cols)
        )

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
    for i in range(rows):
        for j in range(cols):        
            task_obj = Task(selected_values[img_count]['task_pathspec'])
            image = task_obj[selected_values[img_count]['img_val']].data
            if img_count < rows*cols or img_count < len(selected_values):
                axes[i, j].set_title(
                    create_prompt(selected_values[img_count]['prompt'] , selected_values[img_count]['style'])
                )
                axes[i, j].axis('off')
                axes[i, j].imshow(
                    img_reshape(
                        image, width=width, height=height
                    ))
                img_count+=1


def get_successful_run_prompts(max_runs = None):
    namespace(None)
    flow = Flow('DynamicPromptsToImages')
    success_runs = []
    _idx = 0
    for r in flow.runs():
        if max_runs is not None and _idx >= max_runs:
            break
        if r.successful:
            success_runs.append(r)
        _idx+=1
    
    # extract all unique runs
    core_step_pathspecs = []
    for r in success_runs:
        if "None" in r['generate_images'].origin_pathspec:
            core_step_pathspecs.append(r['generate_images'].pathspec)
        else:
            if r['generate_images'].origin_pathspec not in core_step_pathspecs:
                core_step_pathspecs.append(r['generate_images'].origin_pathspec)
    
    # Extract all the prompt values into a json list. 
    prompt_values = []
    for steptsp in list(set(core_step_pathspecs)):
        mf_step = Step(steptsp)
        seed_value = None
        for task in mf_step:
            if seed_value is None:
                seed_value = task.data.seed
            run_id = task.pathspec.split('/')[1]
            image_indx = task.data.image_index            
            prompt_values.extend([
                dict(
                    prompt=prompt,
                    style=style,
                    img_val=img_val,
                    task_pathspec = task.pathspec,
                    run_id=run_id, 
                    seed=seed_value
                ) for prompt, style, img_val in image_indx
            ])
    
    return prompt_values
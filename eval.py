import jsonlines
from base import call_api, get_client_models, expr_to_shot, check_answer
from omegaconf import OmegaConf
import sys
from collections import defaultdict
import numpy as np
import anyio
import wandb
import random
from itertools import groupby

if __name__ == '__main__':    
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    yml_cfg = OmegaConf.load(yaml_path)
    cli_cfg = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(yml_cfg, cli_cfg)
    client, models = get_client_models(cfg)
    
    random.seed(cfg.seed)
    
    run = wandb.init(
        project="Mathador",
        config=OmegaConf.to_container(cfg),
        mode='online' if cfg.use_wandb else 'disabled'
    )
    run.save("./*.py")
    run.save("./*.yaml")
    dataset_artifact = wandb.Artifact('dataset', type='dataset')
    dataset_artifact.add_file(cfg.dataset)
    run.log_artifact(dataset_artifact)
    
    dataset = []
    with jsonlines.open(cfg.dataset) as reader:
        for line in reader:
            dataset.append(line)
    if cfg.shuffle:
        random.shuffle(dataset)
            
    cfg.count = min(cfg.count, len(dataset)) if cfg.count != -1 else len(dataset)
        
    problem_index = 0
    shot_index = len(dataset) - 1
    data = defaultdict(list)
    shots_buffer = []
    call_buffer = []
    reason_count = defaultdict(int)
    while problem_index < shot_index and problem_index < cfg.count:
        if len(shots_buffer) < cfg.shots:
            shots_buffer.append(dataset[shot_index])
            shot_index -= 1
            continue
        
        shots = '\n'.join([
            expr_to_shot(
                s['base_numbers'], s['target'], s['simple_solution'], s['simple_solution_score'], s['mathador_solution'], s['mathador_solution_score']) 
            for s in shots_buffer])
        shots_buffer.clear()
        example = dataset[problem_index]
        message = cfg.prompt.format(shots=shots, target=example['target'], base_numbers=', '.join(map(str, example['base_numbers'])))
        for _ in range(cfg.topk):
            call_buffer.append({'message': message, 'p_idx': problem_index})
        problem_index += 1
        if len(call_buffer) + cfg.topk < cfg.batch_size and (problem_index < shot_index and problem_index < cfg.count):
            continue
        r = anyio.run(call_api, client, models, call_buffer)
        # r = [{**item, 'model': 'llama-2-7b', 'answer': dataset[item['p_idx']]['generated_answer']} for item in call_buffer]
        for item in r:
            example = dataset[item['p_idx']]
            print('-'*30)
            print('Model:', item['model'], '\n\n')
            print(item['answer'], '\n\n')
            score, reason = check_answer(item['answer'], example['target'], example['base_numbers'])
            item['score'], item['reason'] = score, reason
            item.update(example)
            if reason != 'correct':
                reason_count[reason] += 1
            print('Score:', score)
        agg_r = []
        for g, l in groupby(r, key=lambda x: x['p_idx']):
            l = list(l)
            avg_score = max(x['score'] for x in l)
            agg_r.append({**l[0], 'score': avg_score})
        r = agg_r
        for item in r:
            data[item['model']].append(item)
        call_buffer.clear()
        
            
    wandb.log({'data': data}, step=0)
    mean_dict = {}
    for k, v in data.items():
        normalized_scores = [x['score']/18 for x in v]
        mean_dict[k] = np.mean(normalized_scores)
    print(mean_dict)
    run.summary['mean'] = mean_dict
    run.summary['reason_percentage'] = {k: v/sum(reason_count.values()) for k, v in reason_count.items()}


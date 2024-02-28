import test_lib as lib

import models.heuristic_models as H
from pathlib import Path
import json
import negmas
from multiprocessing import Pool

def mulprocess_test(opp_name, domain_ufun, noise_sigma):
    

    history_folder = Path(f'bidding_history/noise_{noise_sigma}') / opp_name

    result_folder = Path('results/heuristic_model/')
    if not result_folder.exists():
        result_folder.mkdir()

    result_folder = result_folder / f'noise_{noise_sigma}'
    if not result_folder.exists():
        result_folder.mkdir()

    result_folder = result_folder / opp_name
    if not result_folder.exists():
        result_folder.mkdir()

    d = domain_ufun[0]
    d_path = Path(d)
    result_file = result_folder / (d_path.stem + '.json')
    if result_file.exists():
        return
    history_file = history_folder / (d_path.stem + '.json')
    bidding_history = json.load(history_file.open(mode = 'r'))
    print(d_path.stem)
        ## load file
    ufun_names = domain_ufun[1]
    scene = negmas.inout.load_genius_domain(d, utility_file_names=ufun_names, ignore_discount=True, ignore_reserved=True, safe_parsing = False)
    issues = scene.issues
    ufun1 = scene.ufuns[0]
    ufun2 = scene.ufuns[1]
    if ufun1.issues[0].type != 'categorical':
        issues = issues[1:]
        ufun1 = negmas.LinearUtilityAggregationFunction(values = ufun1.values[1:], weights = ufun1.weights[1:],issues = issues)
        ufun2 = negmas.LinearUtilityAggregationFunction(values = ufun2.values[1:], weights = ufun2.weights[1:],issues = issues)
    ufun1 = ufun1.normalize()
    ufun2 = ufun2.normalize()
    outcomes = ufun1.outcome_space.enumerate()

    #####

    ins_opp_models = [H.CUHKAgentValueModel(ufun1)]

    evaluation_matrix = lib.evaluation_matrix_pearson(ufun_opp = ufun2, ufun_own = ufun1, outcomes = outcomes)
    up_evl = evaluation_matrix.Multi_OM_update_from_history_10(ins_opp_models = ins_opp_models, history = bidding_history)

    json.dump(up_evl, result_file.open(mode = 'w'))



if __name__ == '__main__':

    max_processes = 30    
    Pool_run = Pool(max_processes)

    domain4train_file = Path('scenarios/test_use.json')
    domains_used = json.load(domain4train_file.open(mode = 'r'))
    noise_sigma_S = [0, 0.005]

    opp_names = ['Time1', 'Time2', 'Time5',
                'Reserveconceding0.3', 'Reserveconceding0.5','Reserveconceding0.9']
    
    results = []
    for noise_sigma in noise_sigma_S:
        for opp_name in opp_names:
            for domain_ufun in domains_used.items():
                print(f'num_parallel_{opp_name}_{domain_ufun[0]}_start_!')

                result = Pool_run.apply_async(mulprocess_test, args=(opp_name, domain_ufun, noise_sigma))
                results.append(result)
                while len(results) >= max_processes:
                    results = [r for r in results if not r.ready()]
    
    Pool_run.close()
    Pool_run.join()

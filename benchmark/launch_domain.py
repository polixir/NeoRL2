import os
import time
import argparse

ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--algo', type=str, help='select from `bc`, `cql`, `edac`, `mcq`, `td3bc`, `mopo`, `combo`, `rambo`, `mobile`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    parser.add_argument('--domain', type=str, default=None, help='domain of tasks')
    args = parser.parse_args()

    if not os.path.exists(ResultDir): os.makedirs(ResultDir)

    if args.domain is None:
        ''' run a single algorithm on all the tasks '''
        domains = ['Pipeline', 'Simglucose', 'RocketRecovery', 'RandomFrictionHopper', 'DMSD', 'Fusion', 'SafetyHalfCheetah']
    else:
        domains = [args.domain]
        

    for domain in domains:
        os.system(f'python pretrain_model.py --domain {domain}  --algo bc_model')
        os.system(f'ray stop')
        time.sleep(20)  # wait ray to release the resource
        for algo in ["bc", "cql", "edac", "mcq", "td3bc", "mopo", "combo", "rambo", "mobile"]:
            if args.address is not None:
                os.system(f'python launch_task.py --domain {domain}  --algo {algo} --address {args.address}')
            else:
                os.system(f'python launch_task.py --domain {domain}  --algo {algo}')
            os.system(f'ray stop')
            time.sleep(20)  # wait ray to release the resource
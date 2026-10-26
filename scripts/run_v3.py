"""Run or verify V3 in a fresh ignored output directory."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fedmaint.models.contracts import ExperimentConfig
from fedmaint.models.experiments import run,verify_experiment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','verify'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v3.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v3/reference')
    args=parser.parse_args()
    if args.action=='run':
        config=ExperimentConfig.model_validate_json(args.config.read_text(encoding='utf-8'))
        run(ROOT,config,args.output)
    else:
        print(verify_experiment(ROOT,args.output))

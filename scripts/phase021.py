"""Bounded optimization stages, explicit freeze, and exactly-once final evaluation."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from fedmaint.optimization.common import load_config
from fedmaint.optimization.audit import diagnostics,verify_preservation
from fedmaint.optimization.selection import run_source
from fedmaint.optimization.azure import run_azure
from fedmaint.optimization.workflow import final_fit,seal,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["diagnostics","source","azure","fit","freeze","evaluate","preservation"])
    args=parser.parse_args()
    output=ROOT/"outputs/phase021"
    output.mkdir(parents=True,exist_ok=True)
    config=load_config(ROOT)
    functions={"diagnostics":diagnostics,"source":run_source,"azure":run_azure,"fit":final_fit,"freeze":seal,"evaluate":evaluate}
    result=verify_preservation(ROOT,output) if args.stage=="preservation" else functions[args.stage](ROOT,output,config)
    print(f"{args.stage}: {result.get('status','completed')}",flush=True)


if __name__=="__main__":
    main()

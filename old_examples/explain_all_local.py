import argparse
import json
from src.simba.explain import Simba
from src.simba.cli.main import explain_instance
import os


def main():
    parser = argparse.ArgumentParser(description="Run SIMBA explain_instance for all schools.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for output")
    parser.add_argument("--model", type=str, required=True, help="Prefix for output")
    args = parser.parse_args()

    # load school coordinates
    with open(f"/scratch/hbaier/data_jsons/{args.prefix}_coords.json", "r") as f:
        phl_coords = json.load(f)

    test_path = os.path.join(args.ckpt_dir, "test_indices.txt")
    with open(test_path, "r") as f:
        test_names = f.read().splitlines()
    print("Num test: ", len(test_names))

    print("Num 1: ", len(phl_coords))
    phl_coords = {k:v for k,v in phl_coords.items() if k in test_names}
    print("Num 2: ", len(phl_coords))

    for imname, im_coords in phl_coords.items():
        try:
            coords_txt = f"{im_coords[1]},{im_coords[0]}"
            print(coords_txt)
            explain_instance(
                model = args.model,
                ckpt_dir=args.ckpt_dir,
                dataset="json",
                data_root="/scratch/hbaier/data_jsons",
                prefix=args.prefix,
                distances="0.25,1,2,5,10,15",
                input_coords=coords_txt,
            )
        except Exception as e:
            print("Error: ", e)


if __name__ == "__main__":
    main()

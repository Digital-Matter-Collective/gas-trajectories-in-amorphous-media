import argparse
import json
import subprocess
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path

from utils.logging_setup import setup_logging
from utils.utils import get_float_img_pattern, kprint

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract PNM from binary images"
    )
    parser.add_argument(
        "raw_img_path", type=Path, help="Data directory (contains raw_images)"
    )
    parser.add_argument(
        "pnm_path", type=Path, help="Output directory (contains pnm)"
    )
    parser.add_argument("extractor", type=Path, help="Path to extractor binary")
    parser.add_argument(
        "config", type=Path, help="Path to extractor config JSON"
    )

    args = parser.parse_args()

    path_to_extractor = str(args.extractor)
    path_to_config = str(args.config)
    pnm_path = str(args.pnm_path)
    raw_img_path = str(args.raw_img_path)

    Path(pnm_path).mkdir(parents=True, exist_ok=True)

    # Opening JSON file
    with open(path_to_config) as f:
        # returns JSON object as
        # a dictionary
        jconfig = json.load(f)

    onlyfiles = [
        f
        for f in listdir(raw_img_path)
        if isfile(join(raw_img_path, f)) and '.raw' in f
    ]

    pattern = get_float_img_pattern()

    for i, file in enumerate(onlyfiles):
        start_time = time.time()
        match = pattern.match(file)
        if not match:
            kprint("No match")
            continue
        data = match.groupdict()
        num = data["step"]
        x_min = data["x_min"]
        x_max = data["x_max"]
        y_min = data["y_min"]
        y_max = data["y_max"]
        z_min = data["z_min"]
        z_max = data["z_max"]
        resolution = float(data["resolution"])

        xs = int(round((float(x_max) - float(x_min)) / resolution))
        ys = int(round((float(y_max) - float(y_min)) / resolution))
        zs = int(round((float(z_max) - float(z_min)) / resolution))

        path_to_images = join(raw_img_path, file)
        path_to_pnm = join(pnm_path, file)

        nm_to_um = 0.001

        pnm_pref = join(pnm_path, "pnm-" + file[11:-4])
        if isfile(pnm_pref + "_node1.dat"):
            kprint(f"Skip {num}")
            continue

        jconfig["input_data"]["filename"] = path_to_images
        jconfig["input_data"]["size"]["x"] = xs
        jconfig["input_data"]["size"]["y"] = ys
        jconfig["input_data"]["size"]["z"] = zs
        jconfig["output_data"]["statoil_prefix"] = pnm_pref
        jconfig["output_data"]["filename"] = pnm_pref
        jconfig["extraction_parameters"]["resolution"] = resolution * nm_to_um
        jconfig["extraction_parameters"]["length_unit_type"] = "UM"

        kprint(f"File name:{join(path_to_images)}")
        kprint(f"Size: {xs}, {ys}, {zs}")
        kprint(f"Output name: {pnm_pref}")
        kprint(f"Resolution: {resolution} nm ({resolution * nm_to_um} um)")

        with open(path_to_config, "w") as outfile:
            json.dump(jconfig, outfile)
        kprint(f"run: {path_to_extractor} {path_to_config}")

        process = subprocess.Popen(
            [
                path_to_extractor,
                path_to_config,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # wait for the process to terminate
        out, err = process.communicate()
        errcode = process.returncode
        if errcode != 0:
            kprint(err.decode("utf-8"))
            break
        else:
            kprint(f"Sucssess: {pnm_pref}")
            kprint(
                f"Ready index = {i+1} from {len(onlyfiles)}, Elapsed time: {time.time() - start_time}s. "
            )

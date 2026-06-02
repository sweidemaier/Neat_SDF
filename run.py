import argparse
import os
import subprocess
import time
import yaml


POINTCLOUD_DIR = "pointclouds"


def dict_to_namespace(d):
    import argparse

    if isinstance(d, argparse.Namespace):
        return d

    ns = argparse.Namespace()

    for k, v in d.items():
        setattr(
            ns,
            k,
            dict_to_namespace(v) if isinstance(v, dict) else v
        )

    return ns


def detect_dimension(file_path):
    """
    Detect whether point cloud is 2D or 3D.
    Assumes:
        x y
    or
        x y z
    """

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values = line.replace(",", " ").split()

            if len(values) == 2:
                return 2

            if len(values) == 3:
                return 3

    raise ValueError(f"Could not determine dimensions: {file_path}")


def get_config_file(dim):
    if dim == 2:
        return "configs/NeuralSDFsfromPF2D.yaml"

    return "configs/NeuralSDFsfromPF3D.yaml"


def update_config(
    config_file,
    point_path,
    epsilon,
    batch_size,
    dim
):
    with open(config_file) as f:
        config = yaml.load(
            f,
            Loader=yaml.Loader
        )

    config = dict_to_namespace(config)

    config.input.point_path = point_path
    config.input.epsilon = epsilon
    config.input.parameters.bs = batch_size

    config.models.decoder.dim = dim
    config.models.decoder_pf.dim = dim

    timestamp = int(time.time())

    logname = (
        f"{os.path.basename(point_path)}"
        f"_eps{epsilon}_{timestamp}"
    )

    config.save_dir = f"logs/PF/{logname}"

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


def train(dim):
    cmd = [
        "python",
        "train_NeatSDF.py",
        "--dim",
        str(dim)
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    return subprocess.run(cmd).returncode == 0


def find_pointclouds():

    files = []

    for file in os.listdir(POINTCLOUD_DIR):

        if file.endswith(".txt") or file.endswith(".csv"):

            full = os.path.join(
                POINTCLOUD_DIR,
                file
            )

            files.append(full)

    return files


def main(args):

    pointclouds = find_pointclouds()

    if not pointclouds:
        print("No point clouds found")
        return

    print(
        f"Found {len(pointclouds)} point clouds"
    )

    # Separate 2D and 3D point clouds
    clouds_2d = []
    clouds_3d = []

    for file in pointclouds:
        try:
            dim = detect_dimension(file)
            if dim == 2:
                clouds_2d.append(file)
            else:
                clouds_3d.append(file)
        except Exception as e:
            print(f"Error detecting dimension for {file}: {e}")

    print(f"2D clouds: {len(clouds_2d)}")
    print(f"3D clouds: {len(clouds_3d)}\n")

    # Check if 2D clouds exist
    if clouds_2d:
        print("[⚠] 2D support is not yet implemented. Skipping 2D point clouds.\n")

    # Process 3D only
    for file in clouds_3d:

        print("\n" + "=" * 60)
        print(f"[3D] {file}")

        dim = 3

        config_file = get_config_file(dim)

        point_path = os.path.splitext(file)[0]

        update_config(
            config_file,
            point_path,
            args.epsilon,
            args.batch_size,
            dim
        )

        success = train(dim)

        if success:
            print("[✓] Training completed")

        else:
            print("[✗] Training failed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000
    )

    args = parser.parse_args()

    main(args)
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(str(c) for c in cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)

    python = sys.executable

    # ------------------------------------------------------------------
    # Part 1: Synthetic GAN
    # ------------------------------------------------------------------
    part1_script = SRC / "part1_synthetic_gan.py"
    if part1_script.exists():
        run([
            python,
            str(part1_script),
            "--output_dir",
            str(OUTPUTS / "part1"),
        ])
    else:
        print(f"Skipping Part 1. Missing script: {part1_script}")

    # ------------------------------------------------------------------
    # Part 2.1: BloodMNIST DCGAN
    # ------------------------------------------------------------------
    part21_script = SRC / "part2_bloodmnist_dcgan.py"
    if part21_script.exists():
        run([
            python,
            str(part21_script),
            "--epochs",
            "20",
            "--batch_size",
            "128",
            "--output_dir",
            str(OUTPUTS / "bloodmnist"),
        ])
    else:
        print(f"Skipping Part 2.1. Missing script: {part21_script}")

    # ------------------------------------------------------------------
    # Part 2.2: CICIDS 2017 Tabular GAN
    # Use Wednesday file only, which is enough for the required task.
    # ------------------------------------------------------------------
    part22_script = SRC / "part2_cicids_tabular_gan.py"
    cicids_file = DATA / "cicids" / "Wednesday-workingHours.pcap_ISCX.csv"

    if part22_script.exists() and cicids_file.exists():
        run([
            python,
            str(part22_script),
            "--input_csv",
            str(cicids_file),
            "--epochs",
            "50",
            "--batch_size",
            "256",
            "--latent_dim",
            "32",
            "--hidden_dim",
            "256",
            "--depth",
            "3",
            "--max_rows",
            "20000",
            "--output_dir",
            str(OUTPUTS / "cicids"),
        ])
    else:
        if not part22_script.exists():
            print(f"Skipping Part 2.2. Missing script: {part22_script}")
        if not cicids_file.exists():
            print(f"Skipping Part 2.2. Missing dataset file: {cicids_file}")

    # ------------------------------------------------------------------
    # Part 2.3: QuickDraw Pizza DCGAN
    # ------------------------------------------------------------------
    part23_script = SRC / "part2_quickdraw_dcgan.py"

    quickdraw_candidates = [
        DATA / "pizza.npy",
        DATA / "quickdraw" / "pizza.npy",
        DATA / "quickdraw" / "full_numpy_bitmap_pizza.npy",
    ]

    quickdraw_file = next((p for p in quickdraw_candidates if p.exists()), None)

    if part23_script.exists() and quickdraw_file is not None:
        run([
            python,
            str(part23_script),
            "--input_npy",
            str(quickdraw_file),
            "--epochs",
            "20",
            "--batch_size",
            "128",
            "--latent_dim",
            "100",
            "--max_samples",
            "50000",
            "--output_dir",
            str(OUTPUTS / "quickdraw"),
        ])
    else:
        if not part23_script.exists():
            print(f"Skipping Part 2.3. Missing script: {part23_script}")
        if quickdraw_file is None:
            print("Skipping Part 2.3. Missing QuickDraw dataset file.")
            print("Checked paths:")
            for path in quickdraw_candidates:
                print(f" - {path}")

    print("\nAll available tasks completed.")


if __name__ == "__main__":
    main()
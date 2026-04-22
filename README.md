# Generative Modelling Assignment – VS Code Project

This project is a VS Code-ready version of the GAN assignment code pack.  
It is organized so you can open the folder in VS Code, create a virtual environment, install requirements, and run each task from the terminal or inside Jupyter notebooks.

## Project structure

```text
gan_vscode_project/
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── data/
│   ├── bloodmnist/              # optional local dataset cache
│   ├── cicids/                 # place CICIDS CSV here
│   └── quickdraw/              # place QuickDraw .npy file here
├── notebooks/
├── outputs/
├── src/
│   ├── common.py
│   ├── part1_synthetic_gan.py
│   ├── part2_bloodmnist_dcgan.py
│   ├── part2_cicids_tabular_gan.py
│   └── part2_quickdraw_dcgan.py
├── requirements.txt
├── run_all.py
├── setup.ps1
├── setup.sh
└── README.md
```

## 1. Open in VS Code

Open the `gan_vscode_project` folder in VS Code.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You can also use the provided helper scripts:

- `setup.ps1` for Windows PowerShell
- `setup.sh` for macOS/Linux

## 3. Select interpreter in VS Code

- Press `Ctrl+Shift+P`
- Search: `Python: Select Interpreter`
- Choose the interpreter inside `.venv`

## 4. Run the tasks

### Part 1 – Synthetic GAN

```bash
python src/part1_synthetic_gan.py
```

### Part 2.1 – BloodMNIST DCGAN

```bash
python src/part2_bloodmnist_dcgan.py
```

### Part 2.2 – CICIDS 2017 Tabular GAN

Place the Wednesday CICIDS CSV file inside:

```text
data/cicids/
```

Then edit the input path in `run_all.py` or run the file directly after adjusting the script arguments if needed.

Example direct run:

```bash
python src/part2_cicids_tabular_gan.py
```

### Part 2.3 – QuickDraw Pizza DCGAN

Download the QuickDraw pizza `.npy` file and place it inside:

```text
data/quickdraw/
```

Then run:

```bash
python src/part2_quickdraw_dcgan.py
```

## 5. Run all tasks from one file

```bash
python run_all.py
```

This uses the default project paths and writes outputs under `outputs/`.

## Dataset notes

### BloodMNIST
The script downloads/uses the MedMNIST dataset through the package when needed.

### CICIDS 2017
Put the Wednesday CSV in:

```text
data/cicids/Wednesday-workingHours.pcap_ISCX.csv
```

### QuickDraw Pizza
Put the numpy file in:

```text
data/quickdraw/full_numpy_bitmap_pizza.npy
```

## Outputs

Generated figures, model outputs, and metrics are saved under the `outputs/` folder.

## Recommended workflow for your assignment

1. Run each task once with a shorter configuration to verify it works
2. Increase epochs for final runs
3. Save the best figures for your report
4. Keep notes on:
   - loss behaviour
   - sample quality
   - FID or PCA/t-SNE results
   - limitations and improvements

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
Run:

```bash
pip install torch torchvision torchaudio
```

### Jupyter kernel not seeing packages
Make sure VS Code is using the same `.venv` interpreter where you installed the requirements.

### Slow training
Use a GPU-enabled environment if available, or reduce epochs during testing.

## Important
Use your own actual outputs, metrics, and observations in the final submission.
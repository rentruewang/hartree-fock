# 🍴 Hartree Fork

## ✍️ Description

The Hartree-Fock algorithm is an important method in the field of quantum chemistry. This project aims to be a clean implementation of the heart of the Hartree-Fock method, the SCF (self consistent field) procedure.

## 🏗️ Requirements & Installation

This project only runs on Python 3.10 and above.

For dependencies, I use poetry. Please use

```bash
poetry install
```

to install the packages.

## 💽 Usage

Use

```bash
poetry shell
```

to launch a virtual shell where you can directly use the virtual environment managed by poetry.

After that, run

```bash
python main.py molecule=<molecule>
```

where `<molecule>` is the target molecule whose configs are provided. Please see sample data for more details.

## 📊 Sample Data

Using helium as an example, run

```bash
python main.py molecule=he
```

The program would look in the `conf/` folder for a `he.yaml` file, where program properties like `electrons`, `iterations` etc would be defined. It will also look for `data/*.txt` for `ijkl` (the 2 integral terms), `kinetic` energy, `overlap` matrix, `potential` matrix, and `density` for optional initialization.


## 🌊 Why Helium?

Originally H2O was planned to be included in the sample data, however, since I was unable to find the 2-electron-integrals terms online (406 terms even with symmetry for the HF/STO-3G basis), the plan was abandoned. Instead, I included He (helium) atom's data for calculation.

## 🏹 Accuracy

The actual He molecule would be −2.90372 hartrees, and this program yields something close to -2.44415 hartrees, which seems plausible for hartree fock calculations. Not too good, not too bad 😁.

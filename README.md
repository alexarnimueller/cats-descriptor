# cats
Python implementation of the CATS molecular descriptor.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
You can either import the function `cats_descriptor` from `cats.py` or directly call `cats.py` on a text file containing molecule IDs and SMILES strings. See `mols.txt`for an example.

Using `cats_descriptor` in Python:
```python
from cats import cats_descriptor

d = cats_descriptor(my_molecule_list)
```

Or directly from the command line:

```bash
python cats.py mols.txt
```

To get descriptions for all possible options, run:

```bash
python cats.py --help
```
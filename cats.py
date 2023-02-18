#! /usr/bin/env python
# -*- coding: utf-8 -*-

import click
from time import time
import numpy as np
import pandas as pd
from typing import Union, List
from multiprocessing import Process, Queue, cpu_count
from rdkit.Chem import ChemicalFeatures, PandasTools
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Pharm2D import SigFactory, Generate
from features import fdef_cats, fdef_rdkit


def get_cats_factory(features: str = "cats", names: bool = False) -> SigFactory.SigFactory:
    """Get the feature combinations paired to all possible distances

    :param features: {str} which pharmacophore features to consider; available: ["cats", "rdkit"]
    :param names: {bool} whether to return an array describing the bits with names of features and distances
    :return: RDKit signature factory to be used for 2D pharmacophore fingerprint calculation
    """
    if features == "cats":
        fdef = fdef_cats
    else:
        fdef = fdef_rdkit
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)
    sigfactory = SigFactory.SigFactory(factory, useCounts=True, minPointCount=2, maxPointCount=2)
    sigfactory.SetBins([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    sigfactory.Init()
    if names:
        descs = [sigfactory.GetBitDescription(i) for i in range(sigfactory.GetSigSize())]
        return sigfactory, descs
    else:
        return sigfactory


def _cats_corr(mols: List[Mol], ids: list, q: Queue):
    """private cats descriptor function to be used in multiprocessing

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :param ids: {list/array} IDs to match calculated descriptors to input molecules
    :param q: {queue} multiprocessing queue instance
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    factory = get_cats_factory()
    fps = []
    for mol in mols:
        arr = np.zeros((1,))
        ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
        scale = np.array([10 * [sum(arr[i : i + 10])] for i in range(0, 210, 10)]).flatten()
        fps.append(np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0))
    q.put((ids, np.array(fps).reshape((len(mols), 210)).astype("float32")))


def _one_cats(mol: Mol) -> np.array:
    """Function to calculate the CATS pharmacophore descriptor for one molecule.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mol: {RDKit molecule} molecule to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vector
    """
    factory = get_cats_factory()
    arr = np.zeros((1,))
    ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
    scale = np.array([10 * [sum(arr[i : i + 10])] for i in range(0, 210, 10)]).flatten()
    return np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0).astype("float32")


def cats_descriptor(mols: List[Mol], ids: Union[list, None] = None) -> np.array:
    """Function to calculate the CATS pharmacophore descriptor for a set of molecules.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    if ids is None:
        ids = list(range(len(mols)))
    assert len(ids) == len(mols), "Number of IDs don't match number of molecules!"

    queue = Queue()

    # if only small array, don't parallelize
    if len(mols) < 4 * cpu_count():
        rslt = pd.DataFrame(map(_one_cats, mols), columns=[f"CATS{i}" for i in range(1, 211)])
        rslt["ID"] = ids

    # if many molecules, do parallelize
    else:
        rslt = pd.DataFrame()
        for i, m in zip(np.array_split(np.array(ids), cpu_count()), np.array_split(np.array(mols), cpu_count())):
            p = Process(target=_cats_corr, args=(m, i, queue))
            p.start()
        for _ in range(cpu_count()):
            ids, desc = queue.get(10)
            desc = pd.DataFrame(desc, columns=[f"CATS{i}" for i in range(1, 211)])
            desc["ID"] = ids
            rslt = pd.concat((rslt, desc))
    return rslt[["ID"] + [f"CATS{i}" for i in range(1, 211)]].sort_values("ID")


@click.command()
@click.argument("smiles_file")
@click.option("-o", "--output_file", type=str, default="cats.txt", help="output filename")
@click.option("-i", "--id_column", type=str, default="ID", help="column header of the ID column")
@click.option("-s", "--smiles_column", type=str, default="SMILES", help="column header of the SMILES column")
@click.option("-v", "--verbose", is_flag=True, default=False, help="verbosity")
def run(smiles_file: str, output_file: str, id_column: str, smiles_column: str, verbose: bool):
    if verbose:
        print("Loading molecules...")
    df = pd.read_csv(smiles_file, delimiter="\t")
    PandasTools.AddMoleculeColumnToFrame(df, smiles_column, "Molecule", includeFingerprints=False)
    if verbose:
        print("Calculating descriptor...")
    start = time()
    d = cats_descriptor(df.Molecule, ids=df[id_column])
    t = time() - start
    if verbose:
        print(f"\nShape of descriptor: {d.shape}")
        print(f"\nCalculation took {t:.3f}sec for {len(df)} molecules")
        print(f"\nSaving descriptor to output file {output_file}")
    d.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    run()

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Process, Queue, cpu_count
from rdkit.Chem import ChemicalFeatures
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Pharm2D import SigFactory, Generate
from features import fdef_cats, fdef_rdkit


def get_cats_factory(features='cats', names=False):
    """ Get the feature combinations paired to all possible distances

    :param features: {str} which pharmacophore features to consider; available: ["cats", "rdkit"]
    :param names: {bool} whether to return an array describing the bits with names of features and distances
    :return: RDKit signature factory to be used for 2D pharmacophore fingerprint calculation
    """
    if features == 'cats':
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


def _cats_corr(mols, q):
    """ private cats descriptor function to be used in multiprocessing

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :param q: {queue} multiprocessing queue instance
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    factory = get_cats_factory()
    fps = []
    for mol in mols:
        arr = np.zeros((1,))
        ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
        scale = np.array([10 * [sum(arr[i:i + 10])] for i in range(0, 210, 10)]).flatten()
        fps.append(np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0))
    q.put(np.array(fps).reshape((len(mols), 210)).astype('float32'))


def _one_cats(mol):
    """ Function to calculate the CATS pharmacophore descriptor for one molecule.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mol: {RDKit molecule} molecule to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vector
    """
    factory = get_cats_factory()
    arr = np.zeros((1,))
    ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
    scale = np.array([10 * [sum(arr[i:i + 10])] for i in range(0, 210, 10)]).flatten()
    return np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0).astype('float32')


def cats_descriptor(mols):
    """ Function to calculate the CATS pharmacophore descriptor for a set of molecules.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    queue = Queue()
    rslt = []
    if len(mols) < 4 * cpu_count():  # if only small array, don't parallelize
        for mol in mols:
            rslt.append(_one_cats(mol))
    else:
        for m in np.array_split(np.array(mols), cpu_count()):
            p = Process(target=_cats_corr, args=(m, queue,))
            p.start()
        for _ in range(cpu_count()):
            rslt.extend(queue.get(10))
    return np.array(rslt).reshape((len(mols), 210)).astype('float32')


if __name__ == "__main__":
    from rdkit.Chem import MolFromSmiles
    from time import time

    smls = list()
    with open('./mols.csv', 'r') as f:
        for line in f:
            smls.append(line.strip())
    print("Loading molecules...")
    molecules = [MolFromSmiles(s) for s in smls]

    print("Calculating descriptor...")
    start = time()
    d = cats_descriptor(molecules)
    t = time() - start
    print(d)
    print("Shape of descriptor: %s" % str(d.shape))
    print("\nCalculation took %.4f for %i molecules" % (t, len(molecules)))

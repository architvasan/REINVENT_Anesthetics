"""Compute scores with ChemProp

scoring_function.type = "product"
scoring_function.parallel = false

[[stage.scoring_function.component]]

type = "chemprop"
name = "ChemProp Score"

weight = 0.7

# component specific parameters
param.checkpoint_dir = "ChemProp/3CLPro_6w63"
param.rdkit_2d_normalized = true

transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4
"""

from __future__ import annotations

#__all__ = ["ChemProp"]
import sklearn
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import List
import logging
#import chemprop
import numpy as np
from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles
import pickle
logger = logging.getLogger('reinvent')

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    checkpoint: List[str]
    #rdkit_2d_normalized: List[bool] = field(default_factory=lambda: [False])

#np.mean()

@add_tag("__component")
class AnesthPot:
    def __init__(self, params: Parameters):
        logger.info(f"Calculating AP using Chandler Brady's script")
        self.ap_params = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        #self.smiles_type = 'rdkit_smiles'

        #for checkpoint_dir, rdkit_2d_normalized in zip(
        #    params.checkpoint_dir, params.rdkit_2d_normalized
        #):
        #    args = [
        #        "--checkpoint_dir",  # ChemProp models directory
        #        checkpoint_dir,
        #        "--test_path",  # required
        #        "/dev/null",
        #        "--preds_path",  # required
        #        "/dev/null",
        #    ]

        #    if rdkit_2d_normalized:
        #        args.extend(
        #            ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
        #        )

        #    with suppress_output():
        #        chemprop_args = chemprop.args.PredictArgs().parse_args(args)
        #        chemprop_model = chemprop.train.load_model(args=chemprop_args)
        
        for obj in params.checkpoint:
            with open(obj, 'rb') as input_file:
                model = pickle.load(input_file)
            self.ap_params.append(model)
        
    #@normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        #smilies_list = [[smiles] for smiles in smilies]
        fingerprint=np.zeros([len(smilies),2000])

        for smiles_idx, smiles in enumerate(smilies):
            mol = Chem.MolFromSmiles(smiles)
            bi = {}
            fp0 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=0, bitInfo=bi)
            for x in fp0.GetOnBits():
              fingerprint[smiles_idx,x]=len(bi[x])

        scores = []

        for model in self.ap_params:
            preds = model.predict(fingerprint)                
            print([p[0] for p in preds])
            scores.extend(
                [p[0] for p in preds]
                )
                #np.array(
                #    preds
                    #[val[0] if "Invalid SMILES" not in val else np.nan for val in preds],
                    #dtype=float,
                #)
            #)

        return (ComponentResults([np.array(scores, dtype=float)]))

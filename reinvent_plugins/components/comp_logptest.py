"""Compute the RDKit molecule volume from a conformer and its grid-encoding

The quality depends on the quality of the conformer.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from rdkit.Chem import AllChem as Chem
import logging
import rdkit.Chem
from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles
from reinvent_plugins.mol_cache import molcache
logger = logging.getLogger('reinvent')


@add_tag("__parameters")
@dataclass
class Parameters:
    grid_spacing: Optional[List[float]] = field(default_factory=lambda: [0.2])
    box_margin: Optional[List[float]] = field(default_factory=lambda: [2.0])

@add_tag("__component")
class LogPTest:
    def __init__(self, params: Parameters):
        self.grid_spacings = params.grid_spacing
        #@normalize_smiles
        self.box_margin = params.box_margin

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
            scores = []
    
            for mol in mols:
                try:
                    score = rdkit.Chem.Crippen.MolLogP(mol)
                    scores.append(score)
                    print(score)
                except:
                    scores.append(0)
    
            return ComponentResults([np.array(scores, dtype=float)])

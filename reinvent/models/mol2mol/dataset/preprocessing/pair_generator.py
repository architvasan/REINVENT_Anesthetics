__all__ = ["PairGenerator"]
from abc import ABC, abstractmethod
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd

from reinvent.chemistry.conversions import Conversions


class PairGenerator(ABC):
    def __init__(
        self, min_cardinality: int = 0, max_cardinality: int = 500, *args, **kwargs
    ) -> None:
        """__init__.

        :param min_cardinality: minimum number of targets for each source
        :type min_cardinality: int
        :param max_cardinality: maximum number of targets for each source
        :type max_cardinality: int
        """
        if min_cardinality > max_cardinality:
            raise ValueError("`min_cardinality` must be lower or equal than `max_cardinality`") 
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    @abstractmethod
    def build_pairs(self, smiles: list, *, processes: int):
        """Abstract method for building pairs (source,target)

        :param smiles: a list of smiles
        :type smiles: list
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        """
        pass

    def filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Keeps all the pairs such that for each source s, min_cardinality <= | { (s, t_i) } | <= max_cardinality.

        :param pairs: DataFrame containing the pairs. It must contain columns "Source_Mol" and "Target_Mol"
        :type pairs: pd.DataFrame
        :rtype: pd.DataFrame
        """

        if ("Source_Mol" not in pairs.columns) or ("Target_Mol" not in pairs.columns):
            raise ValueError(
                "`Source_Mol` and `Target_Mol` columns must be included in the DataFrame"
            )

        locations = defaultdict(list)
        for i, smi in enumerate(pairs["Source_Mol"]):
            locations[smi].append(i)

        good_locations = []
        for k in locations:
            if (len(locations[k]) >= self.min_cardinality) and (
                len(locations[k]) <= self.max_cardinality
            ):
                good_locations += locations[k]
        good_locations = np.array(good_locations)
        return pairs.iloc[good_locations].reset_index(drop=True)

    def _standardize_smiles(self, smiles):
        conversions = Conversions()
        std_smiles = set()
        pbar = tqdm(smiles)
        pbar.set_description("Standardizing smiles")
        for smi in pbar:
            std_smi = conversions.convert_to_standardized_smiles(smi)
            if (std_smi is not None) and (len(std_smi) > 0):
                std_smiles.add(std_smi)
        return np.array(list(std_smiles))

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

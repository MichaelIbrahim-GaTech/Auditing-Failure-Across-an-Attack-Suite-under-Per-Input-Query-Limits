"""Load six frozen outcome populations used in the audit study."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def load_populations(root=None):
    root = ROOT if root is None else Path(root)
    populations = []
    archive = np.load(root/'strongreject_audit_results/primary_score_matrices.npz', allow_pickle=False)
    names = ['Dolphin', 'GPT-3.5', 'GPT-4o-mini', 'Llama-3.1']
    for index, identifier in enumerate(archive['models']):
        matrix = (archive['scores'][index] >= .5).astype(np.uint8)
        populations.append(dict(name=names[index], identifier=str(identifier), family='StrongREJECT',
                                d=36, matrix=matrix, S=matrix.sum(axis=1).astype(int)))
    for identifier, name in [('logistic_regression','Logistic'), ('random_forest','Forest')]:
        archive = np.load(root/f'empirical_results/{identifier}_failure_matrix.npz', allow_pickle=False)
        matrix = archive['failure_matrix']
        populations.append(dict(name=name, identifier=identifier, family='Digits',
                                d=98, matrix=matrix, S=matrix.sum(axis=1).astype(int)))
    for population in populations:
        population['theta'] = float((population['S'] > 0).mean())
        population['mu'] = np.bincount(population['S'], minlength=population['d']+1)/len(population['S'])
    return populations

from kge.model import KgeModel
from kge.util.io import load_checkpoint
import pandas as pd
import torch

data_path = '../../Chapter3/data'

# Load embeddings
model_path = f'{data_path}/raw/simple_selfloops_trial21.pt'
checkpoint = load_checkpoint(model_path)
with open(f'{data_path}/libkge_path.txt', 'r') as f:
    libkge_path = f.read()
checkpoint['config'].set('dataset.name', libkge_path + '/data/selfloops')
model = KgeModel.create_from(checkpoint)

# Get node info
node_embeds = model.state_dict()['_entity_embedder._embeddings.weight']
node_list = pd.read_csv(
    checkpoint['config'].get('dataset.name') + '/entity_ids.del',
    sep='\t', header=None
)
drug_ids = [i for i, row in node_list.iterrows() if row[1].startswith('CID')]
gene_ids = [i for i, row in node_list.iterrows() if not row[1].startswith('C')]

# Save embeds
drug_embeds = node_embeds[drug_ids]
torch.save(drug_embeds, 'drug_vecs.pt')

gene_embeds = node_embeds[gene_ids]
torch.save(gene_embeds, 'gene_vecs.pt')


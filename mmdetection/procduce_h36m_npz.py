import numpy as np

ref_file = '/home/wzeng/mycodes/Transformer_related/multiperson/mmdetection/data/CMU_mosh.npz'
in_file = '/home/wzeng/mydata/DecoMR/h36m_train_new.npz'
out_file = '/home/wzeng/mycodes/Transformer_related/multiperson/mmdetection/data/h36m_mosh.npz'

ref_data = np.load(ref_file)
in_data = np.load(in_file)
np.savez(out_file,
         pose=in_data['pose'],
         shape=in_data['shape'])

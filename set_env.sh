#!/usr/bin/env bash
conda env create -f environment.yml
source activate multiperson
cd pyopengl
pip install .
cd neural_renderer/
python3 setup.py install
cd ../mmcv
python3 setup.py install
cd ../mmdetection
./compile.sh
python setup.py develop
cd ../sdf
python3 setup.py install

rsync -avzu --progress /home/wzeng/mycodes/Transformer_related/multiperson/mmdetection/data wzeng:/home/wzeng/mycodes/transformer/multiperson/mmdetection/
rsync -avzu --progress /home/wzeng/mydata/H36Mnew/c2f_vol/rcnn/ wzeng:/home/wzeng/mydata/H36Mnew/c2f_vol/
rsync -avzu --progress /home/wzeng/mydata/coco/annotations wzeng:/home/wzeng/mydata/coco/
rsync -avzu --progress /home/wzeng/mydata/lsp_dataset_original/train.pkl wzeng:/home/wzeng/mydata/lsp_dataset_original/
rsync -avzu --progress /home/wzeng/mydata/lspet_dataset/train.pkl wzeng:/home/wzeng/mydata/lsp_dataset/
rsync -avzu --progress /home/wzeng/mydata/lsp_dataset_original/images_pretrain wzeng:/home/wzeng/mydata/lsp_dataset_original/
rsync -avzu --progress /home/wzeng/mydata/mpii/rcnn wzeng:/home/wzeng/mydata/mpii/
rsync -avzu --progress /home/wzeng/mydata/mpi_inf_3dhp_new/rcnn wzeng:/home/wzeng/mydata/mpi_inf_3dhp_new/
rsync -avzu --progress /home/wzeng/mydata/panoptic/processed wzeng:/home/wzeng/mydata/panoptic/



python3 tools/train.py configs/smpl/my_pretrain.py --create_dummy
while true; do
    python3 tools/train.py configs/smpl/my_pretrain.py --gpus=8
done

python3 tools/train.py configs/smpl/my_baseline.py --load_pretrain ./work_dirs/pretrain/latest.pth
while true;
do
    python3 tools/train.py configs/smpl/my_baseline.py  --gpus=8
done

python3 tools/train.py configs/smpl/my_tune.py --load_pretrain ./work_dirs/baseline/latest.pth
while true;
do
    python3 tools/train.py configs/smpl/my_tune.py --gpus=8
done



# 240k // 4 = 60k iteration
# 11348 * 6
python3 tools/train.py configs/smpl/my_pretrain.py --create_dummy
while true; do
    python3 tools/train.py configs/smpl/my_pretrain.py --gpus=8
done

# 180k // 4 = 45k iteration
# 15055 * 3
python3 tools/train.py configs/smpl/my_baseline.py --load_pretrain ./work_dirs/my_pretrain/latest.pth
while true;
do
    python3 tools/train.py configs/smpl/my_baseline.py  --gpus=8
done

# 100k // 4 = 25k iteration
# 15055 * 2
python3 tools/train.py configs/smpl/my_tune.py --load_pretrain ./work_dirs/my_baseline/latest.pth
while true;
do
    python3 tools/train.py configs/smpl/my_tune.py --gpus=8
done



# 240k iteration # 45536 * 6
python3 tools/train.py configs/smpl/gpu2/my_pretrain.py --create_dummy
python3 tools/train.py configs/smpl/gpu2/my_pretrain.py --gpus=2

# 180k iteration # 60220 * 3
python3 tools/train.py configs/smpl/gpu2/my_baseline.py --load_pretrain ./work_dirs/gpu2/pretrain/latest.pth
python3 tools/train.py configs/smpl/gpu2/my_baseline.py  --gpus=2

# 100k iteration # 60220 * 2
python3 tools/train.py configs/smpl/gpu2/my_tune.py --load_pretrain ./work_dirs/gpu2/baseline/latest.pth
python3 tools/train.py configs/smpl/gpu2/my_tune.py --gpus=2

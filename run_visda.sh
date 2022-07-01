# python3 train_visda_source.py --batch_size 128  --lr 0.001 --save  trained_model/source/visda/source_rep --train_path data/visda/train/ --val_path data/visda/validation/ --resnet 101 --num_gpu 4

python3 train_visda_target.py --epochs 5 --batch_size 128  --lr 0.0003 --train_path data/visda/train/ --val_path data/visda/validation/ --resnet 101  --save trained_model/target/visda/target_rep --load_weight trained_model/source/visda/source_rep_4.pth

python3 train_visda_final.py --batch_size 128 --lr 0.0003 --train_path data/visda/train/ --val_path data/visda/validation/ --resnet 101  --save trained_model/final/visda/final --load_weight trained_model/target/visda/target_rep_4.pth


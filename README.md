# BF_Style
![alt text](https://github.com/loveis98/BF_Style/blob/master/mask_surf.png)

To download pretrained models for testing (in **ckpt**): 
```
./load_models.sh
```

To test on images from **test.txt**:
```
python eval.py --model lw_refine
```
Other arguments:
```
--test-dir, default='./dataset/', path to the test set directory
--test-list, default='./dataset/test.txt', path to the test set list
--out-dir, default='test_result', path to the test set result
--ckpt-path, default='./ckpt/lw_refine.pth.tar', path to the checkpoint file
```

echo SPIN
echo train
python test/test_dataset.py --dataset spin --root data/spin/images --annotations data/spin/annotations --split train
echo val
python test/test_dataset.py --dataset spin --root data/spin/images --annotations data/spin/annotations --split val
echo test
python test/test_dataset.py --dataset spin --root data/spin/images --annotations data/spin/annotations --split test

echo
echo PACO
echo train
python test/test_dataset.py  --dataset paco  --root data/paco/images  --annotations data/paco/annotations  --split train
echo val
python test/test_dataset.py  --dataset paco  --root data/paco/images  --annotations data/paco/annotations  --split val
echo test
python test/test_dataset.py  --dataset paco  --root data/paco/images  --annotations data/paco/annotations  --split test

echo
echo PartImageNet
echo train
python test/test_dataset.py  --dataset partimagenet  --root data/partimagenet  --annotations data/partimagenet/annotations  --split train
echo val
python test/test_dataset.py  --dataset partimagenet  --root data/partimagenet  --annotations data/partimagenet/annotations  --split val
echo test
python test/test_dataset.py  --dataset partimagenet  --root data/partimagenet  --annotations data/partimagenet/annotations  --split test
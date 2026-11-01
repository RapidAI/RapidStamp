set -e
cd /home/rapidstamp_data/build_cpp
P=/home/znsoft/miniconda3/envs/pytorch/bin/python
export LD_LIBRARY_PATH=/home/rapidstamp_data/third_party/onnxruntime-linux-x64-1.22.0/lib
R=/home/rapidstamp_data/deploy/ring_reader
$P mkppm.py gold
./build/rapidstamp_ring_eval $R/current - 16 gold.tsv cpp_gold_route.tsv 8 | tail -1
./build/rapidstamp_ring_eval $R/releases/20261101_prod_S512 - 0 gold.tsv cpp_gold_prod.tsv 8 | tail -1
$P cmp.py route,cpp_gold_route.tsv,e2e_preds_A_route.jsonl,gold prod,cpp_gold_prod.tsv,e2e_preds_prod_only.jsonl,gold | tee $R/eval/cpp_gold_20261101.txt
tail -c 600 $R/scripts/csd.log

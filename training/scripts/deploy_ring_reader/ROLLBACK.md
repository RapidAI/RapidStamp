# Rollback: A + length routing -> production S@512 (2026-11-01)

The live ring reader is selected by a symlink:
  /home/rapidstamp_data/deploy/ring_reader/current -> releases/20261101_A_route

Rollback (instant, no rebuild, no file deletion):
  cd /home/rapidstamp_data/deploy/ring_reader
  ln -sfn releases/20261101_prod_S512 current
Processes that already loaded the model must be restarted (Python RingReader / C++ engine load at create time).

Roll forward again:
  ln -sfn releases/20261101_A_route current

Disable the ring reader completely (back to the bundled seal_det + CTC recognizer path):
  remove/rename the `ring_reader` link inside the model dir given to the engine/pipeline
  (the engine only uses it when <model_dir>/ring_reader/route.json exists).

Threshold only: edit "long_threshold" in releases/20261101_A_route/route.json (16 = default; very large value = A only).

Backups in this directory:
  checkpoint/{best.pt,parseq_reader.onnx,eval.json}  = checkpoints/parseq_reader_large/finetune (production S@512), SHA256SUMS
  names_for_vl1.txt, ppocr_keys_v1.txt, eval_gold_v2_cli.py  = lexicon/charset/eval as of deploy
Code backups on server: src/RapidStamp/python/src/rapidstamp/pipeline.py.bak_20261101
Git: the change is one commit on RapidAI/RapidStamp main; `git revert <sha>` undoes code.



  # 终端1：初始化环境
  eval "$(conda shell.bash hook)"
  conda activate fi-bench
  export FIB_DATASET_PATH=/home/sjs/mlsys2026-dataset



  # 终端1：运行测试
  python scripts/pack_solution.py
  python scripts/run_local.py

  # 终端1：查看结果
  cat solution.json | python -m json.tool | grep -A5 '"solution"'

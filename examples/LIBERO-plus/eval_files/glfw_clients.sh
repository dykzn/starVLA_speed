#!/usr/bin/env bash
# GLFW eval clients only (servers already running)
# 用法: bash examples/LIBERO-plus/eval_files/glfw_clients.sh

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

P=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
PREV_LOG=results/libero_plus_eval/logs/20260625_103928

export DISPLAY=:99
export MUJOCO_GL=glfw
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export LIBERO_HOME=/data3/dengyongkang/my_project/LIBERO-plus
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"

# 读取上次进度
s10=$(grep -c "Success:" ${PREV_LOG}/libero_10.log 2>/dev/null || echo 0)
sg=$(grep -c "Success:" ${PREV_LOG}/libero_goal.log 2>/dev/null || echo 0)
ss=$(grep -c "Success:" ${PREV_LOG}/libero_spatial.log 2>/dev/null || echo 0)
so=$(grep -c "Success:" ${PREV_LOG}/libero_object.log 2>/dev/null || echo 0)

LOG=results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_glfw
mkdir -p $LOG

CUDA_VISIBLE_DEVICES=1 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 --args.start-idx $s10 --args.log-path $LOG \
  > $LOG/libero_10.log 2>&1 &
echo "libero_10 PID=$! start=$s10"

CUDA_VISIBLE_DEVICES=1 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 --args.task-suite-name libero_goal \
  --args.num-trials-per-task 1 --args.start-idx $sg --args.log-path $LOG \
  > $LOG/libero_goal.log 2>&1 &
echo "libero_goal PID=$! start=$sg"

CUDA_VISIBLE_DEVICES=3 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 --args.start-idx $ss --args.log-path $LOG \
  > $LOG/libero_spatial.log 2>&1 &
echo "libero_spatial PID=$! start=$ss"

CUDA_VISIBLE_DEVICES=3 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 --args.task-suite-name libero_object \
  --args.num-trials-per-task 1 --args.start-idx $so --args.log-path $LOG \
  > $LOG/libero_object.log 2>&1 &
echo "libero_object PID=$! start=$so"

echo "Logs: $LOG"

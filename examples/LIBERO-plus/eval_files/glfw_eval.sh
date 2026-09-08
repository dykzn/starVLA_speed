#!/usr/bin/env bash
# GLFW GPU渲染评估脚本 (GPU 1,3)
# 用法: bash examples/LIBERO-plus/eval_files/glfw_eval.sh

STARVLA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_DIR}"

P=/data3/dengyongkang/.conda/envs/VLA_JEPA/bin/python
CKPT=results/Checkpoints/phase1_wm_warmup_v2/checkpoints/steps_6000_pytorch_model.pt
PREV_LOG=results/libero_plus_eval/logs/20260625_103928

export DISPLAY=:99
export MUJOCO_GL=glfw
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export LIBERO_HOME=/data3/dengyongkang/my_project/LIBERO-plus
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero/libero"

# === 1. 清理旧进程 ===
pkill -f 'server_policy.*988[46]' 2>/dev/null || true
pkill Xvfb 2>/dev/null || true
sleep 1

# === 2. Xvfb ===
Xvfb :99 -screen 0 1024x768x24 &
sleep 1
echo "[1/4] Xvfb ready"

# === 3. Server ===
CUDA_VISIBLE_DEVICES=1 $P deployment/model_server/server_policy.py \
  --ckpt_path $CKPT --port 9884 --use_bf16 &
CUDA_VISIBLE_DEVICES=3 $P deployment/model_server/server_policy.py \
  --ckpt_path $CKPT --port 9886 --use_bf16 &
echo "[2/4] Servers starting (wait ~2min for model loading)..."

# === 4. 等 server ready ===
for port in 9884 9886; do
  while ! ss -tlnp 2>/dev/null | grep -q ":${port} "; do sleep 2; done
  echo "       port ${port} ready"
done
echo "[3/4] Servers ready"

# === 5. Client ===
LOG=results/libero_plus_eval/logs/$(date +"%Y%m%d_%H%M%S")_glfw
mkdir -p $LOG

# 读取上次进度
start_10=$(grep -c "Success:" ${PREV_LOG}/libero_10.log 2>/dev/null || echo 0)
start_goal=$(grep -c "Success:" ${PREV_LOG}/libero_goal.log 2>/dev/null || echo 0)
start_spatial=$(grep -c "Success:" ${PREV_LOG}/libero_spatial.log 2>/dev/null || echo 0)
start_object=$(grep -c "Success:" ${PREV_LOG}/libero_object.log 2>/dev/null || echo 0)

# GPU 1 port 9884
CUDA_VISIBLE_DEVICES=1 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 --args.start-idx $start_10 \
  --args.log-path $LOG > $LOG/libero_10.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9884 --args.task-suite-name libero_goal \
  --args.num-trials-per-task 1 --args.start-idx $start_goal \
  --args.log-path $LOG > $LOG/libero_goal.log 2>&1 &

# GPU 3 port 9886
CUDA_VISIBLE_DEVICES=3 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 --args.start-idx $start_spatial \
  --args.log-path $LOG > $LOG/libero_spatial.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 $P examples/LIBERO-plus/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 9886 --args.task-suite-name libero_object \
  --args.num-trials-per-task 1 --args.start-idx $start_object \
  --args.log-path $LOG > $LOG/libero_object.log 2>&1 &

echo "[4/4] Eval clients running"
echo ""
echo "  Server GPU 1 port 9884 → libero_10(start=$start_10) + libero_goal(start=$start_goal)"
echo "  Server GPU 3 port 9886 → libero_spatial(start=$start_spatial) + libero_object(start=$start_object)"
echo "  Logs: $LOG"
echo "  Monitor: tail -f $LOG/libero_*.log"
echo "  Kill:   pkill -f 'server_policy.*988[46]'; pkill -f 'eval_libero.*988[46]'; pkill Xvfb"

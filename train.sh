# export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

LOG_DIR=logs/log_$(date "+%Y-%m-%d_%H:%M:%S").log
mkdir -p $(dirname $LOG_DIR)

# Initialize log file
echo "Master Log Started at $(date)" | tee $LOG_DIR
echo "" | tee -a $LOG_DIR

run_script() {
    description=$1
    shift  # Remove the description to access the command
    
    echo "===============================================" | tee -a $LOG_DIR
    echo "Starting $description at $(date)" | tee -a $LOG_DIR
    echo "===============================================" | tee -a $LOG_DIR
    
    start_time=$(date +%s)
    
    # Execute the command (all remaining arguments)
    (set -x; "$@") 2>&1 | tee -a $LOG_DIR
    exit_code=${PIPESTATUS[0]}  # Get the exit code of the command, not tee
    
    end_time=$(date +%s)
    
    echo "===============================================" | tee -a $LOG_DIR
    echo "$description completed at $(date)" | tee -a $LOG_DIR
    echo "Execution time: $((end_time - start_time)) seconds" | tee -a $LOG_DIR
    echo "Exit code: $exit_code" | tee -a $LOG_DIR
    echo "===============================================" | tee -a $LOG_DIR
    echo "" | tee -a $LOG_DIR
    
    return $exit_code
}

run_script "cs336 a1 train" uv run python cs336_basics/train.py --name $1
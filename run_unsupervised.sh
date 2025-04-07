#!/bin/bash

# Create logs directory
mkdir -p unsupervised_logs

# Base command with the weaker discriminator as baseline
BASE_CMD="python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --dis_dropout 0.4 --dis_input_dropout 0.3 --dis_steps 3 --dis_most_frequent 20000 --map_beta 0.01 --visualize_final_alignment true"

# Check if we should use CPU instead of GPU
USE_CPU=false
if [[ "$1" == "--cpu" ]]; then
    USE_CPU=true
    BASE_CMD="$BASE_CMD --cuda False"
    echo "Running experiments on CPU"
else
    echo "Running experiments on GPU (use --cpu argument to run on CPU instead)"
fi

# Run experiment and log results to both screen and file
run_experiment() {
    local EXPERIMENT_NAME=$1
    local COMMAND=$2
    
    echo "================================================================="
    echo "Running experiment: $EXPERIMENT_NAME"
    echo "Command: $COMMAND"
    echo "================================================================="
    
    # Create log file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="unsupervised_logs/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
    
    # Save the command to the log file
    echo "Command: $COMMAND" > $LOG_FILE
    echo "Started at: $(date)" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Run the command, output to both screen and log file (capturing both stdout and stderr)
    eval $COMMAND 2>&1 | tee -a $LOG_FILE
    
    # Clear GPU memory if using CUDA
    if [ "$USE_CPU" = false ]; then
        echo "Clearing GPU memory..."
        # Wait a moment to ensure process is completely done
        sleep 2
        # Force Python garbage collection
        python -c "import gc; gc.collect()"
        # Let system stabilize
        sleep 2
    fi
    
    echo "" >> $LOG_FILE
    echo "Finished at: $(date)" >> $LOG_FILE
    
    echo ""
    echo "Experiment $EXPERIMENT_NAME completed and logged to $LOG_FILE"
    echo ""
}

# Run one experiment at a time to avoid memory issues
echo "Which experiment would you like to run? Options:"
echo "1. weaker_discriminator (baseline)"
echo "2. hindi_latin_weak_disc (weak discriminator for Hindi in Latin script)"
echo "3. hindi_latin_extended_training (Hindi Latin with longer training)"
echo "Enter a number (1-3):"
read EXPERIMENT_CHOICE

case $EXPERIMENT_CHOICE in
    1)
        # Weaker discriminator (new baseline)
        echo "Running with weaker discriminator baseline..."
        run_experiment "weaker_disc_baseline" "$BASE_CMD --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5 --dico_eval data/dictionaries/en-es.5000-6500.txt"
        ;;
    2)
        # Hindi Latin with weak discriminator
        echo "Testing with weak discriminator for Hindi in Latin script..."
        run_experiment "hindi_latin_weak_disc" "$BASE_CMD --src_lang en --tgt_lang hi_latin --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi_latin.vec --n_refinement 8 --n_epochs 25 --epoch_size 500000 --map_optimizer sgd,lr=0.1 --dis_optimizer sgd,lr=0.01 --dico_eval data/dictionaries/en-hi_latin.5000-6500_iast.txt --early_stopping_patience 5 --dis_lambda 1.2"
        ;;
    3)
        # Hindi Latin with extended training
        echo "Testing with Hindi Latin and extended training time..."
        run_experiment "hindi_latin_extended" "$BASE_CMD --src_lang en --tgt_lang hi_latin --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi_latin.vec --n_refinement 10 --n_epochs 15 --epoch_size 500000 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.1 --dico_eval data/dictionaries/en-hi_latin.5000-6500_iast.txt"
        ;;
    *)
        echo "Invalid option. Running with weaker discriminator baseline..."
        run_experiment "weaker_disc_baseline" "$BASE_CMD --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
esac

echo "Experiments completed. Logs are in the unsupervised_logs directory."

# Make the script executable
chmod +x run_unsupervised.sh 
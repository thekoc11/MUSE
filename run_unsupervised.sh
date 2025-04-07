#!/bin/bash

# Create logs directory
mkdir -p unsupervised_logs

# Base command with the weaker discriminator as baseline
BASE_CMD="python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --dis_dropout 0.4 --dis_input_dropout 0.3 --dis_steps 3 --dis_most_frequent 100000 --map_beta 0.01"

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
echo "2. learning_rate_tuning (higher mapping, lower disc rate)"
echo "3. longer_training (more epochs and refinement)"
echo "4. dictionary_quality (improved selection methods)"
echo "5. advanced_discriminator (fine-tuned parameters)"
echo "6. combined_optimizations (all improvements)"
echo "7. original_settings (default parameters)"
echo "8. very_weak_discriminator (for distant languages)"
echo "9. aggressive_mapping (high learning rate, low orthogonality)"
echo "10. frequent_words_focus (limit vocab to common terms)"
echo "11. distant_language_combined (all distant language optimizations)"
echo "12. custom_seed_dictionary (uses dictionary initialization with disjoint train/test sets)"
echo "13. custom_seed_with_distant_opts (seed dictionary + distant language optimizations)"
echo "14. hindi_latin_weak_disc (weak discriminator for Hindi in Latin script)"
echo "15. hindi_latin_extended_training (Hindi Latin with longer training)"
echo "Enter a number (1-15):"
read EXPERIMENT_CHOICE

case $EXPERIMENT_CHOICE in
    1)
        # Weaker discriminator (new baseline)
        echo "Running with weaker discriminator baseline..."
        run_experiment "weaker_disc_baseline" "$BASE_CMD --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5 --dico_eval data/dictionaries/en-es.5000-6500.txt"
        ;;
    2)
        # Adjust learning rates
        echo "Testing learning rate tuning..."
        run_experiment "lr_tuning" "$BASE_CMD --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.05 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    3)
        # Increase training duration
        echo "Testing longer training duration..."
        run_experiment "longer_training" "$BASE_CMD --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 8 --n_epochs 10 --epoch_size 500000 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    4)
        # Improve dictionary quality
        echo "Testing improved dictionary quality..."
        run_experiment "dict_quality" "$BASE_CMD --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dico_method csls_knn_10 --dico_build \"S2T&T2S\" --dico_threshold 0.1 --dico_eval \"data/dictionaries/en-hi.5000-6500.txt\""
        ;;
    5)
        # Advanced discriminator settings
        echo "Testing advanced discriminator settings..."
        run_experiment "adv_disc" "$BASE_CMD --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dis_lambda 0.5 --dis_smooth 0.2 --dis_clip_weights 0.01 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    6)
        # Combine all optimizations
        echo "Testing combined optimizations..."
        run_experiment "combined_opt" "python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 8 --dis_dropout 0.3 --dis_input_dropout 0.3 --dis_steps 3 --dis_most_frequent 50000 --map_beta 0.01 --n_epochs 10 --epoch_size 500000 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.05 --dis_lambda 0.5 --dis_smooth 0.2 --dis_clip_weights 0.01 --dico_method csls_knn_10 --dico_build S2T\&T2S --dico_threshold 0.1 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    7)
        # Original settings without modifications
        echo "Running with original unmodified settings..."
        run_experiment "original" "python -u unsupervised.py --batch_size 32 --normalize_embeddings center --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    8)
        # Very weak discriminator for distant languages
        echo "Testing very weak discriminator for distant languages..."
        run_experiment "very_weak_disc" "python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dis_dropout 0.5 --dis_input_dropout 0.5 --dis_steps 2 --dis_lambda 0.2 --dis_most_frequent 30000 --dis_smooth 0.3 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    9)
        # Aggressive mapping for distant languages
        echo "Testing aggressive mapping for distant languages..."
        run_experiment "aggressive_map" "python -u unsupervised.py --batch_size 16 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --map_beta 0.001 --map_optimizer sgd,lr=0.5 --map_id_init False --dis_optimizer sgd,lr=0.05 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    10)
        # Focus on frequent words only - drastically reduce vocabulary
        echo "Testing approach with focus on most frequent words only..."
        run_experiment "frequent_words" "python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --max_vocab 50000 --dis_most_frequent 10000 --dico_max_rank 10000 --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    11)
        # Combined distant language optimizations (updated to use valid normalizations)
        echo "Testing combined distant language optimizations..."
        run_experiment "distant_combined" "python -u unsupervised.py --batch_size 16 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 10 --n_epochs 15 --map_beta 0.001 --map_optimizer sgd,lr=0.5 --map_id_init False --dis_dropout 0.5 --dis_input_dropout 0.5 --dis_steps 2 --dis_lambda 0.2 --dis_most_frequent 30000 --dis_smooth 0.3 --dis_optimizer sgd,lr=0.05 --dico_method csls_knn_10 --dico_build S2T\&T2S --dico_threshold 0.1 --dico_eval data/dictionaries/en-hi.5000-6500.txt --epoch_size 250000"
        ;;
    12)
        # Custom seed dictionary (uses a disjoint training set)
        echo "Testing with custom seed dictionary (disjoint from evaluation set)..."
        run_experiment "custom_seed_dict" "python -u unsupervised.py --batch_size 32 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 5 --dico_train custom_seed --dico_eval data/dictionaries/en-hi.5000-6500.txt"
        ;;
    13)
        # Custom seed dictionary with distant language optimizations
        echo "Testing with custom seed dictionary combined with distant language optimizations..."
        run_experiment "custom_seed_distant" "python -u unsupervised.py --batch_size 16 --normalize_embeddings center,renorm --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement 10 --n_epochs 15 --map_beta 0.001 --map_optimizer sgd,lr=0.5 --map_id_init False --dis_dropout 0.5 --dis_input_dropout 0.5 --dis_steps 2 --dis_lambda 0.2 --dis_most_frequent 30000 --dis_smooth 0.3 --dis_optimizer sgd,lr=0.05 --dico_method csls_knn_10 --dico_build S2T\&T2S --dico_threshold 0.1 --dico_train custom_seed --dico_eval data/dictionaries/en-hi.5000-6500.txt --epoch_size 250000"
        ;;
    14)
        # Hindi Latin with weak discriminator
        echo "Testing with weak discriminator for Hindi in Latin script..."
        run_experiment "hindi_latin_weak_disc" "$BASE_CMD --src_lang en --tgt_lang hi_latin --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi_latin.vec --n_refinement 8 --n_epochs 25 --epoch_size 500000 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.05 --dico_eval data/dictionaries/en-hi_latin.5000-6500_iast.txt --early_stopping_patience 5"
        ;;
    15)
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
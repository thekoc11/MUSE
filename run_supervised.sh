#!/bin/bash

# Create logs directory
mkdir -p experiment_logs

# Base command (with option to disable CUDA)
BASE_CMD="python -u muse_wrapper.py --src_lang en --tgt_lang hi --src_emb data/wiki.en.vec --tgt_emb data/wiki.hi.vec"

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
    
    # Create log file
    LOG_FILE="experiment_logs/${EXPERIMENT_NAME}.log"
    
    # Save the command to the log file
    echo "Command: $COMMAND" > $LOG_FILE
    
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
    
    echo ""
    echo "Experiment $EXPERIMENT_NAME completed and logged to $LOG_FILE"
    echo ""
}

# Run one experiment at a time to avoid memory issues
echo "Which experiment would you like to run? Options:"
echo "1. baseline (default settings)"
echo "2. normalization (center, renorm, both)"
echo "3. refinement iterations (3, 10)"
echo "4. dictionary building methods (union)"
echo "5. threshold values (0.1, 0.2)"
echo "6. orthogonalization strengths (0.001, 0.1)"
echo "7. CSLS k values (5, 15)"
echo "8. combined best settings"
echo "9. all experiments (warning: this may take a long time)"
echo "Enter a number (1-9):"
read EXPERIMENT_CHOICE

case $EXPERIMENT_CHOICE in
    1)
        # Run baseline experiment
        echo "Running baseline experiment..."
        run_experiment "baseline" "$BASE_CMD"
        ;;
    2)
        # Test different normalization options
        echo "Testing normalization options..."
        run_experiment "norm_center" "$BASE_CMD --normalize_embeddings center"
        run_experiment "norm_renorm" "$BASE_CMD --normalize_embeddings renorm"
        run_experiment "norm_center_renorm" "$BASE_CMD --normalize_embeddings center,renorm"
        ;;
    3)
        # Test different refinement iterations
        echo "Testing refinement iterations..."
        run_experiment "refinement_3" "$BASE_CMD --n_refinement 3"
        run_experiment "refinement_10" "$BASE_CMD --n_refinement 10"
        ;;
    4)
        # Test different dictionary building methods
        echo "Testing dictionary building methods..."
        run_experiment "build_S2T_T2S_union" "$BASE_CMD --dico_build \"S2T|T2S\""
        ;;
    5)
        # Test different threshold values
        echo "Testing threshold values..."
        run_experiment "threshold_0.1" "$BASE_CMD --dico_threshold 0.1"
        run_experiment "threshold_0.2" "$BASE_CMD --dico_threshold 0.2"
        ;;
    6)
        # Test different orthogonalization strengths
        echo "Testing orthogonalization strengths..."
        run_experiment "map_beta_0.001" "$BASE_CMD --map_beta 0.001"
        run_experiment "map_beta_0.1" "$BASE_CMD --map_beta 0.1"
        ;;
    7)
        # Test different CSLS k values
        echo "Testing CSLS k values..."
        run_experiment "csls_knn_5" "$BASE_CMD --dico_method csls_knn_5"
        run_experiment "csls_knn_15" "$BASE_CMD --dico_method csls_knn_15"
        ;;
    8)
        # Combine best settings based on individual experiments
        echo "Testing combined best settings..."
        run_experiment "combined_best" "$BASE_CMD --dico_build \"S2T|T2S\" --n_refinement 10 --dico_method csls_knn_15 --normalize_embeddings \"center,renorm\""
        ;;
    9)
        # Run baseline experiment
        echo "Running baseline experiment..."
        run_experiment "baseline" "$BASE_CMD"

        # Test different normalization options
        echo "Testing normalization options..."
        run_experiment "norm_center" "$BASE_CMD --normalize_embeddings center"
        run_experiment "norm_renorm" "$BASE_CMD --normalize_embeddings renorm"
        run_experiment "norm_center_renorm" "$BASE_CMD --normalize_embeddings center,renorm"

        # Test different refinement iterations
        echo "Testing refinement iterations..."
        run_experiment "refinement_3" "$BASE_CMD --n_refinement 3"
        run_experiment "refinement_10" "$BASE_CMD --n_refinement 10"

        # Test different dictionary building methods
        echo "Testing dictionary building methods..."
        run_experiment "build_S2T_T2S_union" "$BASE_CMD --dico_build S2T|T2S"

        # Test different threshold values
        echo "Testing threshold values..."
        run_experiment "threshold_0.1" "$BASE_CMD --dico_threshold 0.1"
        run_experiment "threshold_0.2" "$BASE_CMD --dico_threshold 0.2"

        # Test different orthogonalization strengths
        echo "Testing orthogonalization strengths..."
        run_experiment "map_beta_0.001" "$BASE_CMD --map_beta 0.001"
        run_experiment "map_beta_0.1" "$BASE_CMD --map_beta 0.1"

        # Test different CSLS k values
        echo "Testing CSLS k values..."
        run_experiment "csls_knn_5" "$BASE_CMD --dico_method csls_knn_5"
        run_experiment "csls_knn_15" "$BASE_CMD --dico_method csls_knn_15"

        # Combine best settings based on individual experiments
        echo "Testing combined best settings..."
        run_experiment "combined_best" "$BASE_CMD --normalize_embeddings center,renorm --dico_method csls_knn_15 --dico_threshold 0.1"
        ;;
    *)
        echo "Invalid option. Running baseline experiment..."
        run_experiment "baseline" "$BASE_CMD"
        ;;
esac

echo "Experiments completed. Logs are in the experiment_logs directory."
import json
import os
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from river import cluster
from river import feature_extraction
from river import compose

# ==========================================
# 1. SCIENTIFIC CONFIGURATION
# ==========================================
DATASET_CONFIG = {
    "Tweets-T": {
        "filename": "Tweets-T",
        "fading_factor": 0.01,
        "tgap": 200,
        "horizon": 1000
    },
    "News-T": {
        "filename": "News-T",
        "fading_factor": 0.001,
        "tgap": 200,
        "horizon": 1000
    },
    "NT": {
        "filename": "NT",
        "fading_factor": 0.005,
        "tgap": 200,
        "horizon": 2000
    },
    "NTS": {
        "filename": "NTS",
        "fading_factor": 0.005,
        "tgap": 200,
        "horizon": 2000
    },
    "Trends-T": {
        "filename": "Trends-T",
        "fading_factor": 0.01,
        "tgap": 200,
        "horizon": 5000      
    },
    "SO-T": {
        "filename": "SO-T",
        "fading_factor": 0.001,
        "tgap": 200,
        "horizon": 5000
    }
}

DATASETS_FOLDER = 'datasets' 

# ==========================================
# 2. PIPELINE BUILDER
# ==========================================
def create_pipeline(auto_r_status, config):
    return compose.Pipeline(
        feature_extraction.TFIDF(ngram_range=(1, 1)), 
        cluster.TextClust(
            real_time_fading=False, 
            auto_r=auto_r_status,
            radius=0.3,
            fading_factor=config["fading_factor"], 
            tgap=config["tgap"],
            sigma=0.5
        )
    )

# ==========================================
# 3. DATA LOADING
# ==========================================
def load_stream(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Warning: File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            try:
                record = json.loads(line)
                # Try finding keys
                text = record.get('textCleaned') or record.get('body') or record.get('text') or record.get('title')
                label = record.get('clusterNo') or record.get('class') or record.get('label')
                
                if text and label is not None:
                    yield line_no, text, label
            except json.JSONDecodeError:
                continue 

# ==========================================
# 4. BENCHMARK ENGINE (ROBUST)
# ==========================================
def run_benchmark_on_dataset(name, config):
    raw_path = os.path.join(DATASETS_FOLDER, config["filename"])
    if os.path.exists(raw_path):
        filepath = raw_path
    elif os.path.exists(raw_path + ".json"):
        filepath = raw_path + ".json"
    else:
        print(f"❌ SKIPPING {name}: File not found.")
        return

    print(f"\n=== 🚀 PROCESSING: {name} ===")
    print(f"⚙️ Config: Fading(λ)={config['fading_factor']} | Horizon={config['horizon']}")

    model_fixed = create_pipeline(False, config)
    model_adaptive = create_pipeline(True, config)

    x_indices = []
    y_fixed = []
    y_adaptive = []
    
    buffer_true = []
    buffer_pred_fixed = []
    buffer_pred_adaptive = []
    
    stream = load_stream(filepath)
    count = 0
    errors_caught = 0

    for i, text, true_label in stream:
        count += 1
        
        try:
            # --- ROBUST LEARNING BLOCK ---
            model_fixed.learn_one(text)
            model_adaptive.learn_one(text)
            
            # Prediction
            p_fixed = model_fixed.predict_one(text)
            p_adaptive = model_adaptive.predict_one(text)
            
            buffer_true.append(true_label)
            buffer_pred_fixed.append(p_fixed)
            buffer_pred_adaptive.append(p_adaptive)
            
        except ValueError as e:
            # Catch math domain errors (sqrt of negative number) and skip
            if "math domain error" in str(e):
                errors_caught += 1
                if errors_caught <= 5: # Only print the first few to avoid spam
                    print(f"⚠️ Warning: Math error at tweet {count}. Skipping observation.")
                continue
            else:
                raise e # Re-raise legitimate errors

        # --- Evaluation at Horizon ---
        if count % config["horizon"] == 0:
            clean_fixed = [-1 if p is None else p for p in buffer_pred_fixed]
            clean_adaptive = [-1 if p is None else p for p in buffer_pred_adaptive]
            
            nmi_f = normalized_mutual_info_score(buffer_true, clean_fixed)
            nmi_a = normalized_mutual_info_score(buffer_true, clean_adaptive)
            
            x_indices.append(count)
            y_fixed.append(nmi_f)
            y_adaptive.append(nmi_a)
            
            print(f"   Step {count}: Fixed={nmi_f:.3f} | Adaptive={nmi_a:.3f}")
            
            buffer_true = []
            buffer_pred_fixed = []
            buffer_pred_adaptive = []

    # === PLOTTING ===
    if len(x_indices) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(x_indices, y_fixed, label='Fixed (r=0.3)', color='red', linestyle='--', alpha=0.8)
        plt.plot(x_indices, y_adaptive, label='Adaptive (σ=0.5)', color='green', linewidth=2)
        
        plt.title(f'Robustness Validation: {name} (λ={config["fading_factor"]})')
        plt.xlabel('Stream Position')
        plt.ylabel('NMI Score')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        safe_name = name.replace(" ", "_")
        filename = f"result_{safe_name}.png"
        plt.savefig(filename)
        print(f"✅ Saved Plot: {filename} (Total skipped errors: {errors_caught})")
        plt.close()
    else:
        print(f"⚠️ No data processed for {name}.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- STARTING ROBUST MULTI-DATASET BENCHMARK ---")
    
    for dataset_name, config in DATASET_CONFIG.items():
        try:
            run_benchmark_on_dataset(dataset_name, config)
        except Exception as e:
            print(f"💥 Unrecoverable Error on {dataset_name}: {e}")
            traceback.print_exc() # Print full error for debugging

    print("\n--- ALL BENCHMARKS FINISHED ---")
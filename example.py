"""
DNN_PSO - Example Usage Script
Demonstrates training a Deep Neural Network using Particle Swarm Optimization (PSO)
without backpropagation, using the prodnnv10 class from class_prodnn.py.
"""

import os
import json
from class_prodnn import prodnnv10

def main():
    print("=" * 70)
    print("       DNN + PSO: Deep Neural Network Trained by Particle Swarms")
    print("=" * 70)

    # 1. Resolve paths to data files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    x_path = os.path.join(base_dir, "data", "data_x.json")
    y_path = os.path.join(base_dir, "data", "data_y.json")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found in {os.path.join(base_dir, 'data')}")

    # 2. Configure model hyperparameters
    input_neurons = 3          # Number of input features
    output_neurons = 1         # Number of output targets
    transition_per = 2 / 3     # Layer transition compression ratio
    train_size = 50            # Iterations per PSO run
    particle_count = 15        # Number of particles in the swarm
    loop_size = 2              # Number of PSO restart cycles to find the global optimum

    print("\n[+] Initializing prodnnv10 model...")
    model = prodnnv10(
        input_neuron_numbers=input_neurons,
        output_neuron_numbers=output_neurons,
        transition_per=transition_per,
        x_input=x_path,
        y_input=y_path,
        train_size=train_size,
        particle=particle_count,
        loop_size=loop_size
    )

    print(f"[*] Architecture Overview:")
    print(f"    - Input Layer:      {model.all_layers[0]} neurons")
    print(f"    - Hidden Layer(s):  {model.hidden_neuron_numbers} neurons")
    print(f"    - Output Layer:     {model.all_layers[-1]} neurons")
    print(f"    - Layer Structure:  {model.all_layers}")
    print(f"    - PSO Search Space: {model.dimensions} weight dimensions to optimize")

    # 3. Train the model using PSO
    print(f"\n[+] Starting PSO Optimization ({loop_size} cycle(s) x {train_size} iterations)...")
    cost_history = model.trainer()

    best_cost = list(cost_history.keys())[0]
    print(f"\n[OK] PSO Optimization finished!")
    print(f"[*] Global Best MSE Cost: {best_cost:.6f}")

    # 4. Load the optimal weights into the model
    print("[+] Uploading best swarm weights to neural network...")
    model.bestweight_upload()

    # 5. Evaluate and display predictions
    print("\n" + "=" * 50)
    print("               PREDICTION RESULTS")
    print("=" * 50)
    print(f"{'Sample':<8} {'Input (X)':<18} {'Actual (Y)':<12} {'Raw Output':<14} {'Binary Pred':<12}")
    print("-" * 65)

    with open(x_path, "r") as f:
        x_samples = json.load(f)
    with open(y_path, "r") as f:
        y_actual = json.load(f)

    # Trigger forward pass with best weights
    model.propagate_forward()
    predicted_outputs = model.full_outputs_lastmend

    correct = 0
    for idx, (x_val, y_true, y_pred) in enumerate(zip(x_samples, y_actual, predicted_outputs), 1):
        bin_pred = 1 if y_pred >= 0.5 else 0
        is_correct = (bin_pred == y_true)
        if is_correct:
            correct += 1
        status_str = "CORRECT" if is_correct else "WRONG"
        print(f"#{idx:<7} {str(x_val):<18} {y_true:<12} {y_pred:<14.4f} {bin_pred:<12} {status_str}")

    accuracy = (correct / len(y_actual)) * 100
    print("-" * 65)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(y_actual)} correct)")
    print("=" * 50)

    # 6. Plot cost curve
    try:
        print("\n[+] Plotting cost convergence curve...")
        model.cost_effect()
    except Exception as e:
        print(f"Note: Could not display interactive plot: {e}")

if __name__ == "__main__":
    main()

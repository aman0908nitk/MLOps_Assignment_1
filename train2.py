from sklearn.kernel_ridge import KernelRidge
import misc

def main():
    # Load dataset 
    df = misc.load_data()
    
    # Process features with standard scaling enabled to ensure stable kernel operations
    X_train, X_test, y_train, y_test = misc.preprocess_data(
        df, target_column='MEDV', test_size=0.25, random_state=42, scale=True
    )
    
    # Initialize the KernelRidge model with linear model
    model = KernelRidge(alpha=1.0, kernel='linear')
    
    # Train and evaluate the model
    trained_model = misc.train_model(model, X_train, y_train)
    test_mse = misc.evaluate_model(trained_model, X_test, y_test)
    
    print("Kernel Ridge Regressor pipeline execution complete.")
    print(f"Average MSE score on the test set for this model: {test_mse:.4f}")

if __name__ == "__main__":
    main()

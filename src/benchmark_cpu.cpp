#include <iostream>
#include <chrono>
#include <thread>

#include "cuda_mppi_controller/config.h"
#include "cuda_mppi_controller/mppi_controller.h"

using namespace mppi_controller;

int main(int argc, char** argv) {
    std::cout << "MPPI Controller CPU Benchmark" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Parse command line arguments
    int iterations = 100;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0) {
            std::cerr << "Invalid number of iterations: " << iterations << std::endl;
            return 1;
        }
    }
    
    std::cout << "Number of iterations: " << iterations << std::endl;
    
    try {
        // Initialize MPPI controller
        MPPIController controller;
        
        // Set some initial states for testing
        State current_state;
        current_state << 0.0f, 0.0f, 0.0f;  // x, y, theta
        
        State target_state;
        target_state << 5.0f, 5.0f, 0.0f;   // x, y, theta
        
        controller.SetCurrentState(current_state);
        controller.SetTargetState(target_state);
        
        std::cout << "Controller initialized successfully" << std::endl;
        std::cout << "Current state: [" << current_state.transpose() << "]" << std::endl;
        std::cout << "Target state: [" << target_state.transpose() << "]" << std::endl;
        
        // Run the full benchmark
        controller.RunFullBenchmark(iterations);
        
        std::cout << "\nBenchmark completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
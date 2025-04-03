#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <deque>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>  // for sqrt
#include <cstdlib>
#include <ctime>

// DQN Configuration
#define FRAME_HEIGHT 84
#define FRAME_WIDTH 84
#define FRAME_HISTORY 4
#define STATE_SIZE (FRAME_HEIGHT * FRAME_WIDTH * FRAME_HISTORY)
#define MAX_ACTIONS 18  // Maximum number of actions in Atari games
#define REPLAY_MEMORY_SIZE 1000000
#define BATCH_SIZE 32
#define GAMMA 0.99f
#define INITIAL_EXPLORATION 1.0f
#define FINAL_EXPLORATION 0.1f
#define FINAL_EXPLORATION_FRAME 1000000
#define TARGET_NETWORK_UPDATE_FREQUENCY 10000
#define LEARNING_RATE 0.00025f
#define GRADIENT_MOMENTUM 0.95f
#define SQUARED_GRADIENT_MOMENTUM 0.95f
#define MIN_SQUARED_GRADIENT 0.01f

// CUDA Error Checking
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Experience Replay Memory Structure
struct Experience {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    bool terminal;
};

class ReplayMemory {
private:
    std::deque<Experience> memory;
    size_t max_size;
    std::mt19937 rng;

public:
    ReplayMemory(size_t size) : max_size(size), rng(std::random_device{}()) {}

    void add(const Experience& exp) {
        if (memory.size() >= max_size) {
            memory.pop_front();
        }
        memory.push_back(exp);
    }

    void sample(std::vector<Experience>& batch, size_t batch_size) {
        batch.clear();
        if (memory.size() < batch_size) return;

        std::uniform_int_distribution<size_t> dist(0, memory.size() - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(memory[dist(rng)]);
        }
    }

    size_t size() const {
        return memory.size();
    }
};

// CUDA kernel implementations (free functions)

__global__ void conv2d(float* input, float* weights, float* bias, float* output,
                       int input_h, int input_w, int channels,
                       int kernel_size, int stride, int num_filters) {
    int filter_idx = threadIdx.x;
    int out_x = threadIdx.y;
    int out_y = threadIdx.z;

    if (filter_idx < num_filters) {
        float sum = bias[filter_idx];
        int in_x_start = out_x * stride;
        int in_y_start = out_y * stride;

        for (int c = 0; c < channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = in_x_start + kx;
                    int in_y = in_y_start + ky;
                    if (in_x < input_w && in_y < input_h) {
                        int input_idx = c * input_h * input_w + in_y * input_w + in_x;
                        int weight_idx = filter_idx * channels * kernel_size * kernel_size +
                                         c * kernel_size * kernel_size +
                                         ky * kernel_size + kx;
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        int out_idx = filter_idx * gridDim.y * gridDim.z + out_y * gridDim.y + out_x;
        output[out_idx] = sum;
    }
}

__global__ void relu_activation(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = max(0.0f, input[idx]);
    }
}

__global__ void fully_connected(float* input, float* weights, float* bias, float* output,
                                int input_size, int output_size) {
    int out_idx = threadIdx.x;
    if (out_idx < output_size) {
        float sum = bias[out_idx];
        for (int in_idx = 0; in_idx < input_size; in_idx++) {
            sum += input[in_idx] * weights[in_idx * output_size + out_idx];
        }
        output[out_idx] = sum;
    }
}

// Convolutional Neural Network for DQN
class ConvolutionalNetwork {
private:
    // Device memory pointers for weights and biases
    float *d_conv1_weights, *d_conv1_bias;
    float *d_conv2_weights, *d_conv2_bias;
    float *d_fc1_weights, *d_fc1_bias;
    float *d_fc2_weights, *d_fc2_bias;
    
    // Device memory for intermediate results
    float *d_input;
    float *d_conv1_output, *d_conv2_output;
    float *d_fc1_output, *d_output;
    
    // Network dimensions
    int num_actions;
    
    // CUDA handles
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
    
    // Initialize weights with Xavier/Glorot initialization
    void init_weights(float* d_weights, int fan_in, int fan_out) {
        float scale = sqrt(6.0f / (fan_in + fan_out));
        curandGenerateUniform(curand_gen, d_weights, fan_in * fan_out);
        float h_scale = 2.0f * scale;
        float h_offset = -scale;
        std::vector<float> h_weights(fan_in * fan_out);
        CHECK_CUDA_ERROR(cudaMemcpy(h_weights.data(), d_weights, sizeof(float) * fan_in * fan_out, cudaMemcpyDeviceToHost));
        for (int i = 0; i < fan_in * fan_out; i++) {
            h_weights[i] = h_weights[i] * h_scale + h_offset;
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights.data(), sizeof(float) * fan_in * fan_out, cudaMemcpyHostToDevice));
    }
    
    // Initialize biases to small constant value
    void init_biases(float* d_biases, int size, float value = 0.01f) {
        std::vector<float> h_biases(size, value);
        CHECK_CUDA_ERROR(cudaMemcpy(d_biases, h_biases.data(), sizeof(float) * size, cudaMemcpyHostToDevice));
    }

public:
    ConvolutionalNetwork(int actions) : num_actions(actions) {
        // Create CUDA handles
        cublasCreate(&cublas_handle);
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, static_cast<unsigned long long>(time(NULL)));
        
        // Allocate memory for weights and biases
        // Layer 1: Convolutional 8x8 with stride 4, 16 filters
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_weights, sizeof(float) * 8 * 8 * FRAME_HISTORY * 16));
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_bias, sizeof(float) * 16));
        
        // Layer 2: Convolutional 4x4 with stride 2, 32 filters
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_weights, sizeof(float) * 4 * 4 * 16 * 32));
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_bias, sizeof(float) * 32));
        
        // Layer 3: Fully connected with 256 rectifier units
        int conv_output_size = 9 * 9 * 32; // After two convolutions
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_weights, sizeof(float) * conv_output_size * 256));
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_bias, sizeof(float) * 256));
        
        // Layer 4: Fully connected output layer
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc2_weights, sizeof(float) * 256 * num_actions));
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc2_bias, sizeof(float) * num_actions));
        
        // Allocate memory for intermediate results
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, sizeof(float) * STATE_SIZE));
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_output, sizeof(float) * 20 * 20 * 16)); // After 1st conv
        CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_output, sizeof(float) * 9 * 9 * 32));   // After 2nd conv
        CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_output, sizeof(float) * 256));            // After FC
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float) * num_actions));        // Final output
        
        // Initialize weights and biases
        init_weights(d_conv1_weights, 8 * 8 * FRAME_HISTORY, 16);
        init_biases(d_conv1_bias, 16);
        
        init_weights(d_conv2_weights, 4 * 4 * 16, 32);
        init_biases(d_conv2_bias, 32);
        
        init_weights(d_fc1_weights, conv_output_size, 256);
        init_biases(d_fc1_bias, 256);
        
        init_weights(d_fc2_weights, 256, num_actions);
        init_biases(d_fc2_bias, num_actions);
    }
    
    ~ConvolutionalNetwork() {
        // Free allocated memory
        cudaFree(d_conv1_weights);
        cudaFree(d_conv1_bias);
        cudaFree(d_conv2_weights);
        cudaFree(d_conv2_bias);
        cudaFree(d_fc1_weights);
        cudaFree(d_fc1_bias);
        cudaFree(d_fc2_weights);
        cudaFree(d_fc2_bias);
        
        cudaFree(d_input);
        cudaFree(d_conv1_output);
        cudaFree(d_conv2_output);
        cudaFree(d_fc1_output);
        cudaFree(d_output);
        
        cublasDestroy(cublas_handle);
        curandDestroyGenerator(curand_gen);
    }
    
    // Clone network weights to target network
    void clone_to(ConvolutionalNetwork& target) {
        cudaMemcpy(target.d_conv1_weights, d_conv1_weights, sizeof(float) * 8 * 8 * FRAME_HISTORY * 16, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_conv1_bias, d_conv1_bias, sizeof(float) * 16, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_conv2_weights, d_conv2_weights, sizeof(float) * 4 * 4 * 16 * 32, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_conv2_bias, d_conv2_bias, sizeof(float) * 32, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_fc1_weights, d_fc1_weights, sizeof(float) * 9 * 9 * 32 * 256, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_fc1_bias, d_fc1_bias, sizeof(float) * 256, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_fc2_weights, d_fc2_weights, sizeof(float) * 256 * num_actions, cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.d_fc2_bias, d_fc2_bias, sizeof(float) * num_actions, cudaMemcpyDeviceToDevice);
    }
    
    // Forward pass implementation
    void forward(float* state, float* q_values) {
        // Copy input state to device
        cudaMemcpy(d_input, state, sizeof(float) * STATE_SIZE, cudaMemcpyHostToDevice);
        
        // First convolutional layer + ReLU
        dim3 conv1_grid(1, 1, 1);
        dim3 conv1_block(16, 20, 20); // One thread per output element
        conv2d<<<conv1_grid, conv1_block>>>(d_input, d_conv1_weights, d_conv1_bias, d_conv1_output, 
                                             FRAME_WIDTH, FRAME_HEIGHT, FRAME_HISTORY, 
                                             8, 4, 16);
        relu_activation<<<16, 20*20>>>(d_conv1_output, 20 * 20);
        
        // Second convolutional layer + ReLU
        dim3 conv2_grid(1, 1, 1);
        dim3 conv2_block(32, 9, 9);
        conv2d<<<conv2_grid, conv2_block>>>(d_conv1_output, d_conv2_weights, d_conv2_bias, d_conv2_output,
                                            20, 20, 16,
                                            4, 2, 32);
        relu_activation<<<32, 9*9>>>(d_conv2_output, 9 * 9);
        
        // First fully connected layer + ReLU
        dim3 fc1_grid(1, 1, 1);
        dim3 fc1_block(256, 1, 1);
        fully_connected<<<fc1_grid, fc1_block>>>(d_conv2_output, d_fc1_weights, d_fc1_bias, d_fc1_output,
                                                 9 * 9 * 32, 256);
        relu_activation<<<1, 256>>>(d_fc1_output, 256);
        
        // Output layer
        dim3 fc2_grid(1, 1, 1);
        dim3 fc2_block(num_actions, 1, 1);
        fully_connected<<<fc2_grid, fc2_block>>>(d_fc1_output, d_fc2_weights, d_fc2_bias, d_output,
                                                 256, num_actions);
        
        // Copy results back to host
        cudaMemcpy(q_values, d_output, sizeof(float) * num_actions, cudaMemcpyDeviceToHost);
    }
};

// Deep Q-Network Agent implementation
class DQNAgent {
private:
    std::unique_ptr<ConvolutionalNetwork> q_network;
    std::unique_ptr<ConvolutionalNetwork> target_network;
    std::unique_ptr<ReplayMemory> replay_memory;
    
    int num_actions;
    int frame_count;
    float epsilon;
    std::mt19937 rng;
    
    // Preprocess a frame (for example, grayscale and normalization)
    void preprocess_frame(uint8_t* raw_frame, float* processed_frame) {
        for (int i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; i++) {
            processed_frame[i] = static_cast<float>(raw_frame[i]) / 255.0f;
        }
    }
    
    // Update epsilon value based on frame count
    void update_epsilon() {
        if (frame_count < FINAL_EXPLORATION_FRAME) {
            epsilon = INITIAL_EXPLORATION - 
                      frame_count * (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME;
        } else {
            epsilon = FINAL_EXPLORATION;
        }
    }

public:
    DQNAgent(int actions) : num_actions(actions), frame_count(0), epsilon(INITIAL_EXPLORATION),
                           rng(std::random_device{}()) {
        q_network = std::make_unique<ConvolutionalNetwork>(actions);
        target_network = std::make_unique<ConvolutionalNetwork>(actions);
        replay_memory = std::make_unique<ReplayMemory>(REPLAY_MEMORY_SIZE);
        q_network->clone_to(*target_network);
    }
    
    // Epsilon-greedy action selection
    int select_action(float* state) {
        std::uniform_real_distribution<float> dist(0, 1);
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, num_actions - 1);
            return action_dist(rng);
        } else {
            float q_values[MAX_ACTIONS];
            q_network->forward(state, q_values);
            return std::distance(q_values, std::max_element(q_values, q_values + num_actions));
        }
    }
    
    // Store experience in replay memory
    void store_experience(float* state, int action, float reward, float* next_state, bool terminal) {
        Experience exp;
        std::copy(state, state + STATE_SIZE, exp.state);
        exp.action = action;
        exp.reward = reward;
        std::copy(next_state, next_state + STATE_SIZE, exp.next_state);
        exp.terminal = terminal;
        replay_memory->add(exp);
    }
    
    // Perform one training step
    void train() {
        if (replay_memory->size() < BATCH_SIZE) return;
        
        std::vector<Experience> batch;
        replay_memory->sample(batch, BATCH_SIZE);
        
        float states[BATCH_SIZE][STATE_SIZE];
        int actions[BATCH_SIZE];
        float rewards[BATCH_SIZE];
        float next_states[BATCH_SIZE][STATE_SIZE];
        bool terminals[BATCH_SIZE];
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            const Experience& exp = batch[i];
            std::copy(exp.state, exp.state + STATE_SIZE, states[i]);
            actions[i] = exp.action;
            rewards[i] = exp.reward;
            std::copy(exp.next_state, exp.next_state + STATE_SIZE, next_states[i]);
            terminals[i] = exp.terminal;
        }
        
        float target_q_values[BATCH_SIZE][MAX_ACTIONS];
        for (int i = 0; i < BATCH_SIZE; i++) {
            target_network->forward(next_states[i], target_q_values[i]);
        }
        
        float current_q_values[BATCH_SIZE][MAX_ACTIONS];
        for (int i = 0; i < BATCH_SIZE; i++) {
            q_network->forward(states[i], current_q_values[i]);
        }
        
        float targets[BATCH_SIZE];
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (terminals[i]) {
                targets[i] = rewards[i];
            } else {
                float max_next_q = *std::max_element(target_q_values[i], target_q_values[i] + num_actions);
                targets[i] = rewards[i] + GAMMA * max_next_q;
            }
        }
        
        // RMSProp optimization and weight update would go here
        
        frame_count++;
        update_epsilon();
        if (frame_count % TARGET_NETWORK_UPDATE_FREQUENCY == 0) {
            q_network->clone_to(*target_network);
        }
    }
    
    // Process frame and train the network
    void process_frame_and_train(uint8_t* raw_frame, int action, float reward, bool terminal) {
        static float current_state[STATE_SIZE] = {0};
        static float next_state[STATE_SIZE] = {0};
        
        std::copy(current_state + FRAME_WIDTH * FRAME_HEIGHT, 
                  current_state + STATE_SIZE, 
                  current_state);
        
        float processed_frame[FRAME_WIDTH * FRAME_HEIGHT];
        preprocess_frame(raw_frame, processed_frame);
        std::copy(processed_frame, 
                  processed_frame + FRAME_WIDTH * FRAME_HEIGHT, 
                  current_state + STATE_SIZE - FRAME_WIDTH * FRAME_HEIGHT);
        
        if (frame_count >= FRAME_HISTORY - 1) {
            if (action < 0) {
                action = select_action(current_state);
            }
            std::copy(current_state, current_state + STATE_SIZE, next_state);
            store_experience(current_state, action, reward, next_state, terminal);
            train();
        } else {
            frame_count++;
        }
    }
};

// Simple example of usage
int main() {
    const int NUM_ACTIONS = 4;
    DQNAgent agent(NUM_ACTIONS);
    
    uint8_t dummy_frame[FRAME_WIDTH * FRAME_HEIGHT] = {0};
    
    // Simulate 100 frames
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < FRAME_WIDTH * FRAME_HEIGHT; j++) {
            dummy_frame[j] = rand() % 256;
        }
        
        float reward = (i % 10 == 0) ? 1.0f : 0.0f;
        bool terminal = (i == 99);
        
        agent.process_frame_and_train(dummy_frame, -1, reward, terminal);
    }
    
    std::cout << "Training completed for 100 frames!" << std::endl;
    return 0;
}

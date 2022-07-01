#include <cstdlib>
#include <iostream>

// Maximum number of errors to print out.
#define MAX_PRINT_ERRORS 10

#include "../../src/graph.cuh"
#include "../../src/kernels/cpu/pr.cuh"
#include "../../src/kernels/gpu/pr.cuh"
#include "../../src/kernels/heterogeneous/pr.cuh"

/** Forward decl. */
bool verify(const weight_t *oracle_score, const weight_t *score, 
        const nid_t num_nodes);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.wsg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load graph.
    CSRWGraph g = load_graph_from_file<CSRWGraph>(argv[1]);
    std::cout << "Loaded graph." << std::endl;

    SourcePicker<CSRWGraph> sp(&g);
    nid_t source_id = sp.next_vertex();

    // Compute oracle score.
    weight_t *oracle_score = nullptr;
    /*pr_dijkstras_cpu_serial(g, source_id, &oracle_score);*/
    //version that tests against serial
    //pr_pull_cpu_serial(g, source_id, &oracle_score);
    //std::cout << "Computed oracle score." << std::endl;

    // Generate init score.
    weight_t *init_score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_score[i] = 1.0f/g.num_nodes;
    init_score[source_id] = 0.0f;

    {
        weight_t *score = nullptr;

	//version that test against cpu result
	pr_pull_cpu(g, epoch_pr_pull_cpu_one_to_one, init_score, &oracle_score);
	std::cout << "Computed oracle score." << std::endl;
        pr_pull_cpu(g, epoch_pr_pull_cpu_one_to_one, init_score, &score);

        std::cout << "Verifying PR CPU kernel ..." << std::endl;
        bool success = verify(oracle_score, score, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] score;
    }

    // Check PR GPU naive kernel.
    {
        weight_t *score = nullptr;
        pr_pull_gpu(g, epoch_pr_pull_gpu_one_to_one, init_score, &score);

        std::cout << "Verifying PR GPU naive kernel ..." << std::endl;
        bool success = verify(oracle_score, score, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] score;
    }

    // Check PR GPU warp red kernel.
    {
        weight_t *score = nullptr;
        pr_pull_gpu(g, epoch_pr_pull_gpu_warp_red, init_score, &score);

        std::cout << "Verifying PR GPU warp red kernel ..." << std::endl;
        bool success = verify(oracle_score, score, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] score;
    }

    // Check PR GPU block red kernel.
    {
        weight_t *score = nullptr;
        pr_pull_gpu(g, epoch_pr_pull_gpu_block_red, init_score, &score,
                64, 256);

        std::cout << "Verifying PR GPU block red kernel ..." << std::endl;
        bool success = verify(oracle_score, score, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] score;
    }

    // Check PR heterogeneous kernel.
    {
        enable_all_peer_access_pr();
        weight_t *score = nullptr;
        pr_pull_heterogeneous(g, init_score, &score);

        std::cout << "Verifying PR heterogeneous kernel ..." << std::endl;
        bool success = verify(oracle_score, score, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] score;
    }

    delete[] oracle_score;

    return EXIT_SUCCESS;
}

/**
 * Verifies that the computed scores is the same as the oracle score.
 * Parameters:
 *   - oracle_score <- correct scoreance.
 *   - score        <- computed score (score to check).
 *   - num_nodes   <- number of nodes in the graph.
 * Returns:
 *   true if the computed score is correct.
 */
bool verify(const weight_t *oracle_score, const weight_t *score,
        const nid_t num_nodes
) {
    bool is_correct = true;
    nid_t error_count = 0;
    float epsilon=.0005;

    for (nid_t nid = 0; nid < num_nodes; nid++) {
        if (abs(oracle_score[nid]-score[nid])>epsilon) {
            if (error_count < MAX_PRINT_ERRORS)
                std::cout << " > " << nid << ": " 
                    << score[nid] << " != " << oracle_score[nid] << std::endl;
            is_correct = false;
            error_count++;
        }
    }

    // Print extra error count if any.
    if (error_count >= MAX_PRINT_ERRORS) {
        nid_t more_error_count = error_count - MAX_PRINT_ERRORS;
        std::cout << " > ... " << more_error_count << " more error"
            << (more_error_count != 1 ? "s" : "") << "!" << std::endl;
    }

    return is_correct;
}

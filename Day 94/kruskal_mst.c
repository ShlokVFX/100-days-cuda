#include <stdio.h>
#include <stdlib.h>

// Structure to represent an edge
typedef struct Edge {
    int src, dest, weight;
} Edge;

// Structure for subset in Union-Find
typedef struct Subset {
    int parent;
    int rank;
} Subset;

// Function prototypes
int find(Subset subsets[], int i);
void Union(Subset subsets[], int x, int y);
int compareEdges(const void* a, const void* b);
void KruskalMST(Edge edges[], int V, int E);
int calculateTotalCost(Edge result[], int size);
void sortByNodeNumbers(Edge result[], int size);

int main() {
    FILE *inputFile = fopen("input.txt", "r");
    FILE *outputFile = fopen("output.txt", "w");
    
    if (inputFile == NULL || outputFile == NULL) {
        printf("Error opening files.\n");
        return 1;
    }
    
    int V, E;
    fscanf(inputFile, "%d %d", &V, &E);
    
    Edge *edges = (Edge*) malloc(E * sizeof(Edge));
    
    for (int i = 0; i < E; i++) {
        fscanf(inputFile, "%d %d %d", &edges[i].src, &edges[i].dest, &edges[i].weight);
    }
    
    // Redirect stdout to output file for the results
    freopen("output.txt", "w", stdout);
    
    KruskalMST(edges, V, E);
    
    fclose(inputFile);
    fclose(outputFile);
    free(edges);
    
    return 0;
}

// Find set of an element i (uses path compression)
int find(Subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

// Union of two sets by rank
void Union(Subset subsets[], int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    // Attach smaller rank tree under root of higher rank tree
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        // If ranks are same, make one as root and increment its rank
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

// Compare function for qsort to sort edges by weight
int compareEdges(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

// Function to calculate total cost of MST
int calculateTotalCost(Edge result[], int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += result[i].weight;
    }
    return total;
}

// Function to ensure smaller node is always src and sort by node numbers
void sortByNodeNumbers(Edge result[], int size) {
    // First, ensure smaller node is always src
    for (int i = 0; i < size; i++) {
        if (result[i].src > result[i].dest) {
            int temp = result[i].src;
            result[i].src = result[i].dest;
            result[i].dest = temp;
        }
    }
    
    // Sort by src
    for (int i = 0; i < size-1; i++) {
        for (int j = 0; j < size-i-1; j++) {
            if (result[j].src > result[j+1].src) {
                Edge temp = result[j];
                result[j] = result[j+1];
                result[j+1] = temp;
            } else if (result[j].src == result[j+1].src && result[j].dest > result[j+1].dest) {
                // For same src, sort by dest
                Edge temp = result[j];
                result[j] = result[j+1];
                result[j+1] = temp;
            }
        }
    }
}

// Main function to construct MST using Kruskal's algorithm
void KruskalMST(Edge edges[], int V, int E) {
    // Allocate memory for results
    Edge result[V-1];  // MST will have V-1 edges
    int e = 0;  // Index for result[]
    int i = 0;  // Index for sorted edges
    
    // Sort edges in non-decreasing order of their weight
    qsort(edges, E, sizeof(Edge), compareEdges);
    
    // Create V subsets with single elements
    Subset *subsets = (Subset*) malloc(V * sizeof(Subset));
    for (int v = 0; v < V; v++) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }
    
    // Number of edges to be taken is equal to V-1
    while (e < V - 1 && i < E) {
        // Pick the smallest edge
        Edge next_edge = edges[i++];
        
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);
        
        // If including this edge doesn't cause cycle, include it
        if (x != y) {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
    }
    
    // Calculate total cost
    int total_cost = calculateTotalCost(result, e);
    
    // Sort result by node numbers for output
    sortByNodeNumbers(result, e);
    
    // Print the constructed MST
    printf("Minimum cost: %d\n\n", total_cost);
    printf("Connections to establish:\n");
    
    for (i = 0; i < e; i++) {
        printf("%d-%d\n", result[i].src, result[i].dest);
    }
    
    free(subsets);
}
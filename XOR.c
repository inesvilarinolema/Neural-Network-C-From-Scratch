#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*Inés Vilariño Lema*/

/* XOR PROBLEM using Backpropagation */

//Training Data (XOR Truth Table)
float train_inputs[4][3] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1}
};

//Target Outputs
float train_outputs[4][1] = {{0}, {1},{1}, {0} };

//Weights
float v[2][3]; 

//Output Layer receives y1, y2, y3.
float w[3]; 

float y[3]; //Output of hidden layer (y1, y2, y3=-1)
float z; //Final output

float eta = 0.5;  
float beta = 1.0;  
float eMax = 0.001; 


float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-beta * x));
}

float sigmoid_derivative(float out) {
    return beta * out * (1.0 - out);
}

float random_weight() {
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

int main() {
    srand(time(NULL)); 

    //Initialize Weights Randomly
    for(int i=0; i<2; i++) {
        for(int j=0; j<3; j++) {
            v[i][j] = random_weight();
        }
    }
    for(int j=0; j<3; j++) {
        w[j] = random_weight();
    }

    y[2] = -1.0; 

    int iteraciones = 0;
    float total_error = 0.0;


    while(1) {
        total_error = 0.0;

        for(int p=0; p<4; p++) {
            
            //Calculate y1 e y2
            for(int k=0; k<2; k++) {
                float net = 0.0;

                for(int j=0; j<3; j++) { 
                    net += v[k][j] * train_inputs[p][j];
                }

                y[k] = sigmoid(net);
            }

            //Find the output z of the network
            float net_z = 0.0;

            for(int j=0; j<3; j++) {
                net_z += w[j] * y[j];
            }

            z = sigmoid(net_z);

            //Calculate the error of the output
            float d = train_outputs[p][0];
            total_error += 0.5 * (d - z) * (d - z);

            float delta_z = (d - z) * sigmoid_derivative(z);

            float delta_y[2];

            for(int k=0; k<2; k++) {
                delta_y[k] = sigmoid_derivative(y[k]) * (delta_z * w[k]);
            }

            //Modify the weights of output neuron
            for(int j=0; j<3; j++) {
                w[j] = w[j] + (eta * delta_z * y[j]);
            }

            //Modify the weights of the neurons in the first layer
            for(int k=0; k<2; k++) { 

                for(int j=0; j<3; j++) { 
                    v[k][j] = v[k][j] + (eta * delta_y[k] * train_inputs[p][j]);
                }
            }
        }

        iteraciones++;

        if(iteraciones % 1000 == 0) {
            printf("Epoch: %d, Error: %f\n", iteraciones, total_error);
        }

        if(total_error < eMax) {
            printf("\n---CONVERGENCE REACHED---\n");
            printf("Epoch: %d, Final Error: %f\n", iteraciones, total_error);
            break;
        }

        if(iteraciones > 100000) {
            printf("\n---FAILED CONVERGE---\n");
            break;
        }
    }

    printf("\n---VERIFICATION---\n");
    printf("X1  X2  | Target | Output\n");
    
    for(int p=0; p<4; p++) {

        for(int k=0; k<2; k++) {
            float net = 0.0;

            for(int j=0; j<3; j++) {
                net += v[k][j] * train_inputs[p][j];
            }

            y[k] = sigmoid(net);
        }

        float net_z = 0.0;

        for(int j=0; j<3; j++) {
            net_z += w[j] * y[j];
        }

        z = sigmoid(net_z);

        printf("%.0f   %.0f   |    %.0f   | %.4f\n", train_inputs[p][0], train_inputs[p][1], train_outputs[p][0], z);
    }

    return 0;
}
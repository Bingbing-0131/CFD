#include "Three_Order.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void ThreeOrder::Initialize(int mx_, double CFL_, double a_, std::string folder_) {
    mx = mx_;
    dx = 1.0 / mx;
    CFL = CFL_;
    a = a_;
    dt = CFL * dx / a;

    u_prev.resize(mx + 1);
    u_new.resize(mx + 1);
    
    for (int i = 0; i <= mx; i++) {
        double x = i * dx;
        u_prev[i] = Initial_condition(x);
        u_new[i] = u_prev[i];
    }

    folder = folder_;
}

double ThreeOrder::Initial_condition(double x) {
    // Shift domain to [-0.5, 0.5] for initial condition
    x -= 0.5;

    if (x >= -0.25 && x <= 0.25) return 1.0;
    else return 0.0;
}

void ThreeOrder::Boundary() {
    // Apply periodic boundary conditions
    u_prev[0] = u_prev[mx - 1];
    u_prev[mx] = u_prev[1];
}

void ThreeOrder::Forward() {
    for (int i = 0; i <= mx; i++) {
        int i_add_1 = (i + 1 + mx + 1) % (mx + 1);
        int i_minus_1 = (i - 1 + mx + 1) % (mx + 1);
        int i_minus_2 = (i - 2 + mx + 1) % (mx + 1);

        u_new[i] = u_prev[i] 
            - CFL / 6.0 * ((2 - CFL) * u_prev[i_add_1]
                          + 3 * (1 + CFL) * u_prev[i]
                          - (6 + 3 * CFL) * u_prev[i_minus_1]
                          + (1 + CFL) * u_prev[i_minus_2])
            + CFL * CFL / 6.0 * ((2 - CFL) * u_prev[i_add_1]
                                + (3 * CFL - 3) * u_prev[i]
                                - 3 * CFL * u_prev[i_minus_1]
                                + (1 + CFL) * u_prev[i_minus_2]);
    }
    u_prev = u_new;
    Boundary(); // Enforce periodic BCs
}


void ThreeOrder::Data(int step) {
    std::ostringstream filename;
    filename << folder << "/output_step_" << step << ".txt";

    std::ofstream file(filename.str());
    for (int i = 0; i <= mx; i++) {
        file << i * dx << " " << u_new[i] << "\n";
    }
    file.close();
}
